"""Hermetic differential witnesses for the public VFS contract.

The harness deliberately knows very little about a concrete VFS
implementation.  A surface adapter receives a canonical contract case and an
isolated filesystem fixture.  This keeps adapters for fsspec, managers,
buckets, and handlers small while keeping comparison, provenance, and cleanup
semantics identical.

Only explicitly enumerated transport representations are normalized.  Paths,
content identifiers, sizes, errors, state changes, and backend-selection
signals remain observable contract data.
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
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TypeAlias, runtime_checkable
from unittest import mock

from .vfs_contract_pack import (
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
    CanonicalVector,
    VfsContractPack,
    VfsErrorCode,
    VfsInvariantKind,
    VfsOperation,
    build_vfs_contract_pack,
)

VFS_DIFFERENTIAL_WITNESS_SCHEMA = "vfs/differential-contract-witness@1"
VFS_DIFFERENTIAL_TRACE_SCHEMA = "vfs/canonical-operation-trace@1"
VFS_DIFFERENTIAL_GOAL_ID = "VFS-G091"
VFS_DIFFERENTIAL_TASK_ID = "VFS-077"
VFS_DIFFERENTIAL_OBJECTIVE_REVISION = (
    "baguqeeraum7l4fdbqgbeprgieb3ffifd72a74vivgjjrpy5gdykb4ubuv5pa"
)
VFS_DIFFERENTIAL_PACKET_GOAL_IDS = ("VFS-G091", "VFS-G158")
VFS_DIFFERENTIAL_EVIDENCE_KINDS = (
    VFS_DIFFERENTIAL_WITNESS_SCHEMA,
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
)
MAX_TRACE_STEPS = 256


class VfsDifferentialHarnessError(ValueError):
    """Raised when a requested differential run is not safe or well formed."""


class SurfaceFamily(str, Enum):
    """Concrete public implementation families compared by the harness."""

    VFS = "vfs"
    FSSPEC = "fsspec"
    MANAGER = "manager"
    BUCKET = "bucket"
    HANDLER = "handler"


class PublicSurfaceKind(str, Enum):
    """Closed operation transports required by the differential objective."""

    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"
    MCP_PLUS_PLUS = "mcp++"
    HTTP = "http"
    LIBP2P = "libp2p"
    BACKEND = "backend"


REQUIRED_PUBLIC_SURFACES = tuple(PublicSurfaceKind)


class SurfaceAvailability(str, Enum):
    """Whether a surface can contribute authoritative evidence."""

    REAL = "real"
    MOCK = "mock"
    UNAVAILABLE = "unavailable"


class HermeticNetworkError(PermissionError):
    """Raised when a differential case attempts external network access."""


class ObservationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class DriftKind(str, Enum):
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


class NormalizationRule(str, Enum):
    """Closed set of contract-approved representation normalizations."""

    TRANSPORT_ENVELOPE = "transport_envelope"
    ERROR_ENVELOPE = "error_envelope"
    BYTES_LIKE = "bytes_like"
    EXPLICIT_UTF8_TEXT = "explicit_utf8_text"
    STAT_FIELD_ALIASES = "stat_field_aliases"


DEFAULT_NORMALIZATION_RULES = (
    NormalizationRule.TRANSPORT_ENVELOPE,
    NormalizationRule.ERROR_ENVELOPE,
    NormalizationRule.BYTES_LIKE,
    NormalizationRule.EXPLICIT_UTF8_TEXT,
    NormalizationRule.STAT_FIELD_ALIASES,
)


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _json_value(value: Any) -> JsonValue:
    """Produce lossless, deterministic JSON for provenance records."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise VfsDifferentialHarnessError("non-finite floats are not canonical")
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
                raise VfsDifferentialHarnessError(
                    f"canonical mappings require string keys, got {type(key).__name__}"
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
        raise VfsDifferentialHarnessError("record must be an object")
    return converted


@dataclass(frozen=True)
class TraceStep:
    vector_id: str
    operation: VfsOperation
    description: str
    request: Mapping[str, Any]
    expected: Mapping[str, Any]
    invariant_ids: tuple[str, ...]
    source_contract_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.vector_id, str) or not self.vector_id.strip():
            raise VfsDifferentialHarnessError("trace vector_id must be non-empty")
        if not isinstance(self.operation, VfsOperation):
            raise VfsDifferentialHarnessError(
                "trace operation must be a VfsOperation"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise VfsDifferentialHarnessError(
                "trace description must be non-empty"
            )
        if not isinstance(self.request, Mapping) or not isinstance(
            self.expected, Mapping
        ):
            raise VfsDifferentialHarnessError(
                "trace request and expected values must be mappings"
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
    def from_vector(cls, vector: CanonicalVector) -> "TraceStep":
        return cls(
            vector_id=vector.vector_id,
            operation=vector.operation,
            description=vector.description,
            request=vector.request,
            expected=vector.expected,
            invariant_ids=vector.invariant_ids,
            source_contract_ids=vector.source_contract_ids,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "operation": self.operation.value,
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
    schema: str = VFS_DIFFERENTIAL_TRACE_SCHEMA
    operation_matrix_schema: str = VFS_CANONICAL_OPERATION_MATRIX_SCHEMA

    def __post_init__(self) -> None:
        if not self.steps:
            raise VfsDifferentialHarnessError("a differential trace cannot be empty")
        if len(self.steps) > MAX_TRACE_STEPS:
            raise VfsDifferentialHarnessError(
                f"trace has {len(self.steps)} steps; maximum is {MAX_TRACE_STEPS}"
            )
        ids = [step.vector_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise VfsDifferentialHarnessError("trace vector_ids must be unique")
        if self.schema != VFS_DIFFERENTIAL_TRACE_SCHEMA:
            raise VfsDifferentialHarnessError(
                f"unsupported differential trace schema: {self.schema!r}"
            )
        if self.operation_matrix_schema != VFS_CANONICAL_OPERATION_MATRIX_SCHEMA:
            raise VfsDifferentialHarnessError(
                "unsupported canonical operation matrix schema: "
                f"{self.operation_matrix_schema!r}"
            )
        if not isinstance(self.contract_pack_cid, str) or not self.contract_pack_cid:
            raise VfsDifferentialHarnessError(
                "trace contract_pack_cid must be non-empty"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "operation_matrix_schema": self.operation_matrix_schema,
            "contract_pack_cid": self.contract_pack_cid,
            "steps": [step.to_record() for step in self.steps],
        }

    @property
    def content_id(self) -> str:
        return _content_id(self.to_record())


def build_canonical_operation_trace(
    pack: VfsContractPack | None = None,
    *,
    vector_ids: Iterable[str] | None = None,
) -> CanonicalOperationTrace:
    """Build a finite, deterministic trace from the normative contract pack."""

    selected_pack = pack or build_vfs_contract_pack()
    requested = None if vector_ids is None else tuple(vector_ids)
    if requested is not None:
        if not requested:
            raise VfsDifferentialHarnessError("vector_ids cannot be empty")
        if len(requested) != len(set(requested)):
            raise VfsDifferentialHarnessError("vector_ids cannot contain duplicates")
        available = {vector.vector_id: vector for vector in selected_pack.vectors}
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise VfsDifferentialHarnessError(
                f"unknown contract vector_ids: {', '.join(unknown)}"
            )
        vectors = tuple(available[vector_id] for vector_id in requested)
    else:
        vectors = selected_pack.vectors
    return CanonicalOperationTrace(
        steps=tuple(TraceStep.from_vector(vector) for vector in vectors),
        contract_pack_cid=selected_pack.content_id,
    )


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
            raise VfsDifferentialHarnessError(
                f"fixture path must be relative and contained: {self.path!r}"
            )
        if self.path != unicodedata.normalize("NFC", self.path):
            raise VfsDifferentialHarnessError(
                f"fixture paths must be NFC canonical: {self.path!r}"
            )
        if self.kind not in {"file", "directory"}:
            raise VfsDifferentialHarnessError(
                f"unsupported fixture entry kind: {self.kind!r}"
            )
        if self.kind == "file":
            if self.content_hex is None:
                raise VfsDifferentialHarnessError("fixture files require content_hex")
            try:
                decoded = bytes.fromhex(self.content_hex)
            except ValueError as exc:
                raise VfsDifferentialHarnessError(
                    f"invalid fixture content hex for {self.path!r}"
                ) from exc
            if decoded.hex() != self.content_hex:
                raise VfsDifferentialHarnessError(
                    f"fixture content hex must be canonical lowercase: {self.path!r}"
                )
        elif self.content_hex is not None:
            raise VfsDifferentialHarnessError(
                "fixture directories cannot declare content"
            )
        if (
            not isinstance(self.mode, int)
            or isinstance(self.mode, bool)
            or self.mode < 0
            or self.mode > 0o777
        ):
            raise VfsDifferentialHarnessError(
                f"fixture mode must be an integer from 0o000 to 0o777: {self.mode!r}"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "content_hex": self.content_hex,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    entries: tuple[FixtureEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, str) or not self.fixture_id.strip():
            raise VfsDifferentialHarnessError("fixture_id must be non-empty")
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise VfsDifferentialHarnessError("fixture paths must be unique")
        file_paths = {
            PurePosixPath(entry.path)
            for entry in self.entries
            if entry.kind == "file"
        }
        for entry in self.entries:
            parent = PurePosixPath(entry.path).parent
            while str(parent) != ".":
                if parent in file_paths:
                    raise VfsDifferentialHarnessError(
                        f"fixture file {parent} cannot contain {entry.path}"
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


def build_default_fixture() -> FixtureSpec:
    """Return the compact fixture shared by every canonical trace case."""

    entries = (
        FixtureEntry("a", "directory", mode=0o700),
        FixtureEntry("a/x", "file", "78"),
        FixtureEntry("café", "directory", mode=0o700),
        FixtureEntry("café/data", "file", "63616665"),
        FixtureEntry("dir", "directory", mode=0o700),
        FixtureEntry("dir/child", "file", "6368696c64"),
        FixtureEntry("hello.txt", "file", "68656c6c6f0a"),
        FixtureEntry("many", "directory", mode=0o700),
        FixtureEntry("many/a", "file", "61"),
        FixtureEntry("many/b", "file", "62"),
        FixtureEntry("many/c", "file", "63"),
        FixtureEntry("secret", "file", "746f702d736563726574", mode=0o600),
    )
    return FixtureSpec(fixture_id="vfs-differential-default@1", entries=entries)


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
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
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
                        {"media_type": "application/octet-stream", "hex": content.hex()}
                    ),
                }
            )
        else:
            records.append({"path": relative, "kind": "special", "mode": mode})
    return TreeSnapshot(entries=tuple(records), content_id=_content_id(records))


@dataclass
class SurfaceRunContext:
    """Capabilities exposed to an adapter for one isolated trace case."""

    root: Path
    fixture: FixtureSpec
    step: TraceStep
    state: dict[str, Any] = field(default_factory=dict)
    network_allowed: bool = False

    def resolve_path(self, path: str, *, allow_root: bool = True) -> Path:
        if not isinstance(path, str):
            raise VfsDifferentialHarnessError("surface paths must be strings")
        if "\x00" in path:
            raise VfsDifferentialHarnessError("surface paths cannot contain NUL")
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
    family: SurfaceFamily
    availability: SurfaceAvailability
    implementation: str
    public_surface: PublicSurfaceKind | str
    package_names: tuple[str, ...]
    unavailable_reason: str | None

    def execute(
        self, step: TraceStep, context: SurfaceRunContext
    ) -> Any | Awaitable[Any]: ...


@dataclass(frozen=True)
class CallableSurfaceAdapter:
    """Adapt a real callable, a declared mock, or an unavailable surface."""

    surface_id: str
    family: SurfaceFamily
    executor: SurfaceExecutor | None
    implementation: str
    public_surface: PublicSurfaceKind | str = PublicSurfaceKind.PYTHON
    availability: SurfaceAvailability = SurfaceAvailability.REAL
    package_names: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise VfsDifferentialHarnessError("surface_id must be non-empty")
        if not isinstance(self.family, SurfaceFamily):
            raise VfsDifferentialHarnessError(
                "surface family must be a SurfaceFamily"
            )
        if not isinstance(self.availability, SurfaceAvailability):
            raise VfsDifferentialHarnessError(
                "surface availability must be a SurfaceAvailability"
            )
        if (
            not isinstance(self.implementation, str)
            or not self.implementation.strip()
        ):
            raise VfsDifferentialHarnessError(
                "surface implementation must be non-empty"
            )
        try:
            public_surface = PublicSurfaceKind(self.public_surface)
        except (TypeError, ValueError) as exc:
            raise VfsDifferentialHarnessError(
                f"unsupported public surface: {self.public_surface!r}"
            ) from exc
        object.__setattr__(self, "public_surface", public_surface)
        if self.availability is SurfaceAvailability.UNAVAILABLE:
            if self.executor is not None:
                raise VfsDifferentialHarnessError(
                    "unavailable surfaces cannot have an executor"
                )
            if not self.unavailable_reason:
                raise VfsDifferentialHarnessError(
                    "unavailable surfaces require an unavailable_reason"
                )
        elif self.executor is None:
            raise VfsDifferentialHarnessError(
                "real and mock surfaces require an executor"
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
        family: SurfaceFamily,
        *,
        implementation: str,
        reason: str,
        public_surface: PublicSurfaceKind | str = PublicSurfaceKind.PYTHON,
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

    def execute(
        self, step: TraceStep, context: SurfaceRunContext
    ) -> Any | Awaitable[Any]:
        if self.executor is None:  # pragma: no cover - validated and skipped
            raise RuntimeError("unavailable surface cannot execute")
        return self.executor(step, context)


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


def _error_code_for_exception(error: BaseException) -> str:
    if isinstance(error, PermissionError):
        return VfsErrorCode.PERMISSION_DENIED.value
    if isinstance(error, FileNotFoundError):
        return VfsErrorCode.NOT_FOUND.value
    if isinstance(error, FileExistsError):
        return VfsErrorCode.ALREADY_EXISTS.value
    if isinstance(error, IsADirectoryError):
        return VfsErrorCode.NOT_A_FILE.value
    if isinstance(error, NotADirectoryError):
        return VfsErrorCode.NOT_A_DIRECTORY.value
    if isinstance(error, TimeoutError):
        return VfsErrorCode.DEADLINE_EXCEEDED.value
    if isinstance(error, NotImplementedError):
        return VfsErrorCode.UNSUPPORTED.value
    if isinstance(error, asyncio.CancelledError):
        return VfsErrorCode.CANCELLED.value
    if isinstance(error, (TypeError, ValueError)):
        return VfsErrorCode.INVALID_ARGUMENT.value
    if isinstance(error, OSError):
        by_errno = {
            value: code
            for symbol, code in (
                ("EACCES", VfsErrorCode.PERMISSION_DENIED.value),
                ("EPERM", VfsErrorCode.PERMISSION_DENIED.value),
                ("ENOENT", VfsErrorCode.NOT_FOUND.value),
                ("EEXIST", VfsErrorCode.ALREADY_EXISTS.value),
                ("EISDIR", VfsErrorCode.NOT_A_FILE.value),
                ("ENOTDIR", VfsErrorCode.NOT_A_DIRECTORY.value),
                ("ENOTEMPTY", VfsErrorCode.DIRECTORY_NOT_EMPTY.value),
                ("EINVAL", VfsErrorCode.INVALID_ARGUMENT.value),
                ("ENOSPC", VfsErrorCode.RESOURCE_EXHAUSTED.value),
                ("EDQUOT", VfsErrorCode.RESOURCE_EXHAUSTED.value),
                ("ETIMEDOUT", VfsErrorCode.DEADLINE_EXCEEDED.value),
            )
            if (value := getattr(errno_module, symbol, None)) is not None
        }
        if error.errno in by_errno:
            return by_errno[error.errno]
        unsupported_errnos = {
            getattr(errno_module, name)
            for name in ("ENOSYS", "ENOTSUP", "EOPNOTSUPP")
            if hasattr(errno_module, name)
        }
        if error.errno in unsupported_errnos:
            return VfsErrorCode.UNSUPPORTED.value
    return VfsErrorCode.IO_FAILURE.value


def _exception_identity(error: BaseException) -> ErrorIdentity:
    code = _error_code_for_exception(error)
    errno = getattr(error, "errno", None)
    record = {
        "code": code,
        "exception_module": type(error).__module__,
        "exception_type": type(error).__qualname__,
        "message": str(error),
        "errno": errno if isinstance(errno, int) else None,
    }
    return ErrorIdentity(**record, content_id=_content_id(record))


def _reported_error_identity(value: Mapping[str, Any]) -> ErrorIdentity | None:
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
        code = str(error.get("code", VfsErrorCode.IO_FAILURE.value))
        message = error.get("message")
    elif isinstance(error, str):
        code = error
        message = candidate.get("message")
    else:
        code = str(candidate.get("code", VfsErrorCode.IO_FAILURE.value))
        message = candidate.get("message")
    record = {
        "code": code,
        "exception_module": None,
        "exception_type": None,
        "message": None if message is None else str(message),
        "errno": None,
    }
    return ErrorIdentity(**record, content_id=_content_id(record))


def _has_invariant(step: TraceStep, invariant: VfsInvariantKind) -> bool:
    """Match both enum values and the contract pack's canonical identifiers."""

    value = invariant.value
    return value in step.invariant_ids or f"invariant:{value}" in step.invariant_ids


def _normalization_rules_for(step: TraceStep) -> tuple[NormalizationRule, ...]:
    rules = list(DEFAULT_NORMALIZATION_RULES)
    if not _has_invariant(step, VfsInvariantKind.STAT_LIST):
        rules.remove(NormalizationRule.STAT_FIELD_ALIASES)
    if not _has_invariant(step, VfsInvariantKind.BYTES_TEXT):
        rules.remove(NormalizationRule.EXPLICIT_UTF8_TEXT)
    return tuple(rules)


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


def _normalize_value(value: Any, rules: set[NormalizationRule]) -> JsonValue:
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
            str(key): _normalize_value(item, rules) for key, item in value.items()
        }
        if NormalizationRule.ERROR_ENVELOPE in rules and "error" in result:
            error = result["error"]
            if isinstance(error, str):
                result["error"] = {"code": error}
            elif isinstance(error, Mapping) and "code" not in error:
                result["error"] = {
                    "code": str(error.get("type", VfsErrorCode.IO_FAILURE.value)),
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
        return [_normalize_value(item, rules) for item in value]
    return _json_value(value)


def normalize_contract_result(
    step: TraceStep,
    value: Any,
    *,
    rules: Iterable[NormalizationRule] | None = None,
) -> JsonValue:
    """Normalize only the closed set of approved transport representations."""

    selected = set(_normalization_rules_for(step) if rules is None else rules)
    return _normalize_value(value, selected)


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


def _await_result(value: Any | Awaitable[Any]) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: list[Any] = []
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as exc:  # preserved for exact error identity
            failure.append(exc)

    thread = threading.Thread(target=run, name="vfs-differential-await", daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0]


_NETWORK_GUARD_LOCK = threading.RLock()


@contextmanager
def _deny_network() -> Iterable[None]:
    """Deny common in-process socket paths while a surface case executes."""

    def denied(*_args: Any, **_kwargs: Any) -> Any:
        raise HermeticNetworkError(
            "network access is disabled in differential VFS fixtures"
        )

    with _NETWORK_GUARD_LOCK:
        with (
            mock.patch.object(socket.socket, "connect", denied),
            mock.patch.object(socket.socket, "connect_ex", denied),
            mock.patch.object(socket.socket, "sendto", denied),
            mock.patch.object(socket, "create_connection", denied),
        ):
            yield


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
    family: SurfaceFamily
    availability: SurfaceAvailability
    authoritative: bool
    implementation: str
    public_surface: PublicSurfaceKind
    unavailable_reason: str | None
    runtime: RuntimeIdentity
    implementation_identity: ImplementationIdentity
    observations: tuple[SurfaceObservation, ...]
    content_id: str

    def to_record(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "family": self.family.value,
            "availability": self.availability.value,
            "authoritative": self.authoritative,
            "implementation": self.implementation,
            "public_surface": self.public_surface.value,
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
class WitnessBindings:
    """Exact content bindings needed to reproduce and audit a witness."""

    contract_pack_cid: str
    operation_trace_cid: str
    fixture_spec_cid: str
    fixture_snapshot_cids: tuple[str, ...]
    toolchain_cids: Mapping[str, str]
    implementation_cids: Mapping[str, str]
    surface_run_cids: Mapping[str, str]
    content_id: str

    def __post_init__(self) -> None:
        scalar_cids = {
            "contract_pack_cid": self.contract_pack_cid,
            "operation_trace_cid": self.operation_trace_cid,
            "fixture_spec_cid": self.fixture_spec_cid,
            "content_id": self.content_id,
        }
        for field_name, value in scalar_cids.items():
            if not isinstance(value, str) or not value:
                raise VfsDifferentialHarnessError(
                    f"witness binding {field_name} must be non-empty"
                )
        snapshots = tuple(sorted(set(self.fixture_snapshot_cids)))
        if snapshots != self.fixture_snapshot_cids:
            raise VfsDifferentialHarnessError(
                "fixture snapshot CIDs must be unique and sorted"
            )
        mappings = (
            ("toolchain_cids", "toolchain", self.toolchain_cids),
            ("implementation_cids", "implementation", self.implementation_cids),
            ("surface_run_cids", "surface run", self.surface_run_cids),
        )
        surface_ids: set[str] | None = None
        for field_name, label, values in mappings:
            if not isinstance(values, Mapping):
                raise VfsDifferentialHarnessError(
                    f"{label} CID bindings must be a mapping"
                )
            normalized = {key: value for key, value in sorted(values.items())}
            if any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, str)
                or not value
                for key, value in normalized.items()
            ):
                raise VfsDifferentialHarnessError(
                    f"{label} CID bindings require non-empty string pairs"
                )
            object.__setattr__(self, field_name, normalized)
            current_ids = set(normalized)
            if surface_ids is None:
                surface_ids = current_ids
            elif current_ids != surface_ids:
                raise VfsDifferentialHarnessError(
                    "toolchain, implementation, and surface run bindings "
                    "must cover identical surface IDs"
                )
        expected_content_id = _content_id(self._record_without_cid())
        if self.content_id != expected_content_id:
            raise VfsDifferentialHarnessError(
                "witness binding CID does not match its exact dependencies"
            )

    def _record_without_cid(self) -> dict[str, Any]:
        return {
            "contract_pack_cid": self.contract_pack_cid,
            "operation_trace_cid": self.operation_trace_cid,
            "fixture_spec_cid": self.fixture_spec_cid,
            "fixture_snapshot_cids": list(self.fixture_snapshot_cids),
            "toolchain_cids": dict(self.toolchain_cids),
            "implementation_cids": dict(self.implementation_cids),
            "surface_run_cids": dict(self.surface_run_cids),
        }

    def to_record(self) -> dict[str, Any]:
        return {**self._record_without_cid(), "cid": self.content_id}


def _drift_kinds(step: TraceStep, observation: SurfaceObservation) -> tuple[DriftKind, ...]:
    kinds: set[DriftKind] = set()
    if any(
        _has_invariant(step, invariant)
        for invariant in (
            VfsInvariantKind.VERSIONED_PATH,
            VfsInvariantKind.UNICODE,
            VfsInvariantKind.ROOT,
            VfsInvariantKind.TRAVERSAL,
            VfsInvariantKind.MOUNT,
        )
    ):
        kinds.add(DriftKind.PATH)
    if _has_invariant(step, VfsInvariantKind.BYTES_TEXT):
        kinds.add(DriftKind.BYTES_TEXT)
    if _has_invariant(step, VfsInvariantKind.STAT_LIST):
        kinds.add(DriftKind.STAT_LIST)
    if _has_invariant(step, VfsInvariantKind.ATOMICITY):
        kinds.add(DriftKind.RENAME_ATOMICITY)
    if _has_invariant(step, VfsInvariantKind.JOURNAL_REPLAY):
        kinds.add(DriftKind.JOURNAL)
    if _has_invariant(step, VfsInvariantKind.CACHE_PIN_COHERENCE):
        kinds.add(DriftKind.CACHE)
    if _has_invariant(step, VfsInvariantKind.AUTHORIZATION):
        kinds.add(DriftKind.AUTHORIZATION)
    if any(
        _has_invariant(step, invariant)
        for invariant in (
            VfsInvariantKind.BACKEND_NEGOTIATION,
            VfsInvariantKind.DEGRADATION,
        )
    ):
        kinds.add(DriftKind.FALLBACK)
    if observation.silent_success:
        kinds.add(DriftKind.SILENT_SUCCESS)
    if observation.error is not None:
        kinds.add(DriftKind.ERROR)
    else:
        kinds.add(DriftKind.RESULT)
    return tuple(sorted(kinds, key=lambda kind: kind.value))


@dataclass(frozen=True)
class DifferentialWitness:
    schema: str
    goal_id: str
    task_id: str
    objective_revision: str
    evidence_kinds: tuple[str, ...]
    goal_ids: tuple[str, ...]
    trace: CanonicalOperationTrace
    fixture: FixtureSpec
    surface_runs: tuple[SurfaceRun, ...]
    findings: tuple[DriftFinding, ...]
    authoritative_surface_ids: tuple[str, ...]
    non_authoritative_surface_ids: tuple[str, ...]
    unavailable_surface_ids: tuple[str, ...]
    authoritative_agreement: bool
    all_cleanup_succeeded: bool
    bindings: WitnessBindings
    content_id: str

    def __post_init__(self) -> None:
        expected_identity = (
            (self.schema, VFS_DIFFERENTIAL_WITNESS_SCHEMA, "schema"),
            (self.goal_id, VFS_DIFFERENTIAL_GOAL_ID, "goal_id"),
            (self.task_id, VFS_DIFFERENTIAL_TASK_ID, "task_id"),
            (
                self.objective_revision,
                VFS_DIFFERENTIAL_OBJECTIVE_REVISION,
                "objective_revision",
            ),
            (
                self.evidence_kinds,
                VFS_DIFFERENTIAL_EVIDENCE_KINDS,
                "evidence_kinds",
            ),
            (
                self.goal_ids,
                VFS_DIFFERENTIAL_PACKET_GOAL_IDS,
                "goal_ids",
            ),
        )
        for actual, expected, field_name in expected_identity:
            if actual != expected:
                raise VfsDifferentialHarnessError(
                    f"{field_name} must be {expected!r}, got {actual!r}"
                )
        expected_authoritative_ids = tuple(
            run.surface_id
            for run in self.surface_runs
            if run.availability is SurfaceAvailability.REAL
        )
        expected_non_authoritative_ids = tuple(
            run.surface_id
            for run in self.surface_runs
            if run.availability is SurfaceAvailability.MOCK
        )
        expected_unavailable_ids = tuple(
            run.surface_id
            for run in self.surface_runs
            if run.availability is SurfaceAvailability.UNAVAILABLE
        )
        expected_cleanup = all(
            observation.cleanup.succeeded
            for run in self.surface_runs
            for observation in run.observations
        )
        expected_agreement = (
            bool(expected_authoritative_ids)
            and not any(finding.authoritative for finding in self.findings)
            and expected_cleanup
        )
        expected_classification = (
            (
                self.authoritative_surface_ids,
                expected_authoritative_ids,
                "authoritative surface IDs",
            ),
            (
                self.non_authoritative_surface_ids,
                expected_non_authoritative_ids,
                "non-authoritative surface IDs",
            ),
            (
                self.unavailable_surface_ids,
                expected_unavailable_ids,
                "unavailable surface IDs",
            ),
            (
                self.all_cleanup_succeeded,
                expected_cleanup,
                "cleanup status",
            ),
            (
                self.authoritative_agreement,
                expected_agreement,
                "authoritative agreement",
            ),
        )
        for actual, expected, label in expected_classification:
            if actual != expected:
                raise VfsDifferentialHarnessError(
                    f"witness {label} does not match its surface runs"
                )
        expected_toolchains = {
            run.surface_id: run.runtime.content_id for run in self.surface_runs
        }
        expected_implementations = {
            run.surface_id: run.implementation_identity.content_id
            for run in self.surface_runs
        }
        expected_runs = {
            run.surface_id: run.content_id for run in self.surface_runs
        }
        expected_snapshots = tuple(
            sorted(
                {
                    observation.fixture_before_cid
                    for run in self.surface_runs
                    for observation in run.observations
                }
            )
        )
        expected_bindings = (
            (
                self.bindings.contract_pack_cid,
                self.trace.contract_pack_cid,
                "contract pack",
            ),
            (
                self.bindings.operation_trace_cid,
                self.trace.content_id,
                "operation trace",
            ),
            (
                self.bindings.fixture_spec_cid,
                self.fixture.content_id,
                "fixture spec",
            ),
            (
                self.bindings.fixture_snapshot_cids,
                expected_snapshots,
                "fixture snapshots",
            ),
            (
                dict(self.bindings.toolchain_cids),
                expected_toolchains,
                "toolchains",
            ),
            (
                dict(self.bindings.implementation_cids),
                expected_implementations,
                "implementations",
            ),
            (
                dict(self.bindings.surface_run_cids),
                expected_runs,
                "surface runs",
            ),
        )
        for actual, expected, label in expected_bindings:
            if actual != expected:
                raise VfsDifferentialHarnessError(
                    f"witness {label} are not bound to the recorded run"
                )
        if any(
            observation.fixture_spec_cid != self.fixture.content_id
            for run in self.surface_runs
            for observation in run.observations
        ):
            raise VfsDifferentialHarnessError(
                "an observation is bound to a different fixture recipe"
            )
        unsigned_record = self.to_record()
        unsigned_record.pop("cid")
        if self.content_id != _content_id(unsigned_record):
            raise VfsDifferentialHarnessError(
                "differential witness CID does not match its record"
            )

    @property
    def observed_public_surfaces(self) -> tuple[PublicSurfaceKind, ...]:
        return tuple(
            surface
            for surface in REQUIRED_PUBLIC_SURFACES
            if any(
                run.public_surface is surface
                and run.availability is SurfaceAvailability.REAL
                for run in self.surface_runs
            )
        )

    @property
    def missing_public_surfaces(self) -> tuple[PublicSurfaceKind, ...]:
        observed = set(self.observed_public_surfaces)
        return tuple(
            surface for surface in REQUIRED_PUBLIC_SURFACES if surface not in observed
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_kinds": list(self.evidence_kinds),
            "goal_id": self.goal_id,
            "goal_ids": list(self.goal_ids),
            "task_id": self.task_id,
            "objective_revision": self.objective_revision,
            "authority": {
                "completion": False,
                "correctness": False,
                "repair": False,
            },
            "coverage": {
                "required_public_surfaces": [
                    surface.value for surface in REQUIRED_PUBLIC_SURFACES
                ],
                "observed_public_surfaces": [
                    surface.value for surface in self.observed_public_surfaces
                ],
                "missing_public_surfaces": [
                    surface.value for surface in self.missing_public_surfaces
                ],
                "public_surface_coverage_complete": (
                    not self.missing_public_surfaces
                ),
            },
            "bindings": self.bindings.to_record(),
            "trace": {**self.trace.to_record(), "cid": self.trace.content_id},
            "fixture": {**self.fixture.to_record(), "cid": self.fixture.content_id},
            "surface_runs": [run.to_record() for run in self.surface_runs],
            "findings": [finding.to_record() for finding in self.findings],
            "authoritative_surface_ids": list(self.authoritative_surface_ids),
            "non_authoritative_surface_ids": list(
                self.non_authoritative_surface_ids
            ),
            "unavailable_surface_ids": list(self.unavailable_surface_ids),
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
    fixture: FixtureSpec,
    *,
    temp_parent: Path | None,
) -> SurfaceObservation:
    safe_surface_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in adapter.surface_id[:24]
    )
    root = Path(
        tempfile.mkdtemp(
            prefix=f"vfs-diff-{safe_surface_id}-",
            dir=None if temp_parent is None else str(temp_parent),
        )
    )
    # mkdtemp creates the directory; FixtureSpec owns creation so remove only
    # this empty, resolved directory before materialization.
    root.rmdir()
    fixture_before_cid = fixture.materialize(root)
    expected_fixture_cid = snapshot_tree(root).content_id
    if fixture_before_cid != expected_fixture_cid:  # pragma: no cover - guard
        raise VfsDifferentialHarnessError("fixture snapshot was not reproducible")

    raw_value: Any = None
    raw_record: JsonValue | None = None
    error_identity: ErrorIdentity | None = None
    status = ObservationStatus.SUCCESS
    explicit_success_with_error = False
    context = SurfaceRunContext(root=root, fixture=fixture, step=step)
    try:
        with _deny_network():
            raw_value = _await_result(adapter.execute(step, context))
        raw_record = _json_value(raw_value)
        if isinstance(raw_value, Mapping):
            explicit_success_with_error = (
                (raw_value.get("ok") is True or raw_value.get("success") is True)
                and raw_value.get("error") is not None
            )
            error_identity = _reported_error_identity(raw_value)
            if error_identity is not None:
                status = ObservationStatus.ERROR
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            before_cleanup = snapshot_tree(root).content_id
            _cleanup(root, before_cleanup)
            raise
        error_identity = _exception_identity(exc)
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
    rules = _normalization_rules_for(step)
    normalized_expected = normalize_contract_result(step, step.expected, rules=rules)
    normalized_result = normalize_contract_result(step, raw_record, rules=rules)
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
        "operation": step.operation.value,
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
        "contract_match": not mismatch_paths and cleanup.succeeded,
        "mismatch_paths": list(mismatch_paths),
        "silent_success": silent_success,
        "cleanup": cleanup.to_record(),
    }
    return SurfaceObservation(
        surface_id=adapter.surface_id,
        vector_id=step.vector_id,
        operation=step.operation.value,
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
        contract_match=not mismatch_paths and cleanup.succeeded,
        mismatch_paths=mismatch_paths,
        silent_success=silent_success,
        cleanup=cleanup,
        content_id=_content_id(record_without_cid),
    )


def _surface_run(
    adapter: SurfaceAdapter,
    trace: CanonicalOperationTrace,
    fixture: FixtureSpec,
    *,
    temp_parent: Path | None,
) -> SurfaceRun:
    try:
        public_surface = PublicSurfaceKind(adapter.public_surface)
    except (TypeError, ValueError) as exc:
        raise VfsDifferentialHarnessError(
            f"unsupported public surface: {adapter.public_surface!r}"
        ) from exc
    runtime = capture_runtime_identity(adapter.package_names)
    implementation = _implementation_identity(adapter)
    observations: tuple[SurfaceObservation, ...]
    if adapter.availability is SurfaceAvailability.UNAVAILABLE:
        observations = ()
    else:
        observations = tuple(
            _observation_for_step(
                adapter, step, fixture, temp_parent=temp_parent
            )
            for step in trace.steps
        )
    record = {
        "surface_id": adapter.surface_id,
        "family": adapter.family.value,
        "availability": adapter.availability.value,
        "authoritative": adapter.availability is SurfaceAvailability.REAL,
        "implementation": adapter.implementation,
        "public_surface": public_surface.value,
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
        public_surface=public_surface,
        unavailable_reason=adapter.unavailable_reason,
        runtime=runtime,
        implementation_identity=implementation,
        observations=observations,
        content_id=_content_id(record),
    )


def _build_witness_bindings(
    trace: CanonicalOperationTrace,
    fixture: FixtureSpec,
    runs: Sequence[SurfaceRun],
) -> WitnessBindings:
    record = {
        "contract_pack_cid": trace.contract_pack_cid,
        "operation_trace_cid": trace.content_id,
        "fixture_spec_cid": fixture.content_id,
        "fixture_snapshot_cids": sorted(
            {
                observation.fixture_before_cid
                for run in runs
                for observation in run.observations
            }
        ),
        "toolchain_cids": {
            run.surface_id: run.runtime.content_id
            for run in sorted(runs, key=lambda item: item.surface_id)
        },
        "implementation_cids": {
            run.surface_id: run.implementation_identity.content_id
            for run in sorted(runs, key=lambda item: item.surface_id)
        },
        "surface_run_cids": {
            run.surface_id: run.content_id
            for run in sorted(runs, key=lambda item: item.surface_id)
        },
    }
    return WitnessBindings(
        contract_pack_cid=trace.contract_pack_cid,
        operation_trace_cid=trace.content_id,
        fixture_spec_cid=fixture.content_id,
        fixture_snapshot_cids=tuple(record["fixture_snapshot_cids"]),
        toolchain_cids=record["toolchain_cids"],
        implementation_cids=record["implementation_cids"],
        surface_run_cids=record["surface_run_cids"],
        content_id=_content_id(record),
    )


def _build_findings(
    trace: CanonicalOperationTrace, runs: Sequence[SurfaceRun]
) -> tuple[DriftFinding, ...]:
    by_step = {step.vector_id: step for step in trace.steps}
    findings: list[DriftFinding] = []
    for run in runs:
        for observation in run.observations:
            if observation.contract_match:
                continue
            step = by_step[observation.vector_id]
            expected = normalize_contract_result(step, step.expected)
            observed_cid = observation.normalized_result_cid or _content_id(None)
            record = {
                "vector_id": step.vector_id,
                "operation": step.operation.value,
                "kinds": [
                    kind.value for kind in _drift_kinds(step, observation)
                ],
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
                    operation=step.operation.value,
                    kinds=_drift_kinds(step, observation),
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
                expected = normalize_contract_result(step, step.expected)
                observed_cids = (
                    left_observation.normalized_result_cid or _content_id(None),
                    right_observation.normalized_result_cid or _content_id(None),
                )
                kinds = tuple(
                    sorted(
                        set(_drift_kinds(step, left_observation))
                        | set(_drift_kinds(step, right_observation)),
                        key=lambda kind: kind.value,
                    )
                )
                record = {
                    "vector_id": step.vector_id,
                    "operation": step.operation.value,
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
                        operation=step.operation.value,
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


def run_vfs_differential_harness(
    surfaces: Iterable[SurfaceAdapter],
    *,
    trace: CanonicalOperationTrace | None = None,
    fixture: FixtureSpec | None = None,
    temp_parent: str | os.PathLike[str] | None = None,
) -> DifferentialWitness:
    """Execute selected surfaces and return a self-identifying witness.

    Real surfaces are authoritative for drift.  Declared mocks execute and are
    retained as non-authoritative observations.  Unavailable surfaces never
    execute and remain explicit in the witness.
    """

    adapters = tuple(surfaces)
    if not adapters:
        raise VfsDifferentialHarnessError("at least one surface must be selected")
    for adapter in adapters:
        if not isinstance(adapter, SurfaceAdapter):
            raise VfsDifferentialHarnessError(
                f"surface {adapter!r} does not implement SurfaceAdapter"
            )
    ids = [adapter.surface_id for adapter in adapters]
    if len(ids) != len(set(ids)):
        raise VfsDifferentialHarnessError("surface_ids must be unique")
    selected_trace = trace or build_canonical_operation_trace()
    selected_fixture = fixture or build_default_fixture()
    parent = None if temp_parent is None else Path(temp_parent).resolve()
    if parent is not None:
        if not parent.is_dir():
            raise VfsDifferentialHarnessError(
                f"temp_parent is not an existing directory: {parent}"
            )

    runs = tuple(
        _surface_run(
            adapter, selected_trace, selected_fixture, temp_parent=parent
        )
        for adapter in adapters
    )
    findings = _build_findings(selected_trace, runs)
    bindings = _build_witness_bindings(selected_trace, selected_fixture, runs)
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
    all_cleanup = all(
        observation.cleanup.succeeded
        for run in runs
        for observation in run.observations
    )
    agreement = bool(authoritative_ids) and all(
        not finding.authoritative for finding in findings
    ) and all_cleanup
    observed_public_surfaces = tuple(
        surface
        for surface in REQUIRED_PUBLIC_SURFACES
        if any(
            run.public_surface is surface
            and run.availability is SurfaceAvailability.REAL
            for run in runs
        )
    )
    missing_public_surfaces = tuple(
        surface
        for surface in REQUIRED_PUBLIC_SURFACES
        if surface not in set(observed_public_surfaces)
    )
    record = {
        "schema": VFS_DIFFERENTIAL_WITNESS_SCHEMA,
        "evidence_kinds": list(VFS_DIFFERENTIAL_EVIDENCE_KINDS),
        "goal_id": VFS_DIFFERENTIAL_GOAL_ID,
        "goal_ids": list(VFS_DIFFERENTIAL_PACKET_GOAL_IDS),
        "task_id": VFS_DIFFERENTIAL_TASK_ID,
        "objective_revision": VFS_DIFFERENTIAL_OBJECTIVE_REVISION,
        "authority": {"completion": False, "correctness": False, "repair": False},
        "coverage": {
            "required_public_surfaces": [
                surface.value for surface in REQUIRED_PUBLIC_SURFACES
            ],
            "observed_public_surfaces": [
                surface.value for surface in observed_public_surfaces
            ],
            "missing_public_surfaces": [
                surface.value for surface in missing_public_surfaces
            ],
            "public_surface_coverage_complete": not missing_public_surfaces,
        },
        "bindings": bindings.to_record(),
        "trace": {**selected_trace.to_record(), "cid": selected_trace.content_id},
        "fixture": {**selected_fixture.to_record(), "cid": selected_fixture.content_id},
        "surface_runs": [run.to_record() for run in runs],
        "findings": [finding.to_record() for finding in findings],
        "authoritative_surface_ids": list(authoritative_ids),
        "non_authoritative_surface_ids": list(non_authoritative_ids),
        "unavailable_surface_ids": list(unavailable_ids),
        "authoritative_agreement": agreement,
        "all_cleanup_succeeded": all_cleanup,
    }
    return DifferentialWitness(
        schema=VFS_DIFFERENTIAL_WITNESS_SCHEMA,
        goal_id=VFS_DIFFERENTIAL_GOAL_ID,
        task_id=VFS_DIFFERENTIAL_TASK_ID,
        objective_revision=VFS_DIFFERENTIAL_OBJECTIVE_REVISION,
        evidence_kinds=VFS_DIFFERENTIAL_EVIDENCE_KINDS,
        goal_ids=VFS_DIFFERENTIAL_PACKET_GOAL_IDS,
        trace=selected_trace,
        fixture=selected_fixture,
        surface_runs=runs,
        findings=findings,
        authoritative_surface_ids=authoritative_ids,
        non_authoritative_surface_ids=non_authoritative_ids,
        unavailable_surface_ids=unavailable_ids,
        authoritative_agreement=agreement,
        all_cleanup_succeeded=all_cleanup,
        bindings=bindings,
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
    "MAX_TRACE_STEPS",
    "REQUIRED_PUBLIC_SURFACES",
    "VFS_DIFFERENTIAL_EVIDENCE_KINDS",
    "VFS_DIFFERENTIAL_GOAL_ID",
    "VFS_DIFFERENTIAL_OBJECTIVE_REVISION",
    "VFS_DIFFERENTIAL_PACKET_GOAL_IDS",
    "VFS_DIFFERENTIAL_TASK_ID",
    "VFS_DIFFERENTIAL_TRACE_SCHEMA",
    "VFS_DIFFERENTIAL_WITNESS_SCHEMA",
    "CallableSurfaceAdapter",
    "CanonicalOperationTrace",
    "CleanupReceipt",
    "DifferentialWitness",
    "DriftFinding",
    "DriftKind",
    "ErrorIdentity",
    "FixtureEntry",
    "FixtureSpec",
    "HermeticNetworkError",
    "ImplementationIdentity",
    "NormalizationRule",
    "ObservationStatus",
    "PublicSurfaceKind",
    "RuntimeIdentity",
    "SurfaceAdapter",
    "SurfaceAvailability",
    "SurfaceFamily",
    "SurfaceObservation",
    "SurfaceRun",
    "SurfaceRunContext",
    "TraceStep",
    "TreeSnapshot",
    "VfsDifferentialHarnessError",
    "WitnessBindings",
    "build_canonical_operation_trace",
    "build_default_fixture",
    "capture_runtime_identity",
    "normalize_contract_result",
    "run_vfs_differential_harness",
    "snapshot_tree",
    "write_differential_witness",
]
