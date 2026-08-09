"""Deterministic control-plane state export renderers (DQP-011).

Interface: ``StateExporter@1``

Export is a **read-only rendering job** over a generation-bound snapshot.
Supported profiles:

* human Markdown taskboard/objective report (intentionally lossy)
* bounded status and event JSON
* JSONL audit stream
* CSV / Parquet analysis extracts
* portable release bundle (lossless for non-secret fields)

Every artifact binds store UUID, generation, schema revision, transaction
watermark, view/renderer revision, parameters, destination, and content
digest via :class:`StateExportReceipt`. Re-exporting the same snapshot and
parameters is byte-identical. Destinations are never watched as input and
cannot affect runtime decisions; only an explicit later import may re-admit
a portable bundle.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import (
    REDACTION_MARKER,
    STATE_EXPORT_RECEIPT_INTERFACE,
    STATE_EXPORT_RECEIPT_SCHEMA,
    ControlPlaneAuthorityError,
    ControlPlaneBoundsError,
    ControlPlaneContractError,
    ControlPlaneIdentityError,
    StateAuthorityClass,
    StateExportReceipt,
    StateSnapshot,
    canonical_json_bytes,
    content_identity,
    redact_mapping,
)

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

STATE_EXPORTER_INTERFACE: Final[str] = "StateExporter@1"
STATE_EXPORTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-exporter@1"
)
EXPORT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-export-request@1"
)
EXPORT_SOURCE_DATA_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-export-source-data@1"
)
PORTABLE_BUNDLE_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/portable-export-bundle@1"
)

# View / renderer revisions bound into every receipt.
QUERY_VIEW_REVISION: Final[str] = "view:control-plane-export@1"
RENDERER_REVISION: Final[str] = "renderer:state-export@1"
RENDERER_MARKDOWN: Final[str] = "renderer:markdown@1"
RENDERER_JSON: Final[str] = "renderer:json@1"
RENDERER_JSONL: Final[str] = "renderer:jsonl@1"
RENDERER_CSV: Final[str] = "renderer:csv@1"
RENDERER_PARQUET: Final[str] = "renderer:parquet@1"
RENDERER_BUNDLE: Final[str] = "renderer:portable-bundle@1"

EXPORTER_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = 1

# Exports never authorize mutation, completion, or scheduling.
EXPORT_IS_AUTHORITY: Final[bool] = False
EXPORT_AUTHORIZES_MUTATION: Final[bool] = False
EXPORT_IS_COMPLETION_EVIDENCE: Final[bool] = False
EXPORT_DEFAULT_AUTHORITY_CLASS: Final[StateAuthorityClass] = (
    StateAuthorityClass.EXPORT
)

# Human Markdown declares these fields intentionally omitted.
MARKDOWN_OMITTED_FIELDS: Final[tuple[str, ...]] = (
    "body_json",
    "identity_json",
    "secret_handle",
    "token",
    "password",
    "api_key",
    "authorization",
    "credentials",
    "private_key",
    "session_token",
    "refresh_token",
    "access_token",
    "client_secret",
    "cookie",
    "raw_event_payload",
    "lease_token",
    "provider_prompt",
    "provider_response",
)

MARKDOWN_NON_AUTHORITY_BANNER: Final[str] = (
    "> **NON-AUTHORITATIVE EXPORT** — This Markdown is a human projection of a "
    "database snapshot. It is not control-plane authority. Tampering with or "
    "deleting this file cannot change runtime scheduling, lifecycle, or "
    "completion decisions. Intentional loss: several fields are omitted."
)

MAX_PAGE_LIMIT: Final[int] = 10_000
DEFAULT_PAGE_LIMIT: Final[int] = 1_000
MAX_DESTINATION_BYTES: Final[int] = 4_096
MAX_PARAMETER_KEYS: Final[int] = 64
MAX_RECORD_COUNT: Final[int] = 100_000
MAX_ID_BYTES: Final[int] = 512
MAX_TEXT_BYTES: Final[int] = 8_192

_PORTABLE_MANIFEST_NAME: Final[str] = "manifest.json"
_PORTABLE_SNAPSHOT_NAME: Final[str] = "snapshot.json"
_PORTABLE_PAYLOAD_NAME: Final[str] = "payload.json"
_PORTABLE_RECEIPT_NAME: Final[str] = "receipt.json"

_TASK_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "task_alias",
    "goal_cid",
    "objective_id",
    "status",
    "priority",
    "ordinal",
    "revision",
    "created_at",
    "updated_at",
)

_EVENT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "event_id",
    "stream_id",
    "sequence",
    "global_sequence",
    "event_kind",
    "task_cid",
    "created_at",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StateExportError(RuntimeError):
    """Base class for fail-closed state export errors."""


class StateExportParameterError(StateExportError, ValueError):
    """Export parameters or profile are invalid."""


class StateExportRenderError(StateExportError):
    """A renderer failed to produce a deterministic artifact."""


class StateExportWriteError(StateExportError):
    """Atomic destination write failed."""


class StateExportRoundTripError(StateExportError):
    """Portable bundle round-trip failed integrity checks."""


class StateExportDependencyError(StateExportError):
    """An optional dependency required for a format is unavailable."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ExportFormat(str, Enum):
    """Closed set of export media formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    BUNDLE = "bundle"


class ExportProfile(str, Enum):
    """Closed export profiles (view selections + renderer pairings)."""

    HUMAN_TASKBOARD = "human-taskboard"
    STATUS_JSON = "status-json"
    EVENTS_JSON = "events-json"
    AUDIT_JSONL = "audit-jsonl"
    ANALYSIS_CSV = "analysis-csv"
    ANALYSIS_PARQUET = "analysis-parquet"
    PORTABLE_BUNDLE = "portable-bundle"


_PROFILE_FORMAT: Final[dict[ExportProfile, ExportFormat]] = {
    ExportProfile.HUMAN_TASKBOARD: ExportFormat.MARKDOWN,
    ExportProfile.STATUS_JSON: ExportFormat.JSON,
    ExportProfile.EVENTS_JSON: ExportFormat.JSON,
    ExportProfile.AUDIT_JSONL: ExportFormat.JSONL,
    ExportProfile.ANALYSIS_CSV: ExportFormat.CSV,
    ExportProfile.ANALYSIS_PARQUET: ExportFormat.PARQUET,
    ExportProfile.PORTABLE_BUNDLE: ExportFormat.BUNDLE,
}

_PROFILE_RENDERER: Final[dict[ExportProfile, str]] = {
    ExportProfile.HUMAN_TASKBOARD: RENDERER_MARKDOWN,
    ExportProfile.STATUS_JSON: RENDERER_JSON,
    ExportProfile.EVENTS_JSON: RENDERER_JSON,
    ExportProfile.AUDIT_JSONL: RENDERER_JSONL,
    ExportProfile.ANALYSIS_CSV: RENDERER_CSV,
    ExportProfile.ANALYSIS_PARQUET: RENDERER_PARQUET,
    ExportProfile.PORTABLE_BUNDLE: RENDERER_BUNDLE,
}

_PROFILE_INTENTIONAL_LOSS: Final[dict[ExportProfile, bool]] = {
    ExportProfile.HUMAN_TASKBOARD: True,
    ExportProfile.STATUS_JSON: True,
    ExportProfile.EVENTS_JSON: False,
    ExportProfile.AUDIT_JSONL: False,
    ExportProfile.ANALYSIS_CSV: True,
    ExportProfile.ANALYSIS_PARQUET: True,
    ExportProfile.PORTABLE_BUNDLE: False,
}

_PROFILE_EXTENSION: Final[dict[ExportProfile, str]] = {
    ExportProfile.HUMAN_TASKBOARD: ".md",
    ExportProfile.STATUS_JSON: ".json",
    ExportProfile.EVENTS_JSON: ".json",
    ExportProfile.AUDIT_JSONL: ".jsonl",
    ExportProfile.ANALYSIS_CSV: ".csv",
    ExportProfile.ANALYSIS_PARQUET: ".parquet",
    ExportProfile.PORTABLE_BUNDLE: "",
}

_COMPACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,511}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pyarrow_available() -> bool:
    """Return whether pyarrow can be imported for Parquet renders."""

    try:
        import pyarrow  # type: ignore  # noqa: F401
        import pyarrow.parquet  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_tree(root: Path) -> str:
    """Digest a directory tree by sorted relative path + file digest pairs."""

    if not root.is_dir():
        raise StateExportWriteError(f"bundle root is not a directory: {root}")
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        entries.append({"path": relative, "digest": _sha256_file(path)})
    return _sha256_bytes(canonical_json_bytes(entries))


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_ID_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise StateExportParameterError(f"{field_name} must be a string")
    else:
        text = value
    if text != text.strip():
        raise StateExportParameterError(
            f"{field_name} has leading or trailing whitespace"
        )
    if required and not text:
        raise StateExportParameterError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise StateExportParameterError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise StateExportParameterError(f"{field_name} exceeds its byte bound")
    return text


def _compact_id(value: Any, field_name: str) -> str:
    text = _text(value, field_name, required=True, limit=MAX_ID_BYTES)
    if not _COMPACT_ID_RE.match(text):
        raise StateExportParameterError(
            f"{field_name} is not a compact control-plane id"
        )
    return text


def _enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise StateExportParameterError(
                f"{field_name} is not a closed {enum_cls.__name__} value"
            ) from exc
    raise StateExportParameterError(
        f"{field_name} must be a {enum_cls.__name__} value"
    )


def _bounded_int(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_PAGE_LIMIT,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StateExportParameterError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise StateExportParameterError(
            f"{field_name} out of bounds [{minimum}, {maximum}]"
        )
    return value


def _freeze_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise StateExportParameterError(f"{field_name} must be an object")
    if len(value) > MAX_PARAMETER_KEYS:
        raise StateExportParameterError(
            f"{field_name} exceeds max key count {MAX_PARAMETER_KEYS}"
        )
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise StateExportParameterError(
                f"{field_name} keys must be strings"
            )
        frozen[key] = item
    # Reject floats for determinism; nested structures are redacted later.
    try:
        canonical_json_bytes(frozen)
    except (ControlPlaneBoundsError, ControlPlaneContractError) as exc:
        raise StateExportParameterError(
            f"{field_name} is not canonically serializable: {exc}"
        ) from exc
    return MappingProxyType(dict(sorted(frozen.items())))


def _as_mapping_list(
    value: Any, field_name: str
) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StateExportParameterError(f"{field_name} must be a sequence")
    if len(value) > MAX_RECORD_COUNT:
        raise StateExportParameterError(
            f"{field_name} exceeds max record count {MAX_RECORD_COUNT}"
        )
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise StateExportParameterError(
                f"{field_name}[{index}] must be an object"
            )
        rows.append(MappingProxyType(dict(item)))
    return tuple(rows)


def _stable_sort_key(row: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    parts: list[str] = []
    for key in keys:
        value = row.get(key, "")
        if value is None:
            parts.append("")
        else:
            parts.append(str(value))
    # Tie-break on canonical JSON of the whole row for total order.
    parts.append(_sha256_bytes(canonical_json_bytes(dict(row))))
    return tuple(parts)


def _sort_rows(
    rows: Sequence[Mapping[str, Any]], *keys: str
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    materialized.sort(key=lambda row: _stable_sort_key(row, *keys))
    return materialized


def _page_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cursor: int,
    limit: int,
) -> list[dict[str, Any]]:
    if cursor < 0:
        raise StateExportParameterError("cursor must be >= 0")
    if limit < 1:
        raise StateExportParameterError("limit must be >= 1")
    sliced = list(rows)[cursor : cursor + limit]
    return [dict(item) for item in sliced]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return canonical_json_bytes(value).decode("utf-8")
    return str(value)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    """Write ``payload`` to ``destination`` via temp file + ``os.replace``."""

    destination = Path(destination)
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StateExportWriteError(
            f"cannot create destination directory {parent}: {exc}"
        ) from exc
    fd: int | None = None
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(parent),
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(destination))
        temp_path = None
        # Best-effort directory fsync for durability on crash.
        try:
            dir_fd = os.open(str(parent), os.O_RDONLY)
        except OSError:
            dir_fd = -1
        if dir_fd >= 0:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except OSError as exc:
        raise StateExportWriteError(
            f"atomic write failed for {destination}: {exc}"
        ) from exc
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _atomic_write_tree(destination: Path, files: Mapping[str, bytes]) -> None:
    """Atomically replace a directory bundle with the given relative files."""

    destination = Path(destination)
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StateExportWriteError(
            f"cannot create destination directory {parent}: {exc}"
        ) from exc
    temp_dir: Path | None = None
    try:
        temp_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=str(parent),
            )
        )
        for relative, payload in sorted(files.items()):
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise StateExportWriteError(
                    f"unsafe bundle relative path: {relative!r}"
                )
            target = temp_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            with target.open("rb+") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        # Replace destination: remove existing then rename temp into place.
        # On POSIX, directory rename is atomic within the same filesystem.
        backup: Path | None = None
        if destination.exists():
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.bak.",
                    dir=str(parent),
                )
            )
            # Move existing contents aside.
            os.rename(str(destination), str(backup / "previous"))
        os.rename(str(temp_dir), str(destination))
        temp_dir = None
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except OSError as exc:
        raise StateExportWriteError(
            f"atomic bundle write failed for {destination}: {exc}"
        ) from exc
    finally:
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _json_document_bytes(payload: Mapping[str, Any]) -> bytes:
    """Pretty, sorted, trailing-newline JSON for human/tool inspection."""

    body = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        separators=(",", ": "),
    )
    return (body + "\n").encode("utf-8")


def _canonical_document_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(payload)) + b"\n"


# ---------------------------------------------------------------------------
# Source data bound to a snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportSourceData:
    """Bounded snapshot payload for deterministic export rendering.

    The exporter never mutates this structure. Runtime decisions must continue
    to use the live repository / snapshot identity — not re-read export files.
    """

    SCHEMA: ClassVar[str] = EXPORT_SOURCE_DATA_SCHEMA

    snapshot: StateSnapshot
    tasks: tuple[Mapping[str, Any], ...] = ()
    goals: tuple[Mapping[str, Any], ...] = ()
    objectives: tuple[Mapping[str, Any], ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    leases: tuple[Mapping[str, Any], ...] = ()
    commands: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, StateSnapshot):
            raise StateExportParameterError(
                "snapshot must be a StateSnapshot instance"
            )
        object.__setattr__(
            self, "tasks", _as_mapping_list(self.tasks, "tasks")
        )
        object.__setattr__(
            self, "goals", _as_mapping_list(self.goals, "goals")
        )
        object.__setattr__(
            self,
            "objectives",
            _as_mapping_list(self.objectives, "objectives"),
        )
        object.__setattr__(
            self, "events", _as_mapping_list(self.events, "events")
        )
        object.__setattr__(
            self, "leases", _as_mapping_list(self.leases, "leases")
        )
        object.__setattr__(
            self, "commands", _as_mapping_list(self.commands, "commands")
        )
        object.__setattr__(
            self, "metadata", _freeze_mapping(self.metadata, "metadata")
        )

    def redacted(self) -> "ExportSourceData":
        """Return a deep-redacted copy safe for every export profile."""

        return ExportSourceData(
            snapshot=self.snapshot,
            tasks=tuple(redact_mapping(dict(row)) for row in self.tasks),
            goals=tuple(redact_mapping(dict(row)) for row in self.goals),
            objectives=tuple(
                redact_mapping(dict(row)) for row in self.objectives
            ),
            events=tuple(redact_mapping(dict(row)) for row in self.events),
            leases=tuple(redact_mapping(dict(row)) for row in self.leases),
            commands=tuple(
                redact_mapping(dict(row)) for row in self.commands
            ),
            metadata=redact_mapping(dict(self.metadata)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "snapshot": self.snapshot.to_record(),
            "tasks": [dict(row) for row in self.tasks],
            "goals": [dict(row) for row in self.goals],
            "objectives": [dict(row) for row in self.objectives],
            "events": [dict(row) for row in self.events],
            "leases": [dict(row) for row in self.leases],
            "commands": [dict(row) for row in self.commands],
            "metadata": dict(self.metadata),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExportSourceData":
        if not isinstance(payload, Mapping):
            raise StateExportParameterError(
                "export source data payload must be an object"
            )
        schema = payload.get("schema")
        if schema not in (None, "", cls.SCHEMA):
            raise StateExportParameterError(
                f"unsupported export source data schema: {schema!r}"
            )
        snapshot_payload = payload.get("snapshot")
        if not isinstance(snapshot_payload, Mapping):
            raise StateExportParameterError(
                "export source data requires a snapshot object"
            )
        return cls(
            snapshot=StateSnapshot.from_dict(snapshot_payload),
            tasks=payload.get("tasks") or (),
            goals=payload.get("goals") or (),
            objectives=payload.get("objectives") or (),
            events=payload.get("events") or (),
            leases=payload.get("leases") or (),
            commands=payload.get("commands") or (),
            metadata=payload.get("metadata") or {},
        )

    @classmethod
    def from_repository(cls, repository: Any) -> "ExportSourceData":
        """Collect a bounded export population from a StateRepository.

        Uses only typed repository methods (never arbitrary SQL). Pagination
        drains tasks and events to a stable full population.
        """

        if repository is None:
            raise StateExportParameterError("repository is required")
        snapshot = repository.snapshot()
        if not isinstance(snapshot, StateSnapshot):
            raise StateExportParameterError(
                "repository.snapshot() must return StateSnapshot"
            )
        tasks: list[Mapping[str, Any]] = []
        cursor = 0
        while True:
            page = repository.list_tasks(cursor=cursor, limit=DEFAULT_PAGE_LIMIT)
            items = list(getattr(page, "items", ()) or ())
            tasks.extend(items)
            if getattr(page, "exhausted", True) or getattr(
                page, "next_cursor", None
            ) is None:
                break
            cursor = int(page.next_cursor)
            if len(tasks) > MAX_RECORD_COUNT:
                raise StateExportParameterError(
                    "task population exceeds export bound"
                )
        events: list[Mapping[str, Any]] = []
        event_cursor = 0
        while True:
            page = repository.list_events(
                cursor=event_cursor, limit=DEFAULT_PAGE_LIMIT
            )
            items = list(getattr(page, "items", ()) or ())
            events.extend(items)
            if getattr(page, "exhausted", True) or getattr(
                page, "next_cursor", None
            ) is None:
                break
            event_cursor = int(page.next_cursor)
            if len(events) > MAX_RECORD_COUNT:
                raise StateExportParameterError(
                    "event population exceeds export bound"
                )
        leases = tuple(repository.list_leases() or ())
        commands = tuple(repository.list_commands() or ())
        metadata: dict[str, Any] = {
            "store_id": str(getattr(repository, "store_id", snapshot.store_id)),
        }
        authority = getattr(repository, "authority_mode", None)
        if authority is not None:
            metadata["authority_mode"] = getattr(
                authority, "value", str(authority)
            )
        return cls(
            snapshot=snapshot,
            tasks=tuple(tasks),
            events=tuple(events),
            leases=leases,
            commands=commands,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Export request / result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportRequest:
    """Parameters for one deterministic export render."""

    SCHEMA: ClassVar[str] = EXPORT_REQUEST_SCHEMA

    profile: ExportProfile
    destination: str
    export_id: str = ""
    cursor: int = 0
    limit: int = DEFAULT_PAGE_LIMIT
    domain: str = "tasks"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile",
            _enum(self.profile, ExportProfile, field_name="profile"),
        )
        object.__setattr__(
            self,
            "destination",
            _text(
                self.destination,
                "destination",
                required=True,
                limit=MAX_DESTINATION_BYTES,
            ),
        )
        export_id = self.export_id
        if not export_id:
            export_id = (
                f"export:{self.profile.value}:"
                f"{Path(self.destination).name or 'artifact'}"
            )
        object.__setattr__(
            self, "export_id", _compact_id(export_id, "export_id")
        )
        object.__setattr__(
            self,
            "cursor",
            _bounded_int(self.cursor, "cursor", minimum=0, maximum=MAX_RECORD_COUNT),
        )
        object.__setattr__(
            self,
            "limit",
            _bounded_int(
                self.limit, "limit", minimum=1, maximum=MAX_PAGE_LIMIT
            ),
        )
        object.__setattr__(
            self,
            "domain",
            _text(self.domain, "domain", required=True, limit=64),
        )
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, "parameters"),
        )

    @property
    def format(self) -> ExportFormat:
        return _PROFILE_FORMAT[self.profile]

    @property
    def intentional_loss(self) -> bool:
        return _PROFILE_INTENTIONAL_LOSS[self.profile]

    @property
    def renderer_revision(self) -> str:
        return _PROFILE_RENDERER[self.profile]

    @property
    def query_revision(self) -> str:
        return QUERY_VIEW_REVISION

    def parameter_record(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "format": self.format.value,
            "cursor": self.cursor,
            "limit": self.limit,
            "domain": self.domain,
            "parameters": dict(self.parameters),
            "exporter_version": EXPORTER_VERSION,
            "query_revision": self.query_revision,
            "renderer_revision": self.renderer_revision,
            "intentional_loss": self.intentional_loss,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "export_id": self.export_id,
            "profile": self.profile.value,
            "destination": self.destination,
            "cursor": self.cursor,
            "limit": self.limit,
            "domain": self.domain,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ExportArtifact:
    """In-memory render product before/without durable write."""

    profile: ExportProfile
    format: ExportFormat
    body: bytes
    files: Mapping[str, bytes] = field(default_factory=dict)
    media_type: str = "application/octet-stream"
    intentional_loss: bool = False
    omitted_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", bytes(self.body))
        frozen_files = {
            str(key): bytes(value) for key, value in dict(self.files).items()
        }
        object.__setattr__(
            self, "files", MappingProxyType(dict(sorted(frozen_files.items())))
        )
        object.__setattr__(
            self, "omitted_fields", tuple(str(item) for item in self.omitted_fields)
        )

    @property
    def digest(self) -> str:
        if self.format is ExportFormat.BUNDLE:
            entries = [
                {"path": path, "digest": _sha256_bytes(payload)}
                for path, payload in sorted(self.files.items())
            ]
            return _sha256_bytes(canonical_json_bytes(entries))
        return _sha256_bytes(self.body)


@dataclass(frozen=True)
class ExportResult:
    """Durable export outcome with receipt and artifact digest."""

    receipt: StateExportReceipt
    artifact: ExportArtifact
    destination: str
    written: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt": self.receipt.to_record(),
            "destination": self.destination,
            "written": self.written,
            "artifact_digest": self.artifact.digest,
            "format": self.artifact.format.value,
            "profile": self.artifact.profile.value,
            "intentional_loss": self.artifact.intentional_loss,
            "omitted_fields": list(self.artifact.omitted_fields),
            "authority_class": self.receipt.authority_class.value,
            "is_authority": EXPORT_IS_AUTHORITY,
        }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _human_task_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a task row to the lossy human surface."""

    alias = row.get("task_alias") or row.get("id") or row.get("task_id") or ""
    title = (
        row.get("title")
        or row.get("name")
        or (dict(row.get("body_json") or {}) if isinstance(row.get("body_json"), Mapping) else {}).get("title")
        or ""
    )
    if not title and isinstance(row.get("body_json"), str):
        try:
            parsed = json.loads(row["body_json"])
            if isinstance(parsed, Mapping):
                title = str(parsed.get("title") or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            title = ""
    return {
        "id": str(alias or row.get("task_cid") or ""),
        "task_cid": str(row.get("task_cid") or ""),
        "title": str(title or ""),
        "status": str(row.get("status") or ""),
        "priority": str(row.get("priority") or ""),
        "goal_cid": str(row.get("goal_cid") or ""),
        "objective_id": str(row.get("objective_id") or ""),
        "ordinal": row.get("ordinal", ""),
    }


def render_markdown(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """Render intentionally lossy human Markdown for a snapshot."""

    redacted = source.redacted()
    snapshot = redacted.snapshot
    tasks = _sort_rows(redacted.tasks, "ordinal", "task_cid", "task_alias")
    goals = _sort_rows(redacted.goals, "ordinal", "goal_cid")
    objectives = _sort_rows(redacted.objectives, "ordinal", "objective_id")
    paged_tasks = _page_rows(tasks, cursor=request.cursor, limit=request.limit)

    lines: list[str] = [
        "# Control-plane taskboard export",
        "",
        MARKDOWN_NON_AUTHORITY_BANNER,
        "",
        "## Export identity",
        "",
        f"- Authority class: `{StateAuthorityClass.EXPORT.value}`",
        f"- Is authority: `{str(EXPORT_IS_AUTHORITY).lower()}`",
        f"- Intentional loss: `true`",
        f"- Snapshot id: `{snapshot.snapshot_id}`",
        f"- Store id: `{snapshot.store_id}`",
        f"- Database UUID: `{snapshot.database_uuid}`",
        f"- Generation: `{snapshot.generation}`",
        f"- Schema revision: `{snapshot.schema_revision}`",
        f"- Revision: `{snapshot.revision}`",
        f"- Event watermark: `{snapshot.event_watermark}`",
        f"- Query revision: `{request.query_revision}`",
        f"- Renderer revision: `{request.renderer_revision}`",
        f"- Profile: `{request.profile.value}`",
        f"- Cursor: `{request.cursor}`",
        f"- Limit: `{request.limit}`",
        "",
        "## Intentionally omitted fields",
        "",
    ]
    for name in MARKDOWN_OMITTED_FIELDS:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Objectives",
            "",
        ]
    )
    if not objectives:
        lines.append("_None in snapshot page._")
        lines.append("")
    else:
        for objective in objectives:
            oid = (
                objective.get("objective_id")
                or objective.get("id")
                or objective.get("objective_cid")
                or "unknown"
            )
            title = objective.get("title") or objective.get("name") or ""
            status = objective.get("status") or ""
            lines.append(f"### {oid} {title}".rstrip())
            lines.append("")
            lines.append(f"- Status: {status}")
            lines.append("")

    lines.extend(["## Goals", ""])
    if not goals:
        lines.append("_None in snapshot page._")
        lines.append("")
    else:
        for goal in goals:
            gid = goal.get("goal_cid") or goal.get("goal_alias") or "unknown"
            title = goal.get("title") or ""
            status = goal.get("status") or ""
            lines.append(f"### {gid} {title}".rstrip())
            lines.append("")
            lines.append(f"- Status: {status}")
            lines.append(f"- Objective: {goal.get('objective_id') or ''}")
            lines.append("")

    lines.extend(["## Tasks", ""])
    if not paged_tasks:
        lines.append("_No tasks in this page._")
        lines.append("")
    else:
        for task in paged_tasks:
            projected = _human_task_row(task)
            heading = f"## {projected['id']}"
            if projected["title"]:
                heading = f"{heading} {projected['title']}"
            lines.append(heading)
            lines.append("")
            lines.append(f"- Status: {projected['status']}")
            lines.append(f"- Priority: {projected['priority']}")
            lines.append(f"- Goal: {projected['goal_cid']}")
            lines.append(f"- Objective: {projected['objective_id']}")
            lines.append(f"- Task CID: `{projected['task_cid']}`")
            lines.append(f"- Ordinal: {projected['ordinal']}")
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "End of non-authoritative export. Do not treat this file as a "
            "scheduler input, completion receipt, or lease authority.",
            "",
        ]
    )
    body = "\n".join(lines).encode("utf-8")
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.MARKDOWN,
        body=body,
        media_type="text/markdown; charset=utf-8",
        intentional_loss=True,
        omitted_fields=MARKDOWN_OMITTED_FIELDS,
    )


def render_status_json(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """Bounded status JSON projection (intentionally lossy)."""

    redacted = source.redacted()
    snapshot = redacted.snapshot
    tasks = _sort_rows(redacted.tasks, "ordinal", "task_cid")
    paged = _page_rows(tasks, cursor=request.cursor, limit=request.limit)
    status_rows = [
        {
            "task_cid": str(row.get("task_cid") or ""),
            "task_alias": str(row.get("task_alias") or row.get("id") or ""),
            "status": str(row.get("status") or ""),
            "priority": str(row.get("priority") or ""),
            "goal_cid": str(row.get("goal_cid") or ""),
            "revision": row.get("revision", 0),
        }
        for row in paged
    ]
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/status-export@1",
        "authority_class": StateAuthorityClass.EXPORT.value,
        "is_authority": EXPORT_IS_AUTHORITY,
        "intentional_loss": True,
        "omitted_fields": list(MARKDOWN_OMITTED_FIELDS),
        "snapshot": snapshot.to_record(),
        "query_revision": request.query_revision,
        "renderer_revision": request.renderer_revision,
        "cursor": request.cursor,
        "limit": request.limit,
        "task_count": len(status_rows),
        "tasks": status_rows,
        "lease_count": len(redacted.leases),
        "event_watermark": snapshot.event_watermark,
    }
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.JSON,
        body=_json_document_bytes(payload),
        media_type="application/json",
        intentional_loss=True,
        omitted_fields=MARKDOWN_OMITTED_FIELDS,
    )


def render_events_json(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """Lossless (post-redaction) event JSON page."""

    redacted = source.redacted()
    snapshot = redacted.snapshot
    events = _sort_rows(
        redacted.events, "global_sequence", "sequence", "event_id"
    )
    paged = _page_rows(events, cursor=request.cursor, limit=request.limit)
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/events-export@1",
        "authority_class": StateAuthorityClass.EXPORT.value,
        "is_authority": EXPORT_IS_AUTHORITY,
        "intentional_loss": False,
        "snapshot": snapshot.to_record(),
        "query_revision": request.query_revision,
        "renderer_revision": request.renderer_revision,
        "cursor": request.cursor,
        "limit": request.limit,
        "event_count": len(paged),
        "events": paged,
    }
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.JSON,
        body=_json_document_bytes(payload),
        media_type="application/json",
        intentional_loss=False,
    )


def render_audit_jsonl(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """JSONL audit stream — one event object per line, sorted."""

    redacted = source.redacted()
    snapshot = redacted.snapshot
    events = _sort_rows(
        redacted.events, "global_sequence", "sequence", "event_id"
    )
    paged = _page_rows(events, cursor=request.cursor, limit=request.limit)
    lines: list[str] = []
    # Header line binds the snapshot so the stream is self-describing.
    header = {
        "record_type": "export_header",
        "authority_class": StateAuthorityClass.EXPORT.value,
        "is_authority": EXPORT_IS_AUTHORITY,
        "intentional_loss": False,
        "snapshot": snapshot.to_record(),
        "query_revision": request.query_revision,
        "renderer_revision": request.renderer_revision,
        "cursor": request.cursor,
        "limit": request.limit,
        "event_count": len(paged),
    }
    lines.append(canonical_json_bytes(header).decode("utf-8"))
    for event in paged:
        record = {
            "record_type": "domain_event",
            **event,
        }
        lines.append(canonical_json_bytes(record).decode("utf-8"))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.JSONL,
        body=body,
        media_type="application/x-ndjson",
        intentional_loss=False,
    )


def _tabular_rows(
    source: ExportSourceData,
    request: ExportRequest,
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    redacted = source.redacted()
    domain = request.domain
    if domain in {"tasks", "task", "analysis-tasks"}:
        columns = _TASK_CSV_COLUMNS
        rows = _sort_rows(redacted.tasks, "ordinal", "task_cid")
    elif domain in {"events", "event", "analysis-events"}:
        columns = _EVENT_CSV_COLUMNS
        rows = _sort_rows(
            redacted.events, "global_sequence", "sequence", "event_id"
        )
    else:
        raise StateExportParameterError(
            f"unsupported analysis domain {domain!r}; use tasks or events"
        )
    paged = _page_rows(rows, cursor=request.cursor, limit=request.limit)
    projected: list[dict[str, Any]] = []
    for row in paged:
        projected.append({column: row.get(column, "") for column in columns})
    return columns, projected


def render_analysis_csv(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """Deterministic CSV analysis extract."""

    columns, rows = _tabular_rows(source, request)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(columns),
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _cell(row.get(key)) for key in columns})
    body = buffer.getvalue().encode("utf-8")
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.CSV,
        body=body,
        media_type="text/csv; charset=utf-8",
        intentional_loss=True,
        omitted_fields=("body_json", "identity_json", "raw_event_payload"),
    )


def render_analysis_parquet(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """Deterministic Parquet analysis extract via pyarrow when available."""

    if not pyarrow_available():
        raise StateExportDependencyError(
            "pyarrow is required for analysis-parquet exports; "
            "install the optional pyarrow dependency"
        )
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    columns, rows = _tabular_rows(source, request)
    # All columns as UTF-8 strings for stable schema across row sets.
    arrays = {
        column: pa.array(
            [_cell(row.get(column)) for row in rows], type=pa.string()
        )
        for column in columns
    }
    table = pa.table(arrays)
    buffer = io.BytesIO()
    # Fixed writer options for byte-stable re-export on the same library.
    write_kwargs: dict[str, Any] = {
        "compression": "none",
        "use_dictionary": False,
        "write_statistics": False,
        "data_page_version": "1.0",
        "store_schema": True,
    }
    # Prefer a fixed parquet format version when the installed pyarrow supports it.
    for version in ("2.6", "2.4", "1.0"):
        try:
            pq.write_table(table, buffer, version=version, **write_kwargs)
            break
        except (TypeError, ValueError, OSError):
            buffer.seek(0)
            buffer.truncate(0)
    else:
        pq.write_table(table, buffer, **write_kwargs)
    body = buffer.getvalue()
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.PARQUET,
        body=body,
        media_type="application/vnd.apache.parquet",
        intentional_loss=True,
        omitted_fields=("body_json", "identity_json", "raw_event_payload"),
    )


def render_portable_bundle(
    source: ExportSourceData,
    request: ExportRequest,
) -> ExportArtifact:
    """Lossless (post-redaction) portable release bundle.

    The bundle is a directory of JSON documents. Round-tripping through
    :func:`load_portable_bundle` reconstructs :class:`ExportSourceData` such
    that re-export is byte-identical.
    """

    redacted = source.redacted()
    snapshot = redacted.snapshot
    payload = redacted.to_dict()
    # Portable payload uses compact canonical JSON for lossless identity.
    payload_bytes = _canonical_document_bytes(payload)
    snapshot_bytes = _json_document_bytes(snapshot.to_record())
    # Manifest is written without artifact digests first; digests are filled
    # after known file bytes so the tree digest is self-consistent.
    file_digests = {
        _PORTABLE_SNAPSHOT_NAME: _sha256_bytes(snapshot_bytes),
        _PORTABLE_PAYLOAD_NAME: _sha256_bytes(payload_bytes),
    }
    manifest = {
        "schema": PORTABLE_BUNDLE_MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "authority_class": StateAuthorityClass.EXPORT.value,
        "is_authority": EXPORT_IS_AUTHORITY,
        "intentional_loss": False,
        "profile": request.profile.value,
        "query_revision": request.query_revision,
        "renderer_revision": request.renderer_revision,
        "exporter_version": EXPORTER_VERSION,
        "snapshot_id": snapshot.snapshot_id,
        "store_id": snapshot.store_id,
        "database_uuid": snapshot.database_uuid,
        "generation": snapshot.generation,
        "schema_revision": snapshot.schema_revision,
        "revision": snapshot.revision,
        "event_watermark": snapshot.event_watermark,
        "export_id": request.export_id,
        "parameters": request.parameter_record(),
        "files": file_digests,
        "omitted_fields": [REDACTION_MARKER],  # secrets redacted, not omitted
        "redaction_marker": REDACTION_MARKER,
    }
    manifest_bytes = _json_document_bytes(manifest)
    files = {
        _PORTABLE_MANIFEST_NAME: manifest_bytes,
        _PORTABLE_SNAPSHOT_NAME: snapshot_bytes,
        _PORTABLE_PAYLOAD_NAME: payload_bytes,
    }
    return ExportArtifact(
        profile=request.profile,
        format=ExportFormat.BUNDLE,
        body=manifest_bytes,
        files=files,
        media_type="application/x-control-plane-export-bundle",
        intentional_loss=False,
    )


_RENDERERS: Final[
    dict[ExportProfile, Callable[[ExportSourceData, ExportRequest], ExportArtifact]]
] = {
    ExportProfile.HUMAN_TASKBOARD: render_markdown,
    ExportProfile.STATUS_JSON: render_status_json,
    ExportProfile.EVENTS_JSON: render_events_json,
    ExportProfile.AUDIT_JSONL: render_audit_jsonl,
    ExportProfile.ANALYSIS_CSV: render_analysis_csv,
    ExportProfile.ANALYSIS_PARQUET: render_analysis_parquet,
    ExportProfile.PORTABLE_BUNDLE: render_portable_bundle,
}


# ---------------------------------------------------------------------------
# Portable round-trip
# ---------------------------------------------------------------------------


def load_portable_bundle(path: Path | str) -> ExportSourceData:
    """Load a portable bundle produced by this exporter.

    This is an **explicit** import-side helper for round-trip tests and
    operator recovery tooling. Runtime schedulers must not call this
    implicitly; export destinations are never watched as decision input.
    """

    root = Path(path)
    if not root.is_dir():
        raise StateExportRoundTripError(
            f"portable bundle path is not a directory: {root}"
        )
    manifest_path = root / _PORTABLE_MANIFEST_NAME
    payload_path = root / _PORTABLE_PAYLOAD_NAME
    snapshot_path = root / _PORTABLE_SNAPSHOT_NAME
    if not manifest_path.is_file() or not payload_path.is_file():
        raise StateExportRoundTripError(
            "portable bundle missing manifest.json or payload.json"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StateExportRoundTripError(
            f"portable bundle is unreadable: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping) or not isinstance(payload, Mapping):
        raise StateExportRoundTripError(
            "portable bundle manifest/payload must be objects"
        )
    if manifest.get("schema") not in (None, "", PORTABLE_BUNDLE_MANIFEST_SCHEMA):
        raise StateExportRoundTripError(
            f"unsupported portable bundle schema: {manifest.get('schema')!r}"
        )
    if manifest.get("is_authority") is True:
        raise ControlPlaneAuthorityError(
            "portable bundle claims authority; refusing load"
        )
    if manifest.get("authority_class") == StateAuthorityClass.AUTHORITATIVE.value:
        raise ControlPlaneAuthorityError(
            "portable bundle labeled authoritative; refusing load"
        )
    # Verify per-file digests when present.
    files_meta = manifest.get("files") or {}
    if isinstance(files_meta, Mapping):
        for relative, expected in files_meta.items():
            member = root / str(relative)
            if not member.is_file():
                raise StateExportRoundTripError(
                    f"portable bundle missing declared file {relative!r}"
                )
            actual = _sha256_file(member)
            if str(expected) != actual:
                raise StateExportRoundTripError(
                    f"digest mismatch for {relative}: "
                    f"expected {expected}, got {actual}"
                )
    if snapshot_path.is_file():
        try:
            snapshot_payload = json.loads(
                snapshot_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StateExportRoundTripError(
                f"snapshot.json unreadable: {exc}"
            ) from exc
        # Ensure payload snapshot agrees with standalone snapshot file.
        if payload.get("snapshot") and snapshot_payload:
            if content_identity(
                StateSnapshot.from_dict(payload["snapshot"]).to_dict()
            ) != content_identity(
                StateSnapshot.from_dict(snapshot_payload).to_dict()
            ):
                raise StateExportRoundTripError(
                    "snapshot.json disagrees with payload snapshot"
                )
    return ExportSourceData.from_dict(payload)


def portable_round_trip_identical(
    source: ExportSourceData,
    *,
    destination: Path | str | None = None,
) -> bool:
    """Return whether a portable export → load → re-export is byte-identical.

    When ``destination`` is omitted the comparison is performed in memory.
    """

    exporter = StateExporter()
    request = ExportRequest(
        profile=ExportProfile.PORTABLE_BUNDLE,
        destination=str(destination or "portable-bundle"),
        export_id="export:portable-roundtrip",
    )
    first = exporter.render(source, request)
    if destination is not None:
        result = exporter.export(source, request, write=True)
        restored = load_portable_bundle(result.destination)
        second = exporter.render(restored, request)
    else:
        # In-memory round-trip via temp directory for tree load path.
        with tempfile.TemporaryDirectory(prefix="state-export-rt-") as tmp:
            tmp_path = Path(tmp) / "bundle"
            exporter.write_artifact(first, tmp_path)
            restored = load_portable_bundle(tmp_path)
            second = exporter.render(restored, request)
    if first.format is ExportFormat.BUNDLE:
        return dict(first.files) == dict(second.files)
    return first.body == second.body


# ---------------------------------------------------------------------------
# StateExporter
# ---------------------------------------------------------------------------


class StateExporter:
    """Read-only deterministic multi-format control-plane exporter.

    Interface: ``StateExporter@1``.

    Invariants:

    * never mutates :class:`ExportSourceData` or any repository
    * never labels receipts authoritative
    * never treats destinations as decision input
    * re-export of identical snapshot/parameters is byte-identical
    * secrets are redacted before any media is produced
    """

    INTERFACE: ClassVar[str] = STATE_EXPORTER_INTERFACE
    SCHEMA: ClassVar[str] = STATE_EXPORTER_SCHEMA

    def render(
        self,
        source: ExportSourceData,
        request: ExportRequest | Mapping[str, Any],
    ) -> ExportArtifact:
        """Render an in-memory artifact without writing."""

        req = (
            request
            if isinstance(request, ExportRequest)
            else ExportRequest(**dict(request))  # type: ignore[arg-type]
        )
        if not isinstance(source, ExportSourceData):
            raise StateExportParameterError(
                "source must be ExportSourceData"
            )
        # Snapshot consistency: refuse export authority on the snapshot itself.
        if source.snapshot.authority_class is StateAuthorityClass.EXPORT:
            raise ControlPlaneAuthorityError(
                "refusing to export from an export-authority snapshot"
            )
        renderer = _RENDERERS.get(req.profile)
        if renderer is None:
            raise StateExportParameterError(
                f"no renderer registered for profile {req.profile.value}"
            )
        try:
            artifact = renderer(source, req)
        except StateExportError:
            raise
        except Exception as exc:  # noqa: BLE001 - map to render error
            raise StateExportRenderError(
                f"renderer {req.profile.value} failed: {exc}"
            ) from exc
        if artifact.intentional_loss != req.intentional_loss:
            raise StateExportRenderError(
                "renderer intentional_loss disagrees with profile policy"
            )
        return artifact

    def build_receipt(
        self,
        source: ExportSourceData,
        request: ExportRequest,
        artifact: ExportArtifact,
        *,
        destination: str | None = None,
    ) -> StateExportReceipt:
        """Construct a :class:`StateExportReceipt` bound to snapshot + artifact."""

        snapshot = source.snapshot
        dest = destination if destination is not None else request.destination
        receipt = StateExportReceipt(
            export_id=request.export_id,
            snapshot_id=snapshot.snapshot_id,
            store_id=snapshot.store_id,
            database_uuid=snapshot.database_uuid,
            schema_revision=snapshot.schema_revision,
            generation=snapshot.generation,
            revision=snapshot.revision,
            event_watermark=snapshot.event_watermark,
            renderer_revision=request.renderer_revision,
            query_revision=request.query_revision,
            artifact_digest=artifact.digest,
            destination=str(dest),
            parameters=request.parameter_record(),
            authority_class=StateAuthorityClass.EXPORT,
            intentional_loss=request.intentional_loss,
        )
        if not receipt.binds_snapshot(snapshot):
            raise ControlPlaneIdentityError(
                "export receipt does not bind the source snapshot"
            )
        if receipt.authority_class is StateAuthorityClass.AUTHORITATIVE:
            raise ControlPlaneAuthorityError(
                "export receipt cannot be authoritative"
            )
        return receipt

    def write_artifact(
        self, artifact: ExportArtifact, destination: Path | str
    ) -> Path:
        """Atomically write an artifact to ``destination``."""

        path = Path(destination)
        if artifact.format is ExportFormat.BUNDLE:
            _atomic_write_tree(path, artifact.files)
            return path
        _atomic_write_bytes(path, artifact.body)
        return path

    def export(
        self,
        source: ExportSourceData,
        request: ExportRequest | Mapping[str, Any],
        *,
        write: bool = True,
    ) -> ExportResult:
        """Render, optionally write atomically, and return a receipt-bound result."""

        req = (
            request
            if isinstance(request, ExportRequest)
            else ExportRequest(**dict(request))  # type: ignore[arg-type]
        )
        artifact = self.render(source, req)
        written = False
        destination = req.destination
        if write:
            self.write_artifact(artifact, destination)
            written = True
            # For bundles, recompute digest from durable tree to confirm write.
            dest_path = Path(destination)
            if artifact.format is ExportFormat.BUNDLE:
                tree_digest = _sha256_tree(dest_path)
                if tree_digest != artifact.digest:
                    # Include receipt.json is not part of artifact; compare file set.
                    # Re-hash only the declared artifact files.
                    entries = []
                    for relative in sorted(artifact.files):
                        member = dest_path / relative
                        entries.append(
                            {
                                "path": relative,
                                "digest": _sha256_file(member),
                            }
                        )
                    durable = _sha256_bytes(canonical_json_bytes(entries))
                    if durable != artifact.digest:
                        raise StateExportWriteError(
                            "durable bundle digest diverged from render"
                        )
            else:
                durable = _sha256_file(dest_path)
                if durable != artifact.digest:
                    raise StateExportWriteError(
                        "durable file digest diverged from render"
                    )
            # Optionally co-locate a receipt sidecar for operator inspection.
            # The sidecar is never required for runtime decisions.
            receipt = self.build_receipt(
                source, req, artifact, destination=destination
            )
            if artifact.format is ExportFormat.BUNDLE:
                receipt_path = Path(destination) / _PORTABLE_RECEIPT_NAME
                _atomic_write_bytes(
                    receipt_path, _json_document_bytes(receipt.to_record())
                )
            else:
                receipt_path = Path(str(destination) + ".receipt.json")
                _atomic_write_bytes(
                    receipt_path, _json_document_bytes(receipt.to_record())
                )
        else:
            receipt = self.build_receipt(
                source, req, artifact, destination=destination
            )
        return ExportResult(
            receipt=receipt,
            artifact=artifact,
            destination=destination,
            written=written,
        )

    def export_all_profiles(
        self,
        source: ExportSourceData,
        destination_dir: Path | str,
        *,
        write: bool = True,
        include_parquet: bool | None = None,
    ) -> tuple[ExportResult, ...]:
        """Render every closed profile under ``destination_dir``."""

        root = Path(destination_dir)
        if include_parquet is None:
            include_parquet = pyarrow_available()
        results: list[ExportResult] = []
        for profile in ExportProfile:
            if (
                profile is ExportProfile.ANALYSIS_PARQUET
                and not include_parquet
            ):
                continue
            extension = _PROFILE_EXTENSION[profile]
            if profile is ExportProfile.PORTABLE_BUNDLE:
                dest = str(root / "portable-bundle")
            else:
                dest = str(root / f"{profile.value}{extension}")
            request = ExportRequest(
                profile=profile,
                destination=dest,
                export_id=f"export:{profile.value}",
            )
            results.append(self.export(source, request, write=write))
        return tuple(results)


def runtime_decisions_ignore_exports(
    source: ExportSourceData,
    *,
    decision_fn: Callable[[ExportSourceData], Any] | None = None,
) -> Any:
    """Demonstrate that decisions use snapshot data, not export files.

    The default decision is the snapshot content id — a pure function of
    authoritative identity fields. Callers may supply a custom pure function
    over :class:`ExportSourceData`. Export file presence is intentionally
    unused.
    """

    if decision_fn is None:
        return source.snapshot.content_id
    return decision_fn(source)


def closed_export_profiles() -> tuple[str, ...]:
    return tuple(item.value for item in ExportProfile)


def closed_export_formats() -> tuple[str, ...]:
    return tuple(item.value for item in ExportFormat)


__all__ = (
    "CONTRACT_VERSION",
    "DEFAULT_PAGE_LIMIT",
    "EXPORT_AUTHORIZES_MUTATION",
    "EXPORT_DEFAULT_AUTHORITY_CLASS",
    "EXPORT_IS_AUTHORITY",
    "EXPORT_IS_COMPLETION_EVIDENCE",
    "EXPORT_REQUEST_SCHEMA",
    "EXPORT_SOURCE_DATA_SCHEMA",
    "EXPORTER_VERSION",
    "ExportArtifact",
    "ExportFormat",
    "ExportProfile",
    "ExportRequest",
    "ExportResult",
    "ExportSourceData",
    "MARKDOWN_NON_AUTHORITY_BANNER",
    "MARKDOWN_OMITTED_FIELDS",
    "PORTABLE_BUNDLE_MANIFEST_SCHEMA",
    "QUERY_VIEW_REVISION",
    "RENDERER_REVISION",
    "STATE_EXPORTER_INTERFACE",
    "STATE_EXPORTER_SCHEMA",
    "STATE_EXPORT_RECEIPT_INTERFACE",
    "STATE_EXPORT_RECEIPT_SCHEMA",
    "StateExportDependencyError",
    "StateExportError",
    "StateExportParameterError",
    "StateExportRenderError",
    "StateExportRoundTripError",
    "StateExportWriteError",
    "StateExporter",
    "closed_export_formats",
    "closed_export_profiles",
    "load_portable_bundle",
    "portable_round_trip_identical",
    "pyarrow_available",
    "render_analysis_csv",
    "render_analysis_parquet",
    "render_audit_jsonl",
    "render_events_json",
    "render_markdown",
    "render_portable_bundle",
    "render_status_json",
    "runtime_decisions_ignore_exports",
)
