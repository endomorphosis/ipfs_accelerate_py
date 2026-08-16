"""Deterministic control-plane state export (DQP-011 / StateExporter@1).

Export is a read-only rendering job over a versioned, snapshot-bound payload.
Supported media types are Markdown, JSON, JSONL, CSV, and Parquet. Destinations
are never watched as input; runtime decisions never read export files.

Each receipt binds store UUID, generation, schema revision, event watermark,
query/view revision, renderer revision, parameters, destination, and artifact
digest. Re-export of the same snapshot and parameters is byte-identical.

Portable machine exports are lossless under redaction. Human Markdown reports
declare non-authority and intentional field loss. Secrets are redacted before
any bytes leave the exporter.

Cold import of this module performs no filesystem, database, network, provider,
or process action.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Mapping as TypingMapping

from .control_plane_contracts import (
    STATE_EXPORT_RECEIPT_INTERFACE,
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
STATE_EXPORT_REQUEST_INTERFACE: Final[str] = "StateExportRequest@1"
STATE_EXPORT_PAYLOAD_INTERFACE: Final[str] = "StateExportPayload@1"

STATE_EXPORTER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-exporter@1"
)
STATE_EXPORT_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-export-request@1"
)
STATE_EXPORT_PAYLOAD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-export-payload@1"
)
PORTABLE_EXPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/portable-state-export@1"
)

RENDERER_VERSION: Final[str] = "1"
QUERY_VIEW_VERSION: Final[str] = "1"
EXPORTER_VERSION: Final[str] = "state-export/1"

DEFAULT_PAGE_LIMIT: Final[int] = 1_000
MAX_PAGE_LIMIT: Final[int] = 50_000
MAX_OFFSET: Final[int] = 10_000_000
MAX_DESTINATION_BYTES: Final[int] = 4_096
MAX_PARAMETER_BYTES: Final[int] = 65_536

# Markdown intentionally omits these full domains from the human report.
MARKDOWN_OMITTED_FIELDS: Final[tuple[str, ...]] = (
    "leases",
    "commands",
    "store_identity.metadata",
    "raw_body_json",
    "secret_handles",
    "fence_tokens",
)

NON_AUTHORITY_BANNER: Final[str] = (
    "NON-AUTHORITATIVE EXPORT — runtime decisions must not read this file. "
    "Database snapshot identity is the sole authority."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class StateExportError(RuntimeError):
    """Base class for fail-closed state export errors."""


class StateExportRequestError(StateExportError, ValueError):
    """The export request is malformed or inconsistent."""


class StateExportPayloadError(StateExportError, ValueError):
    """The export payload is malformed or does not bind a snapshot."""


class StateExportFormatError(StateExportError, ValueError):
    """Requested media type / view combination is unsupported."""


class StateExportDependencyError(StateExportError):
    """An optional dependency required for a format is unavailable."""


class StateExportIOError(StateExportError):
    """Filesystem write failed (atomic replace rolled back)."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ExportMediaType(str, Enum):
    """Closed set of export media types."""

    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class ExportView(str, Enum):
    """Closed set of versioned export views / query profiles."""

    PORTABLE = "portable"
    TASKBOARD = "taskboard"
    STATUS = "status"
    EVENTS = "events"
    ANALYSIS = "analysis"


# Views that are intentionally lossy human/tool projections.
_LOSSY_VIEWS: Final[frozenset[ExportView]] = frozenset(
    {
        ExportView.TASKBOARD,
        ExportView.STATUS,
        ExportView.ANALYSIS,
    }
)

# Media types allowed per view.
_VIEW_MEDIA: Final[Mapping[ExportView, frozenset[ExportMediaType]]] = (
    MappingProxyType(
        {
            ExportView.PORTABLE: frozenset(
                {
                    ExportMediaType.JSON,
                    ExportMediaType.JSONL,
                    ExportMediaType.CSV,
                    ExportMediaType.PARQUET,
                }
            ),
            ExportView.TASKBOARD: frozenset({ExportMediaType.MARKDOWN}),
            ExportView.STATUS: frozenset(
                {ExportMediaType.JSON, ExportMediaType.JSONL}
            ),
            ExportView.EVENTS: frozenset(
                {ExportMediaType.JSONL, ExportMediaType.JSON}
            ),
            ExportView.ANALYSIS: frozenset(
                {
                    ExportMediaType.CSV,
                    ExportMediaType.PARQUET,
                    ExportMediaType.JSON,
                }
            ),
        }
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    limit: int = MAX_DESTINATION_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise StateExportRequestError(f"{field_name} must be a string")
    else:
        text = value
    if text != text.strip():
        raise StateExportRequestError(
            f"{field_name} has leading or trailing whitespace"
        )
    if required and not text:
        raise StateExportRequestError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise StateExportRequestError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > limit:
        raise StateExportRequestError(f"{field_name} exceeds its byte bound")
    return text


def _enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise StateExportRequestError(
                f"{field_name} is not a closed {enum_cls.__name__} value"
            ) from exc
    raise StateExportRequestError(
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
        raise StateExportRequestError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise StateExportRequestError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StateExportPayloadError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _freeze_rows(rows: Sequence[Mapping[str, Any]] | None) -> tuple[dict[str, Any], ...]:
    if rows is None:
        return ()
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise StateExportPayloadError("rows must be a sequence of objects")
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StateExportPayloadError(f"row {index} must be an object")
        frozen.append(dict(redact_mapping(dict(row))))
    return tuple(frozen)


def _sort_key_for_domain(domain: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    if domain == "tasks":
        return (str(row.get("task_cid") or row.get("task_id") or ""),)
    if domain == "leases":
        return (str(row.get("task_cid") or row.get("id") or ""),)
    if domain == "events":
        return (
            int(row.get("global_sequence") or row.get("sequence") or 0),
            str(row.get("event_id") or ""),
        )
    if domain == "commands":
        return (str(row.get("idempotency_key") or row.get("command_id") or ""),)
    return (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False),)


def _stable_rows(
    domain: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    material = [dict(row) for row in rows]
    material.sort(key=lambda row: _sort_key_for_domain(domain, row))
    return material


def _page_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    if offset < 0:
        raise StateExportRequestError("offset must be non-negative")
    if limit < 1:
        raise StateExportRequestError("limit must be at least 1")
    return [dict(row) for row in rows[offset : offset + limit]]


def renderer_revision_for(media_type: ExportMediaType) -> str:
    return f"renderer:{media_type.value}@{RENDERER_VERSION}"


def query_revision_for(view: ExportView) -> str:
    return f"view:{view.value}@{QUERY_VIEW_VERSION}"


def media_type_from_path(path: Path | str) -> ExportMediaType:
    """Infer media type from a destination suffix."""

    suffix = Path(path).suffix.lower()
    mapping = {
        ".md": ExportMediaType.MARKDOWN,
        ".markdown": ExportMediaType.MARKDOWN,
        ".json": ExportMediaType.JSON,
        ".jsonl": ExportMediaType.JSONL,
        ".csv": ExportMediaType.CSV,
        ".parquet": ExportMediaType.PARQUET,
    }
    if suffix not in mapping:
        raise StateExportRequestError(
            f"cannot infer media type from destination suffix {suffix!r}"
        )
    return mapping[suffix]


def intentional_loss_for(view: ExportView, media_type: ExportMediaType) -> bool:
    if media_type is ExportMediaType.MARKDOWN:
        return True
    return view in _LOSSY_VIEWS


# ---------------------------------------------------------------------------
# Request / payload contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateExportRequest:
    """Parameters for one deterministic export rendering.

    Interface: ``StateExportRequest@1``.
    """

    SCHEMA: ClassVar[str] = STATE_EXPORT_REQUEST_SCHEMA
    INTERFACE: ClassVar[str] = STATE_EXPORT_REQUEST_INTERFACE

    destination: str
    media_type: ExportMediaType
    view: ExportView = ExportView.PORTABLE
    offset: int = 0
    limit: int = DEFAULT_PAGE_LIMIT
    domains: tuple[str, ...] = ("tasks", "leases", "events", "commands")
    parameters: TypingMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "destination",
            _text(self.destination, "destination", limit=MAX_DESTINATION_BYTES),
        )
        media = _enum(
            self.media_type, ExportMediaType, field_name="media_type"
        )
        view = _enum(self.view, ExportView, field_name="view")
        object.__setattr__(self, "media_type", media)
        object.__setattr__(self, "view", view)
        object.__setattr__(
            self,
            "offset",
            _bounded_int(self.offset, "offset", minimum=0, maximum=MAX_OFFSET),
        )
        object.__setattr__(
            self,
            "limit",
            _bounded_int(self.limit, "limit", minimum=1, maximum=MAX_PAGE_LIMIT),
        )
        if not isinstance(self.domains, Sequence) or isinstance(
            self.domains, (str, bytes, bytearray)
        ):
            raise StateExportRequestError("domains must be a sequence of strings")
        domains = tuple(str(item).strip() for item in self.domains if str(item).strip())
        if not domains:
            raise StateExportRequestError("domains must not be empty")
        object.__setattr__(self, "domains", domains)
        params = dict(self.parameters or {})
        # Parameters are redacted and frozen; secret keys become markers.
        redacted = redact_mapping(params)
        if not isinstance(redacted, dict):
            raise StateExportRequestError("parameters must be an object")
        encoded = canonical_json_bytes(redacted)
        if len(encoded) > MAX_PARAMETER_BYTES:
            raise StateExportRequestError("parameters exceed their byte bound")
        object.__setattr__(self, "parameters", MappingProxyType(redacted))
        allowed = _VIEW_MEDIA.get(view, frozenset())
        if media not in allowed:
            raise StateExportFormatError(
                f"view {view.value!r} does not support media type {media.value!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "destination": self.destination,
            "media_type": self.media_type.value,
            "view": self.view.value,
            "offset": self.offset,
            "limit": self.limit,
            "domains": list(self.domains),
            "parameters": dict(self.parameters),
        }

    def receipt_parameters(self) -> dict[str, Any]:
        """Stable parameter bag recorded on the export receipt.

        Values must remain JSON-canonical after ``StateExportReceipt`` freezes
        them: sequences become tuples under freeze, and receipt content
        identity rejects tuples. Encode multi-valued fields as scalars.
        """

        safe_user: dict[str, Any] = {}
        for key, value in dict(self.parameters).items():
            # Drop secret-bearing keys entirely; freeze rejects them even when
            # values are redaction markers.
            normalized = str(key).lower().replace("-", "_").strip()
            if normalized in {
                "access_token",
                "api_key",
                "authorization",
                "client_secret",
                "cookie",
                "credential",
                "credentials",
                "password",
                "passwd",
                "private_key",
                "refresh_token",
                "secret",
                "session_token",
                "token",
            } or any(
                marker in normalized
                for marker in (
                    "password",
                    "private_key",
                    "access_token",
                    "api_key",
                    "client_secret",
                    "refresh_token",
                )
            ):
                continue
            if isinstance(value, (list, tuple)):
                safe_user[str(key)] = json.dumps(
                    list(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            elif isinstance(value, Mapping):
                safe_user[str(key)] = json.dumps(
                    dict(value),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            elif value is None or isinstance(value, (str, bool, int)):
                safe_user[str(key)] = value
            else:
                safe_user[str(key)] = str(value)
        return {
            "view": self.view.value,
            "media_type": self.media_type.value,
            "offset": self.offset,
            "limit": self.limit,
            # Comma-joined scalar: freeze keeps strings, not tuples.
            "domains": ",".join(self.domains),
            "exporter_version": EXPORTER_VERSION,
            **safe_user,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateExportRequest":
        data = _mapping(payload, "export request")
        return cls(
            destination=str(data.get("destination") or ""),
            media_type=str(data.get("media_type") or ""),
            view=str(data.get("view") or ExportView.PORTABLE.value),
            offset=int(data.get("offset") or 0),
            limit=int(data.get("limit") or DEFAULT_PAGE_LIMIT),
            domains=tuple(data.get("domains") or ("tasks", "leases", "events", "commands")),
            parameters=data.get("parameters") or {},
        )


@dataclass(frozen=True)
class StateExportPayload:
    """Snapshot-bound, redacted population used as the sole export input.

    Interface: ``StateExportPayload@1``.

    The payload is already a consistent bounded snapshot: exporters never
    re-read live mutable state after construction.
    """

    SCHEMA: ClassVar[str] = STATE_EXPORT_PAYLOAD_SCHEMA
    INTERFACE: ClassVar[str] = STATE_EXPORT_PAYLOAD_INTERFACE

    snapshot: StateSnapshot
    store_identity: TypingMapping[str, Any] = field(default_factory=dict)
    generation: TypingMapping[str, Any] = field(default_factory=dict)
    tasks: tuple[dict[str, Any], ...] = ()
    leases: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    commands: tuple[dict[str, Any], ...] = ()
    schema_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, StateSnapshot):
            raise StateExportPayloadError("snapshot must be a StateSnapshot")
        object.__setattr__(
            self,
            "store_identity",
            MappingProxyType(
                dict(redact_mapping(_mapping(self.store_identity, "store_identity")))
            ),
        )
        object.__setattr__(
            self,
            "generation",
            MappingProxyType(
                dict(redact_mapping(_mapping(self.generation, "generation")))
            ),
        )
        object.__setattr__(self, "tasks", _freeze_rows(self.tasks))
        object.__setattr__(self, "leases", _freeze_rows(self.leases))
        object.__setattr__(self, "events", _freeze_rows(self.events))
        object.__setattr__(self, "commands", _freeze_rows(self.commands))
        fingerprint = str(self.schema_fingerprint or "").strip()
        object.__setattr__(self, "schema_fingerprint", fingerprint)

    def domain_rows(self, domain: str) -> list[dict[str, Any]]:
        table = {
            "tasks": self.tasks,
            "leases": self.leases,
            "events": self.events,
            "commands": self.commands,
        }
        if domain not in table:
            raise StateExportRequestError(f"unknown export domain {domain!r}")
        return _stable_rows(domain, table[domain])

    def to_portable_dict(self) -> dict[str, Any]:
        """Lossless (under redaction) portable machine projection."""

        return {
            "schema": PORTABLE_EXPORT_SCHEMA,
            "exporter_version": EXPORTER_VERSION,
            "authority_class": StateAuthorityClass.EXPORT.value,
            "intentional_loss": False,
            "snapshot": self.snapshot.to_dict(),
            "store_identity": dict(self.store_identity),
            "generation": dict(self.generation),
            "schema_fingerprint": self.schema_fingerprint,
            "tasks": _stable_rows("tasks", self.tasks),
            "leases": _stable_rows("leases", self.leases),
            "events": _stable_rows("events", self.events),
            "commands": _stable_rows("commands", self.commands),
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_portable_dict())

    @classmethod
    def from_portable_dict(cls, payload: Mapping[str, Any]) -> "StateExportPayload":
        data = _mapping(payload, "portable export")
        schema = str(data.get("schema") or "")
        if schema and schema != PORTABLE_EXPORT_SCHEMA:
            raise StateExportPayloadError(
                f"unsupported portable export schema {schema!r}"
            )
        snapshot_payload = data.get("snapshot")
        if not isinstance(snapshot_payload, Mapping):
            raise StateExportPayloadError("portable export requires snapshot")
        snapshot = StateSnapshot.from_dict(snapshot_payload)
        return cls(
            snapshot=snapshot,
            store_identity=data.get("store_identity") or {},
            generation=data.get("generation") or {},
            tasks=tuple(data.get("tasks") or ()),
            leases=tuple(data.get("leases") or ()),
            events=tuple(data.get("events") or ()),
            commands=tuple(data.get("commands") or ()),
            schema_fingerprint=str(data.get("schema_fingerprint") or ""),
        )

    @classmethod
    def from_repository(cls, repository: Any) -> "StateExportPayload":
        """Build a consistent payload from a StateRepository-like object.

        Captures ``snapshot()`` and ``canonical_population()`` (or equivalent
        list methods) once; subsequent renders use only this frozen payload.
        """

        if not hasattr(repository, "snapshot"):
            raise StateExportPayloadError(
                "repository must provide snapshot() for export binding"
            )
        snapshot = repository.snapshot()
        if not isinstance(snapshot, StateSnapshot):
            raise StateExportPayloadError(
                "repository.snapshot() must return StateSnapshot"
            )

        population = None
        if hasattr(repository, "canonical_population"):
            population = repository.canonical_population()

        if population is not None and hasattr(population, "to_dict"):
            pop = population.to_dict()
            return cls(
                snapshot=snapshot,
                store_identity=pop.get("store_identity") or {},
                generation=pop.get("generation") or {},
                tasks=tuple(pop.get("tasks") or ()),
                leases=tuple(pop.get("leases") or ()),
                events=tuple(pop.get("events") or ()),
                commands=tuple(pop.get("commands") or ()),
                schema_fingerprint=str(pop.get("schema_fingerprint") or ""),
            )

        # Fallback: page repository methods if population is unavailable.
        tasks = _drain_pages(repository, "list_tasks")
        events = _drain_pages(repository, "list_events")
        leases = list(getattr(repository, "list_leases", lambda: ())())
        commands = list(getattr(repository, "list_commands", lambda: ())())
        generation: dict[str, Any] = {}
        store_identity: dict[str, Any] = {}
        schema_fingerprint = ""
        if hasattr(repository, "load_generation"):
            gen = repository.load_generation()
            generation = gen.to_dict() if hasattr(gen, "to_dict") else dict(gen)
        if hasattr(repository, "store_identity"):
            identity = repository.store_identity()
            store_identity = (
                identity.to_dict()
                if hasattr(identity, "to_dict")
                else dict(identity)
            )
            schema_fingerprint = str(
                store_identity.get("schema_fingerprint") or ""
            )
        return cls(
            snapshot=snapshot,
            store_identity=store_identity,
            generation=generation,
            tasks=tuple(tasks),
            leases=tuple(leases),
            events=tuple(events),
            commands=tuple(commands),
            schema_fingerprint=schema_fingerprint,
        )


def _drain_pages(repository: Any, method_name: str) -> list[dict[str, Any]]:
    method = getattr(repository, method_name, None)
    if method is None:
        return []
    rows: list[dict[str, Any]] = []
    cursor = 0
    while True:
        page = method(cursor=cursor, limit=DEFAULT_PAGE_LIMIT)
        items = getattr(page, "items", page)
        for item in items:
            if isinstance(item, Mapping):
                rows.append(dict(item))
        exhausted = getattr(page, "exhausted", True)
        next_cursor = getattr(page, "next_cursor", None)
        if exhausted or next_cursor is None:
            break
        cursor = int(next_cursor)
    return rows


# ---------------------------------------------------------------------------
# Renderers (pure, deterministic)
# ---------------------------------------------------------------------------


def _status_projection(payload: StateExportPayload) -> dict[str, Any]:
    tasks = _stable_rows("tasks", payload.tasks)
    status_counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema": f"{STATE_EXPORTER_SCHEMA}/status-projection@1",
        "authority_class": StateAuthorityClass.EXPORT.value,
        "intentional_loss": True,
        "snapshot_id": payload.snapshot.snapshot_id,
        "store_id": payload.snapshot.store_id,
        "database_uuid": payload.snapshot.database_uuid,
        "generation": payload.snapshot.generation,
        "schema_revision": payload.snapshot.schema_revision,
        "revision": payload.snapshot.revision,
        "event_watermark": payload.snapshot.event_watermark,
        "task_count": len(tasks),
        "status_counts": dict(sorted(status_counts.items())),
        "omitted_fields": [
            "task bodies",
            "leases",
            "commands",
            "event payloads",
        ],
    }


def _analysis_rows(
    payload: StateExportPayload,
    request: StateExportRequest,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in request.domains:
        for item in payload.domain_rows(domain):
            flat: dict[str, Any] = {"domain": domain}
            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(
                        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                    )
                elif value is None or isinstance(value, (str, bool, int)):
                    flat[key] = value
                else:
                    flat[key] = str(value)
            rows.append(flat)
    rows.sort(
        key=lambda row: (
            str(row.get("domain") or ""),
            str(row.get("task_cid") or row.get("event_id") or row.get("id") or ""),
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )
    )
    return _page_rows(rows, offset=request.offset, limit=request.limit)


def _render_markdown(payload: StateExportPayload, request: StateExportRequest) -> bytes:
    snapshot = payload.snapshot
    tasks = _page_rows(
        payload.domain_rows("tasks"),
        offset=request.offset,
        limit=request.limit,
    )
    lines: list[str] = [
        "# Control-plane export (taskboard view)",
        "",
        f"> {NON_AUTHORITY_BANNER}",
        "",
        "- Authority class: export",
        "- Intentional loss: true",
        f"- Snapshot id: {snapshot.snapshot_id}",
        f"- Store id: {snapshot.store_id}",
        f"- Database UUID: {snapshot.database_uuid}",
        f"- Generation: {snapshot.generation}",
        f"- Schema revision: {snapshot.schema_revision}",
        f"- Revision: {snapshot.revision}",
        f"- Event watermark: {snapshot.event_watermark}",
        f"- Snapshot digest: {snapshot.snapshot_digest}",
        f"- Query revision: {query_revision_for(request.view)}",
        f"- Renderer revision: {renderer_revision_for(request.media_type)}",
        f"- Exporter version: {EXPORTER_VERSION}",
        f"- Offset: {request.offset}",
        f"- Limit: {request.limit}",
        f"- Task count (page): {len(tasks)}",
        "",
        "## Intentionally omitted fields",
        "",
    ]
    for name in MARKDOWN_OMITTED_FIELDS:
        lines.append(f"- {name}")
    lines.extend(["", "## Tasks", ""])
    if not tasks:
        lines.append("_No tasks in this page._")
        lines.append("")
    for task in tasks:
        task_id = str(
            task.get("task_alias")
            or task.get("task_id")
            or task.get("task_cid")
            or "unknown"
        )
        title = str(task.get("title") or task.get("summary") or "").strip()
        heading = f"## {task_id}"
        if title:
            heading = f"{heading} {title}"
        lines.append(heading)
        lines.append("")
        for key in ("status", "priority", "goal_cid", "objective_id", "ordinal"):
            if key in task and task[key] not in (None, ""):
                lines.append(f"- {key}: {task[key]}")
        lines.append("")
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def _render_json(document: Mapping[str, Any]) -> bytes:
    # Canonical separators + sorted keys; trailing newline for POSIX tools.
    body = canonical_json_bytes(dict(document))
    return body + b"\n"


def _render_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    chunks: list[bytes] = []
    for row in rows:
        chunks.append(canonical_json_bytes(dict(row)))
    if not chunks:
        return b""
    return b"\n".join(chunks) + b"\n"


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Stable CSV header: ``domain`` first when present, then sorted keys."""

    keys = {str(key) for row in rows for key in row.keys()}
    fieldnames: list[str] = []
    if "domain" in keys:
        fieldnames.append("domain")
        keys.discard("domain")
    fieldnames.extend(sorted(keys))
    return fieldnames


def _render_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    fieldnames = _csv_fieldnames(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="ignore",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for row in rows:
        serialized: dict[str, Any] = {}
        for key in fieldnames:
            value = row.get(key, "")
            if value is None:
                serialized[key] = ""
            elif isinstance(value, (dict, list)):
                serialized[key] = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            else:
                serialized[key] = value
        writer.writerow(serialized)
    return buffer.getvalue().encode("utf-8")


def _render_parquet(rows: Sequence[Mapping[str, Any]]) -> bytes:
    """Render a deterministic Parquet blob via DuckDB when available."""

    if not duckdb_available():
        raise StateExportDependencyError(
            "DuckDB is required for Parquet export; install the optional duckdb dependency"
        )
    import duckdb  # type: ignore

    # Normalize rows to a stable column set of JSON-canonical scalars.
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key in fieldnames:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                item[key] = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            elif value is None or isinstance(value, (str, bool, int)):
                item[key] = value
            else:
                item[key] = str(value)
        normalized.append(item)

    connection = duckdb.connect(database=":memory:")
    try:
        # Single-threaded write reduces non-determinism across renderers.
        try:
            connection.execute("PRAGMA threads=1")
        except Exception:
            pass
        if not fieldnames:
            connection.execute("CREATE TABLE export_rows (domain VARCHAR)")
        else:
            cols_sql = ", ".join(
                f'"{name.replace(chr(34), chr(34) + chr(34))}" VARCHAR'
                for name in fieldnames
            )
            connection.execute(f"CREATE TABLE export_rows ({cols_sql})")
            for item in normalized:
                placeholders = ", ".join("?" for _ in fieldnames)
                values = [item.get(name) for name in fieldnames]
                connection.execute(
                    f"INSERT INTO export_rows VALUES ({placeholders})",
                    values,
                )
        # COPY to a temp file then read bytes (DuckDB requires a path for parquet).
        fd, temp_name = tempfile.mkstemp(prefix="state-export-", suffix=".parquet")
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            # Quote path for SQL; path is under tempfile and not operator-controlled SQL.
            escaped = temp_path.as_posix().replace("'", "''")
            connection.execute(
                f"COPY export_rows TO '{escaped}' "
                "(FORMAT PARQUET, COMPRESSION 'uncompressed')"
            )
            return temp_path.read_bytes()
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        connection.close()


def render_export_bytes(
    payload: StateExportPayload,
    request: StateExportRequest,
) -> bytes:
    """Pure render of export artifact bytes (no filesystem writes)."""

    view = request.view
    media = request.media_type

    if view is ExportView.TASKBOARD and media is ExportMediaType.MARKDOWN:
        return _render_markdown(payload, request)

    if view is ExportView.STATUS:
        document = _status_projection(payload)
        if media is ExportMediaType.JSON:
            return _render_json(document)
        if media is ExportMediaType.JSONL:
            return _render_jsonl([document])

    if view is ExportView.EVENTS:
        events = _page_rows(
            payload.domain_rows("events"),
            offset=request.offset,
            limit=request.limit,
        )
        if media is ExportMediaType.JSONL:
            return _render_jsonl(events)
        if media is ExportMediaType.JSON:
            return _render_json(
                {
                    "schema": f"{STATE_EXPORTER_SCHEMA}/events-projection@1",
                    "authority_class": StateAuthorityClass.EXPORT.value,
                    "intentional_loss": False,
                    "snapshot_id": payload.snapshot.snapshot_id,
                    "events": events,
                }
            )

    if view is ExportView.ANALYSIS:
        rows = _analysis_rows(payload, request)
        if media is ExportMediaType.CSV:
            return _render_csv(rows)
        if media is ExportMediaType.PARQUET:
            return _render_parquet(rows)
        if media is ExportMediaType.JSON:
            return _render_json(
                {
                    "schema": f"{STATE_EXPORTER_SCHEMA}/analysis-projection@1",
                    "authority_class": StateAuthorityClass.EXPORT.value,
                    "intentional_loss": True,
                    "snapshot_id": payload.snapshot.snapshot_id,
                    "rows": rows,
                }
            )

    if view is ExportView.PORTABLE:
        portable = payload.to_portable_dict()
        if media is ExportMediaType.JSON:
            return _render_json(portable)
        if media is ExportMediaType.JSONL:
            # One domain object per line with a leading envelope line.
            envelope = {
                "schema": PORTABLE_EXPORT_SCHEMA,
                "kind": "envelope",
                "exporter_version": EXPORTER_VERSION,
                "authority_class": StateAuthorityClass.EXPORT.value,
                "intentional_loss": False,
                "snapshot": payload.snapshot.to_dict(),
                "store_identity": dict(payload.store_identity),
                "generation": dict(payload.generation),
                "schema_fingerprint": payload.schema_fingerprint,
            }
            lines: list[dict[str, Any]] = [envelope]
            for domain in request.domains:
                for row in _page_rows(
                    payload.domain_rows(domain),
                    offset=request.offset,
                    limit=request.limit,
                ):
                    lines.append({"kind": "row", "domain": domain, **row})
            return _render_jsonl(lines)
        if media is ExportMediaType.CSV:
            return _render_csv(_analysis_rows(payload, request))
        if media is ExportMediaType.PARQUET:
            return _render_parquet(_analysis_rows(payload, request))

    raise StateExportFormatError(
        f"unsupported render combination view={view.value} media={media.value}"
    )


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def atomic_write_bytes(destination: Path | str, payload: bytes) -> str:
    """Atomically replace ``destination`` with ``payload``; return sha256 digest."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha256_bytes(payload)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise StateExportIOError(
            f"atomic export write failed for {path}: {exc}"
        ) from exc
    return digest


# ---------------------------------------------------------------------------
# StateExporter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateExporter:
    """Read-only deterministic exporter over snapshot-bound payloads.

    Interface: ``StateExporter@1``.

    Construction is pure. The first I/O boundary is :meth:`export` (or the
    explicit :func:`atomic_write_bytes` helper). Export never mutates the
    source payload or any control-plane database.
    """

    INTERFACE: ClassVar[str] = STATE_EXPORTER_INTERFACE
    SCHEMA: ClassVar[str] = STATE_EXPORTER_SCHEMA

    def render(
        self,
        payload: StateExportPayload,
        request: StateExportRequest,
    ) -> bytes:
        """Return deterministic artifact bytes without writing."""

        if not isinstance(payload, StateExportPayload):
            raise StateExportPayloadError("payload must be StateExportPayload")
        if not isinstance(request, StateExportRequest):
            raise StateExportRequestError("request must be StateExportRequest")
        return render_export_bytes(payload, request)

    def export(
        self,
        payload: StateExportPayload,
        request: StateExportRequest,
    ) -> StateExportReceipt:
        """Render and atomically write an export; return a bound receipt.

        Re-export of the same snapshot and parameters is byte-identical.
        Deleting or tampering with the destination cannot affect runtime
        decisions because exports are non-authoritative projections only.
        """

        artifact = self.render(payload, request)
        destination = Path(request.destination)
        digest = atomic_write_bytes(destination, artifact)
        return self.build_receipt(
            payload=payload,
            request=request,
            artifact_digest=digest,
        )

    def build_receipt(
        self,
        *,
        payload: StateExportPayload,
        request: StateExportRequest,
        artifact_digest: str,
        artifact_bytes: bytes | None = None,
    ) -> StateExportReceipt:
        """Build a ``StateExportReceipt@1`` bound to the snapshot."""

        snapshot = payload.snapshot
        if artifact_bytes is not None:
            artifact_digest = _sha256_bytes(artifact_bytes)
        export_id = self._export_id(
            snapshot=snapshot,
            request=request,
            artifact_digest=artifact_digest,
        )
        return StateExportReceipt(
            export_id=export_id,
            snapshot_id=snapshot.snapshot_id,
            store_id=snapshot.store_id,
            database_uuid=snapshot.database_uuid,
            schema_revision=snapshot.schema_revision,
            generation=snapshot.generation,
            revision=snapshot.revision,
            event_watermark=snapshot.event_watermark,
            renderer_revision=renderer_revision_for(request.media_type),
            query_revision=query_revision_for(request.view),
            artifact_digest=artifact_digest,
            destination=str(request.destination),
            parameters=request.receipt_parameters(),
            authority_class=StateAuthorityClass.EXPORT,
            intentional_loss=intentional_loss_for(request.view, request.media_type),
        )

    def load_portable(self, path: Path | str) -> StateExportPayload:
        """Load a lossless portable JSON export (round-trip helper)."""

        destination = Path(path)
        text = destination.read_text(encoding="utf-8")
        document = json.loads(text)
        if not isinstance(document, Mapping):
            raise StateExportPayloadError("portable export root must be an object")
        return StateExportPayload.from_portable_dict(document)

    @staticmethod
    def _export_id(
        *,
        snapshot: StateSnapshot,
        request: StateExportRequest,
        artifact_digest: str,
    ) -> str:
        material = {
            "snapshot_id": snapshot.snapshot_id,
            "store_id": snapshot.store_id,
            "database_uuid": snapshot.database_uuid,
            "generation": snapshot.generation,
            "schema_revision": snapshot.schema_revision,
            "revision": snapshot.revision,
            "event_watermark": snapshot.event_watermark,
            "view": request.view.value,
            "media_type": request.media_type.value,
            "offset": request.offset,
            "limit": request.limit,
            "domains": list(request.domains),
            "parameters": dict(request.parameters),
            "destination": request.destination,
            "artifact_digest": artifact_digest,
            "exporter_version": EXPORTER_VERSION,
        }
        digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()[:32]
        return f"export:{digest}"


def export_state(
    payload: StateExportPayload,
    request: StateExportRequest,
) -> StateExportReceipt:
    """Module-level convenience wrapper around :class:`StateExporter`."""

    return StateExporter().export(payload, request)


def render_state(
    payload: StateExportPayload,
    request: StateExportRequest,
) -> bytes:
    """Module-level pure render wrapper."""

    return StateExporter().render(payload, request)


__all__ = (
    "DEFAULT_PAGE_LIMIT",
    "EXPORTER_VERSION",
    "ExportMediaType",
    "ExportView",
    "MARKDOWN_OMITTED_FIELDS",
    "NON_AUTHORITY_BANNER",
    "PORTABLE_EXPORT_SCHEMA",
    "STATE_EXPORTER_INTERFACE",
    "STATE_EXPORTER_SCHEMA",
    "STATE_EXPORT_PAYLOAD_INTERFACE",
    "STATE_EXPORT_PAYLOAD_SCHEMA",
    "STATE_EXPORT_RECEIPT_INTERFACE",
    "STATE_EXPORT_REQUEST_INTERFACE",
    "STATE_EXPORT_REQUEST_SCHEMA",
    "StateExportDependencyError",
    "StateExportError",
    "StateExportFormatError",
    "StateExportIOError",
    "StateExportPayload",
    "StateExportPayloadError",
    "StateExportRequest",
    "StateExportRequestError",
    "StateExporter",
    "atomic_write_bytes",
    "duckdb_available",
    "export_state",
    "intentional_loss_for",
    "media_type_from_path",
    "query_revision_for",
    "render_export_bytes",
    "render_state",
    "renderer_revision_for",
)
