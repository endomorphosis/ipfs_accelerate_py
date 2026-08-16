"""Side-effect-free adapters for static provider and model inventories.

The adapter deliberately has no default source.  Callers must inject either
an in-memory value or an explicit local JSON/JSONL path, making all I/O visible
at the call site.  Loose legacy rows are projected into the strict catalog v1
records without treating static declarations as runtime health observations.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..identity import REDACTED, is_secret_key, is_secret_value
from ..schema import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    MAX_SNAPSHOT_RECORDS,
    CapabilityDescriptor,
    CatalogSnapshot,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    SchemaValidationError,
)

DEFAULT_STATIC_PRECEDENCE = 10
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_DIAGNOSTICS = 1_000
MAX_SOURCE_REVISION_BYTES = 512
MAX_ROW_FIELDS = 256
MAX_NESTED_ITEMS = 4_096

_NAME_BAD = re.compile(r"[^a-z0-9._/-]+")
_OPERATION_ALIASES = {
    "text-generation": Operation.TEXT_GENERATE,
    "text_generation": Operation.TEXT_GENERATE,
    "generate": Operation.TEXT_GENERATE,
    "completion": Operation.TEXT_GENERATE,
    "conversational": Operation.TEXT_CHAT,
    "chat": Operation.TEXT_CHAT,
    "feature-extraction": Operation.EMBEDDING_GENERATE,
    "feature_extraction": Operation.EMBEDDING_GENERATE,
    "embedding": Operation.EMBEDDING_GENERATE,
    "embeddings": Operation.EMBEDDING_GENERATE,
    "visual-question-answering": Operation.VISION_GENERATE,
    "image-to-text": Operation.VISION_GENERATE,
    "vision": Operation.VISION_GENERATE,
    "automatic-speech-recognition": Operation.AUDIO_TRANSCRIBE,
    "transcription": Operation.AUDIO_TRANSCRIBE,
    "text-to-speech": Operation.AUDIO_SYNTHESIZE,
    "speech-synthesis": Operation.AUDIO_SYNTHESIZE,
    "streaming": Operation.STREAM,
    "function-calling": Operation.TOOL_CALL,
    "function_calling": Operation.TOOL_CALL,
    "language_model": Operation.TEXT_GENERATE,
    "decoder_only": Operation.TEXT_GENERATE,
    "encoder_decoder": Operation.TEXT_GENERATE,
    "encoder_only": Operation.EMBEDDING_GENERATE,
    "embedding_model": Operation.EMBEDDING_GENERATE,
    "vision_model": Operation.VISION_GENERATE,
    "multimodal": Operation.VISION_GENERATE,
    "audio_model": Operation.AUDIO_TRANSCRIBE,
}


@dataclass(frozen=True)
class SourceDiagnostic:
    """A bounded, non-secret description of a rejected or sanitized row."""

    index: Optional[int]
    code: str
    message: str
    source_record_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SourceMetadata:
    """Ordering and revision facts used by the later catalog merge layer."""

    source: str
    precedence: int
    revision: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class CatalogSourceResult:
    """A valid partial snapshot plus diagnostics for rows that did not adapt."""

    snapshot: CatalogSnapshot
    metadata: SourceMetadata
    diagnostics: Tuple[SourceDiagnostic, ...] = ()
    redacted_fields: int = 0

    @property
    def providers(self) -> Tuple[ProviderDescriptor, ...]:
        return self.snapshot.providers

    @property
    def models(self) -> Tuple[ModelDescriptor, ...]:
        return self.snapshot.models

    @property
    def source(self) -> str:
        return self.metadata.source

    @property
    def precedence(self) -> int:
        return self.metadata.precedence

    @property
    def source_revision(self) -> Optional[str]:
        return self.metadata.revision

    @property
    def observed_at(self) -> Optional[str]:
        return self.metadata.updated_at or self.metadata.created_at

    @property
    def revision(self) -> str:
        return self.snapshot.revision  # type: ignore[return-value]

    @property
    def error_count(self) -> int:
        return sum(item.code != "redacted" for item in self.diagnostics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "metadata": self.metadata.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "redacted_fields": self.redacted_fields,
        }


def _bounded_text(
    value: Any, field_name: str, maximum: int, *, optional: bool = True
) -> Optional[str]:
    if value is None and optional:
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % field_name)
    value = value.strip()
    if not value and optional:
        return None
    if len(value.encode("utf-8")) > maximum:
        raise ValueError("%s exceeds %d UTF-8 bytes" % (field_name, maximum))
    if is_secret_value(value):
        return REDACTED
    return value


def _canonical_name(value: Any, field_name: str) -> str:
    value = _bounded_text(value, field_name, MAX_NAME_LENGTH, optional=False)
    assert value is not None
    value = _NAME_BAD.sub("-", value.casefold()).strip("-._/")
    value = re.sub(r"/+", "/", value)
    value = re.sub(r"\.{2,}", ".", value)
    if not value:
        raise ValueError("%s has no canonical name characters" % field_name)
    if len(value.encode("utf-8")) > MAX_NAME_LENGTH:
        raise ValueError("%s exceeds %d UTF-8 bytes" % (field_name, MAX_NAME_LENGTH))
    return value


def _timestamp(value: Any, field_name: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if len(raw.encode("utf-8")) > 64:
            raise ValueError("%s exceeds 64 UTF-8 bytes" % field_name)
        try:
            parsed = datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
        except ValueError as exc:
            raise ValueError("%s is not an ISO 8601 timestamp" % field_name) from exc
    else:
        raise ValueError("%s must be an ISO 8601 timestamp" % field_name)
    # ModelManager historically serialized naive local datetimes.  Treating
    # these legacy values as UTC preserves the instant text deterministically.
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    method = getattr(value, "to_dict", None)
    if callable(method):
        converted = method()
        if isinstance(converted, Mapping):
            return converted
    raise ValueError("row must be an object or model name string")


def _first(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _sequence(value: Any, field_name: str, maximum: int = 64) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    elif isinstance(value, (set, frozenset)):
        values = tuple(value)
    else:
        raise ValueError("%s must be a string or array" % field_name)
    if len(values) > maximum:
        raise ValueError("%s exceeds maximum count" % field_name)
    return tuple(values)


def _aliases(row: Mapping[str, Any], canonical_name: str) -> Tuple[str, ...]:
    values = _first(row, ("aliases", "alias", "model_aliases", "provider_aliases"))
    aliases = {_canonical_name(item, "alias") for item in _sequence(values, "aliases")}
    aliases.discard(canonical_name)
    return tuple(sorted(aliases))


def _operations(row: Mapping[str, Any]) -> Tuple[Operation, ...]:
    raw = _first(
        row,
        ("operations", "pipeline_types", "tasks", "task", "capabilities", "model_type"),
    )
    operations = set()
    for item in _sequence(raw, "operations", maximum=32):
        if isinstance(item, Operation):
            operations.add(item)
            continue
        if not isinstance(item, str):
            raise ValueError("operation names must be strings")
        normalized = item.strip().casefold()
        try:
            operations.add(Operation(normalized))
        except ValueError:
            operation = _OPERATION_ALIASES.get(normalized)
            if operation is None:
                raise ValueError("unknown operation name")
            operations.add(operation)
    if row.get("supports_streaming") is True or row.get("streaming") is True:
        operations.add(Operation.STREAM)
    if row.get("function_calling") is True or row.get("tool_calling") is True:
        operations.add(Operation.TOOL_CALL)
    if row.get("vision_capable") is True and not operations:
        operations.add(Operation.VISION_GENERATE)
    qualifiers = {Operation.STREAM, Operation.BATCH}
    if operations and not operations - qualifiers:
        raise ValueError("stream and batch require an invokable operation")
    return tuple(sorted(operations, key=lambda item: item.value))


def _capabilities(row: Mapping[str, Any]) -> Tuple[CapabilityDescriptor, ...]:
    operations = _operations(row)
    if not operations:
        return ()
    inputs = {Modality.TEXT}
    outputs = {Modality.TEXT}
    if Operation.EMBEDDING_GENERATE in operations:
        outputs.add(Modality.EMBEDDING)
    if Operation.VISION_GENERATE in operations or row.get("vision_capable") is True:
        inputs.add(Modality.IMAGE)
    if Operation.AUDIO_TRANSCRIBE in operations:
        inputs.add(Modality.AUDIO)
    if Operation.AUDIO_SYNTHESIZE in operations:
        outputs.add(Modality.AUDIO)
    context = _first(row, ("max_context_tokens", "context_length", "context_window"))
    if context is not None and (isinstance(context, bool) or not isinstance(context, int)):
        raise ValueError("context length must be an integer")
    return (
        CapabilityDescriptor(
            operations=operations,
            input_modalities=tuple(inputs),
            output_modalities=tuple(outputs),
            max_context_tokens=context,
        ),
    )


def _lifecycle(row: Mapping[str, Any]) -> LifecycleState:
    if row.get("deprecated") is True:
        return LifecycleState.DEPRECATED
    raw = _first(row, ("lifecycle", "lifecycle_state", "status"))
    if raw is None:
        return LifecycleState.DECLARED
    if isinstance(raw, LifecycleState):
        return raw
    if not isinstance(raw, str):
        raise ValueError("lifecycle must be a string")
    normalized = raw.strip().casefold()
    if is_secret_value(normalized):
        return LifecycleState.DECLARED
    # Runtime-looking values in a static file remain declarations.
    if normalized in {"ready", "healthy", "available", "online", "active"}:
        return LifecycleState.DECLARED
    try:
        return LifecycleState(normalized)
    except ValueError as exc:
        raise ValueError("unknown lifecycle") from exc


def _secret_count(value: Any, depth: int = 0) -> int:
    if depth > 24:
        return 0
    if isinstance(value, Mapping):
        count = 0
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_NESTED_ITEMS:
                break
            if is_secret_key(str(key)):
                count += 1
            else:
                count += _secret_count(item, depth + 1)
        return count
    if isinstance(value, (list, tuple, set, frozenset)):
        count = 0
        for index, item in enumerate(value):
            if index >= MAX_NESTED_ITEMS:
                break
            count += _secret_count(item, depth + 1)
        return count
    return int(isinstance(value, str) and is_secret_value(value))


def _safe_record_id(value: Any) -> Optional[str]:
    try:
        return _bounded_text(value, "source_record_id", 512)
    except ValueError:
        return None


def _diagnostic_message(error: Exception) -> str:
    message = str(error)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", message)
    if is_secret_value(message) or any(is_secret_key(token) for token in tokens):
        return "row contains invalid or credential-shaped data"
    return message[:512]


def _row_timestamps(
    row: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    created = _timestamp(_first(row, ("created_at", "created", "date_created")), "created_at")
    updated = _timestamp(
        _first(
            row,
            (
                "updated_at",
                "updated",
                "modified_at",
                "last_modified",
                "revision_created_at",
            ),
        ),
        "updated_at",
    )
    observed = _timestamp(
        _first(row, ("observed_at", "timestamp", "collected_at")),
        "observed_at",
    )
    if created is not None and updated is not None and updated < created:
        raise ValueError("updated_at must not precede created_at")
    return created, updated, observed


def _labels(
    row: Mapping[str, Any],
    precedence: int,
    revision: Optional[str],
    created_at: Optional[str],
    updated_at: Optional[str],
) -> Dict[str, str]:
    labels = {"source.precedence": str(precedence)}
    if revision is not None and len(revision.encode("utf-8")) <= 256:
        labels["source.revision"] = revision
    if created_at is not None:
        labels["source.created-at"] = created_at
    if updated_at is not None:
        labels["source.updated-at"] = updated_at
    raw = row.get("labels")
    if isinstance(raw, Mapping):
        for key, value in sorted(raw.items(), key=lambda pair: str(pair[0])):
            if len(labels) >= 64 or is_secret_key(str(key)):
                continue
            try:
                label_key = _canonical_name(key, "label key").replace("/", ".")
                label_value = _bounded_text(value, "label value", 256)
            except ValueError:
                continue
            if label_value is not None:
                labels[label_key] = label_value
    return labels


def _provenance(
    source: str,
    source_record_id: Optional[str],
    observed_at: Optional[str],
) -> Tuple[Provenance, ...]:
    return (
        Provenance(
            source=source,
            source_record_id=source_record_id,
            observed_at=observed_at,
        ),
    )


def _split_seed(
    row: Mapping[str, Any], default_provider: str
) -> Tuple[str, Optional[str], Optional[str]]:
    provider_raw = _first(row, ("provider", "provider_name", "backend", "api", "vendor"))
    if row.get("__provider_only__") is True:
        model_raw = None
    else:
        # In both ModelManager and APIModel, model_id is the stable external
        # seed while model_name is presentation text.  Rows having only the
        # latter remain supported for older JSON inventories.
        model_raw = _first(row, ("model", "model_id", "id", "name", "model_name"))
    record_id = _safe_record_id(_first(row, ("source_record_id", "record_id", "model_id", "id")))
    if model_raw is None:
        if provider_raw is None:
            raise ValueError("row has neither a provider nor a model name")
        return _canonical_name(provider_raw, "provider"), None, record_id
    model_name = _canonical_name(model_raw, "model name")
    if provider_raw is None and "/" in model_name:
        provider_name, model_name = model_name.split("/", 1)
    else:
        provider_name = _canonical_name(
            provider_raw if provider_raw is not None else default_provider,
            "provider",
        )
        prefix = provider_name + "/"
        if model_name.startswith(prefix):
            model_name = model_name[len(prefix) :]
    return provider_name, model_name, record_id


def _candidate_key(precedence: int, record: Any) -> Tuple[int, str]:
    return precedence, json.dumps(
        record.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _normalize_input(
    value: Any, max_records: int = MAX_SNAPSHOT_RECORDS
) -> Tuple[Tuple[Any, ...], Mapping[str, Any]]:
    """Return bounded rows and optional envelope metadata."""

    if isinstance(value, CatalogSnapshot):
        return tuple(value.providers) + tuple(value.models), {
            "revision": value.revision,
            "created_at": value.created_at,
        }
    if isinstance(value, Mapping):
        envelope_keys = {
            "records",
            "items",
            "providers",
            "models",
            "source_revision",
            "revision",
            "created_at",
            "updated_at",
            "observed_at",
            "precedence",
        }
        if set(value) & {"records", "items"}:
            rows = value.get("records", value.get("items"))
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise ValueError("source records must be an array")
            return tuple(rows), value
        rows = []
        if "providers" in value or "models" in value:
            providers = value.get("providers", ())
            models = value.get("models", ())
            if isinstance(providers, Mapping):
                providers = [
                    dict(item, provider_name=key)
                    if isinstance(item, Mapping)
                    else {"provider_name": key, "name": item}
                    for key, item in providers.items()
                ]
            elif isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
                providers = [
                    (
                        (
                            item
                            if item.get("schema_version") is not None
                            and item.get("provider_id") is not None
                            else dict(
                                item,
                                provider_name=_first(
                                    item,
                                    ("provider_name", "provider", "name", "id"),
                                ),
                                __provider_only__=True,
                            )
                        )
                        if isinstance(item, Mapping)
                        else {
                            "provider_name": item,
                            "__provider_only__": True,
                        }
                    )
                    for item in providers
                ]
            if isinstance(models, Mapping):
                normalized_models = []
                for key, item in models.items():
                    if isinstance(item, ModelDescriptor) or (
                        isinstance(item, Mapping)
                        and item.get("schema_version") is not None
                        and item.get("provider_id") is not None
                        and item.get("model_id") is not None
                    ):
                        normalized_models.append(item)
                        continue
                    try:
                        model_row = _mapping(item)
                    except ValueError:
                        model_row = {"name": item}
                    normalized_models.append(dict(model_row, source_record_id=key))
                models = normalized_models
            if not isinstance(providers, Sequence) or isinstance(providers, (str, bytes)):
                raise ValueError("providers must be an array or object")
            if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
                raise ValueError("models must be an array or object")
            rows.extend(providers)
            rows.extend(models)
            return tuple(rows), value
        if set(value) <= envelope_keys:
            return (), value
        # Legacy forms are either ModelManager's id -> metadata mapping or a
        # provider -> [model names] inventory.
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                rows.extend({"provider": key, "name": model} for model in item)
            elif isinstance(item, (ProviderDescriptor, ModelDescriptor)):
                rows.append(item)
            else:
                try:
                    record = _mapping(item)
                except ValueError:
                    record = {"name": item}
                rows.append(
                    dict(
                        record,
                        source_record_id=record.get("source_record_id", key),
                    )
                )
        return tuple(rows), {}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value), {}
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value, key=repr)), {}
    if isinstance(value, Iterable):
        return tuple(itertools.islice(value, max_records + 1)), {}
    raise ValueError("source must be a catalog object or array")


def _read_explicit_path(path: Path) -> Any:
    if not path.is_file():
        raise ValueError("catalog source path is not a local file")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("catalog source exceeds %d bytes" % MAX_SOURCE_BYTES)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError("catalog source could not be read as UTF-8") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as document_error:
        rows = []
        diagnostics = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                diagnostics.append(
                    {
                        "__adapter_error__": "malformed JSONL row",
                        "__line__": line_number,
                    }
                )
        if rows or diagnostics:
            return rows + diagnostics
        raise ValueError("catalog source is neither JSON nor JSONL") from document_error


class StaticCatalogSource:
    """Adapt an explicitly supplied static inventory into canonical records."""

    def __init__(
        self,
        records: Any = None,
        *,
        path: Optional[Any] = None,
        source: str = "catalog.static",
        precedence: Optional[int] = None,
        revision: Optional[str] = None,
        observed_at: Optional[Any] = None,
        default_provider: str = "local",
        max_records: int = MAX_SNAPSHOT_RECORDS,
        default_precedence: int = DEFAULT_STATIC_PRECEDENCE,
    ) -> None:
        if (records is None) == (path is None):
            raise ValueError("supply exactly one of records or path")
        if isinstance(max_records, bool) or not isinstance(max_records, int):
            raise ValueError("max_records must be an integer")
        if max_records < 0 or max_records > MAX_SNAPSHOT_RECORDS:
            raise ValueError("max_records must be between 0 and %d" % MAX_SNAPSHOT_RECORDS)
        self._records = records
        self._path = None if path is None else Path(path)
        self.source = _canonical_name(source, "source")
        self.precedence = precedence
        self.revision = revision
        self.observed_at = observed_at
        self.default_provider = _canonical_name(default_provider, "default_provider")
        self.max_records = max_records
        self.default_precedence = default_precedence

    def _supplied_value(self) -> Any:
        return self._records if self._path is None else _read_explicit_path(self._path)

    def load(self) -> CatalogSourceResult:
        supplied = self._supplied_value()
        rows, envelope = _normalize_input(supplied, self.max_records)
        if len(rows) > self.max_records:
            raise ValueError("source exceeds maximum record count")

        raw_precedence = (
            self.precedence
            if self.precedence is not None
            else envelope.get("precedence", self.default_precedence)
        )
        if isinstance(raw_precedence, bool) or not isinstance(raw_precedence, int):
            raise ValueError("precedence must be an integer")
        if raw_precedence < -1_000_000 or raw_precedence > 1_000_000:
            raise ValueError("precedence is outside the supported bound")
        revision = self.revision or _first(envelope, ("source_revision", "revision"))
        revision = _bounded_text(revision, "source revision", MAX_SOURCE_REVISION_BYTES)
        created_at = _timestamp(envelope.get("created_at"), "source created_at")
        updated_at = _timestamp(
            _first(envelope, ("updated_at", "observed_at")),
            "source updated_at",
        )
        explicit_observed = _timestamp(self.observed_at, "observed_at")
        source_observed = explicit_observed or updated_at or created_at

        providers: Dict[str, Tuple[Tuple[int, str], ProviderDescriptor]] = {}
        models: Dict[str, Tuple[Tuple[int, str], ModelDescriptor]] = {}
        diagnostics = []
        redacted_fields = 0

        def add_diagnostic(item: SourceDiagnostic) -> None:
            if len(diagnostics) < MAX_DIAGNOSTICS:
                diagnostics.append(item)

        for index, raw_row in enumerate(rows):
            try:
                if (
                    isinstance(raw_row, Mapping)
                    and raw_row.get("schema_version") is not None
                    and raw_row.get("provider_id") is not None
                    and raw_row.get("name") is not None
                ):
                    canonical_row = dict(raw_row)
                    canonical_row.pop("__provider_only__", None)
                    raw_row = (
                        ModelDescriptor.from_dict(canonical_row)
                        if raw_row.get("model_id") is not None
                        else ProviderDescriptor.from_dict(canonical_row)
                    )
                if isinstance(raw_row, ProviderDescriptor):
                    provider = dataclasses.replace(
                        raw_row,
                        state=OperationalState(),
                        lifecycle=(
                            LifecycleState.DEPRECATED
                            if raw_row.lifecycle == LifecycleState.DEPRECATED
                            else LifecycleState.DECLARED
                        ),
                        provenance=_provenance(self.source, raw_row.provider_id, source_observed),
                    )
                    rank = _candidate_key(raw_precedence, provider)
                    if (
                        provider.provider_id not in providers
                        or rank > providers[provider.provider_id][0]
                    ):
                        providers[provider.provider_id] = (rank, provider)
                    continue
                elif isinstance(raw_row, ModelDescriptor):
                    # A canonical model can be retained even when the provider
                    # seed is unavailable; its identity remains unchanged.
                    provenance = _provenance(self.source, raw_row.model_id, source_observed)
                    model = dataclasses.replace(
                        raw_row,
                        state=OperationalState(),
                        lifecycle=(
                            LifecycleState.DEPRECATED
                            if raw_row.lifecycle == LifecycleState.DEPRECATED
                            else LifecycleState.DECLARED
                        ),
                        provenance=provenance,
                    )
                    rank = _candidate_key(raw_precedence, model)
                    if model.model_id not in models or rank > models[model.model_id][0]:
                        models[model.model_id] = (rank, model)
                    continue
                elif isinstance(raw_row, str):
                    row = {"name": raw_row}
                else:
                    row = _mapping(raw_row)
                if "__adapter_error__" in row:
                    raise ValueError(str(row["__adapter_error__"]))
                if len(row) > MAX_ROW_FIELDS:
                    raise ValueError("row exceeds maximum field count")

                count = _secret_count(row)
                redacted_fields += count
                if count:
                    add_diagnostic(
                        SourceDiagnostic(
                            index=index,
                            code="redacted",
                            message="credential-shaped fields were redacted",
                            source_record_id=_safe_record_id(
                                _first(row, ("source_record_id", "model_id", "id"))
                            ),
                        )
                    )

                row_precedence = _first(row, ("precedence", "priority"))
                if row_precedence is None:
                    row_precedence = raw_precedence
                if isinstance(row_precedence, bool) or not isinstance(row_precedence, int):
                    raise ValueError("row precedence must be an integer")
                provider_name, model_name, record_id = _split_seed(row, self.default_provider)
                row_created, row_updated, row_observed = _row_timestamps(row)
                observed = row_observed or row_updated or row_created or source_observed
                row_revision = (
                    _bounded_text(
                        _first(row, ("source_revision", "revision", "model_revision")),
                        "row revision",
                        MAX_SOURCE_REVISION_BYTES,
                    )
                    or revision
                )
                labels = _labels(row, row_precedence, row_revision, row_created, row_updated)
                provenance = _provenance(self.source, record_id, observed)
                provider_only = model_name is None
                display_name = _bounded_text(
                    _first(
                        row,
                        (
                            "provider_display_name",
                            "provider_title",
                            "display_name" if provider_only else "_missing",
                            "title" if provider_only else "_missing",
                        ),
                    ),
                    "provider display name",
                    256,
                )
                provider_description = _bounded_text(
                    _first(
                        row,
                        (
                            "provider_description",
                            "description" if provider_only else "_missing",
                        ),
                    ),
                    "provider description",
                    MAX_DESCRIPTION_LENGTH,
                )
                provider = ProviderDescriptor(
                    name=provider_name,
                    display_name=display_name,
                    aliases=_aliases(row, provider_name) if provider_only else (),
                    description=provider_description or "",
                    capabilities=_capabilities(row),
                    lifecycle=_lifecycle(row),
                    state=OperationalState(),
                    provenance=provenance,
                    labels=labels,
                )
                model = None
                if model_name is None:
                    model_rank = None
                else:
                    model_display = _bounded_text(
                        _first(
                            row,
                            (
                                "display_name",
                                "title",
                                "model_display_name",
                                ("model_name" if row.get("model_id") is not None else "_missing"),
                            ),
                        ),
                        "display name",
                        256,
                    )
                    description = _bounded_text(
                        row.get("description"),
                        "description",
                        MAX_DESCRIPTION_LENGTH,
                    )
                    architecture = _bounded_text(
                        _first(row, ("architecture", "model_type")),
                        "architecture",
                        256,
                    )
                    model = ModelDescriptor(
                        provider_id=provider.provider_id,
                        name=model_name,
                        display_name=model_display,
                        aliases=_aliases(row, model_name),
                        description=description or "",
                        architecture=architecture,
                        capabilities=_capabilities(row),
                        lifecycle=_lifecycle(row),
                        state=OperationalState(),
                        provenance=provenance,
                        labels=labels,
                    )
                    model_rank = _candidate_key(row_precedence, model)

                # Commit the row only after every record it contributes has
                # validated, so a malformed model cannot leave an orphaned
                # provider candidate behind.
                provider_rank = _candidate_key(row_precedence, provider)
                old_provider = providers.get(provider.provider_id)
                if old_provider is None or provider_rank > old_provider[0]:
                    providers[provider.provider_id] = (provider_rank, provider)
                if model is not None and model_rank is not None:
                    old_model = models.get(model.model_id)
                    if old_model is None or model_rank > old_model[0]:
                        models[model.model_id] = (model_rank, model)
            except (SchemaValidationError, TypeError, ValueError) as exc:
                add_diagnostic(
                    SourceDiagnostic(
                        index=index,
                        code="malformed_row",
                        message=_diagnostic_message(exc),
                        source_record_id=(
                            _safe_record_id(raw_row.get("source_record_id"))
                            if isinstance(raw_row, Mapping)
                            else None
                        ),
                    )
                )

        snapshot = CatalogSnapshot(
            providers=tuple(item[1] for item in providers.values()),
            models=tuple(item[1] for item in models.values()),
            created_at=source_observed,
        )
        return CatalogSourceResult(
            snapshot=snapshot,
            metadata=SourceMetadata(
                source=self.source,
                precedence=raw_precedence,
                revision=revision,
                created_at=created_at,
                updated_at=updated_at or explicit_observed,
            ),
            diagnostics=tuple(diagnostics),
            redacted_fields=redacted_fields,
        )

    # Catalog source protocols commonly use snapshot() or read(); keep both
    # aliases side-effect-equivalent to the explicit load operation.
    snapshot = load
    read = load


StaticSourceAdapter = StaticCatalogSource


def adapt_static_source(
    records: Any = None, *, path: Optional[Any] = None, **kwargs: Any
) -> CatalogSourceResult:
    """Adapt injected records or one explicit local path."""

    return StaticCatalogSource(records, path=path, **kwargs).load()


load_static_catalog = adapt_static_source


__all__ = [
    "CatalogSourceResult",
    "DEFAULT_STATIC_PRECEDENCE",
    "MAX_DIAGNOSTICS",
    "MAX_SOURCE_BYTES",
    "MAX_SOURCE_REVISION_BYTES",
    "MAX_ROW_FIELDS",
    "SourceDiagnostic",
    "SourceMetadata",
    "StaticCatalogSource",
    "StaticSourceAdapter",
    "adapt_static_source",
    "load_static_catalog",
]
