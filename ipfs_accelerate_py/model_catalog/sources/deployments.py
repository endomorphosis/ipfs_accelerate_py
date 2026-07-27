"""Pure catalog adapters for served endpoints and registered backends.

This module intentionally does not import ``ModelManager`` or
``InferenceBackendManager``.  Callers inject snapshots of those systems (plain
mappings, their dataclass records, or served-model advertisements).  Merely
constructing, loading, or reading a source performs no network, process,
credential, or model operation.

Dynamic checks are equally explicit: :meth:`DeploymentCatalogSource.refresh`
is the only method that calls an injected probe.  The probe receives a
credential-free :class:`ProbeTarget`; this module never implements an HTTP
client and therefore cannot accidentally turn discovery into inference.
"""

from __future__ import annotations

import dataclasses
import inspect
import ipaddress
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from ..identity import (
    REDACTED,
    canonical_json,
    is_secret_key,
    is_secret_value,
    stable_id,
)
from ..schema import (
    MAX_NAME_LENGTH,
    MAX_SNAPSHOT_RECORDS,
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    Provenance,
    ProviderDescriptor,
    SchemaValidationError,
)
from .static import MAX_DIAGNOSTICS, SourceDiagnostic, SourceMetadata

DEFAULT_DEPLOYMENT_PRECEDENCE = 40
DEFAULT_HEALTH_TTL_SECONDS = 60
MAX_HEALTH_TTL_SECONDS = 86_400
MAX_HEALTH_DIAGNOSTICS = 16
MAX_DIAGNOSTIC_BYTES = 512
MAX_ENDPOINT_BYTES = 2_048
MAX_RECORD_FIELDS = 256

_NAME_BAD = re.compile(r"[^a-z0-9._/-]+")
_PROTOCOL_ALIASES = {
    "openai": "openai.http",
    "openai-compatible": "openai.http",
    "openai_compatible": "openai.http",
    "http": "openai.http",
    "https": "openai.http",
    "rest": "http",
    "ws": "websocket",
    "wss": "websocket",
    "in_process": "in-process",
    "inprocess": "in-process",
}
_OPERATION_ALIASES = {
    "text-generation": Operation.TEXT_GENERATE,
    "text_generation": Operation.TEXT_GENERATE,
    "generate": Operation.TEXT_GENERATE,
    "completion": Operation.TEXT_GENERATE,
    "chat": Operation.TEXT_CHAT,
    "conversational": Operation.TEXT_CHAT,
    "chat-completion": Operation.TEXT_CHAT,
    "chat.completions": Operation.TEXT_CHAT,
    "embedding": Operation.EMBEDDING_GENERATE,
    "embeddings": Operation.EMBEDDING_GENERATE,
    "feature-extraction": Operation.EMBEDDING_GENERATE,
    "vision": Operation.VISION_GENERATE,
    "image-to-text": Operation.VISION_GENERATE,
    "audio": Operation.AUDIO_TRANSCRIBE,
    "transcription": Operation.AUDIO_TRANSCRIBE,
    "automatic-speech-recognition": Operation.AUDIO_TRANSCRIBE,
    "text-to-speech": Operation.AUDIO_SYNTHESIZE,
    "speech-synthesis": Operation.AUDIO_SYNTHESIZE,
    "streaming": Operation.STREAM,
    "tools": Operation.TOOL_CALL,
    "function-calling": Operation.TOOL_CALL,
}
_HEALTHY_STATUSES = frozenset(("healthy", "available", "online", "active"))
_DEGRADED_STATUSES = frozenset(("degraded", "warning"))
_STOPPED_STATUSES = frozenset(("stopped", "offline", "unavailable", "dead"))
_STARTING_STATUSES = frozenset(("starting", "initializing", "loading"))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: Any, field_name: str) -> str:
    if isinstance(value, bool):
        raise ValueError("%s must be a timestamp" % field_name)
    if isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), timezone.utc)
        except (OSError, OverflowError, ValueError) as exc:
            raise ValueError("%s must be a timestamp" % field_name) from exc
    elif isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError("%s must not be empty" % field_name)
        try:
            parsed = datetime.fromisoformat(
                raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            )
        except ValueError as exc:
            raise ValueError("%s must be an RFC 3339 timestamp" % field_name) from exc
    else:
        raise ValueError("%s must be a timestamp" % field_name)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        parsed.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _optional_timestamp(value: Any, field_name: str) -> Optional[str]:
    return None if value is None or value == "" else _timestamp(value, field_name)


def _as_datetime(value: Any, field_name: str) -> datetime:
    return datetime.fromisoformat(_timestamp(value, field_name).replace("Z", "+00:00"))


def _canonical_name(value: Any, field_name: str, *, default: Optional[str] = None) -> str:
    if isinstance(value, Enum):
        value = value.value
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is None:
            raise ValueError("%s must not be empty" % field_name)
        value = default
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % field_name)
    if is_secret_value(value):
        raise ValueError("%s contains credential-shaped data" % field_name)
    result = _NAME_BAD.sub("-", value.strip().casefold()).strip("-._/")
    result = re.sub(r"/+", "/", result)
    result = re.sub(r"\.{2,}", ".", result)
    if not result:
        raise ValueError("%s has no canonical name characters" % field_name)
    if len(result.encode("utf-8")) > MAX_NAME_LENGTH:
        raise ValueError("%s exceeds the catalog name bound" % field_name)
    return result


def _safe_text(value: Any, maximum: int = 256) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None
    if is_secret_value(value):
        return REDACTED
    encoded = value.encode("utf-8")
    if len(encoded) > maximum:
        value = encoded[:maximum].decode("utf-8", "ignore")
    return value


def _safe_diagnostic(value: Any) -> str:
    text = _safe_text(value, MAX_DIAGNOSTIC_BYTES) or "probe failed"
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)
    if text == REDACTED or any(is_secret_key(token) for token in tokens):
        return "credential-shaped diagnostic was redacted"
    return text


def _bool_or_none(value: Any, field_name: str) -> Optional[bool]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("%s must be a boolean or null" % field_name)
    return value


def _ttl(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ttl_seconds must be a positive number")
    if value <= 0 or value > MAX_HEALTH_TTL_SECONDS:
        raise ValueError(
            "ttl_seconds must be between 1 and %d" % MAX_HEALTH_TTL_SECONDS
        )
    return int(value)


def _sequence(value: Any, field_name: str, maximum: int = 64) -> Tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Mapping) or not isinstance(
        value, (Sequence, set, frozenset)
    ):
        raise ValueError("%s must be an array" % field_name)
    else:
        values = tuple(value)
    if len(values) > maximum:
        raise ValueError("%s exceeds maximum count" % field_name)
    return tuple(values)


def _mapping_from_object(value: Any) -> Mapping[str, Any]:
    """Take a shallow, bounded snapshot without traversing backend instances."""

    if isinstance(value, Mapping):
        if len(value) > MAX_RECORD_FIELDS:
            raise ValueError("deployment row exceeds maximum field count")
        return dict(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result = {}
        for item in dataclasses.fields(value):
            if item.name == "instance":
                continue
            result[item.name] = getattr(value, item.name)
        return result
    names = (
        "backend_id",
        "backend_type",
        "name",
        "endpoint",
        "endpoint_uri",
        "status",
        "capabilities",
        "metrics",
        "metadata",
        "registered_at",
        "last_seen",
        "provider",
        "model",
        "model_id",
        "logical_model_id",
        "transport_model_id",
        "transport",
        "served",
        "configured",
        "authorized",
        "reachable",
        "live",
        "ready",
        "healthy",
        "routable",
    )
    result = {name: getattr(value, name) for name in names if hasattr(value, name)}
    if not result:
        raise ValueError("deployment row must be a mapping or record object")
    return result


def _nested_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    try:
        return _mapping_from_object(value)
    except ValueError:
        return {}


def _first(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None and row[name] != "":
            return row[name]
    return None


def _source_rows(records: Any, maximum: int) -> Tuple[Any, ...]:
    if records is None:
        return ()
    if not isinstance(records, Mapping) and hasattr(records, "list_backends"):
        # Calling arbitrary methods would violate the injected-snapshot contract.
        raise ValueError("inject list_backends() output, not a live manager")
    if isinstance(records, Mapping):
        for key in ("served_models", "backends", "deployments", "endpoints", "data"):
            if key in records:
                value = records[key]
                if isinstance(value, Mapping):
                    rows = tuple(value.values())
                else:
                    rows = _sequence(value, key, maximum)
                break
        else:
            record_keys = {
                "endpoint",
                "endpoint_uri",
                "base_url",
                "api_base",
                "backend_id",
                "model_id",
                "transport_model_id",
            }
            rows = (records,) if record_keys.intersection(records) else tuple(records.values())
    else:
        rows = _sequence(records, "deployment records", maximum)
    if len(rows) > maximum:
        raise ValueError("deployment source exceeds maximum record count")
    return rows


def _sanitize_endpoint(
    raw_value: Any,
    *,
    backend_id: str,
    allow_virtual: bool,
) -> Tuple[str, bool]:
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        if not allow_virtual:
            raise ValueError("endpoint URI is required")
        path_name = _canonical_name(backend_id, "backend_id", default="backend")
        return "unix:///in-process/%s" % path_name.replace("/", "-"), False
    if not isinstance(raw_value, str):
        raise ValueError("endpoint URI must be a string")
    if len(raw_value.encode("utf-8")) > MAX_ENDPOINT_BYTES:
        raise ValueError("endpoint URI exceeds maximum size")
    raw = raw_value.strip()
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError as exc:
        raise ValueError("endpoint URI is malformed") from exc
    scheme = parts.scheme.casefold()
    if scheme not in ("http", "https", "unix"):
        raise ValueError("endpoint URI has an unsupported scheme")
    redacted = bool(
        parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    )
    if scheme == "unix":
        if not parts.path.startswith("/"):
            raise ValueError("unix endpoint URI must use an absolute path")
        return urlunsplit(("unix", "", parts.path, "", "")), redacted
    if not parts.hostname:
        raise ValueError("endpoint URI must include a host")
    host = parts.hostname.casefold()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("endpoint URI host is malformed") from exc
    if ":" in host and not host.startswith("["):
        host = "[%s]" % host
    if port is not None and not (
        (scheme == "http" and port == 80)
        or (scheme == "https" and port == 443)
    ):
        host = "%s:%d" % (host, port)
    path = parts.path or "/"
    # Credential-looking path segments are never retained.
    if is_secret_value(path):
        path = "/"
        redacted = True
    return urlunsplit((scheme, host, path, "", "")), redacted


def _locality(endpoint: str, asserted: Any = None) -> str:
    if asserted is not None:
        value = _canonical_name(asserted, "locality")
        if value not in ("local", "remote", "peer"):
            raise ValueError("locality must be local, remote, or peer")
        return value
    parts = urlsplit(endpoint)
    if parts.scheme == "unix":
        return "local"
    host = (parts.hostname or "").casefold().rstrip(".")
    if host == "localhost":
        return "local"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "remote"
    return "local" if address.is_loopback else "remote"


def _protocol(row: Mapping[str, Any], endpoint: str, served: bool) -> str:
    raw = _first(row, ("protocol", "endpoint_protocol", "api_protocol"))
    if raw is None:
        protocols = _nested_mapping(row.get("capabilities")).get("protocols")
        values = _sequence(protocols, "protocols") if protocols is not None else ()
        raw = values[0] if len(values) == 1 else None
    if raw is None:
        raw = "openai.http" if served else (
            "in-process" if urlsplit(endpoint).scheme == "unix" else "http"
        )
    if isinstance(raw, Enum):
        raw = raw.value
    normalized = str(raw).strip().casefold()
    return _canonical_name(
        _PROTOCOL_ALIASES.get(normalized, normalized),
        "protocol",
    )


def _operations(row: Mapping[str, Any]) -> Tuple[Operation, ...]:
    capabilities = _nested_mapping(row.get("capabilities"))
    raw = _first(
        row,
        ("operations", "tasks", "task", "supported_tasks"),
    )
    if raw is None:
        raw = _first(
            capabilities,
            ("operations", "tasks", "supported_tasks"),
        )
    if raw is None and not isinstance(row.get("capabilities"), Mapping):
        raw = row.get("capabilities")
    operations = set()
    for item in _sequence(raw, "operations", 32):
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
                raise ValueError("unknown operation name: %s" % normalized)
            operations.add(operation)
    streaming = _first(row, ("supports_streaming", "streaming"))
    if streaming is None:
        streaming = capabilities.get("supports_streaming")
    batching = _first(row, ("supports_batching", "batching"))
    if batching is None:
        batching = capabilities.get("supports_batching")
    if streaming is True:
        operations.add(Operation.STREAM)
    if batching is True:
        operations.add(Operation.BATCH)
    if operations and not operations - {Operation.STREAM, Operation.BATCH}:
        raise ValueError("stream and batch require an invokable operation")
    return tuple(sorted(operations, key=lambda item: item.value))


def _capability(row: Mapping[str, Any]) -> Tuple[CapabilityDescriptor, ...]:
    operations = _operations(row)
    if not operations:
        return ()
    inputs = {Modality.TEXT}
    outputs = {Modality.TEXT}
    if Operation.EMBEDDING_GENERATE in operations:
        outputs.add(Modality.EMBEDDING)
    if Operation.VISION_GENERATE in operations:
        inputs.add(Modality.IMAGE)
    if Operation.AUDIO_TRANSCRIBE in operations:
        inputs.add(Modality.AUDIO)
    if Operation.AUDIO_SYNTHESIZE in operations:
        outputs.add(Modality.AUDIO)
    capabilities = _nested_mapping(row.get("capabilities"))
    maximum = capabilities.get("max_batch_size")
    if maximum is None:
        maximum = row.get("max_batch_size")
    if Operation.BATCH not in operations:
        maximum = None
    return (
        CapabilityDescriptor(
            operations=operations,
            input_modalities=tuple(inputs),
            output_modalities=tuple(outputs),
            max_batch_size=maximum,
        ),
    )


def _status_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    return str(value).strip().casefold() or None


def _lifecycle(row: Mapping[str, Any]) -> LifecycleState:
    status = _status_name(row.get("status"))
    if row.get("served") is False or status in _STOPPED_STATUSES:
        return LifecycleState.STOPPED
    if status in _STARTING_STATUSES:
        return LifecycleState.STARTING
    if status in _DEGRADED_STATUSES:
        return LifecycleState.DEGRADED
    if status in _HEALTHY_STATUSES or row.get("ready") is True:
        return LifecycleState.READY
    return LifecycleState.CONFIGURED


def _explicit_state(row: Mapping[str, Any], *, configured_default: bool) -> OperationalState:
    return OperationalState(
        known=True,
        configured=_bool_or_none(
            row.get("configured", configured_default), "configured"
        ),
        authorized=_bool_or_none(row.get("authorized"), "authorized"),
        reachable=_bool_or_none(row.get("reachable"), "reachable"),
        healthy=_bool_or_none(row.get("healthy"), "healthy"),
        routable=_bool_or_none(row.get("routable"), "routable"),
    )


@dataclass(frozen=True)
class DeploymentIdentity:
    """All non-secret fields that distinguish one served deployment."""

    service: str
    model: str
    provider: str
    protocol: str
    endpoint_id: str
    locality: str

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ProbeTarget:
    """Credential-free input supplied to an explicit health probe."""

    deployment_id: str
    endpoint_uri: str
    endpoint_id: str
    protocol: str
    locality: str
    service: str
    provider: str
    model: Optional[str]
    purpose: str = "liveness-readiness"
    inference_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class HealthSample:
    """A bounded, expiring observation without cross-field inference."""

    deployment_id: str
    observed_at: str
    ttl_seconds: int
    provenance: str
    configured: Optional[bool] = None
    reachable: Optional[bool] = None
    live: Optional[bool] = None
    ready: Optional[bool] = None
    healthy: Optional[bool] = None
    routable: Optional[bool] = None
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        deployment_id = _safe_text(self.deployment_id, 128)
        if not deployment_id or not re.fullmatch(
            r"deployment_[0-9a-f]{64}", deployment_id
        ):
            raise ValueError("deployment_id is not a stable deployment identity")
        object.__setattr__(self, "deployment_id", deployment_id)
        object.__setattr__(
            self, "observed_at", _timestamp(self.observed_at, "observed_at")
        )
        object.__setattr__(self, "ttl_seconds", _ttl(self.ttl_seconds))
        provenance = _canonical_name(self.provenance, "provenance")
        object.__setattr__(self, "provenance", provenance)
        for name in (
            "configured",
            "reachable",
            "live",
            "ready",
            "healthy",
            "routable",
        ):
            object.__setattr__(
                self, name, _bool_or_none(getattr(self, name), name)
            )
        diagnostics = _sequence(
            self.diagnostics, "health diagnostics", MAX_HEALTH_DIAGNOSTICS
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(_safe_diagnostic(item) for item in diagnostics),
        )

    @property
    def expires_at(self) -> str:
        expiry = _as_datetime(self.observed_at, "observed_at") + timedelta(
            seconds=self.ttl_seconds
        )
        return _timestamp(expiry, "expires_at")

    @property
    def source(self) -> str:
        return self.provenance

    def is_stale(self, at: Any) -> bool:
        return _as_datetime(at, "at") >= _as_datetime(self.expires_at, "expires_at")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "observed_at": self.observed_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "provenance": self.provenance,
            "configured": self.configured,
            "reachable": self.reachable,
            "live": self.live,
            "ready": self.ready,
            "healthy": self.healthy,
            "routable": self.routable,
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HealthSample":
        data = dict(value)
        data.pop("expires_at", None)
        return cls(**data)


@dataclass(frozen=True)
class DeploymentSourceResult:
    snapshot: CatalogSnapshot
    metadata: SourceMetadata
    health_samples: Tuple[HealthSample, ...] = ()
    diagnostics: Tuple[SourceDiagnostic, ...] = ()
    redacted_fields: int = 0

    @property
    def providers(self) -> Tuple[ProviderDescriptor, ...]:
        return self.snapshot.providers

    @property
    def models(self) -> Tuple[ModelDescriptor, ...]:
        return self.snapshot.models

    @property
    def deployments(self) -> Tuple[DeploymentDescriptor, ...]:
        return self.snapshot.deployments

    @property
    def source(self) -> str:
        return self.metadata.source

    @property
    def precedence(self) -> int:
        return self.metadata.precedence

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
            "health_samples": [item.to_dict() for item in self.health_samples],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "redacted_fields": self.redacted_fields,
        }


@dataclass(frozen=True)
class _Candidate:
    provider: ProviderDescriptor
    model: Optional[ModelDescriptor]
    deployment: DeploymentDescriptor
    identity: DeploymentIdentity
    target: ProbeTarget
    input_sample: Optional[HealthSample] = None


def _input_sample(
    row: Mapping[str, Any],
    deployment_id: str,
    *,
    source: str,
    default_ttl: int,
) -> Optional[HealthSample]:
    observation = _nested_mapping(row.get("health"))
    observed = _first(
        observation,
        ("observed_at", "timestamp", "last_health_check"),
    )
    if observed is None:
        metrics = _nested_mapping(row.get("metrics"))
        observed = _first(
            row,
            ("health_observed_at", "observed_at", "last_health_check"),
        )
        if observed is None:
            observed = metrics.get("last_health_check")
    if observed is None:
        return None
    values = {}
    for name in (
        "configured",
        "reachable",
        "live",
        "ready",
        "healthy",
        "routable",
    ):
        values[name] = observation.get(name, row.get(name))
    status = _status_name(row.get("status"))
    # A backend's explicit health status is evidence about health and liveness,
    # but never about reachability, readiness, authorization, or routability.
    if values["healthy"] is None and status in _HEALTHY_STATUSES:
        values["healthy"] = True
    elif values["healthy"] is None and status in _DEGRADED_STATUSES | _STOPPED_STATUSES:
        values["healthy"] = False
    if values["live"] is None and status in _HEALTHY_STATUSES | _DEGRADED_STATUSES:
        values["live"] = True
    elif values["live"] is None and status in _STOPPED_STATUSES:
        values["live"] = False
    diagnostics = observation.get("diagnostics", ())
    if not diagnostics and observation.get("diagnostic"):
        diagnostics = (observation["diagnostic"],)
    return HealthSample(
        deployment_id=deployment_id,
        observed_at=observed,
        ttl_seconds=observation.get(
            "ttl_seconds", row.get("health_ttl_seconds", default_ttl)
        ),
        provenance=observation.get("provenance", source),
        diagnostics=tuple(_sequence(diagnostics, "health diagnostics")),
        **values,
    )


def _project_sample(
    deployment: DeploymentDescriptor,
    sample: Optional[HealthSample],
    now: datetime,
) -> DeploymentDescriptor:
    if sample is None or sample.is_stale(now):
        return deployment
    state = deployment.state
    projected = OperationalState(
        known=state.known,
        configured=(
            sample.configured
            if sample.configured is not None
            else state.configured
        ),
        authorized=state.authorized,
        reachable=(
            sample.reachable if sample.reachable is not None else state.reachable
        ),
        healthy=sample.healthy if sample.healthy is not None else state.healthy,
        routable=(
            sample.routable if sample.routable is not None else state.routable
        ),
    )
    lifecycle = deployment.lifecycle
    if sample.ready is True:
        lifecycle = LifecycleState.READY
    elif sample.live is False:
        lifecycle = LifecycleState.STOPPED
    provenance = tuple(deployment.provenance) + (
        Provenance(
            source=sample.provenance,
            source_record_id=deployment.deployment_id,
            observed_at=sample.observed_at,
            expires_at=sample.expires_at,
        ),
    )
    return dataclasses.replace(
        deployment,
        state=projected,
        lifecycle=lifecycle,
        updated_at=sample.observed_at,
        provenance=provenance,
    )


def _probe_sample(
    value: Any,
    candidate: _Candidate,
    *,
    now: datetime,
    source: str,
    default_ttl: int,
) -> HealthSample:
    if isinstance(value, HealthSample):
        if value.deployment_id != candidate.deployment.deployment_id:
            raise ValueError("probe sample deployment_id does not match target")
        return value
    if isinstance(value, bool):
        data: Mapping[str, Any] = {"reachable": value}
    elif isinstance(value, Mapping):
        data = value
    elif value is None:
        data = {}
    else:
        data = _mapping_from_object(value)
    diagnostics = data.get("diagnostics", ())
    if not diagnostics and data.get("diagnostic") is not None:
        diagnostics = (data["diagnostic"],)
    return HealthSample(
        deployment_id=candidate.deployment.deployment_id,  # type: ignore[arg-type]
        observed_at=data.get("observed_at", now),
        ttl_seconds=data.get("ttl_seconds", default_ttl),
        provenance=data.get("provenance", "%s.probe" % source),
        configured=data.get("configured"),
        reachable=data.get("reachable"),
        live=data.get("live"),
        ready=data.get("ready"),
        healthy=data.get("healthy"),
        routable=data.get("routable"),
        diagnostics=tuple(_sequence(diagnostics, "health diagnostics")),
    )


class DeploymentCatalogSource:
    """Normalize injected endpoint/backend records into deployment descriptors."""

    def __init__(
        self,
        records: Any = None,
        *,
        probe: Optional[Callable[[ProbeTarget], Any]] = None,
        source: str = "deployments.runtime",
        precedence: int = DEFAULT_DEPLOYMENT_PRECEDENCE,
        revision: Optional[str] = None,
        observed_at: Optional[Any] = None,
        default_provider: str = "local",
        default_service: str = "inference",
        default_protocol: Optional[str] = None,
        health_ttl_seconds: int = DEFAULT_HEALTH_TTL_SECONDS,
        max_records: int = MAX_SNAPSHOT_RECORDS,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if isinstance(precedence, bool) or not isinstance(precedence, int):
            raise ValueError("precedence must be an integer")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records < 1
            or max_records > MAX_SNAPSHOT_RECORDS
        ):
            raise ValueError("max_records is outside the catalog bound")
        if probe is not None and not callable(probe):
            raise ValueError("probe must be callable")
        if not callable(clock):
            raise ValueError("clock must be callable")
        self._records = records
        self.probe = probe
        self.source = _canonical_name(source, "source")
        self.precedence = precedence
        self.source_revision = _safe_text(revision, 512)
        self.observed_at = (
            _timestamp(observed_at, "observed_at") if observed_at is not None else None
        )
        self.default_provider = _canonical_name(
            default_provider, "default_provider"
        )
        self.default_service = _canonical_name(default_service, "default_service")
        self.default_protocol = (
            _canonical_name(default_protocol, "default_protocol")
            if default_protocol is not None
            else None
        )
        self.health_ttl_seconds = _ttl(health_ttl_seconds)
        self.max_records = max_records
        self.clock = clock

    def _candidate(
        self,
        raw: Any,
        *,
        model_override: Any = None,
    ) -> Tuple[_Candidate, bool]:
        row = _mapping_from_object(raw)
        metadata = _nested_mapping(row.get("metadata"))
        backend_id = _canonical_name(
            _first(row, ("backend_id", "deployment_name", "instance_id", "id")),
            "backend_id",
            default="endpoint",
        )
        served = bool(
            row.get("served") is not None
            or "transport_model_id" in row
            or "logical_model_id" in row
        )
        endpoint_raw = _first(
            row,
            ("endpoint_uri", "endpoint", "base_url", "api_base", "url"),
        )
        if endpoint_raw is None:
            endpoint_raw = _first(
                metadata,
                ("endpoint_uri", "endpoint", "base_url", "api_base"),
            )
        backend_type = _status_name(row.get("backend_type")) or ""
        endpoint, redacted = _sanitize_endpoint(
            endpoint_raw,
            backend_id=backend_id,
            allow_virtual=backend_type not in ("api", "remote") and not served,
        )
        locality = _locality(endpoint, row.get("locality") or metadata.get("locality"))
        protocol = self.default_protocol or _protocol(row, endpoint, served)
        service = _canonical_name(
            _first(row, ("service", "service_name", "api_service")),
            "service",
            default=("openai-compatible" if served else self.default_service),
        )
        provider_raw = _first(
            row,
            ("provider", "owned_by", "vendor"),
        )
        if provider_raw is None:
            provider_raw = _first(metadata, ("provider", "owned_by", "vendor"))
        if provider_raw is None and served:
            provider_raw = row.get("transport")
        if provider_raw is None and backend_type == "api":
            provider_raw = backend_id.removeprefix("api_").removeprefix("api-")
        provider_name = _canonical_name(
            provider_raw, "provider", default=self.default_provider
        )

        model_raw = model_override
        if model_raw is None:
            model_raw = _first(
                row,
                (
                    "logical_model_id",
                    "model_id",
                    "model",
                    "served_model",
                    "transport_model_id",
                ),
            )
        model_name = (
            _canonical_name(model_raw, "model") if model_raw is not None else None
        )
        provider = ProviderDescriptor(
            name=provider_name,
            lifecycle=LifecycleState.CONFIGURED,
            state=OperationalState(known=True, configured=True),
            provenance=(
                Provenance(source=self.source, source_record_id=backend_id),
            ),
            labels={
                "service": service,
                "locality": locality,
                "source.precedence": str(self.precedence),
            },
        )
        capabilities = _capability(row)
        model = None
        if model_name is not None:
            alias_values = []
            for field_name in ("transport_model_id", "id", "name"):
                alias = row.get(field_name)
                if alias is None:
                    continue
                try:
                    canonical_alias = _canonical_name(alias, "model alias")
                except ValueError:
                    continue
                if canonical_alias != model_name:
                    alias_values.append(canonical_alias)
            model = ModelDescriptor(
                provider_id=provider.provider_id,
                name=model_name,
                aliases=tuple(sorted(set(alias_values))),
                capabilities=capabilities,
                lifecycle=LifecycleState.READY if served else LifecycleState.CONFIGURED,
                state=OperationalState(known=True, configured=True),
                provenance=(
                    Provenance(source=self.source, source_record_id=_safe_text(model_raw, 512)),
                ),
                labels={"service": service, "locality": locality},
            )

        endpoint_id = stable_id(
            "endpoint", service, protocol, endpoint, locality
        )
        identity = DeploymentIdentity(
            service=service,
            model=model_name or "",
            provider=provider_name,
            protocol=protocol,
            endpoint_id=endpoint_id,
            locality=locality,
        )
        suffix = endpoint_id.rsplit("_", 1)[-1][:16]
        deployment_name = "%s/%s/%s/%s" % (
            service,
            protocol,
            locality,
            suffix,
        )
        source_record_id = _safe_text(
            _first(row, ("backend_id", "transport_model_id", "id")), 512
        )
        state = _explicit_state(row, configured_default=True)
        deployment = DeploymentDescriptor(
            provider_id=provider.provider_id,
            model_id=model.model_id if model is not None else None,
            name=deployment_name,
            endpoint_uri=endpoint,
            capabilities=capabilities,
            lifecycle=_lifecycle(row),
            state=state,
            created_at=_optional_timestamp(
                _first(row, ("created_at", "registered_at")), "created_at"
            ),
            updated_at=_optional_timestamp(
                _first(row, ("updated_at", "last_seen")), "updated_at"
            ),
            provenance=(
                Provenance(
                    source=self.source,
                    source_record_id=source_record_id or backend_id,
                    observed_at=self.observed_at,
                ),
            ),
            labels={
                "service": service,
                "provider": provider_name,
                "model": model_name or "unknown",
                "protocol": protocol,
                "endpoint-id": endpoint_id,
                "locality": locality,
                "source.precedence": str(self.precedence),
            },
        )
        target = ProbeTarget(
            deployment_id=deployment.deployment_id,  # type: ignore[arg-type]
            endpoint_uri=deployment.endpoint_uri,
            endpoint_id=endpoint_id,
            protocol=protocol,
            locality=locality,
            service=service,
            provider=provider_name,
            model=model_name,
        )
        sample = _input_sample(
            row,
            deployment.deployment_id,  # type: ignore[arg-type]
            source=self.source,
            default_ttl=self.health_ttl_seconds,
        )
        return _Candidate(provider, model, deployment, identity, target, sample), redacted

    def _candidates(
        self,
    ) -> Tuple[Tuple[_Candidate, ...], Tuple[SourceDiagnostic, ...], int]:
        diagnostics = []
        candidates = []
        redacted_fields = 0
        rows = _source_rows(self._records, self.max_records)
        for index, raw in enumerate(rows):
            try:
                row = _mapping_from_object(raw)
                capabilities = _nested_mapping(row.get("capabilities"))
                models = _first(row, ("supported_models", "models"))
                if models is None:
                    models = capabilities.get("supported_models")
                model_values = _sequence(models, "supported_models", self.max_records)
                if not model_values:
                    model_values = (None,)
                remaining = self.max_records - len(candidates)
                if len(model_values) > remaining:
                    raise ValueError(
                        "expanded deployment source exceeds maximum record count"
                    )
                for model_value in sorted(
                    model_values,
                    key=lambda value: str(value),
                ):
                    candidate, redacted = self._candidate(
                        raw, model_override=model_value
                    )
                    candidates.append(candidate)
                    if redacted:
                        redacted_fields += 1
                        if len(diagnostics) < MAX_DIAGNOSTICS:
                            diagnostics.append(
                                SourceDiagnostic(
                                    index=index,
                                    code="redacted",
                                    message="endpoint credentials or query data were removed",
                                    source_record_id=_safe_text(
                                        row.get("backend_id") or row.get("id"), 512
                                    ),
                                )
                            )
            except (SchemaValidationError, TypeError, ValueError) as exc:
                if len(diagnostics) < MAX_DIAGNOSTICS:
                    diagnostics.append(
                        SourceDiagnostic(
                            index=index,
                            code="malformed_endpoint",
                            message=_safe_diagnostic(exc),
                            source_record_id=None,
                        )
                    )
        # Coalesce exact deployment identities independent of input order.
        selected: Dict[str, _Candidate] = {}
        for candidate in candidates:
            key = candidate.deployment.deployment_id  # type: ignore[assignment]
            current = selected.get(key)
            candidate_rank = (
                canonical_json(candidate.deployment),
                canonical_json(candidate.input_sample.to_dict(), reject_secrets=False)
                if candidate.input_sample is not None
                else "",
            )
            current_rank = (
                (
                    canonical_json(current.deployment),
                    canonical_json(
                        current.input_sample.to_dict(), reject_secrets=False
                    )
                    if current.input_sample is not None
                    else "",
                )
                if current is not None
                else None
            )
            if current_rank is None or candidate_rank > current_rank:
                selected[key] = candidate
        return (
            tuple(selected[key] for key in sorted(selected)),
            tuple(diagnostics),
            redacted_fields,
        )

    def _result(
        self,
        candidates: Sequence[_Candidate],
        diagnostics: Sequence[SourceDiagnostic],
        redacted_fields: int,
        *,
        probe_samples: Optional[Mapping[str, HealthSample]] = None,
        now: Optional[datetime] = None,
    ) -> DeploymentSourceResult:
        current = now or self.clock()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        providers: Dict[str, ProviderDescriptor] = {}
        models: Dict[str, ModelDescriptor] = {}
        deployments: Dict[str, DeploymentDescriptor] = {}
        samples: Dict[str, HealthSample] = {}
        for candidate in candidates:
            providers[candidate.provider.provider_id] = candidate.provider  # type: ignore[index]
            if candidate.model is not None:
                models[candidate.model.model_id] = candidate.model  # type: ignore[index]
            sample = (
                probe_samples.get(candidate.deployment.deployment_id)  # type: ignore[union-attr]
                if probe_samples is not None
                else candidate.input_sample
            )
            if sample is not None:
                existing = samples.get(sample.deployment_id)
                if existing is None or (
                    sample.observed_at,
                    canonical_json(sample.to_dict(), reject_secrets=False),
                ) > (
                    existing.observed_at,
                    canonical_json(existing.to_dict(), reject_secrets=False),
                ):
                    samples[sample.deployment_id] = sample
            deployments[candidate.deployment.deployment_id] = _project_sample(  # type: ignore[index]
                candidate.deployment, sample, current
            )
        snapshot = CatalogSnapshot(
            providers=tuple(providers.values()),
            models=tuple(models.values()),
            deployments=tuple(deployments.values()),
            created_at=self.observed_at,
        )
        return DeploymentSourceResult(
            snapshot=snapshot,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=self.source_revision,
                updated_at=self.observed_at,
            ),
            health_samples=tuple(samples[key] for key in sorted(samples)),
            diagnostics=tuple(diagnostics),
            redacted_fields=redacted_fields,
        )

    def load(self) -> DeploymentSourceResult:
        """Return a pure snapshot; the injected probe is not consulted."""

        candidates, diagnostics, redacted = self._candidates()
        return self._result(candidates, diagnostics, redacted)

    snapshot = load
    read = load

    def refresh(
        self,
        *,
        probe: Optional[Callable[[ProbeTarget], Any]] = None,
    ) -> DeploymentSourceResult:
        """Explicitly invoke an injected, inference-forbidden health probe."""

        selected_probe = probe or self.probe
        if selected_probe is None:
            raise ValueError("explicit refresh requires an injected probe")
        candidates, diagnostics, redacted = self._candidates()
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        samples: Dict[str, HealthSample] = {}
        mutable_diagnostics = list(diagnostics)
        for candidate in candidates:
            try:
                value = selected_probe(candidate.target)
                if inspect.isawaitable(value):
                    if inspect.iscoroutine(value):
                        value.close()
                    raise ValueError("async probes require an async source adapter")
                sample = _probe_sample(
                    value,
                    candidate,
                    now=now,
                    source=self.source,
                    default_ttl=self.health_ttl_seconds,
                )
            except Exception as exc:
                sample = HealthSample(
                    deployment_id=candidate.deployment.deployment_id,  # type: ignore[arg-type]
                    observed_at=now,
                    ttl_seconds=self.health_ttl_seconds,
                    provenance="%s.probe" % self.source,
                    diagnostics=(_safe_diagnostic(exc),),
                )
                if len(mutable_diagnostics) < MAX_DIAGNOSTICS:
                    mutable_diagnostics.append(
                        SourceDiagnostic(
                            index=None,
                            code="probe_failed",
                            message=_safe_diagnostic(exc),
                            source_record_id=candidate.deployment.deployment_id,
                        )
                    )
            samples[sample.deployment_id] = sample
        return self._result(
            candidates,
            mutable_diagnostics,
            redacted,
            probe_samples=samples,
            now=now,
        )


class ServedEndpointDeploymentSource(DeploymentCatalogSource):
    """Defaults for ModelManager/MCP++ served-model advertisements."""

    def __init__(self, records: Any = None, **kwargs: Any) -> None:
        kwargs.setdefault("source", "model-manager.served")
        kwargs.setdefault("default_service", "openai-compatible")
        kwargs.setdefault("default_protocol", "openai.http")
        kwargs.setdefault("default_provider", "llamacpp")
        super().__init__(records, **kwargs)


class BackendDeploymentSource(DeploymentCatalogSource):
    """Defaults for injected InferenceBackendManager registration snapshots."""

    def __init__(self, records: Any = None, **kwargs: Any) -> None:
        kwargs.setdefault("source", "inference-backend-manager")
        kwargs.setdefault("default_service", "inference-backend")
        super().__init__(records, **kwargs)


def adapt_deployment_source(records: Any = None, **kwargs: Any) -> DeploymentSourceResult:
    return DeploymentCatalogSource(records, **kwargs).load()


def adapt_served_endpoints(records: Any = None, **kwargs: Any) -> DeploymentSourceResult:
    return ServedEndpointDeploymentSource(records, **kwargs).load()


def adapt_backend_deployments(records: Any = None, **kwargs: Any) -> DeploymentSourceResult:
    return BackendDeploymentSource(records, **kwargs).load()


# Compatibility-friendly names for the catalog aggregation work that consumes
# source adapters through duck typing.
DeploymentSourceAdapter = DeploymentCatalogSource
ServedEndpointSource = ServedEndpointDeploymentSource
BackendSourceAdapter = BackendDeploymentSource
load_deployments = adapt_deployment_source


__all__ = [
    "BackendDeploymentSource",
    "BackendSourceAdapter",
    "DEFAULT_DEPLOYMENT_PRECEDENCE",
    "DEFAULT_HEALTH_TTL_SECONDS",
    "DeploymentCatalogSource",
    "DeploymentIdentity",
    "DeploymentSourceAdapter",
    "DeploymentSourceResult",
    "HealthSample",
    "MAX_DIAGNOSTIC_BYTES",
    "MAX_HEALTH_DIAGNOSTICS",
    "MAX_HEALTH_TTL_SECONDS",
    "ProbeTarget",
    "ServedEndpointDeploymentSource",
    "ServedEndpointSource",
    "adapt_backend_deployments",
    "adapt_deployment_source",
    "adapt_served_endpoints",
    "load_deployments",
]
