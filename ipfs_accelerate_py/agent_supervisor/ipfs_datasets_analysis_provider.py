"""Lazy, bounded adapter for optional :mod:`ipfs_datasets_py` analysis.

The supervisor may use ``ipfs_datasets_py`` to nominate analysis evidence, but
that optional package is not part of the supervisor's trusted completion
boundary.  This module therefore has three deliberately narrow properties:

* constructing the provider and inspecting its local capability declaration
  never imports the optional package;
* dispatch is limited to a closed operation vocabulary and bounded canonical
  JSON requests and responses; and
* every result is explicitly non-authoritative.  Missing, incompatible, or
  unhealthy optional capabilities produce a typed local-fallback result.

Backends can be injected directly for tests and embedded deployments.  A
backend may be a callable, expose ``analyze(request)``, or expose one of the
allowlisted operation methods.  Both ordinary and awaitable return values are
supported without requiring an async test/runtime dependency.

The adapter intentionally owns no cache or single-flight map.  Offload,
retrieval, and local analysis are one authority-changing operation and are
therefore coordinated together by ``AnalysisPipeline`` using its complete
seven-dimension cache key.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import queue
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .formal_verification_contracts import content_identity


IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION: Final = 1
IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION: Final = 1
IPFS_DATASETS_ANALYSIS_PROVIDER_ID: Final = "ipfs_datasets_py.analysis"
IPFS_DATASETS_OFFLOAD_COORDINATION_BOUNDARY: Final = (
    "analysis_pipeline.single_flight"
)
IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID: Final = (
    "184801846437522667882915494501685213497"
)

# Compatibility spelling used by objective/evidence scanners.
OPTIONAL_DATASETS_DEGRADATION_REQUIREMENT_ID = (
    IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID
)

PROVIDER_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-analysis-capability@1"
)
PROVIDER_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-analysis-request@1"
)
PROVIDER_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/ipfs-datasets-analysis-result@2"
)
PROVIDER_DEGRADATION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "ipfs-datasets-lazy-degradation-evidence@2"
)

DEFAULT_OPTIONAL_MODULE: Final = "ipfs_datasets_py"
DEFAULT_MAX_QUERY_BYTES: Final = 16 * 1024
DEFAULT_MAX_REQUEST_BYTES: Final = 64 * 1024
DEFAULT_MAX_RESPONSE_BYTES: Final = 128 * 1024
DEFAULT_MAX_RESULTS: Final = 32
DEFAULT_MAX_REFERENCE_BYTES: Final = 4096
DEFAULT_TIMEOUT_MS: Final = 30_000
MAX_CONCURRENT_PROVIDER_DISPATCHES: Final = 4

_FORBIDDEN_FIELDS = frozenset(
    {
        "source",
        "source_body",
        "source_code",
        "source_text",
        "file_contents",
        "content",
        "raw",
        "raw_output",
        "decoded_output",
        "decoded_model_output",
        "model_output",
        "model_response",
        "prompt",
        "completion",
        "transcript",
        "ast",
        "ast_body",
        "graph",
        "artifact_graph",
        "nested_graph",
        "embedding",
    }
)
_REFERENCE_FIELDS = frozenset(
    {
        "reference_id",
        "evidence_id",
        "artifact_id",
        "record_id",
        "receipt_id",
        "cid",
        "digest",
        "uri",
        "path",
        "symbol",
        "kind",
        "summary",
        "detail",
        "score",
        "score_millionths",
        "provenance_id",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "artifact_id",
        "record_id",
        "receipt_id",
        "cid",
        "digest",
        "uri",
        "path",
        "kind",
    }
)


class IpfsDatasetsAnalysisProviderError(ValueError):
    """A provider request or policy violates the bounded adapter contract."""


class AnalysisProviderOperation(str, Enum):
    GRAPH_RETRIEVAL = "graph_retrieval"
    DATASET_QUERY = "dataset_query"
    PROVENANCE_QUERY = "provenance_query"
    PREMISE_SELECTION = "premise_selection"
    PROOF_CANDIDATE_SELECTION = "proof_candidate_selection"
    LEGAL_LOGIC_ANALYSIS = "legal_logic_analysis"
    BATCH_ANALYSIS = "batch_analysis"


DEFAULT_OPERATIONS: Final = tuple(AnalysisProviderOperation)


class AnalysisProviderStatus(str, Enum):
    COMPLETED = "completed"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    MALFORMED = "malformed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class AnalysisProviderHealth(str, Enum):
    LAZY = "lazy"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"


# Only adapter states in this closed table may prove lazy, explicit
# degradation.  Other failures remain useful typed diagnostics, but a
# self-consistent caller-created witness with an arbitrary reason, health, or
# import history must not emit the objective requirement.
_PROVING_DEGRADATION_STATES: Final = MappingProxyType(
    {
        "provider_disabled": (
            AnalysisProviderStatus.DISABLED,
            frozenset({False}),
            AnalysisProviderHealth.DEGRADED,
        ),
        "operation_not_allowlisted": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        "optional_module_unavailable": (
            AnalysisProviderStatus.UNAVAILABLE,
            frozenset({True}),
            AnalysisProviderHealth.UNAVAILABLE,
        ),
        "optional_capability_unavailable": (
            AnalysisProviderStatus.UNAVAILABLE,
            frozenset({True}),
            AnalysisProviderHealth.UNAVAILABLE,
        ),
        "optional_dispatch_dependency_unavailable": (
            AnalysisProviderStatus.UNAVAILABLE,
            frozenset({False, True}),
            AnalysisProviderHealth.UNAVAILABLE,
        ),
        "protocol_incompatible": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        "backend_unhealthy": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.DEGRADED,
        ),
        "no_supported_operations": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        "operation_not_supported": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        "operation_dispatch_unavailable": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
    }
)


def _canonical_value(value: Any, *, name: str, depth: int = 0) -> Any:
    if depth > 8:
        raise IpfsDatasetsAnalysisProviderError(f"{name} exceeds maximum depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IpfsDatasetsAnalysisProviderError(f"{name} must be finite")
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value, name=name, depth=depth + 1)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise IpfsDatasetsAnalysisProviderError(
                f"{name} object keys must be strings"
            )
        return {
            key: _canonical_value(item, name=name, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [
            _canonical_value(item, name=name, depth=depth + 1)
            for item in value
        ]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical_value(converter(), name=name, depth=depth + 1)
    raise IpfsDatasetsAnalysisProviderError(
        f"{name} contains unsupported {type(value).__name__}"
    )


def _json_bytes(value: Any, *, name: str) -> bytes:
    try:
        normalized = _canonical_value(value, name=name)
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, IpfsDatasetsAnalysisProviderError):
            raise
        raise IpfsDatasetsAnalysisProviderError(
            f"{name} must be canonical JSON"
        ) from exc


def _content_id(value: Any, *, name: str) -> str:
    """Content address canonical adapter JSON, including finite query floats."""

    return f"{name}:sha256:" + hashlib.sha256(
        _json_bytes(value, name=name)
    ).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = 4096,
) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise IpfsDatasetsAnalysisProviderError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise IpfsDatasetsAnalysisProviderError(f"{name} is required")
    if "\x00" in result:
        raise IpfsDatasetsAnalysisProviderError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > max_bytes:
        raise IpfsDatasetsAnalysisProviderError(f"{name} exceeds {max_bytes} bytes")
    return result


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise IpfsDatasetsAnalysisProviderError(
            f"{name} must be an integer between 1 and {maximum}"
        )
    return value


def _operation(value: Any) -> AnalysisProviderOperation:
    if isinstance(value, AnalysisProviderOperation):
        return value
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    aliases = {
        "graphrag": AnalysisProviderOperation.GRAPH_RETRIEVAL,
        "retrieve": AnalysisProviderOperation.GRAPH_RETRIEVAL,
        "search": AnalysisProviderOperation.GRAPH_RETRIEVAL,
        "dataset": AnalysisProviderOperation.DATASET_QUERY,
        "provenance": AnalysisProviderOperation.PROVENANCE_QUERY,
        "premises": AnalysisProviderOperation.PREMISE_SELECTION,
        "proof_candidates": AnalysisProviderOperation.PROOF_CANDIDATE_SELECTION,
        "logic": AnalysisProviderOperation.LEGAL_LOGIC_ANALYSIS,
        "batch": AnalysisProviderOperation.BATCH_ANALYSIS,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return AnalysisProviderOperation(raw)
    except ValueError as exc:
        raise IpfsDatasetsAnalysisProviderError(
            f"unsupported analysis provider operation: {raw or '<empty>'}"
        ) from exc


def normalize_analysis_provider_operation(value: Any) -> AnalysisProviderOperation:
    """Return the adapter's canonical operation for a public name or alias."""

    return _operation(value)


def _status(value: Any) -> AnalysisProviderStatus:
    if isinstance(value, AnalysisProviderStatus):
        return value
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    aliases = {
        "success": AnalysisProviderStatus.COMPLETED,
        "successful": AnalysisProviderStatus.COMPLETED,
        "succeeded": AnalysisProviderStatus.COMPLETED,
        "ok": AnalysisProviderStatus.COMPLETED,
        "error": AnalysisProviderStatus.FAILED,
        "timeout": AnalysisProviderStatus.TIMED_OUT,
    }
    if raw in aliases:
        return aliases[raw]
    try:
        return AnalysisProviderStatus(raw)
    except ValueError as exc:
        raise IpfsDatasetsAnalysisProviderError(
            "backend returned an unsupported status"
        ) from exc


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    value = getattr(token, "cancelled", False)
    if callable(value):
        value = value()
    if not value:
        checker = getattr(token, "is_cancelled", None)
        value = checker() if callable(checker) else False
    return bool(value)


def _resource_use(value: Any) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise IpfsDatasetsAnalysisProviderError(
            "backend resource_use must be an object"
        )
    if len(value) > 32:
        raise IpfsDatasetsAnalysisProviderError(
            "backend resource_use exceeds 32 counters"
        )
    result: dict[str, int] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise IpfsDatasetsAnalysisProviderError(
                "backend resource_use counters must be non-negative integers"
            )
        result[_text(key, "resource_use key", max_bytes=64)] = item
    return result


@dataclass(frozen=True)
class AnalysisProviderBounds:
    max_results: int = DEFAULT_MAX_RESULTS
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_reference_bytes: int = DEFAULT_MAX_REFERENCE_BYTES
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def __post_init__(self) -> None:
        limits = {
            "max_results": 1000,
            "max_query_bytes": 1024 * 1024,
            "max_request_bytes": 4 * 1024 * 1024,
            "max_response_bytes": 16 * 1024 * 1024,
            "max_reference_bytes": 256 * 1024,
            "timeout_ms": 10 * 60 * 1000,
        }
        for name, maximum in limits.items():
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum=maximum),
            )
        if self.max_query_bytes > self.max_request_bytes:
            raise IpfsDatasetsAnalysisProviderError(
                "max_query_bytes cannot exceed max_request_bytes"
            )
        if self.max_reference_bytes > self.max_response_bytes:
            raise IpfsDatasetsAnalysisProviderError(
                "max_reference_bytes cannot exceed max_response_bytes"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_results": self.max_results,
            "max_query_bytes": self.max_query_bytes,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_reference_bytes": self.max_reference_bytes,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_value(
        cls, value: "AnalysisProviderBounds | Mapping[str, Any] | None"
    ) -> "AnalysisProviderBounds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError("bounds must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise IpfsDatasetsAnalysisProviderError(
                "unknown bounds: " + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class AnalysisProviderPolicy:
    enabled: bool = True
    module_name: str = DEFAULT_OPTIONAL_MODULE
    operations: tuple[AnalysisProviderOperation, ...] = DEFAULT_OPERATIONS
    bounds: AnalysisProviderBounds = field(default_factory=AnalysisProviderBounds)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise IpfsDatasetsAnalysisProviderError("enabled must be a boolean")
        object.__setattr__(
            self, "module_name", _text(self.module_name, "module_name", max_bytes=255)
        )
        if isinstance(self.operations, (str, bytes)) or not isinstance(
            self.operations, Sequence
        ):
            raise IpfsDatasetsAnalysisProviderError("operations must be a sequence")
        operations = tuple(sorted({_operation(item) for item in self.operations}, key=lambda x: x.value))
        if not operations:
            raise IpfsDatasetsAnalysisProviderError("operations must not be empty")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self, "bounds", AnalysisProviderBounds.from_value(self.bounds)
        )

    @classmethod
    def from_value(
        cls, value: "AnalysisProviderPolicy | Mapping[str, Any] | None"
    ) -> "AnalysisProviderPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError("policy must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise IpfsDatasetsAnalysisProviderError(
                "unknown policy fields: " + ", ".join(sorted(unknown))
            )
        return cls(
            enabled=value.get("enabled", True),
            module_name=value.get("module_name", DEFAULT_OPTIONAL_MODULE),
            operations=(
                tuple(value["operations"])
                if "operations" in value
                else DEFAULT_OPERATIONS
            ),
            bounds=AnalysisProviderBounds.from_value(value.get("bounds")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "module_name": self.module_name,
            "operations": [item.value for item in self.operations],
            "bounds": self.bounds.to_dict(),
        }

    @property
    def policy_id(self) -> str:
        return _content_id(self.to_dict(), name="analysis-provider-policy")


@dataclass(frozen=True)
class AnalysisProviderRequest:
    operation: AnalysisProviderOperation
    repository_id: str
    tree_id: str
    objective_revision: str
    query: Any
    artifact_references: tuple[Mapping[str, Any], ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    bounds: AnalysisProviderBounds = field(default_factory=AnalysisProviderBounds)
    request_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(
            self, "bounds", AnalysisProviderBounds.from_value(self.bounds)
        )
        for name in ("repository_id", "tree_id", "objective_revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, max_bytes=1024)
            )
        query = _canonical_value(self.query, name="query")
        if len(_json_bytes(query, name="query")) > self.bounds.max_query_bytes:
            raise IpfsDatasetsAnalysisProviderError("query exceeds max_query_bytes")
        object.__setattr__(self, "query", query)
        if not isinstance(self.payload, Mapping):
            raise IpfsDatasetsAnalysisProviderError("payload must be an object")
        payload = _canonical_value(dict(self.payload), name="payload")
        if _find_forbidden_fields(payload):
            raise IpfsDatasetsAnalysisProviderError(
                "payload contains forbidden heavy fields"
            )
        object.__setattr__(self, "payload", payload)
        references = tuple(
            _compact_artifact_reference(item, self.bounds.max_reference_bytes)
            for item in self.artifact_references
        )
        if len(references) > self.bounds.max_results:
            raise IpfsDatasetsAnalysisProviderError(
                "artifact_references exceeds max_results"
            )
        object.__setattr__(self, "artifact_references", references)
        derived_request_id = _content_id(
            self._identity_payload(), name="analysis-provider-request"
        )
        if self.request_id:
            claimed_request_id = _text(
                self.request_id, "request_id", max_bytes=256
            )
            if claimed_request_id != derived_request_id:
                raise IpfsDatasetsAnalysisProviderError(
                    "analysis provider request identity does not match content"
                )
        object.__setattr__(self, "request_id", derived_request_id)
        if len(_json_bytes(self.to_dict(), name="request")) > self.bounds.max_request_bytes:
            raise IpfsDatasetsAnalysisProviderError("request exceeds max_request_bytes")

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_REQUEST_SCHEMA,
            "protocol_version": IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION,
            "operation": self.operation.value,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "query": self.query,
            "artifact_references": list(self.artifact_references),
            "payload": dict(self.payload),
            "bounds": self.bounds.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id, **self._identity_payload()}

    @classmethod
    def from_value(
        cls, value: "AnalysisProviderRequest | Mapping[str, Any]"
    ) -> "AnalysisProviderRequest":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "analysis provider request must be an object"
            )
        aliases = dict(value)
        if "repository_id" not in aliases and "repo_id" in aliases:
            aliases["repository_id"] = aliases.pop("repo_id")
        if "tree_id" not in aliases and "repository_tree_identity" in aliases:
            aliases["tree_id"] = aliases.pop("repository_tree_identity")
        allowed = {
            "schema",
            "protocol_version",
            "operation",
            "repository_id",
            "tree_id",
            "objective_revision",
            "query",
            "artifact_references",
            "payload",
            "bounds",
            "request_id",
        }
        unknown = set(aliases) - allowed
        if unknown:
            raise IpfsDatasetsAnalysisProviderError(
                "unknown request fields: " + ", ".join(sorted(unknown))
            )
        schema = aliases.pop("schema", PROVIDER_REQUEST_SCHEMA)
        protocol = aliases.pop(
            "protocol_version", IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION
        )
        if schema != PROVIDER_REQUEST_SCHEMA:
            raise IpfsDatasetsAnalysisProviderError("unsupported request schema")
        if protocol != IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION:
            raise IpfsDatasetsAnalysisProviderError("unsupported protocol version")
        return cls(
            operation=aliases.get("operation", ""),
            repository_id=aliases.get("repository_id", ""),
            tree_id=aliases.get("tree_id", ""),
            objective_revision=aliases.get("objective_revision", ""),
            query=aliases.get("query", ""),
            artifact_references=tuple(aliases.get("artifact_references") or ()),
            payload=aliases.get("payload") or {},
            bounds=AnalysisProviderBounds.from_value(aliases.get("bounds")),
            request_id=str(aliases.get("request_id") or ""),
        )


def _compact_artifact_reference(value: Any, max_bytes: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        converter = getattr(value, "to_dict", None)
        value = converter() if callable(converter) else value
    if not isinstance(value, Mapping):
        raise IpfsDatasetsAnalysisProviderError(
            "artifact reference must be an object"
        )
    forbidden = set(value).intersection(_FORBIDDEN_FIELDS)
    unknown = set(value) - _ARTIFACT_FIELDS
    if forbidden or unknown:
        raise IpfsDatasetsAnalysisProviderError(
            "artifact reference contains unsupported fields"
        )
    result = {
        key: _text(item, f"artifact reference {key}", required=False, max_bytes=2048)
        for key, item in sorted(value.items())
        if item not in (None, "")
    }
    if not result:
        raise IpfsDatasetsAnalysisProviderError("artifact reference is empty")
    if len(_json_bytes(result, name="artifact reference")) > max_bytes:
        raise IpfsDatasetsAnalysisProviderError(
            "artifact reference exceeds max_reference_bytes"
        )
    return result


@dataclass(frozen=True)
class AnalysisProviderCapability:
    health: AnalysisProviderHealth
    operations: tuple[AnalysisProviderOperation, ...]
    imported: bool = False
    reason_code: str = "lazy_not_probed"
    provider_version: str = "unknown"
    bounds: AnalysisProviderBounds = field(default_factory=AnalysisProviderBounds)

    def __post_init__(self) -> None:
        if not isinstance(self.health, AnalysisProviderHealth):
            object.__setattr__(self, "health", AnalysisProviderHealth(str(self.health)))
        object.__setattr__(
            self,
            "operations",
            tuple(sorted({_operation(item) for item in self.operations}, key=lambda x: x.value)),
        )
        if not isinstance(self.imported, bool):
            raise IpfsDatasetsAnalysisProviderError("imported must be a boolean")
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", required=False, max_bytes=128),
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(
                self.provider_version,
                "provider_version",
                required=False,
                max_bytes=128,
            ),
        )
        object.__setattr__(
            self, "bounds", AnalysisProviderBounds.from_value(self.bounds)
        )

    @property
    def available(self) -> bool:
        return self.health is AnalysisProviderHealth.HEALTHY

    @property
    def non_authoritative(self) -> bool:
        return True

    def supports(self, operation: Any) -> bool:
        try:
            candidate = _operation(operation)
        except IpfsDatasetsAnalysisProviderError:
            return False
        return self.available and candidate in self.operations

    @property
    def capability_id(self) -> str:
        return _content_id(self._payload(), name="analysis-provider-capability")

    @property
    def content_id(self) -> str:
        return self.capability_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CAPABILITY_SCHEMA,
            "protocol_version": IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION,
            "provider_id": IPFS_DATASETS_ANALYSIS_PROVIDER_ID,
            "adapter_version": IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION,
            "provider_version": self.provider_version,
            "health": self.health.value,
            "available": self.available,
            "imported": self.imported,
            "operations": [item.value for item in self.operations],
            "bounds": self.bounds.to_dict(),
            "reason_code": self.reason_code,
            "lazy_import": True,
            "non_authoritative": True,
            "completion_authority": False,
            "proof_success": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"capability_id": self.capability_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisProviderCapability":
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "provider capability must be an object"
            )
        allowed = {
            "capability_id",
            "schema",
            "protocol_version",
            "provider_id",
            "adapter_version",
            "provider_version",
            "health",
            "available",
            "imported",
            "operations",
            "bounds",
            "reason_code",
            "lazy_import",
            "non_authoritative",
            "completion_authority",
            "proof_success",
        }
        if set(value) - allowed:
            raise IpfsDatasetsAnalysisProviderError(
                "provider capability contains unknown fields"
            )
        if (
            value.get("schema") != PROVIDER_CAPABILITY_SCHEMA
            or value.get("protocol_version")
            != IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION
            or value.get("provider_id") != IPFS_DATASETS_ANALYSIS_PROVIDER_ID
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "unsupported provider capability"
            )
        result = cls(
            health=value.get("health", ""),
            operations=tuple(value.get("operations") or ()),
            imported=value.get("imported", False),
            reason_code=value.get("reason_code", ""),
            provider_version=value.get("provider_version", "unknown"),
            bounds=AnalysisProviderBounds.from_value(value.get("bounds")),
        )
        claimed = value.get("capability_id")
        if claimed != result.capability_id:
            raise IpfsDatasetsAnalysisProviderError(
                "provider capability identity does not match"
            )
        available_claim = value.get("available", result.available)
        if (
            not isinstance(available_claim, bool)
            or available_claim != result.available
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "provider capability availability claim does not match"
            )
        fixed_claims = {
            "adapter_version": IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION,
            "lazy_import": True,
            "non_authoritative": True,
            "completion_authority": False,
            "proof_success": False,
        }
        for name, expected in fixed_claims.items():
            if value.get(name) != expected:
                raise IpfsDatasetsAnalysisProviderError(
                    f"provider capability {name} claim does not match"
                )
        return result


@dataclass(frozen=True)
class IpfsDatasetsProviderDegradationEvidence:
    status: AnalysisProviderStatus
    operation: AnalysisProviderOperation
    reason_code: str
    import_attempted: bool
    request_id: str = ""
    repository_id: str = ""
    tree_id: str = ""
    objective_revision: str = ""
    policy_id: str = ""
    backend_health: AnalysisProviderHealth = AnalysisProviderHealth.DEGRADED
    fallback: str = "local_deterministic_analysis"
    requirement_id: str = IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _status(self.status))
        object.__setattr__(self, "operation", _operation(self.operation))
        if self.status is AnalysisProviderStatus.COMPLETED:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence cannot have completed status"
            )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", max_bytes=128),
        )
        if not isinstance(self.import_attempted, bool):
            raise IpfsDatasetsAnalysisProviderError(
                "import_attempted must be a boolean"
            )
        for name in (
            "request_id",
            "repository_id",
            "tree_id",
            "objective_revision",
            "policy_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=False,
                    max_bytes=1024,
                ),
            )
        if not isinstance(self.backend_health, AnalysisProviderHealth):
            object.__setattr__(
                self,
                "backend_health",
                AnalysisProviderHealth(str(self.backend_health)),
            )
        object.__setattr__(
            self, "fallback", _text(self.fallback, "fallback", max_bytes=128)
        )
        if self.requirement_id != IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID:
            raise IpfsDatasetsAnalysisProviderError(
                "unexpected optional-provider degradation requirement"
            )

    @property
    def proves_requirement(self) -> bool:
        # A bare or detached occurrence of the requirement ID is never proof.
        # Only the adapter-created witness for a concrete bounded request may
        # claim the requirement.  Other failure states remain typed and
        # non-authoritative, but do not by themselves establish lazy
        # degradation.
        expected = _PROVING_DEGRADATION_STATES.get(self.reason_code)
        return bool(
            self.proof_bound
            and self.fallback == "local_deterministic_analysis"
            and expected is not None
            and self.status is expected[0]
            and self.import_attempted in expected[1]
            and self.backend_health is expected[2]
        )

    @property
    def request_bound(self) -> bool:
        return all(
            (
                self.request_id,
                self.repository_id,
                self.tree_id,
                self.objective_revision,
            )
        )

    @property
    def proof_bound(self) -> bool:
        return self.request_bound and bool(self.policy_id)

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        """Fail closed because no active policy is available at this surface.

        Use :meth:`proved_requirement_ids_for` with the active request and
        policy.  :attr:`diagnostic_requirement_ids` exposes the shaped witness
        claim for observability without presenting it as proof.
        """

        return ()

    @property
    def diagnostic_requirement_ids(self) -> tuple[str, ...]:
        """Return shaped-but-not-active-context-qualified requirement IDs."""

        return (self.requirement_id,) if self.proves_requirement else ()

    def proves_for(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any],
        policy: AnalysisProviderPolicy | Mapping[str, Any] | None,
    ) -> bool:
        """Independently bind this witness to the active request and policy."""

        normalized_request = AnalysisProviderRequest.from_value(request)
        normalized_policy = AnalysisProviderPolicy.from_value(policy)
        within_policy_bounds = all(
            getattr(normalized_request.bounds, name)
            <= getattr(normalized_policy.bounds, name)
            for name in AnalysisProviderBounds.__dataclass_fields__
        )
        if self.reason_code == "provider_disabled":
            reason_matches_policy = not normalized_policy.enabled
        elif self.reason_code == "operation_not_allowlisted":
            reason_matches_policy = (
                normalized_policy.enabled
                and normalized_request.operation
                not in normalized_policy.operations
            )
        else:
            # All remaining proving states occur after the provider's enabled
            # and operation-allowlist gates in ``_execute``.
            reason_matches_policy = (
                normalized_policy.enabled
                and normalized_request.operation
                in normalized_policy.operations
            )
        return bool(
            self.proves_requirement
            and within_policy_bounds
            and reason_matches_policy
            and self.request_id == normalized_request.request_id
            and self.repository_id == normalized_request.repository_id
            and self.tree_id == normalized_request.tree_id
            and self.objective_revision
            == normalized_request.objective_revision
            and self.operation is normalized_request.operation
            and self.policy_id == normalized_policy.policy_id
        )

    def proved_requirement_ids_for(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any],
        policy: AnalysisProviderPolicy | Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        """Return requirement IDs only after active-context verification."""

        return (
            (IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,)
            if self.proves_for(request, policy)
            else ()
        )

    @property
    def evidence_id(self) -> str:
        return content_identity(self._payload())

    @property
    def content_id(self) -> str:
        return self.evidence_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_DEGRADATION_EVIDENCE_SCHEMA,
            "version": IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION,
            "requirement_id": self.requirement_id,
            "provider_id": IPFS_DATASETS_ANALYSIS_PROVIDER_ID,
            "status": self.status.value,
            "operation": self.operation.value,
            "reason_code": self.reason_code,
            "request_id": self.request_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "backend_health": self.backend_health.value,
            "request_bound": self.request_bound,
            "proof_bound": self.proof_bound,
            "lazy_import": True,
            "import_attempted": self.import_attempted,
            "explicit_fallback": True,
            "fallback": self.fallback,
            "completion_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, **self._payload()}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "IpfsDatasetsProviderDegradationEvidence":
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence must be an object"
            )
        allowed = {
            "evidence_id",
            "schema",
            "version",
            "requirement_id",
            "provider_id",
            "status",
            "operation",
            "reason_code",
            "request_id",
            "repository_id",
            "tree_id",
            "objective_revision",
            "policy_id",
            "backend_health",
            "request_bound",
            "proof_bound",
            "lazy_import",
            "import_attempted",
            "explicit_fallback",
            "fallback",
            "completion_authority",
        }
        if set(value) - allowed:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence contains unknown fields"
            )
        if (
            value.get("schema") != PROVIDER_DEGRADATION_EVIDENCE_SCHEMA
            or value.get("version") != IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION
            or value.get("provider_id") != IPFS_DATASETS_ANALYSIS_PROVIDER_ID
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "unsupported degradation evidence"
            )
        result = cls(
            status=value.get("status", ""),
            operation=value.get("operation", ""),
            reason_code=value.get("reason_code", ""),
            import_attempted=value.get("import_attempted", False),
            request_id=value.get("request_id", ""),
            repository_id=value.get("repository_id", ""),
            tree_id=value.get("tree_id", ""),
            objective_revision=value.get("objective_revision", ""),
            policy_id=value.get("policy_id", ""),
            backend_health=value.get(
                "backend_health", AnalysisProviderHealth.DEGRADED
            ),
            fallback=value.get("fallback", "local_deterministic_analysis"),
            requirement_id=value.get("requirement_id", ""),
        )
        claimed = value.get("evidence_id")
        if claimed != result.evidence_id:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence identity does not match"
            )
        if value.get("lazy_import") is not True:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence must record lazy import"
            )
        if value.get("explicit_fallback") is not True:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence must record explicit fallback"
            )
        if value.get("completion_authority") is not False:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence cannot claim completion authority"
            )
        if value.get("request_bound") is not result.request_bound:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence request binding claim does not match"
            )
        if value.get("proof_bound") is not result.proof_bound:
            raise IpfsDatasetsAnalysisProviderError(
                "degradation evidence proof binding claim does not match"
            )
        return result


@dataclass(frozen=True)
class AnalysisProviderResult:
    request_id: str
    operation: AnalysisProviderOperation
    repository_id: str
    tree_id: str
    objective_revision: str
    status: AnalysisProviderStatus
    reason_code: str
    evidence_references: tuple[Mapping[str, Any], ...] = ()
    provenance_references: tuple[Mapping[str, Any], ...] = ()
    truncated: bool = False
    backend_health: AnalysisProviderHealth = AnalysisProviderHealth.DEGRADED
    provider_version: str = "unknown"
    resource_use: Mapping[str, int] = field(default_factory=dict)
    degradation_evidence: IpfsDatasetsProviderDegradationEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "status", _status(self.status))
        for name in (
            "request_id",
            "repository_id",
            "tree_id",
            "objective_revision",
            "reason_code",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, max_bytes=1024)
            )
        if not isinstance(self.backend_health, AnalysisProviderHealth):
            object.__setattr__(
                self,
                "backend_health",
                AnalysisProviderHealth(str(self.backend_health)),
            )
        if not isinstance(self.truncated, bool):
            raise IpfsDatasetsAnalysisProviderError("truncated must be a boolean")
        default_bounds = AnalysisProviderBounds()
        evidence, evidence_truncated = _compact_references(
            self.evidence_references, default_bounds
        )
        provenance, provenance_truncated = _compact_references(
            self.provenance_references, default_bounds
        )
        object.__setattr__(self, "evidence_references", evidence)
        object.__setattr__(self, "provenance_references", provenance)
        object.__setattr__(
            self,
            "truncated",
            self.truncated or evidence_truncated or provenance_truncated,
        )
        object.__setattr__(
            self,
            "provider_version",
            _text(
                self.provider_version,
                "provider_version",
                required=False,
                max_bytes=128,
            ),
        )
        if not isinstance(self.resource_use, Mapping) or any(
            not isinstance(key, str) for key in self.resource_use
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "resource_use must be an object with string keys"
            )
        resource_use: dict[str, int] = {}
        for key, value in sorted(self.resource_use.items()):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IpfsDatasetsAnalysisProviderError(
                    "resource_use values must be non-negative integers"
                )
            resource_use[_text(key, "resource_use key", max_bytes=64)] = value
        object.__setattr__(self, "resource_use", resource_use)
        if self.status is AnalysisProviderStatus.COMPLETED and self.degradation_evidence:
            raise IpfsDatasetsAnalysisProviderError(
                "completed results cannot contain degradation evidence"
            )
        if (
            self.status is not AnalysisProviderStatus.COMPLETED
            and self.degradation_evidence is None
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "degraded results require typed degradation evidence"
            )
        if (
            self.degradation_evidence is not None
            and not isinstance(
                self.degradation_evidence,
                IpfsDatasetsProviderDegradationEvidence,
            )
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "degradation_evidence must be typed"
            )
        evidence = self.degradation_evidence
        if evidence is not None:
            if not evidence.proof_bound:
                raise IpfsDatasetsAnalysisProviderError(
                    "degradation evidence is not request/policy-bound"
                )
            expected = {
                "request_id": self.request_id,
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
                "objective_revision": self.objective_revision,
                "operation": self.operation,
                "status": self.status,
                "reason_code": self.reason_code,
                "backend_health": self.backend_health,
            }
            for name, expected_value in expected.items():
                if getattr(evidence, name) != expected_value:
                    raise IpfsDatasetsAnalysisProviderError(
                        f"degradation evidence {name} is not result-bound"
                    )

    @property
    def successful(self) -> bool:
        return self.status is AnalysisProviderStatus.COMPLETED

    @property
    def degraded(self) -> bool:
        return not self.successful

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def is_completion_evidence(self) -> bool:
        return False

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        """Fail closed because no active request and policy are supplied.

        Use :meth:`proved_requirement_ids_for` for proof decisions, or
        :attr:`diagnostic_requirement_ids` for shaped witness diagnostics.
        """

        return ()

    @property
    def diagnostic_requirement_ids(self) -> tuple[str, ...]:
        """Return shaped-but-not-active-context-qualified requirement IDs."""

        return (
            self.degradation_evidence.diagnostic_requirement_ids
            if self.degradation_evidence
            else ()
        )

    def proves_requirement_for(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any],
        policy: AnalysisProviderPolicy | Mapping[str, Any] | None,
    ) -> bool:
        """Verify the degradation requirement against active execution state."""

        return bool(
            self.degradation_evidence
            and self.degradation_evidence.proves_for(request, policy)
        )

    def proved_requirement_ids_for(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any],
        policy: AnalysisProviderPolicy | Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        """Return requirement IDs only after active-context verification."""

        return (
            (IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,)
            if self.proves_requirement_for(request, policy)
            else ()
        )

    @property
    def result_id(self) -> str:
        return content_identity(self._payload())

    @property
    def content_id(self) -> str:
        return self.result_id

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_RESULT_SCHEMA,
            "version": IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION,
            "provider_id": IPFS_DATASETS_ANALYSIS_PROVIDER_ID,
            "provider_version": self.provider_version,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "evidence_references": [dict(item) for item in self.evidence_references],
            "provenance_references": [
                dict(item) for item in self.provenance_references
            ],
            "truncated": self.truncated,
            "backend_health": self.backend_health.value,
            "resource_use": dict(self.resource_use),
            "non_authoritative": True,
            "safe_for_completion_reasoning": False,
            "proof_success": False,
            "degradation_evidence": (
                self.degradation_evidence.to_dict()
                if self.degradation_evidence
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"result_id": self.result_id, **self._payload()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisProviderResult":
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "analysis provider result must be an object"
            )
        allowed = {
            "result_id",
            "schema",
            "version",
            "provider_id",
            "provider_version",
            "request_id",
            "operation",
            "repository_id",
            "tree_id",
            "objective_revision",
            "status",
            "reason_code",
            "evidence_references",
            "provenance_references",
            "truncated",
            "backend_health",
            "resource_use",
            "non_authoritative",
            "safe_for_completion_reasoning",
            "proof_success",
            "degradation_evidence",
        }
        if set(value) - allowed:
            raise IpfsDatasetsAnalysisProviderError(
                "analysis provider result contains unknown fields"
            )
        if (
            value.get("schema") != PROVIDER_RESULT_SCHEMA
            or value.get("version") != IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION
            or value.get("provider_id") != IPFS_DATASETS_ANALYSIS_PROVIDER_ID
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "unsupported analysis provider result"
            )
        degradation_raw = value.get("degradation_evidence")
        degradation = (
            IpfsDatasetsProviderDegradationEvidence.from_dict(degradation_raw)
            if isinstance(degradation_raw, Mapping)
            else None
        )
        result = cls(
            request_id=value.get("request_id", ""),
            operation=value.get("operation", ""),
            repository_id=value.get("repository_id", ""),
            tree_id=value.get("tree_id", ""),
            objective_revision=value.get("objective_revision", ""),
            status=value.get("status", ""),
            reason_code=value.get("reason_code", ""),
            evidence_references=tuple(value.get("evidence_references") or ()),
            provenance_references=tuple(
                value.get("provenance_references") or ()
            ),
            truncated=value.get("truncated", False),
            backend_health=value.get(
                "backend_health", AnalysisProviderHealth.DEGRADED
            ),
            provider_version=value.get("provider_version", "unknown"),
            resource_use=value.get("resource_use") or {},
            degradation_evidence=degradation,
        )
        fixed_claims = {
            "non_authoritative": True,
            "safe_for_completion_reasoning": False,
            "proof_success": False,
        }
        for name, expected in fixed_claims.items():
            if value.get(name) != expected:
                raise IpfsDatasetsAnalysisProviderError(
                    f"analysis provider result {name} claim does not match"
                )
        claimed = value.get("result_id")
        if claimed != result.result_id:
            raise IpfsDatasetsAnalysisProviderError(
                "analysis provider result identity does not match"
            )
        if (
            value.get("non_authoritative", True) is not True
            or value.get("safe_for_completion_reasoning", False) is not False
            or value.get("proof_success", False) is not False
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "analysis provider result contains an invalid authority claim"
            )
        return result

    def __bool__(self) -> bool:
        raise TypeError(
            "AnalysisProviderResult has no truth value; inspect status explicitly"
        )


_OPERATION_METHODS: Final = {
    AnalysisProviderOperation.GRAPH_RETRIEVAL: (
        "retrieve_analysis_evidence",
        "graphrag_retrieve",
        "retrieve",
    ),
    AnalysisProviderOperation.DATASET_QUERY: ("query_dataset", "dataset_query"),
    AnalysisProviderOperation.PROVENANCE_QUERY: (
        "query_provenance",
        "provenance_query",
    ),
    AnalysisProviderOperation.PREMISE_SELECTION: (
        "select_premises",
        "premise_selection",
    ),
    AnalysisProviderOperation.PROOF_CANDIDATE_SELECTION: (
        "select_proof_candidates",
        "proof_candidate_selection",
    ),
    AnalysisProviderOperation.LEGAL_LOGIC_ANALYSIS: (
        "analyze_legal_logic",
        "legal_logic_analysis",
    ),
    AnalysisProviderOperation.BATCH_ANALYSIS: ("analyze_batch", "batch_analysis"),
}


class IpfsDatasetsAnalysisProvider:
    """Capability-negotiated adapter with no eager optional imports."""

    provider_id = IPFS_DATASETS_ANALYSIS_PROVIDER_ID
    provider_version = str(IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION)
    protocol_version = IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION

    def __init__(
        self,
        policy: AnalysisProviderPolicy | Mapping[str, Any] | None = None,
        *,
        importer: Callable[[str], Any] | None = None,
        backend: Any = None,
        enabled: bool | None = None,
        module_name: str | None = None,
        operations: Sequence[AnalysisProviderOperation | str] | None = None,
        bounds: AnalysisProviderBounds | Mapping[str, Any] | None = None,
    ) -> None:
        selected_policy = AnalysisProviderPolicy.from_value(policy)
        overrides = any(
            item is not None for item in (enabled, module_name, operations, bounds)
        )
        if policy is not None and overrides:
            raise IpfsDatasetsAnalysisProviderError(
                "policy cannot be combined with policy field overrides"
            )
        if overrides:
            selected_policy = AnalysisProviderPolicy(
                enabled=True if enabled is None else enabled,
                module_name=module_name or DEFAULT_OPTIONAL_MODULE,
                operations=(
                    tuple(operations)
                    if operations is not None
                    else DEFAULT_OPERATIONS
                ),
                bounds=AnalysisProviderBounds.from_value(bounds),
            )
        if importer is not None and not callable(importer):
            raise IpfsDatasetsAnalysisProviderError("importer must be callable")
        self.policy = selected_policy
        self._importer = importer or importlib.import_module
        self._backend = backend
        # Timed-out Python threads cannot be killed safely.  Bound the number
        # that may remain in a non-cooperative backend so repeated timeouts do
        # not create an unbounded thread/resource leak.
        self._dispatch_slots = threading.BoundedSemaphore(
            MAX_CONCURRENT_PROVIDER_DISPATCHES
        )

    def capabilities(self) -> AnalysisProviderCapability:
        """Return the local lazy declaration without importing the backend."""

        return AnalysisProviderCapability(
            health=(
                AnalysisProviderHealth.LAZY
                if self.policy.enabled
                else AnalysisProviderHealth.DEGRADED
            ),
            operations=self.policy.operations,
            imported=False,
            reason_code=(
                "lazy_not_probed" if self.policy.enabled else "provider_disabled"
            ),
            provider_version="unknown",
            bounds=self.policy.bounds,
        )

    capability = capabilities

    def _degraded(
        self,
        request: AnalysisProviderRequest,
        status: AnalysisProviderStatus,
        reason_code: str,
        *,
        import_attempted: bool,
        health: AnalysisProviderHealth,
        provider_version: str = "unknown",
    ) -> AnalysisProviderResult:
        evidence = IpfsDatasetsProviderDegradationEvidence(
            status=status,
            operation=request.operation,
            reason_code=reason_code,
            import_attempted=import_attempted,
            request_id=request.request_id,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_revision=request.objective_revision,
            policy_id=self.policy.policy_id,
            backend_health=health,
        )
        return AnalysisProviderResult(
            request_id=request.request_id,
            operation=request.operation,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_revision=request.objective_revision,
            status=status,
            reason_code=reason_code,
            backend_health=health,
            provider_version=provider_version,
            degradation_evidence=evidence,
        )

    def _load_backend(self) -> tuple[Any, bool]:
        if self._backend is not None:
            return self._backend, False
        return self._importer(self.policy.module_name), True

    def _bounds_from_limits(self, value: Any) -> AnalysisProviderBounds:
        """Translate retrieval-style limits into the provider's strict bounds."""

        if value is None:
            return self.policy.bounds
        if isinstance(value, AnalysisProviderBounds):
            return value
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            value = converter()
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "limits must be provider bounds or a retrieval-limits object"
            )
        if set(value).issubset(AnalysisProviderBounds.__dataclass_fields__):
            merged = {**self.policy.bounds.to_dict(), **dict(value)}
            return AnalysisProviderBounds(**merged)
        # ``analysis_retrieval.RetrievalLimits`` intentionally has a different
        # vocabulary.  Only semantically equivalent bounds are projected.
        unknown = set(value) - {
            "max_results",
            "max_bytes",
            "max_candidates",
            "max_hops",
            "max_backend_results",
        }
        if unknown:
            raise IpfsDatasetsAnalysisProviderError(
                "unknown retrieval limits: " + ", ".join(sorted(unknown))
            )
        projected = self.policy.bounds.to_dict()
        if "max_results" in value:
            projected["max_results"] = int(value["max_results"])
        if "max_bytes" in value:
            projected["max_response_bytes"] = int(value["max_bytes"])
        return AnalysisProviderBounds(**projected)

    def _validate_policy_bounds(self, bounds: AnalysisProviderBounds) -> None:
        for name in AnalysisProviderBounds.__dataclass_fields__:
            if getattr(bounds, name) > getattr(self.policy.bounds, name):
                raise IpfsDatasetsAnalysisProviderError(
                    f"request {name} cannot expand provider policy"
                )

    def _negotiate(
        self, backend: Any, *, imported: bool
    ) -> tuple[AnalysisProviderCapability, Any]:
        capability_method = getattr(backend, "capabilities", None)
        if not callable(capability_method):
            capability_method = getattr(backend, "capability", None)
        raw = capability_method() if callable(capability_method) else None
        if inspect.isawaitable(raw):
            raw = asyncio.run(raw)
        if raw is None:
            operations = tuple(
                operation
                for operation in self.policy.operations
                if callable(backend)
                or callable(getattr(backend, "analyze", None))
                or any(
                    callable(getattr(backend, name, None))
                    for name in _OPERATION_METHODS[operation]
                )
            )
            health = (
                AnalysisProviderHealth.HEALTHY
                if operations
                else AnalysisProviderHealth.INCOMPATIBLE
            )
            version = str(getattr(backend, "__version__", "unknown"))
            return (
                AnalysisProviderCapability(
                    health=health,
                    operations=operations,
                    imported=imported,
                    reason_code=(
                        "capability_inferred"
                        if operations
                        else "no_supported_operations"
                    ),
                    provider_version=version,
                    bounds=self.policy.bounds,
                ),
                backend,
            )
        if not isinstance(raw, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability must be an object"
            )
        encoded = _json_bytes(raw, name="backend capability")
        if len(encoded) > self.policy.bounds.max_response_bytes:
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability exceeds max_response_bytes"
            )
        protocol_versions = raw.get(
            "protocol_versions", (raw.get("protocol_version", 1),)
        )
        if isinstance(protocol_versions, (str, bytes)) or not isinstance(
            protocol_versions, Sequence
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "backend protocol_versions must be a sequence"
            )
        compatible = self.protocol_version in {
            int(item) for item in protocol_versions
            if not isinstance(item, bool) and str(item).isdigit()
        }
        operations_raw = raw.get("operations") or ()
        if isinstance(operations_raw, (str, bytes)) or not isinstance(
            operations_raw, Sequence
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "backend operations must be a sequence"
            )
        operations: list[AnalysisProviderOperation] = []
        for item in operations_raw:
            try:
                operations.append(_operation(item))
            except IpfsDatasetsAnalysisProviderError:
                continue
        available = raw.get("available", raw.get("healthy", True))
        if not isinstance(available, bool):
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability available must be a boolean"
            )
        health = (
            AnalysisProviderHealth.INCOMPATIBLE
            if not compatible
            else (
                AnalysisProviderHealth.HEALTHY
                if available
                else AnalysisProviderHealth.DEGRADED
            )
        )
        capability = AnalysisProviderCapability(
            health=health,
            operations=tuple(
                operation
                for operation in operations
                if operation in self.policy.operations
            ),
            imported=imported,
            reason_code=(
                "capability_negotiated"
                if health is AnalysisProviderHealth.HEALTHY
                else (
                    "protocol_incompatible"
                    if not compatible
                    else "backend_unhealthy"
                )
            ),
            provider_version=str(raw.get("provider_version") or "unknown"),
            bounds=self.policy.bounds,
        )
        return capability, backend

    def _dispatcher(self, backend: Any, operation: AnalysisProviderOperation) -> Any:
        for name in _OPERATION_METHODS[operation]:
            method = getattr(backend, name, None)
            if callable(method):
                return method
        method = getattr(backend, "analyze", None)
        if callable(method):
            return method
        return backend if callable(backend) else None

    def _normalize_response(
        self,
        request: AnalysisProviderRequest,
        response: Any,
        capability: AnalysisProviderCapability,
    ) -> AnalysisProviderResult:
        if isinstance(response, AnalysisProviderResult):
            if (
                response.request_id != request.request_id
                or response.repository_id != request.repository_id
                or response.tree_id != request.tree_id
                or response.objective_revision != request.objective_revision
                or response.operation is not request.operation
            ):
                return self._degraded(
                    request,
                    AnalysisProviderStatus.MALFORMED,
                    "response_identity_mismatch",
                    import_attempted=capability.imported,
                    health=AnalysisProviderHealth.DEGRADED,
                    provider_version=capability.provider_version,
                )
            # Treat a typed object as untrusted backend output too.  Re-enter
            # the bounded mapping projection so custom backends cannot bypass
            # response-size, reference-count, or forbidden-field checks merely
            # by constructing our public result type.
            response = response.to_dict()
        if isinstance(response, Sequence) and not isinstance(
            response, (str, bytes, bytearray)
        ):
            response = {"status": "completed", "results": list(response)}
        if not isinstance(response, Mapping):
            return self._degraded(
                request,
                AnalysisProviderStatus.MALFORMED,
                "response_not_object",
                import_attempted=capability.imported,
                health=AnalysisProviderHealth.DEGRADED,
                provider_version=capability.provider_version,
            )
        try:
            if len(_json_bytes(response, name="backend response")) > request.bounds.max_response_bytes:
                raise IpfsDatasetsAnalysisProviderError(
                    "response exceeds max_response_bytes"
                )
            forbidden = _find_forbidden_fields(response)
            if forbidden:
                raise IpfsDatasetsAnalysisProviderError(
                    "response contains forbidden heavy fields"
                )
            expected_identity = {
                "request_id": request.request_id,
                "repository_id": request.repository_id,
                "tree_id": request.tree_id,
                "objective_revision": request.objective_revision,
                "operation": request.operation.value,
            }
            for name, expected in expected_identity.items():
                claimed = response.get(name)
                if claimed not in (None, "", expected):
                    raise IpfsDatasetsAnalysisProviderError(
                        f"backend response {name} does not match request"
                    )
            response_status = _status(response.get("status", "completed"))
            if response_status is not AnalysisProviderStatus.COMPLETED:
                return self._degraded(
                    request,
                    response_status,
                    "backend_" + response_status.value,
                    import_attempted=capability.imported,
                    health=AnalysisProviderHealth.DEGRADED,
                    provider_version=capability.provider_version,
                )
            raw_references = (
                response.get("evidence_references")
                or response.get("references")
                or response.get("results")
                or ()
            )
            raw_provenance = response.get("provenance_references") or response.get(
                "provenance"
            ) or ()
            references, truncated_refs = _compact_references(
                raw_references, request.bounds
            )
            provenance, truncated_provenance = _compact_references(
                raw_provenance, request.bounds
            )
            truncated = bool(response.get("truncated", False)) or truncated_refs or truncated_provenance
            resource_use = _resource_use(
                response.get("resource_use", response.get("cost", {}))
            )
        except (TypeError, ValueError, IpfsDatasetsAnalysisProviderError):
            return self._degraded(
                request,
                AnalysisProviderStatus.MALFORMED,
                "malformed_backend_response",
                import_attempted=capability.imported,
                health=AnalysisProviderHealth.DEGRADED,
                provider_version=capability.provider_version,
            )
        return AnalysisProviderResult(
            request_id=request.request_id,
            operation=request.operation,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_revision=request.objective_revision,
            status=AnalysisProviderStatus.COMPLETED,
            reason_code="bounded_provider_result",
            evidence_references=references,
            provenance_references=provenance,
            truncated=truncated,
            backend_health=AnalysisProviderHealth.HEALTHY,
            provider_version=capability.provider_version,
            resource_use=resource_use,
        )

    def _execute(self, request: AnalysisProviderRequest, cancellation_token: Any) -> AnalysisProviderResult:
        if _cancelled(cancellation_token):
            return self._degraded(
                request,
                AnalysisProviderStatus.CANCELLED,
                "cancelled_before_import",
                import_attempted=False,
                health=AnalysisProviderHealth.DEGRADED,
            )
        if not self.policy.enabled:
            return self._degraded(
                request,
                AnalysisProviderStatus.DISABLED,
                "provider_disabled",
                import_attempted=False,
                health=AnalysisProviderHealth.DEGRADED,
            )
        if request.operation not in self.policy.operations:
            return self._degraded(
                request,
                AnalysisProviderStatus.UNSUPPORTED,
                "operation_not_allowlisted",
                import_attempted=False,
                health=AnalysisProviderHealth.INCOMPATIBLE,
            )
        try:
            backend, imported = self._load_backend()
        except (ImportError, ModuleNotFoundError):
            return self._degraded(
                request,
                AnalysisProviderStatus.UNAVAILABLE,
                "optional_module_unavailable",
                import_attempted=True,
                health=AnalysisProviderHealth.UNAVAILABLE,
            )
        except Exception:
            return self._degraded(
                request,
                AnalysisProviderStatus.FAILED,
                "optional_import_failed",
                import_attempted=True,
                health=AnalysisProviderHealth.DEGRADED,
            )
        try:
            capability, backend = self._negotiate(backend, imported=imported)
        except (ImportError, ModuleNotFoundError):
            return self._degraded(
                request,
                AnalysisProviderStatus.UNAVAILABLE,
                "optional_capability_unavailable",
                import_attempted=imported,
                health=AnalysisProviderHealth.UNAVAILABLE,
            )
        except Exception:
            return self._degraded(
                request,
                AnalysisProviderStatus.MALFORMED,
                "malformed_capability",
                import_attempted=imported,
                health=AnalysisProviderHealth.INCOMPATIBLE,
            )
        if capability.health is not AnalysisProviderHealth.HEALTHY:
            status = (
                AnalysisProviderStatus.UNAVAILABLE
                if capability.health is AnalysisProviderHealth.UNAVAILABLE
                else AnalysisProviderStatus.UNSUPPORTED
            )
            return self._degraded(
                request,
                status,
                capability.reason_code,
                import_attempted=imported,
                health=capability.health,
                provider_version=capability.provider_version,
            )
        if request.operation not in capability.operations:
            return self._degraded(
                request,
                AnalysisProviderStatus.UNSUPPORTED,
                "operation_not_supported",
                import_attempted=imported,
                health=AnalysisProviderHealth.INCOMPATIBLE,
                provider_version=capability.provider_version,
            )
        if _cancelled(cancellation_token):
            return self._degraded(
                request,
                AnalysisProviderStatus.CANCELLED,
                "cancelled_before_dispatch",
                import_attempted=imported,
                health=AnalysisProviderHealth.DEGRADED,
                provider_version=capability.provider_version,
            )
        dispatcher = self._dispatcher(backend, request.operation)
        if dispatcher is None:
            return self._degraded(
                request,
                AnalysisProviderStatus.UNSUPPORTED,
                "operation_dispatch_unavailable",
                import_attempted=imported,
                health=AnalysisProviderHealth.INCOMPATIBLE,
                provider_version=capability.provider_version,
            )
        try:
            response = dispatcher(request.to_dict())
            if inspect.isawaitable(response):
                response = asyncio.run(response)
        except (ImportError, ModuleNotFoundError):
            return self._degraded(
                request,
                AnalysisProviderStatus.UNAVAILABLE,
                "optional_dispatch_dependency_unavailable",
                import_attempted=imported,
                health=AnalysisProviderHealth.UNAVAILABLE,
                provider_version=capability.provider_version,
            )
        except Exception:
            return self._degraded(
                request,
                AnalysisProviderStatus.FAILED,
                "backend_execution_failed",
                import_attempted=imported,
                health=AnalysisProviderHealth.DEGRADED,
                provider_version=capability.provider_version,
            )
        if _cancelled(cancellation_token):
            return self._degraded(
                request,
                AnalysisProviderStatus.CANCELLED,
                "cancelled_after_dispatch",
                import_attempted=imported,
                health=AnalysisProviderHealth.DEGRADED,
                provider_version=capability.provider_version,
            )
        return self._normalize_response(request, response, capability)

    def build_request(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any] | Any | None = None,
        **request_fields: Any,
    ) -> AnalysisProviderRequest:
        """Normalize pipeline-compatible input within the provider policy.

        This is the public request-construction boundary for callers that use
        retrieval-style limits.  It performs no backend import or capability
        probe and rejects any request that would expand the configured policy.
        """

        if request is None:
            if "limits" in request_fields:
                request_fields["bounds"] = self._bounds_from_limits(
                    request_fields.pop("limits")
                )
            elif "bounds" not in request_fields:
                request_fields["bounds"] = self.policy.bounds
            request = request_fields
        elif (
            not isinstance(request, (AnalysisProviderRequest, Mapping))
            or (
                isinstance(request, Mapping)
                and request_fields
                and not {
                    "operation",
                    "repository_id",
                    "tree_id",
                    "objective_revision",
                    "query",
                }.intersection(request)
            )
        ):
            # Pipeline compatibility: ``analyze(query, operation=..., ...)``.
            query = request
            limits = request_fields.pop("limits", None)
            bounds = request_fields.pop(
                "bounds", self._bounds_from_limits(limits)
            )
            request = {
                **request_fields,
                "query": query,
                "bounds": bounds,
            }
        elif request_fields:
            raise IpfsDatasetsAnalysisProviderError(
                "request fields cannot accompany an explicit request"
            )
        normalized = AnalysisProviderRequest.from_value(request)
        self._validate_policy_bounds(normalized.bounds)
        return normalized

    def analyze(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any] | Any | None = None,
        *,
        cancellation_token: Any = None,
        **request_fields: Any,
    ) -> AnalysisProviderResult:
        """Run one bounded request, returning typed degradation on failure."""

        normalized = self.build_request(request, **request_fields)
        if not self.policy.enabled or normalized.operation not in self.policy.operations:
            return self._execute(normalized, cancellation_token)
        if not self._dispatch_slots.acquire(blocking=False):
            return self._degraded(
                normalized,
                AnalysisProviderStatus.TIMED_OUT,
                "provider_capacity_exhausted",
                import_attempted=False,
                health=AnalysisProviderHealth.DEGRADED,
            )

        output: "queue.Queue[AnalysisProviderResult]" = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                try:
                    result = self._execute(normalized, cancellation_token)
                except Exception:
                    result = self._degraded(
                        normalized,
                        AnalysisProviderStatus.FAILED,
                        "adapter_execution_failed",
                        import_attempted=self._backend is None,
                        health=AnalysisProviderHealth.DEGRADED,
                    )
                output.put(result)
            finally:
                self._dispatch_slots.release()

        thread = threading.Thread(
            target=run,
            name="ipfs-datasets-analysis-provider",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            self._dispatch_slots.release()
            return self._degraded(
                normalized,
                AnalysisProviderStatus.FAILED,
                "provider_thread_start_failed",
                import_attempted=False,
                health=AnalysisProviderHealth.DEGRADED,
            )
        try:
            return output.get(timeout=normalized.bounds.timeout_ms / 1000)
        except queue.Empty:
            return self._degraded(
                normalized,
                AnalysisProviderStatus.TIMED_OUT,
                "provider_timeout",
                import_attempted=self._backend is None,
                health=AnalysisProviderHealth.DEGRADED,
            )

    async def analyze_async(
        self,
        request: AnalysisProviderRequest | Mapping[str, Any] | Any | None = None,
        *,
        cancellation_token: Any = None,
        **request_fields: Any,
    ) -> AnalysisProviderResult:
        """Async facade that keeps imports and synchronous backends off-loop."""

        return await asyncio.to_thread(
            self.analyze,
            request,
            cancellation_token=cancellation_token,
            **request_fields,
        )

    dispatch = analyze
    run = analyze


def _find_forbidden_fields(value: Any) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(item: Any, depth: int = 0) -> None:
        if depth > 8:
            raise IpfsDatasetsAnalysisProviderError(
                "backend response exceeds maximum depth"
            )
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).casefold() in _FORBIDDEN_FIELDS:
                    found.add(str(key))
                visit(nested, depth + 1)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for nested in item:
                visit(nested, depth + 1)

    visit(value)
    return tuple(sorted(found))


def _compact_references(
    value: Any, bounds: AnalysisProviderBounds
) -> tuple[tuple[Mapping[str, Any], ...], bool]:
    if value is None:
        source: Sequence[Any] = ()
    elif isinstance(value, Mapping):
        source = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        source = value
    else:
        raise IpfsDatasetsAnalysisProviderError(
            "backend references must be a sequence"
        )
    truncated = len(source) > bounds.max_results
    result: list[Mapping[str, Any]] = []
    for raw in source:
        if not isinstance(raw, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "backend evidence reference must be an object"
            )
        if set(raw).intersection(_FORBIDDEN_FIELDS) or set(raw) - _REFERENCE_FIELDS:
            raise IpfsDatasetsAnalysisProviderError(
                "backend evidence reference contains unsupported fields"
            )
        item: dict[str, Any] = {}
        for key, nested in sorted(raw.items()):
            if nested in (None, ""):
                continue
            if key in {"score", "score_millionths"}:
                try:
                    score = float(nested)
                except (TypeError, ValueError) as exc:
                    raise IpfsDatasetsAnalysisProviderError(
                        "reference score must be numeric"
                    ) from exc
                if not math.isfinite(score):
                    raise IpfsDatasetsAnalysisProviderError(
                        "reference score must be finite"
                    )
                item["score_millionths"] = (
                    int(round(score * 1_000_000))
                    if key == "score"
                    else int(score)
                )
                if not 0 <= item["score_millionths"] <= 1_000_000:
                    raise IpfsDatasetsAnalysisProviderError(
                        "reference score is out of range"
                    )
            else:
                item[key] = _text(
                    nested,
                    f"reference {key}",
                    required=False,
                    max_bytes=2048,
                )
        if not item:
            raise IpfsDatasetsAnalysisProviderError(
                "backend evidence reference is empty"
            )
        if len(_json_bytes(item, name="backend evidence reference")) > bounds.max_reference_bytes:
            raise IpfsDatasetsAnalysisProviderError(
                "backend evidence reference exceeds max_reference_bytes"
            )
        result.append(item)
    # Canonical order and identity deduplication make backend scheduling order
    # irrelevant to supervisor state.
    unique = {
        _json_bytes(item, name="backend evidence reference"): item for item in result
    }
    ordered = tuple(unique[key] for key in sorted(unique))
    return (
        ordered[: bounds.max_results],
        truncated
        or len(ordered) < len(result)
        or len(ordered) > bounds.max_results,
    )


# Public compatibility aliases.  The project historically uses both IPFS and
# Ipfs class spellings in adapters.
IPFSDatasetsAnalysisProvider = IpfsDatasetsAnalysisProvider
IPFSDatasetsAnalysisProviderPolicy = AnalysisProviderPolicy
IpfsDatasetsAnalysisProviderConfig = AnalysisProviderPolicy
IPFSDatasetsAnalysisProviderConfig = AnalysisProviderPolicy
IPFSDatasetsAnalysisRequest = AnalysisProviderRequest
IPFSDatasetsAnalysisResult = AnalysisProviderResult
IPFSDatasetsAnalysisCapability = AnalysisProviderCapability
IPFSDatasetsProviderDegradationEvidence = IpfsDatasetsProviderDegradationEvidence
ProviderOperation = AnalysisProviderOperation
ProviderStatus = AnalysisProviderStatus
ProviderHealth = AnalysisProviderHealth
ProviderBounds = AnalysisProviderBounds
ProviderPolicy = AnalysisProviderPolicy
ProviderConfig = AnalysisProviderPolicy
ProviderRequest = AnalysisProviderRequest
ProviderResult = AnalysisProviderResult
ProviderCapability = AnalysisProviderCapability


def create_ipfs_datasets_analysis_provider(
    policy: AnalysisProviderPolicy | None = None,
    **kwargs: Any,
) -> IpfsDatasetsAnalysisProvider:
    """Construct the lazy provider without importing the optional backend."""

    return IpfsDatasetsAnalysisProvider(policy, **kwargs)


build_ipfs_datasets_analysis_provider = create_ipfs_datasets_analysis_provider


__all__ = [
    "IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION",
    "IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION",
    "IPFS_DATASETS_ANALYSIS_PROVIDER_ID",
    "IPFS_DATASETS_OFFLOAD_COORDINATION_BOUNDARY",
    "IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID",
    "OPTIONAL_DATASETS_DEGRADATION_REQUIREMENT_ID",
    "PROVIDER_CAPABILITY_SCHEMA",
    "PROVIDER_REQUEST_SCHEMA",
    "PROVIDER_RESULT_SCHEMA",
    "PROVIDER_DEGRADATION_EVIDENCE_SCHEMA",
    "normalize_analysis_provider_operation",
    "AnalysisProviderOperation",
    "AnalysisProviderStatus",
    "AnalysisProviderHealth",
    "AnalysisProviderBounds",
    "AnalysisProviderPolicy",
    "AnalysisProviderRequest",
    "AnalysisProviderCapability",
    "IpfsDatasetsProviderDegradationEvidence",
    "AnalysisProviderResult",
    "IpfsDatasetsAnalysisProviderError",
    "IpfsDatasetsAnalysisProvider",
    "IPFSDatasetsAnalysisProvider",
    "IPFSDatasetsAnalysisProviderPolicy",
    "IpfsDatasetsAnalysisProviderConfig",
    "IPFSDatasetsAnalysisProviderConfig",
    "IPFSDatasetsAnalysisRequest",
    "IPFSDatasetsAnalysisResult",
    "IPFSDatasetsAnalysisCapability",
    "IPFSDatasetsProviderDegradationEvidence",
    "ProviderOperation",
    "ProviderStatus",
    "ProviderHealth",
    "ProviderBounds",
    "ProviderPolicy",
    "ProviderConfig",
    "ProviderRequest",
    "ProviderResult",
    "ProviderCapability",
    "create_ipfs_datasets_analysis_provider",
    "build_ipfs_datasets_analysis_provider",
]
