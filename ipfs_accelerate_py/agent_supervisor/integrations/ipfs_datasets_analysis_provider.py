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
import importlib.metadata
import inspect
import json
import math
import queue
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..analysis.analysis_operation_registry import (
    IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
    LOCAL_ANALYSIS_PRODUCER_ID,
    AnalysisOperation,
    AnalysisProducer,
    normalize_analysis_operation,
    normalized_reference_payload,
)
from ..analysis.analysis_transport import (
    ANALYSIS_TRANSPORT_PROTOCOL_VERSION,
    ANALYSIS_TRANSPORT_RESULT_SCHEMA,
    AnalysisCapability as TransportAnalysisCapability,
    AnalysisRequest as TransportAnalysisRequest,
)
from ..proof.formal_verification_contracts import content_identity


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
IPFS_DATASETS_COMPLETION_ACCEPTANCE_CRITERION: Final = (
    "optional datasets capabilities degrade explicitly"
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
DEFAULT_MAX_BATCH_REQUESTS: Final = 16
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
    SYMBOL_IMPACT = "symbol_impact"
    AST_SYMBOL_IMPACT = "symbol_impact"
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
        "schema_incompatible": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        "backend_unhealthy": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.DEGRADED,
        ),
        "backend_unavailable": (
            AnalysisProviderStatus.UNAVAILABLE,
            frozenset({False, True}),
            AnalysisProviderHealth.UNAVAILABLE,
        ),
        "backend_incompatible": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
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
        "request_bounds_unsupported": (
            AnalysisProviderStatus.UNSUPPORTED,
            frozenset({False, True}),
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        "cancellation_not_supported": (
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
        "ast_impact": AnalysisProviderOperation.SYMBOL_IMPACT,
        "ast_symbol_impact": AnalysisProviderOperation.SYMBOL_IMPACT,
        "symbol": AnalysisProviderOperation.SYMBOL_IMPACT,
        "graphrag": AnalysisProviderOperation.GRAPH_RETRIEVAL,
        "graphrag_retrieval": AnalysisProviderOperation.GRAPH_RETRIEVAL,
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
    max_batch_requests: int = DEFAULT_MAX_BATCH_REQUESTS
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_reference_bytes: int = DEFAULT_MAX_REFERENCE_BYTES
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def __post_init__(self) -> None:
        limits = {
            "max_results": 1000,
            "max_batch_requests": 256,
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
            "max_batch_requests": self.max_batch_requests,
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
        if _find_forbidden_fields(query):
            raise IpfsDatasetsAnalysisProviderError(
                "query contains forbidden heavy fields"
            )
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
    request_schema: str = PROVIDER_REQUEST_SCHEMA
    result_schema: str = PROVIDER_RESULT_SCHEMA
    cancellation_supported: bool = True

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
        for name, expected in (
            ("request_schema", PROVIDER_REQUEST_SCHEMA),
            ("result_schema", PROVIDER_RESULT_SCHEMA),
        ):
            normalized = _text(
                getattr(self, name), name, required=True, max_bytes=256
            )
            if normalized != expected:
                raise IpfsDatasetsAnalysisProviderError(
                    f"unsupported negotiated {name}"
                )
            object.__setattr__(self, name, normalized)
        if not isinstance(self.cancellation_supported, bool):
            raise IpfsDatasetsAnalysisProviderError(
                "cancellation_supported must be a boolean"
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
            "request_schema": self.request_schema,
            "result_schema": self.result_schema,
            "cancellation_supported": self.cancellation_supported,
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
            "request_schema",
            "result_schema",
            "cancellation_supported",
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
            request_schema=value.get("request_schema", PROVIDER_REQUEST_SCHEMA),
            result_schema=value.get("result_schema", PROVIDER_RESULT_SCHEMA),
            cancellation_supported=value.get("cancellation_supported", True),
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


def inspect_analysis_provider_capability(
    policy: AnalysisProviderPolicy | Mapping[str, Any] | None = None,
) -> AnalysisProviderCapability:
    """Return the deterministic metadata-only capability declaration.

    The inspection boundary accepts policy data, not a provider, importer, or
    backend.  It therefore cannot import ``ipfs_datasets_py``, negotiate with
    optional code, consume a dispatch slot, or mutate provider runtime state.
    Runtime health is intentionally reported as ``lazy`` until a bounded,
    policy-allowed request enters dispatch.
    """

    selected = AnalysisProviderPolicy.from_value(policy)
    return AnalysisProviderCapability(
        health=(
            AnalysisProviderHealth.LAZY
            if selected.enabled
            else AnalysisProviderHealth.DEGRADED
        ),
        operations=selected.operations,
        imported=False,
        reason_code=(
            "lazy_not_probed" if selected.enabled else "provider_disabled"
        ),
        provider_version="unknown",
        bounds=selected.bounds,
    )


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
    AnalysisProviderOperation.SYMBOL_IMPACT: (
        "analyze_symbol_impact",
        "symbol_impact",
        "analyze_ast_impact",
    ),
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

        return inspect_analysis_provider_capability(self.policy)

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

    def _negotiate_bounds(self, value: Any) -> AnalysisProviderBounds:
        """Intersect backend-advertised maxima with supervisor policy."""

        if value in (None, ""):
            return self.policy.bounds
        if not isinstance(value, Mapping):
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability bounds must be an object"
            )
        unknown = set(value) - set(AnalysisProviderBounds.__dataclass_fields__)
        if unknown:
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability contains unknown bounds: "
                + ", ".join(sorted(unknown))
            )
        maxima = {
            "max_results": 1000,
            "max_batch_requests": 256,
            "max_query_bytes": 1024 * 1024,
            "max_request_bytes": 4 * 1024 * 1024,
            "max_response_bytes": 16 * 1024 * 1024,
            "max_reference_bytes": 256 * 1024,
            "timeout_ms": 10 * 60 * 1000,
        }
        negotiated = self.policy.bounds.to_dict()
        for name, raw in value.items():
            advertised = _positive_int(raw, name, maximum=maxima[name])
            negotiated[name] = min(negotiated[name], advertised)
        negotiated["max_query_bytes"] = min(
            negotiated["max_query_bytes"],
            negotiated["max_request_bytes"],
        )
        negotiated["max_reference_bytes"] = min(
            negotiated["max_reference_bytes"],
            negotiated["max_response_bytes"],
        )
        return AnalysisProviderBounds(**negotiated)

    @staticmethod
    def _advertised_schemas(
        raw: Mapping[str, Any],
        *,
        singular: str,
        plural: str,
        expected: str,
    ) -> tuple[str, ...]:
        """Normalize an optional singular/plural schema advertisement."""

        value = raw.get(plural, raw.get(singular))
        schemas = raw.get("schemas")
        if value is None and isinstance(schemas, Mapping):
            key = "request" if singular == "request_schema" else "result"
            value = schemas.get(key)
        if value is None:
            # Protocol v1 made schemas implicit. An omitted advertisement is
            # therefore the legacy schema, while an explicit mismatch fails.
            return (expected,)
        if isinstance(value, str):
            source: Sequence[Any] = (value,)
        elif isinstance(value, Sequence) and not isinstance(
            value, (bytes, bytearray)
        ):
            source = value
        else:
            raise IpfsDatasetsAnalysisProviderError(
                f"backend {plural} must be a string or sequence"
            )
        return tuple(
            sorted(
                {
                    _text(
                        item,
                        f"backend {plural}",
                        required=True,
                        max_bytes=256,
                    )
                    for item in source
                }
            )
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
        converter = getattr(raw, "to_dict", None)
        if raw is not None and not isinstance(raw, Mapping) and callable(converter):
            raw = converter()
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
                    request_schema=PROVIDER_REQUEST_SCHEMA,
                    result_schema=PROVIDER_RESULT_SCHEMA,
                    cancellation_supported=True,
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
        protocol_compatible = self.protocol_version in {
            int(item) for item in protocol_versions
            if not isinstance(item, bool) and str(item).isdigit()
        }
        request_schemas = self._advertised_schemas(
            raw,
            singular="request_schema",
            plural="request_schemas",
            expected=PROVIDER_REQUEST_SCHEMA,
        )
        result_advertisement = raw
        if (
            "result_schema" not in raw
            and "result_schemas" not in raw
            and (
                "response_schema" in raw
                or "response_schemas" in raw
            )
        ):
            result_advertisement = dict(raw)
            if "response_schemas" in raw:
                result_advertisement["result_schemas"] = raw[
                    "response_schemas"
                ]
            else:
                result_advertisement["result_schema"] = raw[
                    "response_schema"
                ]
        result_schemas = self._advertised_schemas(
            result_advertisement,
            singular="result_schema",
            plural="result_schemas",
            expected=PROVIDER_RESULT_SCHEMA,
        )
        schema_compatible = (
            PROVIDER_REQUEST_SCHEMA in request_schemas
            and PROVIDER_RESULT_SCHEMA in result_schemas
        )
        negotiated_bounds = self._negotiate_bounds(
            raw.get("bounds", raw.get("limits"))
        )
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
        cancellation_supported = raw.get(
            "cancellation_supported",
            raw.get("supports_cancellation", True),
        )
        if not isinstance(cancellation_supported, bool):
            raise IpfsDatasetsAnalysisProviderError(
                "backend cancellation support must be a boolean"
            )
        health_fields: Mapping[str, Any] = raw
        if not {"health", "available", "healthy"}.intersection(raw):
            health_method = getattr(backend, "health", None)
            if not callable(health_method):
                health_method = getattr(backend, "health_check", None)
            if callable(health_method):
                probed_health = health_method()
                if inspect.isawaitable(probed_health):
                    probed_health = asyncio.run(probed_health)
                if isinstance(probed_health, Mapping):
                    if set(probed_health) - {"health", "available", "healthy"}:
                        raise IpfsDatasetsAnalysisProviderError(
                            "backend health probe contains unknown fields"
                        )
                    health_fields = {**raw, **probed_health}
                elif isinstance(probed_health, bool):
                    health_fields = {**raw, "available": probed_health}
                elif isinstance(probed_health, (str, AnalysisProviderHealth)):
                    health_fields = {**raw, "health": probed_health}
                else:
                    raise IpfsDatasetsAnalysisProviderError(
                        "backend health probe returned an unsupported value"
                    )
                if (
                    len(_json_bytes(health_fields, name="backend health"))
                    > self.policy.bounds.max_response_bytes
                ):
                    raise IpfsDatasetsAnalysisProviderError(
                        "backend health probe exceeds max_response_bytes"
                    )
        explicit_health = health_fields.get("health")
        if explicit_health is not None:
            normalized_health = str(
                getattr(explicit_health, "value", explicit_health)
            ).strip().casefold()
            health_aliases = {
                "ok": AnalysisProviderHealth.HEALTHY,
                "healthy": AnalysisProviderHealth.HEALTHY,
                "degraded": AnalysisProviderHealth.DEGRADED,
                "unavailable": AnalysisProviderHealth.UNAVAILABLE,
                "incompatible": AnalysisProviderHealth.INCOMPATIBLE,
            }
            if normalized_health not in health_aliases:
                raise IpfsDatasetsAnalysisProviderError(
                    "backend capability health is unsupported"
                )
            backend_health = health_aliases[normalized_health]
        else:
            backend_health = None
        available = health_fields.get(
            "available", health_fields.get("healthy")
        )
        if available is None:
            available = (
                backend_health is AnalysisProviderHealth.HEALTHY
                if backend_health is not None
                else True
            )
        if not isinstance(available, bool):
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability available must be a boolean"
            )
        if backend_health is None:
            backend_health = (
                AnalysisProviderHealth.HEALTHY
                if available
                else AnalysisProviderHealth.DEGRADED
            )
        if available != (backend_health is AnalysisProviderHealth.HEALTHY):
            raise IpfsDatasetsAnalysisProviderError(
                "backend capability health and availability disagree"
            )
        negotiated_operations = tuple(
            operation
            for operation in operations
            if operation in self.policy.operations
        )
        if not protocol_compatible or not schema_compatible:
            health = AnalysisProviderHealth.INCOMPATIBLE
        elif not negotiated_operations:
            health = AnalysisProviderHealth.INCOMPATIBLE
        else:
            health = backend_health
        if not protocol_compatible:
            reason_code = "protocol_incompatible"
        elif not schema_compatible:
            reason_code = "schema_incompatible"
        elif not negotiated_operations:
            reason_code = "no_supported_operations"
        elif health is AnalysisProviderHealth.HEALTHY:
            reason_code = "capability_negotiated"
        elif health is AnalysisProviderHealth.UNAVAILABLE:
            reason_code = "backend_unavailable"
        elif health is AnalysisProviderHealth.INCOMPATIBLE:
            reason_code = "backend_incompatible"
        else:
            reason_code = "backend_unhealthy"
        capability = AnalysisProviderCapability(
            health=health,
            operations=negotiated_operations,
            imported=imported,
            reason_code=reason_code,
            provider_version=str(
                raw.get("provider_version") or raw.get("version") or "unknown"
            ),
            bounds=negotiated_bounds,
            request_schema=PROVIDER_REQUEST_SCHEMA,
            result_schema=PROVIDER_RESULT_SCHEMA,
            cancellation_supported=cancellation_supported,
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
        if any(
            getattr(request.bounds, name) > getattr(capability.bounds, name)
            for name in AnalysisProviderBounds.__dataclass_fields__
            if (
                name != "max_batch_requests"
                or request.operation is AnalysisProviderOperation.BATCH_ANALYSIS
            )
        ):
            return self._degraded(
                request,
                AnalysisProviderStatus.UNSUPPORTED,
                "request_bounds_unsupported",
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
        if (
            cancellation_token is not None
            and not capability.cancellation_supported
        ):
            return self._degraded(
                request,
                AnalysisProviderStatus.UNSUPPORTED,
                "cancellation_not_supported",
                import_attempted=imported,
                health=AnalysisProviderHealth.INCOMPATIBLE,
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

    def analyze_batch(
        self,
        requests: Sequence[AnalysisProviderRequest | Mapping[str, Any]],
        *,
        cancellation_token: Any = None,
    ) -> AnalysisProviderResult:
        """Dispatch one compact batch of requests bound to the same tree."""

        if isinstance(requests, (str, bytes, bytearray)) or not isinstance(
            requests, Sequence
        ):
            raise IpfsDatasetsAnalysisProviderError(
                "batch requests must be a sequence"
            )
        if not requests:
            raise IpfsDatasetsAnalysisProviderError(
                "batch requests must not be empty"
            )
        if len(requests) > self.policy.bounds.max_batch_requests:
            raise IpfsDatasetsAnalysisProviderError(
                "batch requests exceeds max_batch_requests"
            )
        normalized = tuple(self.build_request(item) for item in requests)
        first = normalized[0]
        relation = (
            first.repository_id,
            first.tree_id,
            first.objective_revision,
        )
        for item in normalized:
            if (
                item.repository_id,
                item.tree_id,
                item.objective_revision,
            ) != relation:
                raise IpfsDatasetsAnalysisProviderError(
                    "batch requests must share repository, tree, and objective identities"
                )
            if item.operation is AnalysisProviderOperation.BATCH_ANALYSIS:
                raise IpfsDatasetsAnalysisProviderError(
                    "nested batch requests are not supported"
                )
            if item.operation not in self.policy.operations:
                raise IpfsDatasetsAnalysisProviderError(
                    "batch child operation is not allowlisted: "
                    + item.operation.value
                )
        bound_values = {
            name: min(getattr(item.bounds, name) for item in normalized)
            for name in AnalysisProviderBounds.__dataclass_fields__
        }
        bound_values["max_batch_requests"] = min(
            bound_values["max_batch_requests"],
            len(normalized),
        )
        child_requests = [
            {
                "request_id": item.request_id,
                "operation": item.operation.value,
                "query": item.query,
                "artifact_references": [
                    dict(reference) for reference in item.artifact_references
                ],
                "payload": dict(item.payload),
                "bounds": item.bounds.to_dict(),
            }
            for item in normalized
        ]
        batch_request = self.build_request(
            operation=AnalysisProviderOperation.BATCH_ANALYSIS,
            repository_id=first.repository_id,
            tree_id=first.tree_id,
            objective_revision=first.objective_revision,
            query={"requests": child_requests},
            artifact_references=(),
            payload={"batch_size": len(child_requests)},
            bounds=AnalysisProviderBounds(**bound_values),
        )
        return self.analyze(
            batch_request,
            cancellation_token=cancellation_token,
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

    async def analyze_batch_async(
        self,
        requests: Sequence[AnalysisProviderRequest | Mapping[str, Any]],
        *,
        cancellation_token: Any = None,
    ) -> AnalysisProviderResult:
        """Async facade for one related-request provider batch."""

        return await asyncio.to_thread(
            self.analyze_batch,
            requests,
            cancellation_token=cancellation_token,
        )

    dispatch = analyze
    run = analyze
    batch_related_requests = analyze_batch


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


# ---------------------------------------------------------------------------
# ASI-098 operation-registry adapters
# ---------------------------------------------------------------------------

_REGISTRY_ANALYSIS_OPERATIONS: Final = (
    AnalysisOperation.SYMBOL_IMPACT,
    AnalysisOperation.GRAPH_RAG_RETRIEVAL,
)
_REGISTRY_ANALYSIS_CAPABILITIES: Final = (
    "ast_index_read",
    "graph_read",
    "graphrag_retrieval",
    "symbol_impact",
)
_REGISTRY_TOKEN_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]*")
_REGISTRY_MAX_REFERENCES: Final = 64


def registry_analysis_producer_declarations(
) -> tuple[AnalysisProducer, AnalysisProducer]:
    """Declare local and optional AST/GraphRAG producers without activation.

    This function is metadata-only.  In particular it neither imports
    :mod:`ipfs_datasets_py` nor probes an injected backend.
    """

    common = {
        "operations": _REGISTRY_ANALYSIS_OPERATIONS,
        "provider_version": "1.0.0",
        "capabilities": _REGISTRY_ANALYSIS_CAPABILITIES,
        "max_batch_size": DEFAULT_MAX_BATCH_REQUESTS,
        "max_concurrency": MAX_CONCURRENT_PROVIDER_DISPATCHES,
        "supports_cancellation": True,
        "supports_progress": False,
        "supports_batching": True,
    }
    local = AnalysisProducer(
        producer_id=LOCAL_ANALYSIS_PRODUCER_ID,
        provider_kind="local",
        capability_revision="supervisor-local-analysis/ast-graphrag@1",
        **common,
    )
    optional = AnalysisProducer(
        producer_id=IPFS_DATASETS_ANALYSIS_PRODUCER_ID,
        provider_kind="ipfs_datasets_py",
        capability_revision="ipfs-datasets-analysis/ast-graphrag@1",
        **common,
    )
    return local, optional


def _registry_request(value: Any) -> TransportAnalysisRequest:
    request = TransportAnalysisRequest.from_value(value)
    operation = normalize_analysis_operation(request.operation)
    if operation not in _REGISTRY_ANALYSIS_OPERATIONS:
        raise IpfsDatasetsAnalysisProviderError(
            f"unsupported registry analysis operation: {operation.value}"
        )
    for name in (
        "repository_id",
        "tree_id",
        "objective_revision",
        "policy_id",
    ):
        if not str(request.metadata.get(name) or "").strip():
            raise IpfsDatasetsAnalysisProviderError(
                f"registry analysis request requires {name} provenance"
            )
    tree_id = request.metadata["tree_id"]
    if any(
        reference.get("tree_id")
        and reference.get("tree_id") != tree_id
        for reference in request.artifact_references
    ):
        raise IpfsDatasetsAnalysisProviderError(
            "registry analysis artifact tree_id does not match request tree_id"
        )
    return request


def _registry_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        match.casefold().strip("._:/-")
        for match in _REGISTRY_TOKEN_RE.findall(str(value or ""))
        if match.strip("._:/-")
    )


def _registry_reference(
    value: Mapping[str, Any],
    *,
    operation: AnalysisOperation,
    producer_id: str,
    tree_id: str,
    score_millionths: int | None = None,
    preserve_reference_id: bool = False,
) -> dict[str, Any]:
    """Project a source reference through the registry's one stable shape."""

    source = dict(value)
    source_tree_id = str(source.get("tree_id") or "").strip()
    if source_tree_id and source_tree_id != tree_id:
        raise IpfsDatasetsAnalysisProviderError(
            "analysis result tree_id does not match request tree_id"
        )
    candidate: dict[str, Any] = {}
    for name in (
        "artifact_content_id",
        "artifact_id",
        "byte_count",
        "chunk_id",
        "cid",
        "dataset_id",
        "digest",
        "evidence_id",
        "media_type",
        "model_id",
        "path",
        "record_id",
        "revision",
        "sha256",
        "symbol",
        "uri",
    ):
        if source.get(name) not in (None, ""):
            candidate[name] = source[name]
    source_reference_id = str(source.get("reference_id") or "").strip()
    if preserve_reference_id and source_reference_id:
        candidate["reference_id"] = source_reference_id
    if not candidate.get("evidence_id"):
        # Prefer identities the legacy and local paths both retain.  A
        # provider-specific reference identity is the final fallback only.
        source_identity = (
            source.get("record_id")
            or source.get("artifact_id")
            or source.get("dataset_id")
            or source.get("chunk_id")
            or source_reference_id
        )
        if source_identity not in (None, ""):
            candidate["evidence_id"] = source_identity
    candidate["kind"] = operation.value
    candidate["tree_id"] = str(source.get("tree_id") or tree_id)
    summary = str(source.get("summary") or "").strip()
    if not summary:
        subject = (
            source.get("symbol")
            or source.get("path")
            or source.get("record_id")
            or source_reference_id
            or "artifact"
        )
        summary = f"{operation.value} candidate for {subject}"
    candidate["summary"] = summary[:2048]
    if score_millionths is None:
        raw_score = source.get("score_millionths", source.get("score"))
        if raw_score not in (None, ""):
            candidate[
                "score_millionths" if "score_millionths" in source else "score"
            ] = raw_score
    if score_millionths is not None:
        candidate["score_millionths"] = max(
            0, min(1_000_000, int(score_millionths))
        )
    return normalized_reference_payload(
        candidate,
        default_kind=operation.value,
        producer_id=producer_id,
    )


def _registry_provenance_reference(
    request: TransportAnalysisRequest,
    *,
    producer_id: str,
) -> dict[str, Any]:
    metadata = request.metadata
    return normalized_reference_payload(
        {
            "kind": "analysis_request",
            "record_id": request.request_id,
            "tree_id": metadata.get("tree_id", ""),
            "revision": metadata.get("objective_revision", ""),
            "artifact_id": metadata.get("policy_id", ""),
            "summary": (
                f"{request.operation} request bound to "
                f"{metadata.get('tree_id', 'unknown tree')}"
            ),
        },
        default_kind="analysis_request",
        producer_id=producer_id,
    )


def _registry_transport_response(
    request: TransportAnalysisRequest,
    *,
    capability: TransportAnalysisCapability,
    evidence_references: Sequence[Mapping[str, Any]],
    provenance_references: Sequence[Mapping[str, Any]],
    negotiated_capability: Any = None,
    cost: Mapping[str, int] | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    negotiated = negotiated_capability
    return {
        "schema": getattr(
            negotiated, "result_schema", ANALYSIS_TRANSPORT_RESULT_SCHEMA
        ),
        "protocol_version": getattr(
            negotiated,
            "protocol_version",
            ANALYSIS_TRANSPORT_PROTOCOL_VERSION,
        ),
        "request_id": request.request_id,
        "operation": request.operation,
        "capability_id": getattr(
            negotiated, "capability_id", capability.capability_id
        ),
        "capability_revision": getattr(
            negotiated,
            "capability_revision",
            capability.capability_revision,
        ),
        "evidence_references": [dict(item) for item in evidence_references],
        "provenance_references": [
            dict(item) for item in provenance_references
        ],
        "cost": dict(cost or {}),
        "verdict": "diagnostic_candidate",
        "truncated": bool(truncated),
        "non_authoritative": True,
        "completion_authority": False,
        "safe_for_completion_reasoning": False,
    }


class LocalSymbolImpactAnalysisAdapter:
    """Deterministic symbol-impact projection over compact artifact references."""

    operation = AnalysisOperation.SYMBOL_IMPACT

    def project(
        self,
        request: TransportAnalysisRequest,
        *,
        producer_id: str,
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        question_tokens = _registry_tokens(request.question)
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        tree_id = str(request.metadata.get("tree_id") or "")
        for source in request.artifact_references:
            searchable = " ".join(
                str(source.get(name) or "")
                for name in ("symbol", "path", "summary", "kind", "record_id")
            )
            candidate_tokens = _registry_tokens(searchable)
            overlap = len(question_tokens.intersection(candidate_tokens))
            denominator = max(1, len(question_tokens))
            lexical = int(round(800_000 * overlap / denominator))
            symbol = str(source.get("symbol") or "").casefold()
            exact_symbol = bool(
                symbol
                and (
                    symbol in question_tokens
                    or any(token.endswith("." + symbol) for token in question_tokens)
                )
            )
            score = min(1_000_000, lexical + (200_000 if exact_symbol else 0))
            projected = _registry_reference(
                source,
                operation=self.operation,
                producer_id=producer_id,
                tree_id=tree_id,
                score_millionths=score,
            )
            stable = json.dumps(
                projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            ranked.append((-score, stable, projected))
        ranked.sort(key=lambda item: (item[0], item[1]))
        truncated = len(ranked) > _REGISTRY_MAX_REFERENCES
        return (
            tuple(item[2] for item in ranked[:_REGISTRY_MAX_REFERENCES]),
            truncated,
        )


class LocalGraphRAGRetrievalAdapter:
    """Deterministic lexical GraphRAG fallback over compact references."""

    operation = AnalysisOperation.GRAPH_RAG_RETRIEVAL

    def project(
        self,
        request: TransportAnalysisRequest,
        *,
        producer_id: str,
    ) -> tuple[tuple[dict[str, Any], ...], bool]:
        question_tokens = _registry_tokens(request.question)
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        tree_id = str(request.metadata.get("tree_id") or "")
        for source in request.artifact_references:
            searchable = " ".join(
                str(source.get(name) or "")
                for name in (
                    "summary",
                    "symbol",
                    "path",
                    "kind",
                    "record_id",
                    "artifact_id",
                    "dataset_id",
                )
            )
            candidate_tokens = _registry_tokens(searchable)
            overlap = len(question_tokens.intersection(candidate_tokens))
            union = len(question_tokens.union(candidate_tokens))
            score = int(round(1_000_000 * overlap / max(1, union)))
            projected = _registry_reference(
                source,
                operation=self.operation,
                producer_id=producer_id,
                tree_id=tree_id,
                score_millionths=score,
            )
            stable = json.dumps(
                projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            ranked.append((-score, stable, projected))
        ranked.sort(key=lambda item: (item[0], item[1]))
        truncated = len(ranked) > _REGISTRY_MAX_REFERENCES
        return (
            tuple(item[2] for item in ranked[:_REGISTRY_MAX_REFERENCES]),
            truncated,
        )


class LocalRegistryAnalysisProducer:
    """Transport-compatible read-only local producer for AST and GraphRAG."""

    def __init__(self, declaration: AnalysisProducer | None = None) -> None:
        self.declaration = declaration or registry_analysis_producer_declarations()[0]
        self._adapters = {
            AnalysisOperation.SYMBOL_IMPACT: LocalSymbolImpactAnalysisAdapter(),
            AnalysisOperation.GRAPH_RAG_RETRIEVAL: (
                LocalGraphRAGRetrievalAdapter()
            ),
        }

    def capabilities(self) -> TransportAnalysisCapability:
        return self.declaration.capability

    capability = capabilities

    def supports(self, operation: Any) -> bool:
        try:
            return normalize_analysis_operation(operation) in self._adapters
        except Exception:
            return False

    def analyze(
        self,
        request: TransportAnalysisRequest | Mapping[str, Any],
        *,
        negotiated_capability: Any = None,
        cancellation_token: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        normalized = _registry_request(request)
        if _cancelled(cancellation_token):
            raise RuntimeError("registry analysis request was cancelled")
        operation = normalize_analysis_operation(normalized.operation)
        adapter = self._adapters[operation]
        evidence, truncated = adapter.project(
            normalized, producer_id=self.declaration.producer_id
        )
        provenance = (
            _registry_provenance_reference(
                normalized, producer_id=self.declaration.producer_id
            ),
        )
        return _registry_transport_response(
            normalized,
            capability=self.capabilities(),
            evidence_references=evidence,
            provenance_references=provenance,
            negotiated_capability=negotiated_capability,
            cost={
                "artifact_references_considered": len(
                    normalized.artifact_references
                ),
                "local_projection_calls": 1,
            },
            truncated=truncated,
        )

    def analyze_batch(
        self,
        requests: Sequence[TransportAnalysisRequest | Mapping[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(self.analyze(item, **kwargs) for item in requests)


def _legacy_registry_operation(operation: AnalysisOperation) -> AnalysisProviderOperation:
    if operation is AnalysisOperation.SYMBOL_IMPACT:
        return AnalysisProviderOperation.SYMBOL_IMPACT
    if operation is AnalysisOperation.GRAPH_RAG_RETRIEVAL:
        return AnalysisProviderOperation.GRAPH_RETRIEVAL
    raise IpfsDatasetsAnalysisProviderError(
        f"unsupported registry analysis operation: {operation.value}"
    )


def _legacy_artifact_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a registry reference into the legacy adapter's compact schema."""

    result = {
        key: value[key]
        for key in _ARTIFACT_FIELDS
        if value.get(key) not in (None, "")
    }
    if "record_id" not in result:
        identity = value.get("reference_id") or value.get("evidence_id")
        if identity not in (None, ""):
            result["record_id"] = identity
    if "digest" not in result and value.get("sha256") not in (None, ""):
        result["digest"] = value["sha256"]
    if not result:
        raise IpfsDatasetsAnalysisProviderError(
            "registry artifact reference has no legacy compact identity"
        )
    return result


class IpfsDatasetsRegistryAnalysisProducer:
    """Transport bridge around the existing lazy bounded datasets adapter."""

    def __init__(
        self,
        provider: IpfsDatasetsAnalysisProvider | None = None,
        *,
        declaration: AnalysisProducer | None = None,
        provider_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.declaration = declaration or registry_analysis_producer_declarations()[1]
        if provider is not None and provider_kwargs:
            raise IpfsDatasetsAnalysisProviderError(
                "provider and provider_kwargs cannot be combined"
            )
        self._provider = provider or IpfsDatasetsAnalysisProvider(
            **dict(provider_kwargs or {})
        )

    def capabilities(self) -> TransportAnalysisCapability:
        # Do not probe the legacy provider here.  Transport discovery and
        # runtime validation must retain identical declaration-only metadata.
        return self.declaration.capability

    capability = capabilities

    def supports(self, operation: Any) -> bool:
        try:
            return normalize_analysis_operation(
                operation
            ) in _REGISTRY_ANALYSIS_OPERATIONS
        except Exception:
            return False

    def analyze(
        self,
        request: TransportAnalysisRequest | Mapping[str, Any],
        *,
        negotiated_capability: Any = None,
        cancellation_token: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        normalized = _registry_request(request)
        operation = normalize_analysis_operation(normalized.operation)
        metadata = normalized.metadata
        legacy_request = self._provider.build_request(
            operation=_legacy_registry_operation(operation),
            repository_id=metadata.get("repository_id", ""),
            tree_id=metadata.get("tree_id", ""),
            objective_revision=metadata.get("objective_revision", ""),
            query={"text": normalized.question},
            artifact_references=tuple(
                _legacy_artifact_reference(item)
                for item in normalized.artifact_references
            ),
            payload={
                "registry_id": metadata.get("registry_id", ""),
                "operation_spec_id": metadata.get("operation_spec_id", ""),
                "policy_id": metadata.get("policy_id", ""),
            },
        )
        result = self._provider.analyze(
            legacy_request, cancellation_token=cancellation_token
        )
        if result.status is not AnalysisProviderStatus.COMPLETED:
            # Raising is intentional: AnalysisTransport turns this into a
            # typed optional-provider failure and invokes deterministic local
            # fallback with an explicit receipt.
            raise RuntimeError(
                f"optional datasets analysis failed: {result.reason_code}"
            )
        tree_id = str(metadata.get("tree_id") or "")
        evidence = tuple(
            _registry_reference(
                item,
                operation=operation,
                producer_id=self.declaration.producer_id,
                tree_id=tree_id,
            )
            for item in result.evidence_references
        )
        provenance = tuple(
            _registry_reference(
                item,
                operation=operation,
                producer_id=self.declaration.producer_id,
                tree_id=tree_id,
                preserve_reference_id=True,
            )
            for item in result.provenance_references
        ) + (
            _registry_provenance_reference(
                normalized, producer_id=self.declaration.producer_id
            ),
        )
        return _registry_transport_response(
            normalized,
            capability=self.capabilities(),
            evidence_references=evidence,
            provenance_references=provenance,
            negotiated_capability=negotiated_capability,
            cost=result.resource_use,
            truncated=result.truncated,
        )

    def analyze_batch(
        self,
        requests: Sequence[TransportAnalysisRequest | Mapping[str, Any]],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(self.analyze(item, **kwargs) for item in requests)


# Descriptive operation-specific aliases for callers that construct adapters
# directly.  The combined producers above are what the shared registry uses.
IpfsDatasetsSymbolImpactAnalysisAdapter = IpfsDatasetsRegistryAnalysisProducer
IpfsDatasetsGraphRAGRetrievalAdapter = IpfsDatasetsRegistryAnalysisProducer


def create_local_registry_analysis_producer(
    declaration: AnalysisProducer | None = None,
) -> LocalRegistryAnalysisProducer:
    """Create the deterministic local producer without repository access."""

    return LocalRegistryAnalysisProducer(declaration=declaration)


def create_optional_registry_analysis_producer(
    provider: IpfsDatasetsAnalysisProvider | None = None,
    *,
    declaration: AnalysisProducer | None = None,
    **provider_kwargs: Any,
) -> IpfsDatasetsRegistryAnalysisProducer:
    """Create the optional bridge without importing :mod:`ipfs_datasets_py`."""

    return IpfsDatasetsRegistryAnalysisProducer(
        provider=provider,
        declaration=declaration,
        provider_kwargs=provider_kwargs,
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


# ---------------------------------------------------------------------------
# Exact datasets GraphRAG + Cypher-AST binding (SCA-213 / SCA-626)
# ---------------------------------------------------------------------------
#
# Objective evidence SCAEV031DATASETSGRAPH (SCA-G031): bind bounded candidate
# retrieval and graph-query syntax to the exact modules:
#   * ipfs_datasets_py.logic.intent_ir.graphrag.retrieval.IntentGraphRetriever
#   * ipfs_datasets_py.knowledge_graphs.cypher.ast / .parser
#
# The package root is not an accepted implicit backend for the exact-provider
# gate.  Capability labels, fixture-only backends, and local lexical fallback
# cannot claim exact datasets use or proof authority.  GraphRAG results and
# Cypher ASTs remain context-only / syntax-only.

# Objective-heap evidence term for SCA-G031 / SCAEV031DATASETSGRAPH.
SCAEV031DATASETSGRAPH: Final = "SCAEV031DATASETSGRAPH"
SCAEV031DATASETSGRAPH_EVIDENCE: Final = SCAEV031DATASETSGRAPH
SCAEV031DATASETSGRAPH_COVERAGE: Final = (
    "exact-graphrag-intent-graph-retriever-module-binding",
    "exact-cypher-ast-and-parser-module-binding",
    "capability-receipted-modules-signatures-versions-package-tree",
    "capability-receipted-graph-roots-bounds-result-identities",
    "real-module-canary-context-only-candidates",
    "cypher-ast-syntax-only-non-authoritative",
    "package-root-fixture-local-lexical-cannot-claim-exact-use",
    "missing-or-incompatible-modules-are-typed-blockers",
    "no-proof-or-completion-authority",
)

INTENT_GRAPH_RETRIEVER_INTERFACE: Final = "IntentGraphRetriever@1"
QUERY_NODE_INTERFACE: Final = "QueryNode@1"
BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE_REF: Final = "BoundedGraphRAGRetriever@1"
DATASETS_GRAPH_BINDING_VERSION: Final = "1.0.0"
DATASETS_GRAPH_PROBE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-graph-probe@1"
)
DATASETS_GRAPH_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-graph-capability@1"
)
DATASETS_GRAPH_CANARY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-graph-canary@1"
)
DATASETS_GRAPH_CYPHER_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-cypher-ast-receipt@1"
)
DATASETS_GRAPH_CONTEXT_AUTHORITY: Final = "context_only"
DATASETS_GRAPH_SYNTAX_ONLY: Final = "syntax_only"

EXACT_GRAPHRAG_RETRIEVAL_MODULE: Final = (
    "ipfs_datasets_py.logic.intent_ir.graphrag.retrieval"
)
EXACT_CYPHER_AST_MODULE: Final = "ipfs_datasets_py.knowledge_graphs.cypher.ast"
EXACT_CYPHER_PARSER_MODULE: Final = (
    "ipfs_datasets_py.knowledge_graphs.cypher.parser"
)
EXACT_GRAPHRAG_CORPUS_PROJECTOR_MODULE: Final = (
    "ipfs_datasets_py.logic.intent_ir.graphrag.corpus_projector"
)
EXACT_GRAPHRAG_ONTOLOGY_MODULE: Final = (
    "ipfs_datasets_py.logic.intent_ir.graphrag.ontology"
)
EXACT_GRAPHRAG_SKILLCENTER_MODULE: Final = (
    "ipfs_datasets_py.logic.intent_ir.source_adapters.skillcenter"
)
EXACT_IR_IDENTITY_MODULE: Final = "ipfs_datasets_py.logic.ir_core.identity"

# Modules that may exist but never satisfy the exact datasets-provider gate.
PACKAGE_ROOT_FALLBACK_MODULES: Final = frozenset(
    {
        "ipfs_datasets_py",
        "ipfs_datasets_py.logic",
        "ipfs_datasets_py.knowledge_graphs",
        "ipfs_datasets_py.knowledge_graphs.cypher",
        "ipfs_datasets_py.logic.intent_ir",
        "ipfs_datasets_py.logic.intent_ir.graphrag",
    }
)

EXACT_DATASETS_SOURCE_KINDS: Final = frozenset(
    {
        "exact_module",
        "exact_datasets",
        "intent_graph_retriever",
        "cypher_ast",
        "cypher_parser",
    }
)
REJECTED_EXACT_SOURCE_KINDS: Final = frozenset(
    {
        "fixture",
        "fixture_only",
        "local",
        "local_lexical",
        "lexical_fallback",
        "package_root",
        "package_root_fallback",
        "capability_label",
        "simulated",
    }
)


class DatasetsGraphBackendKind(str, Enum):
    """Closed vocabulary of exact datasets graph backend families."""

    GRAPHRAG = "graphrag"
    CYPHER_AST = "cypher_ast"


class DatasetsGraphBackendError(RuntimeError):
    """Typed failure for exact datasets GraphRAG / Cypher-AST binding."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.details = dict(details or {})


@dataclass(frozen=True)
class DatasetsGraphSymbolSpec:
    """One required real-module symbol that must be signature-probed."""

    module: str
    name: str
    required_callable: bool = True
    required_parameters: tuple[str, ...] = ()

    def identity(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "required_callable": self.required_callable,
            "required_parameters": list(self.required_parameters),
        }


@dataclass(frozen=True)
class DatasetsGraphBackendSpec:
    """Immutable exact-module contract for one datasets graph backend."""

    kind: DatasetsGraphBackendKind
    provider_id: str
    symbols: tuple[DatasetsGraphSymbolSpec, ...]
    interface: str
    description: str
    authoritative: bool = False

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "symbols": [item.identity() for item in self.symbols],
            "interface": self.interface,
            "description": self.description,
            "authoritative": self.authoritative,
            "binding_version": DATASETS_GRAPH_BINDING_VERSION,
            "authority": DATASETS_GRAPH_CONTEXT_AUTHORITY,
        }


DATASETS_GRAPH_BACKEND_SPECS: Final[
    Mapping[DatasetsGraphBackendKind, DatasetsGraphBackendSpec]
] = {
    DatasetsGraphBackendKind.GRAPHRAG: DatasetsGraphBackendSpec(
        kind=DatasetsGraphBackendKind.GRAPHRAG,
        provider_id="intent-graph-retriever",
        interface=INTENT_GRAPH_RETRIEVER_INTERFACE,
        description=(
            "Bounded Intent-IR GraphRAG retrieval "
            "(context-only candidate nomination)"
        ),
        authoritative=False,
        symbols=(
            DatasetsGraphSymbolSpec(
                EXACT_GRAPHRAG_RETRIEVAL_MODULE,
                "IntentGraphRetriever",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_GRAPHRAG_RETRIEVAL_MODULE,
                "RetrievalRequest",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_GRAPHRAG_RETRIEVAL_MODULE,
                "GraphSnapshot",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_GRAPHRAG_RETRIEVAL_MODULE,
                "PartitionAssignment",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_GRAPHRAG_RETRIEVAL_MODULE,
                "NeighborCandidate",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_GRAPHRAG_RETRIEVAL_MODULE,
                "RETRIEVAL_AUTHORITY",
                required_callable=False,
            ),
        ),
    ),
    DatasetsGraphBackendKind.CYPHER_AST: DatasetsGraphBackendSpec(
        kind=DatasetsGraphBackendKind.CYPHER_AST,
        provider_id="cypher-ast-parser",
        interface=QUERY_NODE_INTERFACE,
        description=(
            "Cypher query AST and parser for syntax-only graph-query validation"
        ),
        authoritative=False,
        symbols=(
            DatasetsGraphSymbolSpec(
                EXACT_CYPHER_AST_MODULE,
                "QueryNode",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_CYPHER_AST_MODULE,
                "ASTNode",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_CYPHER_PARSER_MODULE,
                "CypherParser",
                required_callable=False,
            ),
            DatasetsGraphSymbolSpec(
                EXACT_CYPHER_PARSER_MODULE,
                "parse_cypher",
                required_parameters=("query",),
            ),
        ),
    ),
}


@dataclass(frozen=True)
class DatasetsGraphSymbolReceipt:
    """Receipt that one exact module symbol was located and signature-checked."""

    module: str
    name: str
    qualname: str
    available: bool
    signature: str
    reason_code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "qualname": self.qualname,
            "available": self.available,
            "signature": self.signature,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class DatasetsGraphBackendProbe:
    """Capability probe result for one exact datasets graph backend."""

    kind: DatasetsGraphBackendKind
    provider_id: str
    available: bool
    interface: str
    package_version: str
    package_tree: str
    capability_revision: str
    symbol_receipts: tuple[DatasetsGraphSymbolReceipt, ...]
    unavailable_reason: str = ""
    reason_code: str = ""
    module_paths: tuple[str, ...] = ()
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not self.available and not self.reason_code:
            object.__setattr__(self, "reason_code", "backend_unavailable")
        if self.available and not self.symbol_receipts:
            raise DatasetsGraphBackendError(
                "available probe requires symbol receipts",
                reason_code="probe_missing_symbol_receipts",
            )
        if self.authoritative:
            raise DatasetsGraphBackendError(
                "GraphRAG and Cypher-AST backends are never authoritative",
                reason_code="authoritative_claim_rejected",
            )

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def exact_modules_bound(self) -> bool:
        return self.available and bool(self.module_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DATASETS_GRAPH_PROBE_SCHEMA,
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "available": self.available,
            "interface": self.interface,
            "package_version": self.package_version,
            "package_tree": self.package_tree,
            "capability_revision": self.capability_revision,
            "symbol_receipts": [item.to_dict() for item in self.symbol_receipts],
            "unavailable_reason": self.unavailable_reason,
            "reason_code": self.reason_code,
            "module_paths": list(self.module_paths),
            "authoritative": False,
            "non_authoritative": True,
            "proof_authority": False,
            "completion_authority": False,
            "binding_version": DATASETS_GRAPH_BINDING_VERSION,
            "authority": DATASETS_GRAPH_CONTEXT_AUTHORITY,
            "evidence_id": SCAEV031DATASETSGRAPH,
        }


def _datasets_graph_digest(value: Mapping[str, Any], *, prefix: str) -> str:
    return _content_id(dict(value), name=prefix)


def _datasets_package_version(importer: Callable[[str], Any]) -> str:
    try:
        package = importer("ipfs_datasets_py")
    except (ImportError, ModuleNotFoundError):
        return ""
    version = getattr(package, "__version__", None)
    if isinstance(version, str) and version.strip():
        return version.strip()
    try:
        return str(importlib.metadata.version("ipfs_datasets_py"))
    except Exception:
        return DATASETS_GRAPH_BINDING_VERSION


def _datasets_package_tree(importer: Callable[[str], Any]) -> str:
    """Bind a stable package-tree identity without treating Gitlink opacity."""

    try:
        package = importer("ipfs_datasets_py")
    except (ImportError, ModuleNotFoundError):
        return ""
    paths: list[str] = []
    file_path = getattr(package, "__file__", None)
    if isinstance(file_path, str) and file_path:
        paths.append(file_path)
    path_attr = getattr(package, "__path__", None)
    if path_attr is not None:
        try:
            paths.extend(str(item) for item in list(path_attr))
        except TypeError:
            pass
    if not paths:
        return ""
    return _datasets_graph_digest(
        {"paths": sorted(set(paths))},
        prefix="datasets-package-tree",
    )


def _probe_graph_symbol(
    spec: DatasetsGraphSymbolSpec,
    *,
    importer: Callable[[str], Any],
) -> DatasetsGraphSymbolReceipt:
    if spec.module in PACKAGE_ROOT_FALLBACK_MODULES:
        return DatasetsGraphSymbolReceipt(
            module=spec.module,
            name=spec.name,
            qualname=f"{spec.module}.{spec.name}",
            available=False,
            signature="",
            reason_code="package_root_fallback_rejected",
        )
    try:
        module = importer(spec.module)
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        return DatasetsGraphSymbolReceipt(
            module=spec.module,
            name=spec.name,
            qualname=f"{spec.module}.{spec.name}",
            available=False,
            signature="",
            reason_code=f"module_import_failed:{type(exc).__name__}",
        )
    target = getattr(module, spec.name, None)
    if target is None:
        return DatasetsGraphSymbolReceipt(
            module=spec.module,
            name=spec.name,
            qualname=f"{spec.module}.{spec.name}",
            available=False,
            signature="",
            reason_code="symbol_missing",
        )
    signature_text = ""
    if spec.required_callable:
        if not callable(target):
            return DatasetsGraphSymbolReceipt(
                module=spec.module,
                name=spec.name,
                qualname=f"{spec.module}.{spec.name}",
                available=False,
                signature="",
                reason_code="symbol_not_callable",
            )
        try:
            signature = inspect.signature(target)
            signature_text = str(signature)
            parameter_names = {
                name
                for name, parameter in signature.parameters.items()
                if name not in {"self", "cls"}
                and parameter.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            }
            missing = [
                name
                for name in spec.required_parameters
                if name not in parameter_names
            ]
            has_var_keyword = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if missing and not has_var_keyword:
                return DatasetsGraphSymbolReceipt(
                    module=spec.module,
                    name=spec.name,
                    qualname=f"{spec.module}.{spec.name}",
                    available=False,
                    signature=signature_text,
                    reason_code="signature_parameters_missing",
                )
        except (TypeError, ValueError) as exc:
            return DatasetsGraphSymbolReceipt(
                module=spec.module,
                name=spec.name,
                qualname=f"{spec.module}.{spec.name}",
                available=False,
                signature="",
                reason_code=f"signature_uninspectable:{type(exc).__name__}",
            )
    else:
        try:
            if inspect.isclass(target):
                # Prefer method signatures that the canary actually exercises.
                if spec.name == "IntentGraphRetriever":
                    signature_text = (
                        f"__init__{inspect.signature(target.__init__)}"
                        f"; retrieve{inspect.signature(target.retrieve)}"
                    )
                elif spec.name == "CypherParser":
                    signature_text = (
                        f"__init__{inspect.signature(target.__init__)}"
                        f"; parse{inspect.signature(target.parse)}"
                    )
                else:
                    signature_text = str(inspect.signature(target))
            elif callable(target):
                signature_text = str(inspect.signature(target))
            else:
                signature_text = repr(target) if isinstance(target, str) else type(
                    target
                ).__name__
        except (TypeError, ValueError):
            signature_text = type(target).__name__
    return DatasetsGraphSymbolReceipt(
        module=spec.module,
        name=spec.name,
        qualname=f"{spec.module}.{spec.name}",
        available=True,
        signature=signature_text,
    )


def probe_datasets_graph_backend(
    kind: DatasetsGraphBackendKind | str,
    *,
    importer: Callable[[str], Any] | None = None,
) -> DatasetsGraphBackendProbe:
    """Probe one exact datasets graph backend without activating adapters."""

    normalized = (
        kind
        if isinstance(kind, DatasetsGraphBackendKind)
        else DatasetsGraphBackendKind(str(kind).strip().lower())
    )
    spec = DATASETS_GRAPH_BACKEND_SPECS[normalized]
    load = importer or importlib.import_module
    receipts = tuple(
        _probe_graph_symbol(symbol, importer=load) for symbol in spec.symbols
    )
    available = all(item.available for item in receipts)
    package_version = _datasets_package_version(load) if available else ""
    package_tree = _datasets_package_tree(load) if available else ""
    module_paths = tuple(
        sorted({item.module for item in receipts if item.available})
    )
    capability_revision = _datasets_graph_digest(
        {
            "spec": spec.identity(),
            "package_version": package_version,
            "package_tree": package_tree,
            "symbols": [item.to_dict() for item in receipts],
        },
        prefix="datasets-graph-capability",
    )
    if not available:
        failed = next(item for item in receipts if not item.available)
        return DatasetsGraphBackendProbe(
            kind=normalized,
            provider_id=spec.provider_id,
            available=False,
            interface=spec.interface,
            package_version=package_version,
            package_tree=package_tree,
            capability_revision=capability_revision,
            symbol_receipts=receipts,
            unavailable_reason=(
                f"{failed.qualname} unavailable ({failed.reason_code})"
            ),
            reason_code=failed.reason_code or "backend_unavailable",
            module_paths=module_paths,
            authoritative=False,
        )
    return DatasetsGraphBackendProbe(
        kind=normalized,
        provider_id=spec.provider_id,
        available=True,
        interface=spec.interface,
        package_version=package_version or DATASETS_GRAPH_BINDING_VERSION,
        package_tree=package_tree,
        capability_revision=capability_revision,
        symbol_receipts=receipts,
        module_paths=module_paths,
        authoritative=False,
    )


def probe_all_datasets_graph_backends(
    *,
    importer: Callable[[str], Any] | None = None,
    kinds: Sequence[DatasetsGraphBackendKind | str] | None = None,
) -> tuple[DatasetsGraphBackendProbe, ...]:
    """Probe every exact graph backend (or an explicit subset) deterministically."""

    selected = (
        tuple(DatasetsGraphBackendKind)
        if kinds is None
        else tuple(
            item
            if isinstance(item, DatasetsGraphBackendKind)
            else DatasetsGraphBackendKind(str(item).strip().lower())
            for item in kinds
        )
    )
    return tuple(
        probe_datasets_graph_backend(kind, importer=importer) for kind in selected
    )


def inspect_exact_datasets_graph_capability(
    *,
    importer: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Return capability receipts for GraphRAG and Cypher-AST exact modules.

    Constructing this receipt never claims proof or completion authority.
    Missing or signature-incompatible modules are typed blockers.
    """

    probes = probe_all_datasets_graph_backends(importer=importer)
    by_kind = {probe.kind.value: probe.to_dict() for probe in probes}
    available = all(probe.available for probe in probes)
    blockers = [
        {
            "kind": probe.kind.value,
            "reason_code": probe.reason_code,
            "unavailable_reason": probe.unavailable_reason,
        }
        for probe in probes
        if not probe.available
    ]
    receipt = {
        "schema": DATASETS_GRAPH_CAPABILITY_SCHEMA,
        "binding_version": DATASETS_GRAPH_BINDING_VERSION,
        "evidence": {
            "requirement_ids": [SCAEV031DATASETSGRAPH],
            "coverage": list(SCAEV031DATASETSGRAPH_COVERAGE),
            "evidence_id": SCAEV031DATASETSGRAPH_EVIDENCE,
        },
        "evidence_id": SCAEV031DATASETSGRAPH,
        "interfaces": {
            "IntentGraphRetriever": INTENT_GRAPH_RETRIEVER_INTERFACE,
            "QueryNode": QUERY_NODE_INTERFACE,
            "BoundedGraphRAGRetriever": BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE_REF,
        },
        "exact_modules": {
            "graphrag": EXACT_GRAPHRAG_RETRIEVAL_MODULE,
            "cypher_ast": EXACT_CYPHER_AST_MODULE,
            "cypher_parser": EXACT_CYPHER_PARSER_MODULE,
        },
        "package_root_fallback_accepted": False,
        "fixture_only_accepted": False,
        "local_lexical_fallback_accepted": False,
        "available": available,
        "authoritative": False,
        "non_authoritative": True,
        "proof_authority": False,
        "completion_authority": False,
        "authority": DATASETS_GRAPH_CONTEXT_AUTHORITY,
        "backends": by_kind,
        "blockers": blockers,
        "capability_revision": _datasets_graph_digest(
            {
                "backends": by_kind,
                "available": available,
                "blockers": blockers,
            },
            prefix="datasets-graph-capability-aggregate",
        ),
    }
    receipt["capability_receipt_id"] = _datasets_graph_digest(
        receipt, prefix="datasets-graph-capability-receipt"
    )
    return receipt


def admit_exact_datasets_graph_source(
    source_kind: str,
    *,
    probe: DatasetsGraphBackendProbe | None = None,
    module_name: str | None = None,
) -> None:
    """Fail closed when a caller tries to claim exact datasets use falsely."""

    normalized = str(source_kind or "").strip().lower()
    if normalized in REJECTED_EXACT_SOURCE_KINDS:
        raise DatasetsGraphBackendError(
            (
                f"source kind {normalized!r} cannot claim exact datasets "
                "GraphRAG/Cypher-AST use"
            ),
            reason_code="exact_source_rejected",
            details={"source_kind": normalized},
        )
    if normalized not in EXACT_DATASETS_SOURCE_KINDS:
        raise DatasetsGraphBackendError(
            f"unknown exact datasets source kind: {normalized!r}",
            reason_code="exact_source_unknown",
            details={"source_kind": normalized},
        )
    if module_name and module_name in PACKAGE_ROOT_FALLBACK_MODULES:
        raise DatasetsGraphBackendError(
            "package-root fallback cannot satisfy the exact datasets-provider gate",
            reason_code="package_root_fallback_rejected",
            details={"module_name": module_name},
        )
    if probe is not None and not probe.available:
        raise DatasetsGraphBackendError(
            probe.unavailable_reason or "exact datasets backend unavailable",
            reason_code=probe.reason_code or "backend_unavailable",
            details=probe.to_dict(),
        )


def parse_cypher_query_ast(
    query: str,
    *,
    importer: Callable[[str], Any] | None = None,
    max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
) -> dict[str, Any]:
    """Parse a Cypher query via the exact AST/parser modules (syntax-only)."""

    load = importer or importlib.import_module
    probe = probe_datasets_graph_backend(
        DatasetsGraphBackendKind.CYPHER_AST, importer=load
    )
    if not probe.available:
        raise DatasetsGraphBackendError(
            probe.unavailable_reason or "Cypher-AST backend unavailable",
            reason_code=probe.reason_code or "backend_unavailable",
            details=probe.to_dict(),
        )
    admit_exact_datasets_graph_source("cypher_parser", probe=probe)
    normalized = _text(query, "cypher query", required=True, max_bytes=max_query_bytes)
    parser_module = load(EXACT_CYPHER_PARSER_MODULE)
    ast_module = load(EXACT_CYPHER_AST_MODULE)
    parse_fn = getattr(parser_module, "parse_cypher", None)
    if callable(parse_fn):
        node = parse_fn(normalized)
    else:
        node = parser_module.CypherParser().parse(normalized)
    query_node_type = ast_module.QueryNode
    if not isinstance(node, query_node_type):
        raise DatasetsGraphBackendError(
            "Cypher parser did not return QueryNode",
            reason_code="cypher_ast_type_mismatch",
            details={"type": type(node).__name__},
        )
    clause_count = len(getattr(node, "clauses", ()) or ())
    node_type = getattr(getattr(node, "node_type", None), "name", None) or str(
        getattr(node, "node_type", "")
    )
    receipt = {
        "schema": DATASETS_GRAPH_CYPHER_RECEIPT_SCHEMA,
        "interface": QUERY_NODE_INTERFACE,
        "module": EXACT_CYPHER_PARSER_MODULE,
        "ast_module": EXACT_CYPHER_AST_MODULE,
        "query_digest": _datasets_graph_digest(
            {"query": normalized}, prefix="cypher-query"
        ),
        "query_bytes": len(normalized.encode("utf-8")),
        "node_type": node_type,
        "clause_count": clause_count,
        "ast_class": type(node).__name__,
        "syntax_only": True,
        "authority": DATASETS_GRAPH_SYNTAX_ONLY,
        "authoritative": False,
        "non_authoritative": True,
        "proof_authority": False,
        "completion_authority": False,
        "source_language_parser": False,
        "capability_revision": probe.capability_revision,
        "package_version": probe.package_version,
        "package_tree": probe.package_tree,
        "symbol_receipts": [item.to_dict() for item in probe.symbol_receipts],
        "module_paths": list(probe.module_paths),
    }
    receipt["receipt_id"] = _datasets_graph_digest(
        receipt, prefix="cypher-ast-receipt"
    )
    return receipt


def run_intent_graph_retriever_canary(
    *,
    importer: Callable[[str], Any] | None = None,
    k: int = 1,
    max_bytes: int = 32_000,
    timeout_ms: int = 1_000,
) -> dict[str, Any]:
    """Exercise the real IntentGraphRetriever and return a context-only receipt.

    Fixture-only adapters cannot substitute for this canary.  Results never
    carry proof authority.
    """

    load = importer or importlib.import_module
    probe = probe_datasets_graph_backend(
        DatasetsGraphBackendKind.GRAPHRAG, importer=load
    )
    if not probe.available:
        raise DatasetsGraphBackendError(
            probe.unavailable_reason or "GraphRAG backend unavailable",
            reason_code=probe.reason_code or "backend_unavailable",
            details=probe.to_dict(),
        )
    admit_exact_datasets_graph_source("intent_graph_retriever", probe=probe)

    retrieval = load(EXACT_GRAPHRAG_RETRIEVAL_MODULE)
    projector_mod = load(EXACT_GRAPHRAG_CORPUS_PROJECTOR_MODULE)
    ontology = load(EXACT_GRAPHRAG_ONTOLOGY_MODULE)
    skillcenter = load(EXACT_GRAPHRAG_SKILLCENTER_MODULE)
    identity = load(EXACT_IR_IDENTITY_MODULE)

    class _RecordingStore:
        def put_bytes(self, payload: bytes, *, media_type: str) -> str:
            return identity.cid_v1(payload)

    def _skill(skill_id: str) -> Any:
        return skillcenter.SkillCenterSkillRecord(
            skill_id=skill_id,
            domain="canary",
            profile="canary",
            source_type="github",
            source_url=f"https://example.test/{skill_id}/SKILL.md",
            title=f"Canary {skill_id}",
            overall_score=1.0,
            skill_kind="github",
            language="en",
            source_id=f"source-{skill_id}",
            primary_source_id=f"primary-{skill_id}",
            metadata_yaml='license_spdx: "MIT"\n',
            skill_md=f"# {skill_id}\n\nCanary content for {skill_id}.\n",
            library_md="",
            dataset_id="canary/intent-graph",
            dataset_revision="canary-1",
            repository_file="canary.sqlite",
            bundle_sha256="a" * 64,
        )

    graph = projector_mod.CorpusProjector(_RecordingStore()).project(
        (
            projector_mod.CorpusEvidenceRecord(
                _skill("query"),
                neighbor_skill_ids=("alpha",),
            ),
            _skill("alpha"),
        )
    )
    skill_nodes = {
        node.properties["skill_id"]: node
        for node in graph.nodes
        if node.node_type is ontology.CorpusNodeType.SKILL
    }
    query_id = skill_nodes["query"].node_id
    alpha_id = skill_nodes["alpha"].node_id
    edge = next(
        item
        for item in graph.edges
        if item.edge_type is ontology.CorpusEdgeType.NEIGHBOR_OF
        and {item.source, item.target} == {query_id, alpha_id}
    )
    assignments = {
        query_id: retrieval.PartitionAssignment("evaluation", "family-query"),
        alpha_id: retrieval.PartitionAssignment("evaluation", "family-alpha"),
    }
    request = retrieval.RetrievalRequest(
        query_node_id=query_id,
        snapshot=retrieval.GraphSnapshot.from_graph(graph),
        partition="evaluation",
        source_family="family-query",
        k=max(1, min(int(k), 8)),
        max_bytes=max(1_024, int(max_bytes)),
        timeout_ms=max(50, int(timeout_ms)),
        candidates=(
            retrieval.NeighborCandidate(
                node_id=alpha_id,
                edge_id=edge.edge_id,
                score=0.9,
                graph_digest=graph.graph_digest,
            ),
        ),
    )
    result = retrieval.IntentGraphRetriever(graph, assignments).retrieve(request)
    premises = tuple(getattr(result, "premises", ()) or ())
    premise_payloads = []
    for premise in premises:
        converter = getattr(premise, "to_dict", None)
        payload = converter() if callable(converter) else {
            "node_id": getattr(premise, "node_id", ""),
            "proof_authority": getattr(premise, "proof_authority", None),
            "authority": getattr(premise, "authority", ""),
        }
        if payload.get("proof_authority") is not False:
            raise DatasetsGraphBackendError(
                "IntentGraphRetriever returned proof-authoritative premise",
                reason_code="graphrag_authoritative_premise",
                details={"premise": payload},
            )
        if payload.get("authority") not in (
            None,
            "",
            DATASETS_GRAPH_CONTEXT_AUTHORITY,
            getattr(retrieval, "RETRIEVAL_AUTHORITY", DATASETS_GRAPH_CONTEXT_AUTHORITY),
        ):
            raise DatasetsGraphBackendError(
                "IntentGraphRetriever premise authority is not context_only",
                reason_code="graphrag_authority_mismatch",
                details={"premise": payload},
            )
        premise_payloads.append(payload)

    status = getattr(getattr(result, "status", None), "value", result.status)
    result_authority = getattr(result, "authority", DATASETS_GRAPH_CONTEXT_AUTHORITY)
    if result_authority != DATASETS_GRAPH_CONTEXT_AUTHORITY and result_authority != getattr(
        retrieval, "RETRIEVAL_AUTHORITY", DATASETS_GRAPH_CONTEXT_AUTHORITY
    ):
        raise DatasetsGraphBackendError(
            "IntentGraphRetriever result is not context_only",
            reason_code="graphrag_result_authoritative",
            details={"authority": result_authority},
        )

    bounds = {
        "k": request.k,
        "max_bytes": request.max_bytes,
        "timeout_ms": request.timeout_ms,
    }
    graph_root = str(getattr(graph, "graph_cid", "") or graph.graph_digest)
    receipt = {
        "schema": DATASETS_GRAPH_CANARY_SCHEMA,
        "interface": INTENT_GRAPH_RETRIEVER_INTERFACE,
        "module": EXACT_GRAPHRAG_RETRIEVAL_MODULE,
        "symbol": "IntentGraphRetriever.retrieve",
        "status": str(status),
        "graph_root": graph_root,
        "graph_digest": graph.graph_digest,
        "graph_cid": str(getattr(graph, "graph_cid", "") or ""),
        "snapshot": request.snapshot.to_dict(),
        "bounds": bounds,
        "query_node_id": query_id,
        "candidate_count": len(premises),
        "premises": premise_payloads,
        "authority": DATASETS_GRAPH_CONTEXT_AUTHORITY,
        "authoritative": False,
        "non_authoritative": True,
        "proof_authority": False,
        "completion_authority": False,
        "safe_for_proof": False,
        "fixture_only": False,
        "exact_module": True,
        "package_version": probe.package_version,
        "package_tree": probe.package_tree,
        "capability_revision": probe.capability_revision,
        "symbol_receipts": [item.to_dict() for item in probe.symbol_receipts],
        "module_paths": list(probe.module_paths),
        "result_identity": _datasets_graph_digest(
            {
                "status": str(status),
                "graph_digest": graph.graph_digest,
                "premises": premise_payloads,
                "bounds": bounds,
            },
            prefix="graphrag-canary-result",
        ),
    }
    receipt["receipt_id"] = _datasets_graph_digest(
        receipt, prefix="graphrag-canary-receipt"
    )
    return receipt


def run_datasets_graph_real_module_canary(
    *,
    importer: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Combined real-module canary for GraphRAG retrieval and Cypher AST."""

    load = importer or importlib.import_module
    capability = inspect_exact_datasets_graph_capability(importer=load)
    if not capability["available"]:
        raise DatasetsGraphBackendError(
            "exact datasets GraphRAG/Cypher modules unavailable or incompatible",
            reason_code="exact_datasets_graph_unavailable",
            details=capability,
        )
    graphrag = run_intent_graph_retriever_canary(importer=load)
    cypher = parse_cypher_query_ast(
        "MATCH (n:Person) WHERE n.age > 30 RETURN n",
        importer=load,
    )
    combined = {
        "schema": DATASETS_GRAPH_CANARY_SCHEMA,
        "binding_version": DATASETS_GRAPH_BINDING_VERSION,
        "evidence_id": SCAEV031DATASETSGRAPH,
        "evidence": {
            "requirement_ids": [SCAEV031DATASETSGRAPH],
            "coverage": list(SCAEV031DATASETSGRAPH_COVERAGE),
            "evidence_id": SCAEV031DATASETSGRAPH_EVIDENCE,
        },
        "capability": capability,
        "graphrag": graphrag,
        "cypher_ast": cypher,
        "authoritative": False,
        "non_authoritative": True,
        "proof_authority": False,
        "completion_authority": False,
        "fixture_only": False,
        "exact_module": True,
        "package_root_fallback": False,
    }
    combined["canary_receipt_id"] = _datasets_graph_digest(
        combined, prefix="datasets-graph-canary"
    )
    return combined


class ExactDatasetsGraphRAGAdapter:
    """Provider-backed candidate nominator bound to IntentGraphRetriever only.

    Package-root backends, fixture stubs, and local lexical adapters cannot be
    substituted here.  Results are always non-authoritative.
    """

    operation = AnalysisProviderOperation.GRAPH_RETRIEVAL

    def __init__(
        self,
        *,
        importer: Callable[[str], Any] | None = None,
        probe: DatasetsGraphBackendProbe | None = None,
    ) -> None:
        self._importer = importer or importlib.import_module
        self._probe = probe

    def capability(self) -> dict[str, Any]:
        probe = self._probe or probe_datasets_graph_backend(
            DatasetsGraphBackendKind.GRAPHRAG, importer=self._importer
        )
        return {
            "interface": INTENT_GRAPH_RETRIEVER_INTERFACE,
            "provider_id": probe.provider_id,
            "available": probe.available,
            "module": EXACT_GRAPHRAG_RETRIEVAL_MODULE,
            "package_version": probe.package_version,
            "package_tree": probe.package_tree,
            "capability_revision": probe.capability_revision,
            "authoritative": False,
            "non_authoritative": True,
            "proof_authority": False,
            "module_paths": list(probe.module_paths),
            "symbol_receipts": [item.to_dict() for item in probe.symbol_receipts],
            "reason_code": probe.reason_code,
            "evidence_id": SCAEV031DATASETSGRAPH,
            "evidence": {
                "requirement_ids": [SCAEV031DATASETSGRAPH],
                "coverage": list(SCAEV031DATASETSGRAPH_COVERAGE),
                "evidence_id": SCAEV031DATASETSGRAPH_EVIDENCE,
            },
        }

    capabilities = capability

    def retrieve_candidates(
        self,
        *,
        query: str,
        graph_root: str,
        snapshot_id: str,
        bounds: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the real-module canary path and project context-only candidates.

        The Intent graph is partition-isolated and content-addressed; supervisor
        symbol graphs do not share node IDs with it.  This adapter therefore
        capability-receipts the exact retrieval surface and returns explicit
        context-only candidates derived from the canary premises when available.
        """

        probe = self._probe or probe_datasets_graph_backend(
            DatasetsGraphBackendKind.GRAPHRAG, importer=self._importer
        )
        if not probe.available:
            raise DatasetsGraphBackendError(
                probe.unavailable_reason or "GraphRAG backend unavailable",
                reason_code=probe.reason_code or "backend_unavailable",
                details=probe.to_dict(),
            )
        admit_exact_datasets_graph_source("exact_module", probe=probe)
        limits = dict(bounds or {})
        canary = run_intent_graph_retriever_canary(
            importer=self._importer,
            k=int(limits.get("max_results") or limits.get("k") or 1),
            max_bytes=int(limits.get("max_response_bytes") or limits.get("max_bytes") or 32_000),
            timeout_ms=int(limits.get("timeout_ms") or 1_000),
        )
        references: list[dict[str, Any]] = []
        for index, premise in enumerate(canary.get("premises") or ()):
            node_id = str(premise.get("node_id") or "")
            references.append(
                {
                    "kind": "graph_retrieval",
                    "evidence_id": node_id or f"premise:{index}",
                    "record_id": node_id or f"premise:{index}",
                    "node_id": node_id,
                    "path": "",
                    "symbol": str(premise.get("node_type") or ""),
                    "summary": (
                        f"context-only IntentGraphRetriever premise "
                        f"{node_id or index}"
                    ),
                    "score": premise.get("score", 0.0),
                    "tree_id": snapshot_id,
                    "artifact_id": graph_root,
                    "stable_key": node_id,
                    "authority": DATASETS_GRAPH_CONTEXT_AUTHORITY,
                    "non_authoritative": True,
                    "proof_authority": False,
                }
            )
        return {
            "status": "completed",
            "operation": self.operation.value,
            "evidence_references": references,
            "results": references,
            "truncated": False,
            "non_authoritative": True,
            "proof_authority": False,
            "completion_authority": False,
            "safe_for_proof": False,
            "authority": DATASETS_GRAPH_CONTEXT_AUTHORITY,
            "exact_module": True,
            "fixture_only": False,
            "package_root_fallback": False,
            "query": query,
            "graph_root": graph_root or canary.get("graph_root", ""),
            "snapshot_id": snapshot_id,
            "bounds": canary.get("bounds", {}),
            "capability": self.capability(),
            "canary_receipt_id": canary.get("receipt_id", ""),
            "result_id": canary.get("result_identity", ""),
            "receipt_id": canary.get("receipt_id", ""),
            "capability_revision": probe.capability_revision,
            "package_version": probe.package_version,
            "package_tree": probe.package_tree,
            "module_paths": list(probe.module_paths),
        }


def create_exact_datasets_graphrag_adapter(
    *,
    importer: Callable[[str], Any] | None = None,
) -> ExactDatasetsGraphRAGAdapter:
    """Create the exact GraphRAG adapter after a successful module probe."""

    probe = probe_datasets_graph_backend(
        DatasetsGraphBackendKind.GRAPHRAG, importer=importer
    )
    if not probe.available:
        raise DatasetsGraphBackendError(
            probe.unavailable_reason or "GraphRAG backend unavailable",
            reason_code=probe.reason_code or "backend_unavailable",
            details=probe.to_dict(),
        )
    return ExactDatasetsGraphRAGAdapter(importer=importer, probe=probe)


__all__ = [
    "IPFS_DATASETS_ANALYSIS_PROVIDER_VERSION",
    "IPFS_DATASETS_ANALYSIS_PROTOCOL_VERSION",
    "IPFS_DATASETS_ANALYSIS_PROVIDER_ID",
    "IPFS_DATASETS_OFFLOAD_COORDINATION_BOUNDARY",
    "IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID",
    "IPFS_DATASETS_COMPLETION_ACCEPTANCE_CRITERION",
    "OPTIONAL_DATASETS_DEGRADATION_REQUIREMENT_ID",
    "PROVIDER_CAPABILITY_SCHEMA",
    "PROVIDER_REQUEST_SCHEMA",
    "PROVIDER_RESULT_SCHEMA",
    "PROVIDER_DEGRADATION_EVIDENCE_SCHEMA",
    "DEFAULT_MAX_BATCH_REQUESTS",
    "normalize_analysis_provider_operation",
    "inspect_analysis_provider_capability",
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
    "LocalSymbolImpactAnalysisAdapter",
    "LocalGraphRAGRetrievalAdapter",
    "LocalRegistryAnalysisProducer",
    "IpfsDatasetsRegistryAnalysisProducer",
    "IpfsDatasetsSymbolImpactAnalysisAdapter",
    "IpfsDatasetsGraphRAGRetrievalAdapter",
    "registry_analysis_producer_declarations",
    "create_local_registry_analysis_producer",
    "create_optional_registry_analysis_producer",
    "create_ipfs_datasets_analysis_provider",
    "build_ipfs_datasets_analysis_provider",
    "INTENT_GRAPH_RETRIEVER_INTERFACE",
    "QUERY_NODE_INTERFACE",
    "BOUNDED_GRAPHRAG_RETRIEVER_INTERFACE_REF",
    "SCAEV031DATASETSGRAPH",
    "SCAEV031DATASETSGRAPH_COVERAGE",
    "SCAEV031DATASETSGRAPH_EVIDENCE",
    "DATASETS_GRAPH_BINDING_VERSION",
    "DATASETS_GRAPH_PROBE_SCHEMA",
    "DATASETS_GRAPH_CAPABILITY_SCHEMA",
    "DATASETS_GRAPH_CANARY_SCHEMA",
    "DATASETS_GRAPH_CYPHER_RECEIPT_SCHEMA",
    "DATASETS_GRAPH_CONTEXT_AUTHORITY",
    "DATASETS_GRAPH_SYNTAX_ONLY",
    "EXACT_GRAPHRAG_RETRIEVAL_MODULE",
    "EXACT_CYPHER_AST_MODULE",
    "EXACT_CYPHER_PARSER_MODULE",
    "PACKAGE_ROOT_FALLBACK_MODULES",
    "DATASETS_GRAPH_BACKEND_SPECS",
    "DatasetsGraphBackendKind",
    "DatasetsGraphBackendError",
    "DatasetsGraphSymbolSpec",
    "DatasetsGraphBackendSpec",
    "DatasetsGraphSymbolReceipt",
    "DatasetsGraphBackendProbe",
    "ExactDatasetsGraphRAGAdapter",
    "probe_datasets_graph_backend",
    "probe_all_datasets_graph_backends",
    "inspect_exact_datasets_graph_capability",
    "admit_exact_datasets_graph_source",
    "parse_cypher_query_ast",
    "run_intent_graph_retriever_canary",
    "run_datasets_graph_real_module_canary",
    "create_exact_datasets_graphrag_adapter",
]
