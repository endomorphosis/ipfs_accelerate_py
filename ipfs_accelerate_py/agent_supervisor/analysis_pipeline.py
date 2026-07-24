"""Authority-safe integration of supervisor analysis, retrieval, and caching.

The pipeline composes the existing :mod:`analysis_cache`,
:mod:`analysis_ast_index`, :mod:`analysis_retrieval`, and
:mod:`analysis_contracts` implementations.  Cache authority is deliberately
stricter than finding a historically successful entry: an entry is reused
only after an exact seven-dimension key lookup, expiry checking, compact
receipt verification, content-addressed packet loading, and a second binding
check against the current request.

Optional ``ipfs_datasets_py`` analysis is advisory.  It can enrich the context
given to a local analyzer, but it cannot manufacture a conclusive supervisor
receipt and its absence never changes local cache semantics.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from copy import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .analysis_ast_index import AnalysisASTIndex, build_analysis_ast_index
from .analysis_cache import (
    ANALYSIS_CACHE_ENTRY_SCHEMA,
    AnalysisCache,
    AnalysisCacheKey,
    AnalysisCacheLookupResult,
    AnalysisCacheLookupStatus,
    AnalysisOutcome as CacheOutcome,
    build_analysis_cache_key,
    canonical_analysis_json,
    digest_analysis_input,
)
from .analysis_contracts import (
    ANALYSIS_CONTRACT_VERSION,
    AnalysisCacheDisposition,
    AnalysisEvidencePacket,
    AnalysisFreshness,
    AnalysisOutcome as ContractOutcome,
    AnalysisStageReceipt,
    AnalysisStageStatus,
)
from .analysis_retrieval import (
    RetrievalLimits,
    RetrievalQuery,
    RetrievalResponse,
    retrieve_analysis_evidence,
)
from .ipfs_datasets_analysis_provider import (
    AnalysisProviderPolicy,
    AnalysisProviderRequest,
    AnalysisProviderResult,
    IpfsDatasetsAnalysisProvider,
)


EXACT_TREE_REUSE_REQUIREMENT_ID: Final = (
    "189057730455837902155591890661235220962"
)
EXACT_TREE_ANALYSIS_REUSE_REQUIREMENT_ID: Final = EXACT_TREE_REUSE_REQUIREMENT_ID
ANALYSIS_PIPELINE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/analysis-pipeline-result@1"
)
EXACT_TREE_REUSE_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/exact-tree-reuse-evidence@1"
)
ANALYSIS_PIPELINE_VERSION: Final = "analysis-pipeline@1"
DEFAULT_ANALYZER_VERSION: Final = "supervisor-integrated-analysis@1"
DEFAULT_POLICY_DIGEST: Final = digest_analysis_input(
    {"policy": "supervisor-analysis-default@1"}
)
DEFAULT_CONFIGURATION_DIGEST: Final = digest_analysis_input(
    {"configuration": "supervisor-analysis-default@1"}
)
_PACKET_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/analysis-packet-artifact@1"
)


class AnalysisPipelineError(RuntimeError):
    """Base class for integrated analysis failures."""


class AnalysisBindingError(AnalysisPipelineError, ValueError):
    """A producer or cache artifact is not bound to the active request."""


class AnalysisProducerError(AnalysisPipelineError):
    """A local analyzer failed without returning a typed receipt."""


class PipelineCacheStatus(str, Enum):
    """Observable source of a pipeline result."""

    EXACT_HIT = "exact_hit"
    PRODUCED = "produced"
    JOINED = "joined"
    INCONCLUSIVE = "inconclusive"


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    normalized = value.strip()
    if "\x00" in normalized:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return normalized


def _canonical_digest(value: Any, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return default
        if normalized.startswith(("sha256:", "analysis-")):
            return normalized
    return digest_analysis_input(value)


def _json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_analysis_json(value).encode("utf-8")
    ).hexdigest()


def _identity_projection(value: Any) -> Any:
    """Project authority-changing inputs without executing opaque objects."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _identity_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        projected = [_identity_projection(item) for item in value]
        if isinstance(value, (set, frozenset)):
            projected.sort(key=canonical_analysis_json)
        return projected
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _identity_projection(getattr(value, item.name))
            for item in fields(value)
        }
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _identity_projection(converter())
    descriptor = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    for name in (
        "content_id",
        "index_id",
        "provider_id",
        "provider_version",
        "version",
        "__version__",
        "cache_identity",
    ):
        item = getattr(value, name, None)
        if item not in (None, "") and not callable(item):
            descriptor[name] = _identity_projection(item)
    if callable(value):
        descriptor["callable"] = (
            f"{getattr(value, '__module__', type(value).__module__)}."
            f"{getattr(value, '__qualname__', type(value).__qualname__)}"
        )
    return descriptor


@dataclass(frozen=True)
class AnalysisPipelinePolicy:
    """Hard bounds and optional-provider controls for one pipeline."""

    retrieval_limits: RetrievalLimits = field(default_factory=RetrievalLimits)
    enable_datasets_provider: bool = True
    require_local_completion: bool = True
    cache_negative_results: bool = True
    negative_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        limits = RetrievalLimits.from_value(self.retrieval_limits)
        object.__setattr__(self, "retrieval_limits", limits)
        for name in (
            "enable_datasets_provider",
            "require_local_completion",
            "cache_negative_results",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            isinstance(self.negative_ttl_seconds, bool)
            or int(self.negative_ttl_seconds) < 1
        ):
            raise ValueError("negative_ttl_seconds must be a positive integer")
        object.__setattr__(
            self, "negative_ttl_seconds", int(self.negative_ttl_seconds)
        )

    @classmethod
    def from_value(
        cls, value: "AnalysisPipelinePolicy | Mapping[str, Any] | None"
    ) -> "AnalysisPipelinePolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("analysis pipeline policy must be a mapping")
        values = dict(value)
        if "retrieval_limits" in values:
            values["retrieval_limits"] = RetrievalLimits.from_value(
                values["retrieval_limits"]
            )
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown analysis pipeline policy fields: " + ", ".join(unknown)
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_limits": self.retrieval_limits.to_dict(),
            "enable_datasets_provider": self.enable_datasets_provider,
            "require_local_completion": self.require_local_completion,
            "cache_negative_results": self.cache_negative_results,
            "negative_ttl_seconds": self.negative_ttl_seconds,
        }


@dataclass(frozen=True)
class AnalysisPipelineRequest:
    """All authority-bearing inputs to an integrated analysis run."""

    repository_id: str
    tree_id: str
    objective_revision: str
    query: RetrievalQuery | str | Mapping[str, Any]
    analyzer_id: str = "supervisor.integrated_analysis"
    analyzer_version: str = DEFAULT_ANALYZER_VERSION
    schema_version: str = str(ANALYSIS_CONTRACT_VERSION)
    configuration: Any = None
    policy: Any = None
    configuration_digest: str = ""
    query_digest: str = ""
    policy_digest: str = ""
    ast_records: Any = ()
    previous_ast_index: AnalysisASTIndex | Mapping[str, Any] | None = None
    retrieval_inputs: Mapping[str, Any] = field(default_factory=dict)
    provider_operation: str = "graph_retrieval"
    provider_payload: Mapping[str, Any] = field(default_factory=dict)
    pipeline_policy_digest: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_revision",
            "analyzer_id",
            "analyzer_version",
            "schema_version",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        query = RetrievalQuery.from_value(self.query)
        object.__setattr__(self, "query", query)
        declared_configuration = (
            self.configuration_digest
            if self.configuration_digest
            else self.configuration
        )
        object.__setattr__(
            self,
            "configuration_digest",
            digest_analysis_input(
                {
                    "declared_configuration": _identity_projection(
                        declared_configuration
                    ),
                    "analyzer_id": self.analyzer_id,
                    "ast_records": _identity_projection(self.ast_records),
                    "previous_ast_index": _identity_projection(
                        self.previous_ast_index
                    ),
                    "retrieval_inputs": _identity_projection(
                        self.retrieval_inputs
                    ),
                    "provider_operation": self.provider_operation,
                    "provider_payload": _identity_projection(
                        self.provider_payload
                    ),
                }
            ),
        )
        declared_query = self.query_digest
        object.__setattr__(
            self,
            "query_digest",
            digest_analysis_input(
                {
                    "declared_query_digest": declared_query,
                    "query": query.to_dict(),
                }
            ),
        )
        declared_policy = self.policy_digest if self.policy_digest else self.policy
        object.__setattr__(
            self,
            "policy_digest",
            _canonical_digest(
                declared_policy,
                default=DEFAULT_POLICY_DIGEST,
            ),
        )
        object.__setattr__(
            self,
            "pipeline_policy_digest",
            _canonical_digest(
                self.pipeline_policy_digest,
                default=DEFAULT_POLICY_DIGEST,
            ),
        )
        if not isinstance(self.retrieval_inputs, Mapping):
            raise TypeError("retrieval_inputs must be a mapping")
        if not isinstance(self.provider_payload, Mapping):
            raise TypeError("provider_payload must be a mapping")
        object.__setattr__(
            self, "retrieval_inputs", dict(self.retrieval_inputs)
        )
        object.__setattr__(self, "provider_payload", dict(self.provider_payload))
        object.__setattr__(
            self,
            "provider_operation",
            _required_text(self.provider_operation, "provider_operation"),
        )

    @property
    def effective_policy_digest(self) -> str:
        return digest_analysis_input(
            {
                "request_policy_digest": self.policy_digest,
                "pipeline_policy_digest": self.pipeline_policy_digest,
            }
        )

    def bind_pipeline_policy(self, value: Any) -> "AnalysisPipelineRequest":
        digest = digest_analysis_input(_identity_projection(value))
        if digest == self.pipeline_policy_digest:
            return self
        # __post_init__ folds declared digests into actual inputs exactly once.
        # A normal dataclass replacement would fold the derived values again.
        bound = copy(self)
        object.__setattr__(bound, "pipeline_policy_digest", digest)
        return bound

    @property
    def cache_key(self) -> AnalysisCacheKey:
        return build_analysis_cache_key(
            repository_tree_identity={
                "repository_id": self.repository_id,
                "tree_id": self.tree_id,
            },
            objective_revision=self.objective_revision,
            analyzer_version=self.analyzer_version,
            schema_version=self.schema_version,
            configuration_digest=self.configuration_digest,
            query_digest=self.query_digest,
            policy_digest=self.effective_policy_digest,
        )

    @property
    def request_id(self) -> str:
        return "analysis-request:sha256:" + hashlib.sha256(
            canonical_analysis_json(self.identity_dict()).encode("utf-8")
        ).hexdigest()

    def identity_dict(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "policy_digest": self.effective_policy_digest,
            "query_id": self.query.query_id,
        }


AnalysisRequest = AnalysisPipelineRequest
AnalysisPolicy = AnalysisPipelinePolicy


@dataclass(frozen=True)
class AnalysisStageContext:
    """Bounded local inputs assembled before invoking the analyzer."""

    request: AnalysisPipelineRequest
    ast_index: AnalysisASTIndex | None
    retrieval: RetrievalResponse
    provider_result: Any = None
    provider_evidence_claim_references: tuple[str, ...] = ()
    provider_request: AnalysisProviderRequest | None = field(
        default=None, repr=False, compare=False
    )
    provider_policy: AnalysisProviderPolicy | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class OptionalProviderFailure:
    """Non-authoritative projection when an optional adapter itself raises."""

    repository_id: str
    tree_id: str
    objective_revision: str
    reason_code: str = "optional_provider_invocation_failed"
    status: str = "failed"

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def is_completion_evidence(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason_code": self.reason_code,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "non_authoritative": True,
            "safe_for_completion_reasoning": False,
        }


def _optional_provider_violation(
    value: Any,
    request: AnalysisPipelineRequest,
    provider_request: AnalysisProviderRequest | None = None,
) -> str:
    """Return a fail-closed reason for an unsafe provider projection.

    Optional providers are outside the supervisor authority boundary.  The
    pipeline therefore rejects, rather than merely ignores, a provider value
    that is rebound to another request or claims any form of completion
    authority.  The rejection itself remains advisory so healthy local
    analysis can continue.
    """

    try:
        if isinstance(value, Mapping):
            projected: Mapping[str, Any] = value
        else:
            converter = getattr(value, "to_dict", None)
            projected = converter() if callable(converter) else {}
            if inspect.isawaitable(projected):
                _dispose_optional_awaitable(projected)
                return "optional_provider_inspection_failed"
            if not isinstance(projected, Mapping):
                return "optional_provider_inspection_failed"
    except Exception:
        return "optional_provider_inspection_failed"

    expected_bindings = {
        "repository_id": request.repository_id,
        "tree_id": request.tree_id,
        "objective_revision": request.objective_revision,
        "operation": (
            provider_request.operation.value
            if provider_request is not None
            else request.provider_operation
        ),
    }
    for name, expected in expected_bindings.items():
        try:
            actual = projected.get(name)
            if actual in (None, ""):
                actual = getattr(value, name, None)
            if isinstance(actual, Enum):
                actual = actual.value
        except Exception:
            return "optional_provider_inspection_failed"
        if actual in (None, ""):
            return "optional_provider_identity_missing"
        if actual != expected:
            return "optional_provider_identity_mismatch"

    if provider_request is not None:
        try:
            actual_request_id = projected.get("request_id")
            if actual_request_id in (None, ""):
                actual_request_id = getattr(value, "request_id", None)
        except Exception:
            return "optional_provider_inspection_failed"
        if actual_request_id in (None, ""):
            return "optional_provider_identity_missing"
        if actual_request_id != provider_request.request_id:
            return "optional_provider_request_mismatch"

    for name in (
        "authoritative",
        "completion_authority",
        "is_completion_evidence",
        "proof_success",
        "safe_for_completion_reasoning",
    ):
        try:
            claim = projected.get(name)
            if claim is None:
                claim = getattr(value, name, None)
            if callable(claim):
                # An optional object exposing an authority method is itself an
                # attempted authority surface.  Do not execute untrusted code
                # to ask whether that claim happens to return true.
                return "optional_provider_authority_claim_rejected"
            if inspect.isawaitable(claim):
                _dispose_optional_awaitable(claim)
                return "optional_provider_inspection_failed"
        except Exception:
            return "optional_provider_inspection_failed"
        if claim is not None and claim is not False:
            return "optional_provider_authority_claim_rejected"
    return ""


def _dispose_optional_awaitable(value: Any) -> None:
    """Best-effort disposal of an unsupported optional-provider awaitable."""

    try:
        close = getattr(value, "close", None)
    except Exception:
        close = None
    if callable(close):
        try:
            close()
        except Exception:
            pass
        return
    try:
        cancel = getattr(value, "cancel", None)
    except Exception:
        cancel = None
    if callable(cancel):
        try:
            cancel()
        except Exception:
            pass


@dataclass(frozen=True)
class ExactTreeReuseEvidence:
    """Content-addressed witness for an authoritative exact-key cache hit."""

    request_id: str
    cache_key_id: str
    packet_id: str
    cache_entry_digest: str
    repository_id: str
    tree_id: str
    objective_revision: str
    analyzer_version: str
    schema_version: str
    configuration_digest: str
    query_digest: str
    policy_digest: str
    lookup_reason: str = "exact_key_hit"
    requirement_id: str = EXACT_TREE_REUSE_REQUIREMENT_ID

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "cache_key_id",
            "packet_id",
            "cache_entry_digest",
            "repository_id",
            "tree_id",
            "objective_revision",
            "analyzer_version",
            "schema_version",
            "configuration_digest",
            "query_digest",
            "policy_digest",
            "lookup_reason",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        if self.requirement_id != EXACT_TREE_REUSE_REQUIREMENT_ID:
            raise AnalysisBindingError("unexpected exact-tree requirement ID")
        if self.lookup_reason != "exact_key_hit":
            raise AnalysisBindingError(
                "exact-tree evidence requires an exact authoritative cache hit"
            )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": EXACT_TREE_REUSE_EVIDENCE_SCHEMA,
            "requirement_id": self.requirement_id,
            "request_id": self.request_id,
            "cache_key_id": self.cache_key_id,
            "packet_id": self.packet_id,
            "cache_entry_digest": self.cache_entry_digest,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "analyzer_version": self.analyzer_version,
            "schema_version": self.schema_version,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "policy_digest": self.policy_digest,
            "lookup_reason": self.lookup_reason,
        }

    @property
    def evidence_id(self) -> str:
        return "exact-tree-reuse:sha256:" + hashlib.sha256(
            canonical_analysis_json(self._content()).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "evidence_id": self.evidence_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactTreeReuseEvidence":
        """Restore a witness while checking schema and content identity."""

        if not isinstance(value, Mapping):
            raise AnalysisBindingError("exact-tree evidence must be an object")
        allowed = {
            "schema",
            "evidence_id",
            "requirement_id",
            "request_id",
            "cache_key_id",
            "packet_id",
            "cache_entry_digest",
            "repository_id",
            "tree_id",
            "objective_revision",
            "analyzer_version",
            "schema_version",
            "configuration_digest",
            "query_digest",
            "policy_digest",
            "lookup_reason",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise AnalysisBindingError(
                "exact-tree evidence has unknown fields: "
                + ", ".join(unknown)
            )
        if value.get("schema") != EXACT_TREE_REUSE_EVIDENCE_SCHEMA:
            raise AnalysisBindingError("unsupported exact-tree evidence schema")
        restored = cls(
            request_id=value.get("request_id", ""),
            cache_key_id=value.get("cache_key_id", ""),
            packet_id=value.get("packet_id", ""),
            cache_entry_digest=value.get("cache_entry_digest", ""),
            repository_id=value.get("repository_id", ""),
            tree_id=value.get("tree_id", ""),
            objective_revision=value.get("objective_revision", ""),
            analyzer_version=value.get("analyzer_version", ""),
            schema_version=value.get("schema_version", ""),
            configuration_digest=value.get("configuration_digest", ""),
            query_digest=value.get("query_digest", ""),
            policy_digest=value.get("policy_digest", ""),
            lookup_reason=value.get("lookup_reason", ""),
            requirement_id=value.get("requirement_id", ""),
        )
        if value.get("evidence_id") != restored.evidence_id:
            raise AnalysisBindingError(
                "exact-tree evidence identity does not match its content"
            )
        return restored

    @classmethod
    def from_lookup(
        cls,
        request: AnalysisPipelineRequest,
        packet: AnalysisEvidencePacket,
        lookup: AnalysisCacheLookupResult,
    ) -> "ExactTreeReuseEvidence":
        """Build a witness only from a verified, exact completion lookup."""

        entry = (
            lookup.entry
            if isinstance(lookup, AnalysisCacheLookupResult)
            else None
        )
        if entry is None:
            raise AnalysisBindingError(
                "exact-tree evidence requires a typed cache entry"
            )
        witness = cls(
            request_id=request.request_id,
            cache_key_id=request.cache_key.key_id,
            packet_id=packet.packet_id,
            cache_entry_digest=entry.entry_digest,
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_revision=request.objective_revision,
            analyzer_version=request.analyzer_version,
            schema_version=request.schema_version,
            configuration_digest=request.configuration_digest,
            query_digest=request.query_digest,
            policy_digest=request.effective_policy_digest,
            lookup_reason=lookup.reason_code,
        )
        if not witness.proves_for(request, packet, lookup):
            raise AnalysisBindingError(
                "exact-tree evidence is not bound to the verified cache lookup"
            )
        return witness

    def proves_for(
        self,
        request: AnalysisPipelineRequest,
        packet: AnalysisEvidencePacket,
        lookup: AnalysisCacheLookupResult,
    ) -> bool:
        """Independently verify this witness against its authority sources."""

        if (
            not isinstance(request, AnalysisPipelineRequest)
            or not isinstance(packet, AnalysisEvidencePacket)
            or not isinstance(lookup, AnalysisCacheLookupResult)
            or lookup.status is not AnalysisCacheLookupStatus.HIT
            or lookup.key != request.cache_key
            or not lookup.is_completion_evidence
            or lookup.reason_code != "exact_key_hit"
            or lookup.entry is None
            or not packet.safe_for_completion_reasoning
        ):
            return False
        entry = lookup.entry
        if (
            not entry.entry_digest
            or entry.entry_digest != entry.computed_digest
            or self.cache_entry_digest != entry.entry_digest
        ):
            return False
        try:
            _validate_packet_binding(packet, request)
        except (TypeError, ValueError, AnalysisPipelineError):
            return False
        summary = entry.receipt.get("summary")
        if not isinstance(summary, Mapping):
            return False
        expected_summary = {
            "repository_id": request.repository_id,
            "tree_id": request.tree_id,
            "objective_revision": request.objective_revision,
            "analyzer_version": request.analyzer_version,
            "schema_version": request.schema_version,
            "configuration_digest": request.configuration_digest,
            "query_digest": request.query_digest,
            "policy_digest": request.effective_policy_digest,
            "packet_id": packet.packet_id,
            "safe_for_completion_reasoning": True,
        }
        if any(
            summary.get(name) != expected
            for name, expected in expected_summary.items()
        ):
            return False
        expected_witness = {
            "request_id": request.request_id,
            "cache_key_id": request.cache_key.key_id,
            "packet_id": packet.packet_id,
            "cache_entry_digest": entry.entry_digest,
            "repository_id": request.repository_id,
            "tree_id": request.tree_id,
            "objective_revision": request.objective_revision,
            "analyzer_version": request.analyzer_version,
            "schema_version": request.schema_version,
            "configuration_digest": request.configuration_digest,
            "query_digest": request.query_digest,
            "policy_digest": request.effective_policy_digest,
            "lookup_reason": "exact_key_hit",
            "requirement_id": EXACT_TREE_REUSE_REQUIREMENT_ID,
        }
        return all(
            getattr(self, name) == expected
            for name, expected in expected_witness.items()
        )


@dataclass(frozen=True)
class AnalysisPipelineMetrics:
    requests: int = 0
    exact_hits: int = 0
    produced: int = 0
    joined: int = 0
    invalidated: int = 0
    negative_rejections: int = 0
    stale_authoritative_hits: int = 0

    @property
    def reuse_ratio(self) -> float:
        return self.exact_hits / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "exact_hits": self.exact_hits,
            "produced": self.produced,
            "joined": self.joined,
            "invalidated": self.invalidated,
            "negative_rejections": self.negative_rejections,
            "stale_authoritative_hits": self.stale_authoritative_hits,
            "reuse_ratio": self.reuse_ratio,
        }


@dataclass(frozen=True)
class AnalysisPipelineResult:
    """One integrated analysis result and its authority decision."""

    request: AnalysisPipelineRequest
    packet: AnalysisEvidencePacket
    cache_status: PipelineCacheStatus
    cache_lookup_status: AnalysisCacheLookupStatus
    cache_reason_codes: tuple[str, ...] = ()
    producer_executed: bool = False
    joined_existing_flight: bool = False
    exact_tree_reuse_evidence: ExactTreeReuseEvidence | None = None
    cache_lookup: AnalysisCacheLookupResult | None = field(
        default=None, repr=False, compare=False
    )
    ast_index_id: str = ""
    retrieval_response_id: str = ""
    provider_result: Any = None
    advisory_evidence_claim_references: tuple[str, ...] = ()
    provider_request: AnalysisProviderRequest | None = field(
        default=None, repr=False, compare=False
    )
    provider_policy: AnalysisProviderPolicy | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.cache_status, PipelineCacheStatus):
            object.__setattr__(
                self, "cache_status", PipelineCacheStatus(str(self.cache_status))
            )
        if not isinstance(
            self.cache_lookup_status, AnalysisCacheLookupStatus
        ):
            object.__setattr__(
                self,
                "cache_lookup_status",
                AnalysisCacheLookupStatus(str(self.cache_lookup_status)),
            )
        object.__setattr__(
            self,
            "cache_reason_codes",
            tuple(
                _required_text(item, "cache reason code")
                for item in self.cache_reason_codes
            ),
        )
        advisory_claims = tuple(
            dict.fromkeys(
                _required_text(item, "advisory evidence claim reference")
                for item in self.advisory_evidence_claim_references
            )
        )
        object.__setattr__(
            self, "advisory_evidence_claim_references", advisory_claims
        )
        if advisory_claims:
            if not isinstance(self.provider_result, AnalysisProviderResult):
                raise AnalysisBindingError(
                    "advisory provider claims require a typed provider result"
                )
            if not isinstance(self.provider_request, AnalysisProviderRequest):
                raise AnalysisBindingError(
                    "advisory provider claims require the exact provider request"
                )
            if not isinstance(self.provider_policy, AnalysisProviderPolicy):
                raise AnalysisBindingError(
                    "advisory provider claims require the provider policy"
                )
            provider_request = self.provider_request
            try:
                expected_provider_request = AnalysisProviderRequest(
                    operation=self.request.provider_operation,
                    repository_id=self.request.repository_id,
                    tree_id=self.request.tree_id,
                    objective_revision=self.request.objective_revision,
                    query=self.request.query,
                    payload=self.request.provider_payload,
                    bounds=provider_request.bounds,
                )
            except (TypeError, ValueError) as exc:
                raise AnalysisBindingError(
                    "pipeline request cannot reproduce the provider request"
                ) from exc
            if expected_provider_request.request_id != provider_request.request_id:
                raise AnalysisBindingError(
                    "provider request is detached from pipeline request"
                )
            policy_bounds = self.provider_policy.bounds
            if any(
                getattr(provider_request.bounds, name)
                > getattr(policy_bounds, name)
                for name in provider_request.bounds.__dataclass_fields__
            ):
                raise AnalysisBindingError(
                    "provider request bounds expand the provider policy"
                )
            proved = tuple(
                self.provider_result.proved_requirement_ids_for(
                    provider_request, self.provider_policy
                )
            )
            if any(item not in proved for item in advisory_claims):
                raise AnalysisBindingError(
                    "advisory provider claims are not active-request bound"
                )
            if EXACT_TREE_REUSE_REQUIREMENT_ID in advisory_claims:
                raise AnalysisBindingError(
                    "exact-tree authority cannot be an advisory provider claim"
                )
        _validate_packet_binding(self.packet, self.request)
        witness = self.exact_tree_reuse_evidence
        if self.cache_status is PipelineCacheStatus.EXACT_HIT:
            if witness is None:
                raise AnalysisBindingError(
                    "exact cache hits require exact-tree reuse evidence"
                )
            if self.cache_lookup_status is not AnalysisCacheLookupStatus.HIT:
                raise AnalysisBindingError(
                    "exact cache hits require an authoritative HIT lookup"
                )
            if "exact_key_hit" not in self.cache_reason_codes:
                raise AnalysisBindingError(
                    "exact cache hits require the exact_key_hit reason"
                )
            if self.producer_executed or self.joined_existing_flight:
                raise AnalysisBindingError(
                    "exact cache hits cannot claim producer or follower work"
                )
            lookup = self.cache_lookup
            if not isinstance(lookup, AnalysisCacheLookupResult):
                raise AnalysisBindingError(
                    "exact cache hits require the typed authoritative lookup"
                )
            if (
                lookup.status is not self.cache_lookup_status
                or tuple(lookup.reason_codes) != self.cache_reason_codes
            ):
                raise AnalysisBindingError(
                    "exact cache hit diagnostics are detached from the lookup"
                )
            if not self.packet.safe_for_completion_reasoning:
                raise AnalysisBindingError(
                    "inconclusive packets cannot carry exact-tree reuse evidence"
                )
            if not witness.proves_for(self.request, self.packet, lookup):
                raise AnalysisBindingError(
                    "exact-tree evidence is not bound to the typed lookup"
                )
        elif witness is not None:
            if self.cache_status is not PipelineCacheStatus.EXACT_HIT:
                raise AnalysisBindingError(
                    "exact-tree evidence is only valid for exact cache hits"
                )
        if (
            self.cache_status is PipelineCacheStatus.JOINED
            and not self.joined_existing_flight
        ):
            raise AnalysisBindingError("joined results must record flight joining")

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.packet.safe_for_completion_reasoning

    @property
    def reused(self) -> bool:
        return self.cache_status is PipelineCacheStatus.EXACT_HIT

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        """Established completion-authority evidence channel."""

        return self.authoritative_evidence_claim_references

    @property
    def all_evidence_claim_references(self) -> tuple[str, ...]:
        """All claims, including non-authoritative provider diagnostics."""

        authoritative = self.evidence_claim_references
        return tuple(
            dict.fromkeys(
                (*authoritative, *self.advisory_evidence_claim_references)
            )
        )

    @property
    def authoritative_evidence_claim_references(self) -> tuple[str, ...]:
        """Claims that participate in supervisor completion authority."""

        if self.exact_tree_reuse_evidence is None:
            return ()
        return (self.exact_tree_reuse_evidence.requirement_id,)

    @property
    def result_id(self) -> str:
        payload = self.to_dict(include_result_id=False)
        return "analysis-pipeline-result:sha256:" + hashlib.sha256(
            canonical_analysis_json(payload).encode("utf-8")
        ).hexdigest()

    def to_dict(self, *, include_result_id: bool = True) -> dict[str, Any]:
        provider = self.provider_result
        converter = getattr(provider, "to_dict", None)
        if callable(converter):
            provider = converter()
        payload = {
            "schema": ANALYSIS_PIPELINE_SCHEMA,
            "request_id": self.request.request_id,
            "cache_key_id": self.request.cache_key.key_id,
            "packet": self.packet.to_record(),
            "cache_status": self.cache_status.value,
            "cache_lookup_status": self.cache_lookup_status.value,
            "cache_reason_codes": list(self.cache_reason_codes),
            "producer_executed": self.producer_executed,
            "joined_existing_flight": self.joined_existing_flight,
            "exact_tree_reuse_evidence": (
                self.exact_tree_reuse_evidence.to_dict()
                if self.exact_tree_reuse_evidence is not None
                else None
            ),
            "authoritative_evidence_claim_references": list(
                self.authoritative_evidence_claim_references
            ),
            "advisory_evidence_claim_references": list(
                self.advisory_evidence_claim_references
            ),
            "evidence_claim_references": list(self.evidence_claim_references),
            "all_evidence_claim_references": list(
                self.all_evidence_claim_references
            ),
            "ast_index_id": self.ast_index_id,
            "retrieval_response_id": self.retrieval_response_id,
            "provider_result": provider,
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
        }
        if include_result_id:
            payload["result_id"] = self.result_id
        return payload


def _validate_packet_binding(
    packet: AnalysisEvidencePacket, request: AnalysisPipelineRequest
) -> None:
    expected = {
        "repository_id": request.repository_id,
        "tree_id": request.tree_id,
        "objective_revision": request.objective_revision,
    }
    for name, value in expected.items():
        if getattr(packet, name) != value:
            raise AnalysisBindingError(
                f"analysis packet {name} is not bound to the active request"
            )
    for receipt in packet.stage_receipts:
        bindings = {
            "analyzer_version": request.analyzer_version,
            "configuration_digest": request.configuration_digest,
            "query_digest": request.query_digest,
            "policy_digest": request.effective_policy_digest,
        }
        for name, value in bindings.items():
            if getattr(receipt, name) != value:
                raise AnalysisBindingError(
                    f"analysis receipt {name} is not bound to the active request"
                )


class _PacketArtifactStore:
    """Small content-addressed store for packet bodies excluded from the cache."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, digest: str) -> Path:
        value = digest.removeprefix("sha256:")
        return self.path / value[:2] / f"{value}.json"

    def put(self, packet: AnalysisEvidencePacket) -> dict[str, str]:
        payload = {
            "schema": _PACKET_ARTIFACT_SCHEMA,
            "packet": packet.to_record(),
        }
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        path = self._path(digest)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not path.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
        return {
            "artifact_id": packet.packet_id,
            "digest": digest,
            "path": str(path),
            "schema": _PACKET_ARTIFACT_SCHEMA,
        }

    def get(self, reference: Mapping[str, Any]) -> AnalysisEvidencePacket:
        digest = _required_text(reference.get("digest"), "artifact digest")
        path = Path(_required_text(reference.get("path"), "artifact path"))
        expected = self._path(digest).resolve()
        if path.resolve() != expected:
            raise AnalysisBindingError("packet artifact path is not content addressed")
        raw = path.read_bytes()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
            raise AnalysisBindingError("packet artifact digest mismatch")
        payload = json.loads(raw)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != _PACKET_ARTIFACT_SCHEMA
            or not isinstance(payload.get("packet"), Mapping)
        ):
            raise AnalysisBindingError("malformed analysis packet artifact")
        packet = AnalysisEvidencePacket.from_dict(payload["packet"])
        if reference.get("artifact_id") != packet.packet_id:
            raise AnalysisBindingError("packet artifact identity mismatch")
        return packet


def _cache_receipt(
    packet: AnalysisEvidencePacket,
    artifact: Mapping[str, str],
    request: AnalysisPipelineRequest,
) -> dict[str, Any]:
    successful = packet.safe_for_completion_reasoning
    return {
        "status": (
            CacheOutcome.SUCCESSFUL.value
            if successful
            else CacheOutcome.INCONCLUSIVE.value
        ),
        "receipt_id": packet.packet_id,
        "summary": {
            "repository_id": request.repository_id,
            "tree_id": request.tree_id,
            "objective_revision": request.objective_revision,
            "analyzer_version": request.analyzer_version,
            "schema_version": request.schema_version,
            "configuration_digest": request.configuration_digest,
            "query_digest": request.query_digest,
            "policy_digest": request.effective_policy_digest,
            "packet_id": packet.packet_id,
            "safe_for_completion_reasoning": successful,
        },
        "artifact_refs": [dict(artifact)],
    }


def _packet_from_producer(
    value: Any, request: AnalysisPipelineRequest
) -> AnalysisEvidencePacket:
    if isinstance(value, AnalysisEvidencePacket):
        packet = value
    elif isinstance(value, AnalysisStageReceipt):
        packet = AnalysisEvidencePacket(
            repository_id=request.repository_id,
            tree_id=request.tree_id,
            objective_revision=request.objective_revision,
            outcome=(
                ContractOutcome.CONCLUSIVE
                if value.safe_for_completion_reasoning
                else ContractOutcome.INCONCLUSIVE
            ),
            conclusion_code=(
                "local_analysis_complete"
                if value.safe_for_completion_reasoning
                else "local_analysis_inconclusive"
            ),
            coverage_complete=value.coverage_complete,
            truncated=value.truncated,
            stage_receipts=(value,),
        )
    elif isinstance(value, Mapping):
        candidate = value.get("packet", value)
        if isinstance(candidate, AnalysisEvidencePacket):
            packet = candidate
        elif isinstance(candidate, Mapping) and (
            candidate.get("schema") == AnalysisEvidencePacket.SCHEMA
            or "stage_receipts" in candidate
        ):
            packet = AnalysisEvidencePacket.from_dict(candidate)
        else:
            receipt_candidate = value.get("receipt", value)
            if isinstance(receipt_candidate, AnalysisStageReceipt):
                receipt = receipt_candidate
            elif isinstance(receipt_candidate, Mapping) and (
                receipt_candidate.get("schema") == AnalysisStageReceipt.SCHEMA
                or "analyzer_id" in receipt_candidate
            ):
                receipt = AnalysisStageReceipt.from_dict(receipt_candidate)
            else:
                raise AnalysisProducerError(
                    "analysis producer mapping must contain a typed packet or receipt"
                )
            return _packet_from_producer(receipt, request)
    else:
        raise AnalysisProducerError(
            "analysis producer must return AnalysisEvidencePacket or "
            "AnalysisStageReceipt"
        )
    _validate_packet_binding(packet, request)
    return packet


def make_analysis_stage_receipt(
    request: AnalysisPipelineRequest,
    *,
    successful: bool,
    coverage_complete: bool = True,
    truncated: bool = False,
    reason_code: str = "",
    error_code: str = "",
    stage: str = "integrated_local_analysis",
    **values: Any,
) -> AnalysisStageReceipt:
    """Build a correctly request-bound receipt for simple local analyzers."""

    status = (
        AnalysisStageStatus.COMPLETED
        if successful
        else AnalysisStageStatus.FAILED
    )
    outcome = (
        ContractOutcome.CONCLUSIVE
        if successful and coverage_complete and not truncated and not error_code
        else ContractOutcome.INCONCLUSIVE
    )
    return AnalysisStageReceipt(
        stage=stage,
        status=status,
        outcome=outcome,
        analyzer_id=request.analyzer_id,
        analyzer_version=request.analyzer_version,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_revision=request.objective_revision,
        configuration_digest=request.configuration_digest,
        query_digest=request.query_digest,
        policy_digest=request.effective_policy_digest,
        freshness=AnalysisFreshness.FRESH,
        cache_disposition=AnalysisCacheDisposition.MISS,
        coverage_complete=coverage_complete,
        truncated=truncated,
        reason_code=reason_code,
        error_code=error_code,
        **values,
    )


class AnalysisPipeline:
    """Run and cache bounded integrated analysis with fail-closed authority."""

    def __init__(
        self,
        cache: AnalysisCache,
        analyzer: Callable[[AnalysisStageContext], Any] | Any,
        *,
        provider: Any = None,
        coordinator: Any = None,
        policy: AnalysisPipelinePolicy | Mapping[str, Any] | None = None,
        artifact_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not isinstance(cache, AnalysisCache):
            raise TypeError("cache must be an AnalysisCache")
        if analyzer is None:
            raise ValueError("a local analyzer is required")
        self.cache = cache
        self.analyzer = analyzer
        self.provider = provider
        self.policy = AnalysisPipelinePolicy.from_value(policy)
        self._execution_identity = {
            "pipeline_version": ANALYSIS_PIPELINE_VERSION,
            "pipeline_policy": self.policy.to_dict(),
            "analyzer": _identity_projection(analyzer),
            "provider": _identity_projection(provider),
            "provider_policy": _identity_projection(
                getattr(provider, "policy", None)
            ),
        }
        if coordinator is None:
            from .cache_coordinator import AnalysisCacheCoordinator

            coordinator = AnalysisCacheCoordinator(cache)
        self.coordinator = coordinator
        root = (
            Path(artifact_path)
            if artifact_path is not None
            else cache.path / "pipeline-artifacts"
        )
        self._artifacts = _PacketArtifactStore(root)
        self._metrics = {
            "requests": 0,
            "exact_hits": 0,
            "produced": 0,
            "joined": 0,
            "invalidated": 0,
            "negative_rejections": 0,
            "stale_authoritative_hits": 0,
        }

    @property
    def metrics(self) -> AnalysisPipelineMetrics:
        return AnalysisPipelineMetrics(**self._metrics)

    def _load_packet_receipt(
        self,
        receipt: Mapping[str, Any],
        request: AnalysisPipelineRequest,
        *,
        require_completion_evidence: bool,
    ) -> AnalysisEvidencePacket:
        if not isinstance(receipt, Mapping):
            raise AnalysisBindingError("cache result has no compact receipt")
        summary = receipt.get("summary")
        references = receipt.get("artifact_refs")
        if not isinstance(summary, Mapping) or not isinstance(references, Sequence):
            raise AnalysisBindingError("cache hit lacks packet binding metadata")
        expected = {
            "repository_id": request.repository_id,
            "tree_id": request.tree_id,
            "objective_revision": request.objective_revision,
            "analyzer_version": request.analyzer_version,
            "schema_version": request.schema_version,
            "configuration_digest": request.configuration_digest,
            "query_digest": request.query_digest,
            "policy_digest": request.effective_policy_digest,
        }
        if any(summary.get(name) != value for name, value in expected.items()):
            raise AnalysisBindingError("cached packet summary is stale")
        if len(references) != 1 or not isinstance(references[0], Mapping):
            raise AnalysisBindingError("cache hit requires one packet artifact")
        packet = self._artifacts.get(references[0])
        _validate_packet_binding(packet, request)
        if packet.packet_id != summary.get("packet_id"):
            raise AnalysisBindingError("cached packet ID does not match receipt")
        if require_completion_evidence:
            packet.require_completion_evidence()
        return packet

    def _load_cached_packet(
        self,
        lookup: AnalysisCacheLookupResult,
        request: AnalysisPipelineRequest,
    ) -> AnalysisEvidencePacket:
        # Do not inspect lookup.entry until both derived authority gates pass.
        if not lookup.hit or not lookup.is_completion_evidence:
            raise AnalysisBindingError("cache lookup is not an authoritative hit")
        receipt = lookup.receipt
        if not isinstance(receipt, Mapping):
            raise AnalysisBindingError("cache hit has no compact receipt")
        return self._load_packet_receipt(
            receipt, request, require_completion_evidence=True
        )

    def _exact_hit_result(
        self,
        lookup: AnalysisCacheLookupResult,
        request: AnalysisPipelineRequest,
    ) -> AnalysisPipelineResult:
        """Revalidate and project one exact cache lookup as pipeline authority."""

        packet = self._load_cached_packet(lookup, request)
        witness = ExactTreeReuseEvidence.from_lookup(request, packet, lookup)
        return AnalysisPipelineResult(
            request=request,
            packet=packet,
            cache_status=PipelineCacheStatus.EXACT_HIT,
            cache_lookup_status=lookup.status,
            cache_reason_codes=tuple(lookup.reason_codes),
            exact_tree_reuse_evidence=witness,
            cache_lookup=lookup,
        )

    def _accept_completion_lookup(
        self,
        lookup: AnalysisCacheLookupResult,
        request: AnalysisPipelineRequest,
    ) -> bool:
        """Fail closed when a compact hit's external packet is unusable."""

        try:
            self._exact_hit_result(lookup, request)
        except (OSError, TypeError, ValueError, AnalysisPipelineError):
            return False
        return True

    def _build_context(
        self, request: AnalysisPipelineRequest
    ) -> AnalysisStageContext:
        ast_index: AnalysisASTIndex | None = None
        records = request.ast_records
        if records not in (None, (), [], {}):
            ast_index = build_analysis_ast_index(
                records,
                previous=request.previous_ast_index,
            )
        retrieval_values = dict(request.retrieval_inputs)
        allowed = {
            "evidence_graph",
            "records",
            "todo_records",
            "dependency_graph",
            "goal_coverage",
            "proof_scope_index",
            "vector_backend",
            "vector_embedder",
            "ast_backend",
            "artifact_id",
            "signal_weights",
        }
        unknown = sorted(set(retrieval_values) - allowed)
        if unknown:
            raise ValueError(
                "unknown retrieval input fields: " + ", ".join(unknown)
            )
        if ast_index is not None and "ast_backend" not in retrieval_values:
            # Feed compact AST query hits as retrieval records.  This avoids
            # adapting incompatible backend signatures or exposing AST bodies.
            ast_hits = ast_index.query_objective_terms(
                request.query.text,
                max_results=self.policy.retrieval_limits.max_backend_results,
                max_bytes=min(
                    self.policy.retrieval_limits.max_bytes,
                    256 * 1024,
                ),
            )
            projected = [
                {
                    "record_id": item.evidence_id,
                    "title": item.value,
                    "path": item.path,
                    "ast_symbols": [item.symbol] if item.symbol else [],
                    "artifact_id": item.record_id,
                }
                for item in ast_hits.evidence
            ]
            retrieval_values["records"] = (
                *tuple(retrieval_values.get("records") or ()),
                *projected,
            )
        retrieval = retrieve_analysis_evidence(
            request.query,
            limits=self.policy.retrieval_limits,
            **retrieval_values,
        )
        provider_result = None
        provider_evidence_claim_references: tuple[str, ...] = ()
        provider_request: AnalysisProviderRequest | None = None
        provider_policy: AnalysisProviderPolicy | None = None
        if self.provider is not None and self.policy.enable_datasets_provider:
            analyze = getattr(self.provider, "analyze", None)
            if not callable(analyze):
                analyze = (
                    self.provider if callable(self.provider) else None
                )
            if analyze is not None:
                try:
                    if isinstance(self.provider, IpfsDatasetsAnalysisProvider):
                        # Construct the adapter's exact bounded request here so
                        # its degradation witness can be independently rebound
                        # before the pipeline advertises the advisory claim.
                        provider_request = self.provider.build_request(
                            request.query,
                            operation=request.provider_operation,
                            repository_id=request.repository_id,
                            tree_id=request.tree_id,
                            objective_revision=request.objective_revision,
                            payload=request.provider_payload,
                            limits=self.policy.retrieval_limits,
                        )
                        provider_policy = self.provider.policy
                        provider_result = analyze(provider_request)
                    else:
                        provider_result = analyze(
                            request.query,
                            operation=request.provider_operation,
                            repository_id=request.repository_id,
                            tree_id=request.tree_id,
                            objective_revision=request.objective_revision,
                            payload=request.provider_payload,
                            limits=self.policy.retrieval_limits,
                        )
                except Exception:
                    provider_result = OptionalProviderFailure(
                        repository_id=request.repository_id,
                        tree_id=request.tree_id,
                        objective_revision=request.objective_revision,
                    )
                if inspect.isawaitable(provider_result):
                    _dispose_optional_awaitable(provider_result)
                    provider_result = OptionalProviderFailure(
                        repository_id=request.repository_id,
                        tree_id=request.tree_id,
                        objective_revision=request.objective_revision,
                        reason_code=(
                            "optional_provider_async_result_unsupported"
                        ),
                    )
                if isinstance(provider_result, OptionalProviderFailure):
                    violation = ""
                else:
                    try:
                        violation = _optional_provider_violation(
                            provider_result, request, provider_request
                        )
                    except Exception:
                        violation = "optional_provider_inspection_failed"
                if violation:
                    provider_result = OptionalProviderFailure(
                        repository_id=request.repository_id,
                        tree_id=request.tree_id,
                        objective_revision=request.objective_revision,
                        reason_code=violation,
                    )
                elif (
                    provider_request is not None
                    and provider_policy is not None
                    and isinstance(provider_result, AnalysisProviderResult)
                ):
                    provider_evidence_claim_references = tuple(
                        provider_result.proved_requirement_ids_for(
                            provider_request, provider_policy
                        )
                    )
        return AnalysisStageContext(
            request=request,
            ast_index=ast_index,
            retrieval=retrieval,
            provider_result=provider_result,
            provider_evidence_claim_references=(
                provider_evidence_claim_references
            ),
            provider_request=provider_request,
            provider_policy=provider_policy,
        )

    def _invoke_analyzer(
        self, context: AnalysisStageContext
    ) -> AnalysisEvidencePacket:
        analyzer = self.analyzer
        method = getattr(analyzer, "analyze", None)
        if callable(method):
            value = method(context)
        elif callable(analyzer):
            value = analyzer(context)
        else:
            raise TypeError("analyzer must be callable or expose analyze()")
        if inspect.isawaitable(value):
            raise AnalysisProducerError(
                "async analyzers require AnalysisPipeline.aanalyze"
            )
        return _packet_from_producer(value, context.request)

    def _prepare_receipt(
        self, request: AnalysisPipelineRequest
    ) -> tuple[
        dict[str, Any], AnalysisEvidencePacket, AnalysisStageContext
    ]:
        context = self._build_context(request)
        packet = self._invoke_analyzer(context)
        artifact = self._artifacts.put(packet)
        receipt = _cache_receipt(packet, artifact, request)
        return receipt, packet, context

    def analyze(
        self,
        request: AnalysisPipelineRequest | Mapping[str, Any],
    ) -> AnalysisPipelineResult:
        if not isinstance(request, AnalysisPipelineRequest):
            if not isinstance(request, Mapping):
                raise TypeError("analysis request must be a mapping")
            request = AnalysisPipelineRequest(**dict(request))
        request = request.bind_pipeline_policy(self._execution_identity)
        self._metrics["requests"] += 1
        lookup = self.cache.lookup(
            request.cache_key, require_completion_evidence=True
        )
        initial_reasons = tuple(lookup.reason_codes)
        if lookup.status is AnalysisCacheLookupStatus.INVALIDATED:
            self._metrics["invalidated"] += 1
            if (
                lookup.entry is not None
                and not lookup.entry.is_completion_evidence
            ):
                self._metrics["negative_rejections"] += 1
        if lookup.hit and lookup.is_completion_evidence:
            try:
                result = self._exact_hit_result(lookup, request)
            except (OSError, TypeError, ValueError, AnalysisPipelineError):
                # A compact entry whose referenced packet is unavailable or
                # fails its second binding check is a miss, never authority.
                lookup = AnalysisCacheLookupResult(
                    AnalysisCacheLookupStatus.INVALIDATED,
                    request.cache_key,
                    reason_codes=("packet_artifact_invalid",),
                )
                initial_reasons = tuple(lookup.reason_codes)
                self._metrics["invalidated"] += 1
            else:
                self._metrics["exact_hits"] += 1
                return result

        produced: dict[str, Any] = {}

        def produce() -> Mapping[str, Any]:
            receipt, packet, context = self._prepare_receipt(request)
            produced["packet"] = packet
            produced["context"] = context
            return receipt

        def coordinated_produce() -> Any:
            from .cache_coordinator import CachePublication

            receipt = produce()
            packet = produced["packet"]
            if packet.safe_for_completion_reasoning:
                return CachePublication(receipt)
            return CachePublication(
                receipt,
                store=self.policy.cache_negative_results,
                ttl_seconds=(
                    self.policy.negative_ttl_seconds
                    if self.policy.cache_negative_results
                    else None
                ),
            )

        coordinate = (
            getattr(self.coordinator, "single_flight", None)
        )
        if coordinate is None:
            coordinate = getattr(self.coordinator, "run", None)
        if coordinate is None:
            receipt = produce()
            ttl = (
                None
                if produced["packet"].safe_for_completion_reasoning
                else self.policy.negative_ttl_seconds
            )
            if (
                produced["packet"].safe_for_completion_reasoning
                or self.policy.cache_negative_results
            ):
                stored = self.cache.put(
                    request.cache_key, receipt, ttl_seconds=ttl
                )
                if (
                    produced["packet"].safe_for_completion_reasoning
                    and not stored.stored
                ):
                    raise AnalysisPipelineError(
                        "authoritative analysis receipt could not be cached: "
                        + stored.reason_code
                    )
            packet = produced["packet"]
            context = produced["context"]
            joined = False
            coordinated_status = ""
        else:
            coordinated = coordinate(
                request.cache_key,
                coordinated_produce,
                ttl_seconds=None,
                completion_validator=lambda candidate: (
                    self._accept_completion_lookup(candidate, request)
                ),
            )
            joined = bool(
                getattr(coordinated, "shared", False)
                or getattr(coordinated, "waited", False)
            )
            coordinated_status = str(
                getattr(getattr(coordinated, "status", ""), "value", "")
            )
            refreshed = self.cache.lookup(
                request.cache_key, require_completion_evidence=True
            )
            if refreshed.hit and refreshed.is_completion_evidence:
                packet = self._load_cached_packet(refreshed, request)
            else:
                receipt = getattr(coordinated, "receipt", None)
                if not isinstance(receipt, Mapping):
                    value = getattr(coordinated, "value", coordinated)
                    receipt = value if isinstance(value, Mapping) else None
                if not isinstance(receipt, Mapping):
                    raise AnalysisPipelineError(
                        "coordinator returned no compact analysis receipt"
                    )
                packet = self._load_packet_receipt(
                    receipt, request, require_completion_evidence=False
                )
            context = produced.get("context")
            if coordinated_status == "cache_hit":
                # A leader may have filled the cache between our optimistic
                # lookup and flight registration.  Treat this as exact reuse.
                self._metrics["exact_hits"] += 1
                return self._exact_hit_result(refreshed, request)
        if joined:
            self._metrics["joined"] += 1
            status = PipelineCacheStatus.JOINED
        elif packet.safe_for_completion_reasoning:
            self._metrics["produced"] += 1
            status = PipelineCacheStatus.PRODUCED
        else:
            status = PipelineCacheStatus.INCONCLUSIVE
        return AnalysisPipelineResult(
            request=request,
            packet=packet,
            cache_status=status,
            cache_lookup_status=lookup.status,
            cache_reason_codes=initial_reasons,
            producer_executed=not joined,
            joined_existing_flight=joined,
            ast_index_id=(
                context.ast_index.index_id
                if context is not None and context.ast_index
                else ""
            ),
            retrieval_response_id=(
                context.retrieval.response_id if context is not None else ""
            ),
            provider_result=(
                context.provider_result if context is not None else None
            ),
            advisory_evidence_claim_references=(
                context.provider_evidence_claim_references
                if context is not None
                else ()
            ),
            provider_request=(
                context.provider_request if context is not None else None
            ),
            provider_policy=(
                context.provider_policy if context is not None else None
            ),
        )

    run = analyze
    execute = analyze

    async def aanalyze(
        self,
        request: AnalysisPipelineRequest | Mapping[str, Any],
    ) -> AnalysisPipelineResult:
        """Async convenience wrapper without leaking awaitables.

        The authority and durable-cache path remains synchronous and identical
        to :meth:`analyze`; callers that need a non-blocking filesystem can run
        it in their executor.  Async analyzer/provider values are intentionally
        not guessed into a second contract.
        """

        import asyncio

        return await asyncio.to_thread(self.analyze, request)

    arun = aanalyze


IntegratedAnalysisPipeline = AnalysisPipeline
SupervisorAnalysisPipeline = AnalysisPipeline


__all__ = [
    "ANALYSIS_PIPELINE_SCHEMA",
    "ANALYSIS_PIPELINE_VERSION",
    "EXACT_TREE_ANALYSIS_REUSE_REQUIREMENT_ID",
    "EXACT_TREE_REUSE_EVIDENCE_SCHEMA",
    "EXACT_TREE_REUSE_REQUIREMENT_ID",
    "AnalysisBindingError",
    "AnalysisPipeline",
    "AnalysisPipelineError",
    "AnalysisPipelineMetrics",
    "AnalysisPipelinePolicy",
    "AnalysisPipelineRequest",
    "AnalysisPipelineResult",
    "AnalysisPolicy",
    "AnalysisProducerError",
    "AnalysisRequest",
    "AnalysisStageContext",
    "ExactTreeReuseEvidence",
    "IntegratedAnalysisPipeline",
    "OptionalProviderFailure",
    "PipelineCacheStatus",
    "SupervisorAnalysisPipeline",
    "make_analysis_stage_receipt",
]
