"""Canonical receipts for local and optional-datasets analysis outcomes.

This module is deliberately a *normalization* boundary, not a voting system.
Local and ``ipfs_datasets_py`` producers remain diagnostic inputs.  Matching
claims may be recorded as agreement, but neither agreement nor model
confidence creates proof or completion authority.  Conflicting claims can be
selected only by an explicit deterministic policy or by a third, independent
validator; otherwise their disagreement and residual uncertainty are retained.

Receipt bodies contain compact identifiers and references only.  Source text,
graphs, decoded model output, and other large payloads belong in the artifact
store and are rejected here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final


ANALYSIS_CONSENSUS_VERSION: Final[int] = 1
ANALYSIS_CONSENSUS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-consensus-receipt@1"
)
ANALYSIS_CONSENSUS_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-consensus-claim@1"
)
ANALYSIS_CONSENSUS_PROVENANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-consensus-provenance@1"
)
ANALYSIS_CONSENSUS_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/analysis-consensus-policy@1"
)

DEFAULT_MAX_CONSENSUS_RECEIPT_BYTES: Final[int] = 64 * 1024
ABSOLUTE_MAX_CONSENSUS_RECEIPT_BYTES: Final[int] = 256 * 1024
DEFAULT_MAX_CONSENSUS_CLAIMS: Final[int] = 8
DEFAULT_MAX_CONSENSUS_REFERENCES: Final[int] = 64
DEFAULT_MAX_CONSENSUS_REFERENCE_BYTES: Final[int] = 8 * 1024
DEFAULT_MAX_RESIDUAL_UNCERTAINTIES: Final[int] = 32
_MAX_TEXT_BYTES: Final[int] = 8 * 1024

_FORBIDDEN_REFERENCE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "ast_body",
        "body",
        "completion",
        "content",
        "decoded_model_output",
        "embedding",
        "file_contents",
        "graph",
        "model_output",
        "model_response",
        "patch",
        "prompt",
        "raw",
        "raw_output",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "transcript",
    }
)
_REFERENCE_FIELDS: Final[tuple[str, ...]] = (
    "reference_id",
    "artifact_content_id",
    "artifact_id",
    "byte_count",
    "chunk_id",
    "cid",
    "dataset_id",
    "digest",
    "evidence_id",
    "graph_id",
    "kind",
    "media_type",
    "model_id",
    "path",
    "producer_id",
    "provider_id",
    "provenance_id",
    "record_id",
    "receipt_id",
    "revision",
    "score_millionths",
    "sha256",
    "source_id",
    "summary",
    "detail",
    "symbol",
    "tree_id",
    "uri",
)
_REFERENCE_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "id": "reference_id",
        "content_id": "artifact_content_id",
        "dataset": "dataset_id",
        "graph": "graph_id",
        "chunk": "chunk_id",
        "model": "model_id",
        "source": "source_id",
    }
)


class AnalysisConsensusError(ValueError):
    """A consensus input or receipt violates the normalization contract."""


class AnalysisConsensusOutcome(str, Enum):
    """Closed terminal vocabulary for one normalized receipt."""

    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    DEGRADED_FALLBACK = "degraded_fallback"
    PARTIAL_RESULT = "partial_result"
    PARTIAL = "partial_result"
    INDEPENDENT_VALIDATION = "independent_validation"


class AnalysisConsensusResolution(str, Enum):
    """How a receipt selected (or deliberately did not select) a claim."""

    AGREEMENT = "agreement"
    EXPLICIT_UNCERTAINTY = "explicit_uncertainty"
    DETERMINISTIC_POLICY = "deterministic_policy"
    INDEPENDENT_VALIDATOR = "independent_validator"
    LOCAL_FALLBACK = "local_fallback"
    PARTIAL_ONLY = "partial_only"


class AnalysisClaimStatus(str, Enum):
    """Authority-relevant producer state; confidence is intentionally absent."""

    CONCLUSIVE = "conclusive"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"
    INCONCLUSIVE = "inconclusive"


class AnalysisProducerKind(str, Enum):
    LOCAL = "local"
    DATASETS = "datasets"
    VALIDATOR = "validator"


class DeterministicDisagreementPolicy(str, Enum):
    """Allowlisted deterministic conflict rules.

    ``EXPLICIT_UNCERTAINTY`` is the fail-closed default.  No rule considers a
    confidence or score field.
    """

    EXPLICIT_UNCERTAINTY = "explicit_uncertainty"
    PREFER_LOCAL = "prefer_local"
    PREFER_DATASETS = "prefer_datasets"
    LEXICOGRAPHIC_CLAIM_ID = "lexicographic_claim_id"


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = _MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise AnalysisConsensusError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise AnalysisConsensusError(f"{name} must not be empty")
    if "\x00" in result:
        raise AnalysisConsensusError(f"{name} must not contain NUL bytes")
    if len(result.encode("utf-8")) > maximum:
        raise AnalysisConsensusError(f"{name} exceeds {maximum} UTF-8 bytes")
    return result


def _positive_int(value: Any, name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise AnalysisConsensusError(
            f"{name} must be an integer from 1 through {maximum}"
        )
    return value


def _canonical(value: Any, *, name: str = "value", depth: int = 0) -> Any:
    if depth > 12:
        raise AnalysisConsensusError(f"{name} exceeds maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisConsensusError(f"{name} must be finite")
        return format(value, ".17g")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AnalysisConsensusError(f"{name} keys must be strings")
        return {
            key: _canonical(item, name=name, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_canonical(item, name=name, depth=depth + 1) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical(converter(), name=name, depth=depth + 1)
    raise AnalysisConsensusError(
        f"{name} contains unsupported {type(value).__name__}"
    )


def canonical_consensus_json(value: Any) -> str:
    return json.dumps(
        _canonical(value, name="consensus value"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _content_id(namespace: str, value: Any) -> str:
    encoded = canonical_consensus_json(value).encode("utf-8")
    return f"{namespace}:sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_reference(
    value: Mapping[str, Any] | Any,
    *,
    max_reference_bytes: int,
) -> Mapping[str, Any]:
    converter = getattr(value, "to_dict", None)
    if not isinstance(value, Mapping) and callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise AnalysisConsensusError("analysis reference must be an object")
    lowered = {str(key).casefold() for key in value}
    forbidden = lowered.intersection(_FORBIDDEN_REFERENCE_FIELDS)
    if forbidden:
        raise AnalysisConsensusError(
            "analysis reference embeds forbidden payload fields: "
            + ", ".join(sorted(forbidden))
        )
    canonical_keys = {
        _REFERENCE_ALIASES.get(str(raw_key), str(raw_key))
        for raw_key in value
    }
    unsupported = canonical_keys.difference(_REFERENCE_FIELDS)
    if unsupported:
        raise AnalysisConsensusError(
            "analysis reference contains unsupported fields: "
            + ", ".join(sorted(unsupported))
        )
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = _REFERENCE_ALIASES.get(str(raw_key), str(raw_key))
        if key not in _REFERENCE_FIELDS or raw_value in (None, ""):
            continue
        if key in {"byte_count", "score_millionths"}:
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, int)
                or raw_value < 0
            ):
                raise AnalysisConsensusError(f"reference {key} must be non-negative")
            if key == "score_millionths" and raw_value > 1_000_000:
                raise AnalysisConsensusError(
                    "reference score_millionths exceeds one million"
                )
            normalized[key] = raw_value
        else:
            normalized[key] = _text(
                raw_value,
                f"reference {key}",
                required=False,
                maximum=2048,
            )
    if not normalized:
        raise AnalysisConsensusError("analysis reference has no compact identity")
    if not any(
        normalized.get(name)
        for name in (
            "reference_id",
            "artifact_content_id",
            "artifact_id",
            "cid",
            "digest",
            "evidence_id",
            "record_id",
            "sha256",
            "source_id",
            "uri",
        )
    ):
        normalized["reference_id"] = _content_id(
            "analysis-consensus-reference", normalized
        )
    ordered = {
        key: normalized[key] for key in _REFERENCE_FIELDS if key in normalized
    }
    if len(canonical_consensus_json(ordered).encode("utf-8")) > max_reference_bytes:
        raise AnalysisConsensusError(
            "analysis reference exceeds max_reference_bytes"
        )
    return MappingProxyType(ordered)


def _normalize_references(
    values: Sequence[Mapping[str, Any] | Any],
    *,
    maximum: int,
    max_reference_bytes: int,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise AnalysisConsensusError("analysis references must be a sequence")
    if len(values) > maximum:
        raise AnalysisConsensusError("analysis references exceed policy bound")
    unique: dict[str, Mapping[str, Any]] = {}
    for value in values:
        normalized = _normalize_reference(
            value, max_reference_bytes=max_reference_bytes
        )
        encoded = canonical_consensus_json(normalized)
        unique[encoded] = normalized
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class AnalysisConsensusPolicy:
    """Versioned bounds and deterministic disagreement behavior."""

    policy_id: str = "analysis-consensus-policy:default"
    policy_revision: str = "analysis-consensus-policy@1"
    disagreement_policy: DeterministicDisagreementPolicy = (
        DeterministicDisagreementPolicy.EXPLICIT_UNCERTAINTY
    )
    max_receipt_bytes: int = DEFAULT_MAX_CONSENSUS_RECEIPT_BYTES
    max_claims: int = DEFAULT_MAX_CONSENSUS_CLAIMS
    max_references_per_claim: int = DEFAULT_MAX_CONSENSUS_REFERENCES
    max_reference_bytes: int = DEFAULT_MAX_CONSENSUS_REFERENCE_BYTES
    max_residual_uncertainties: int = DEFAULT_MAX_RESIDUAL_UNCERTAINTIES

    def __post_init__(self) -> None:
        for name in ("policy_id", "policy_revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=512)
            )
        object.__setattr__(
            self,
            "disagreement_policy",
            DeterministicDisagreementPolicy(self.disagreement_policy),
        )
        maxima = {
            "max_receipt_bytes": ABSOLUTE_MAX_CONSENSUS_RECEIPT_BYTES,
            "max_claims": 64,
            "max_references_per_claim": 1024,
            "max_reference_bytes": ABSOLUTE_MAX_CONSENSUS_RECEIPT_BYTES,
            "max_residual_uncertainties": 256,
        }
        for name, maximum in maxima.items():
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name, maximum),
            )
        if self.max_reference_bytes > self.max_receipt_bytes:
            raise AnalysisConsensusError(
                "max_reference_bytes cannot exceed max_receipt_bytes"
            )

    @property
    def content_id(self) -> str:
        return _content_id("analysis-consensus-policy", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CONSENSUS_POLICY_SCHEMA,
            "version": ANALYSIS_CONSENSUS_VERSION,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "disagreement_policy": self.disagreement_policy.value,
            "max_receipt_bytes": self.max_receipt_bytes,
            "max_claims": self.max_claims,
            "max_references_per_claim": self.max_references_per_claim,
            "max_reference_bytes": self.max_reference_bytes,
            "max_residual_uncertainties": self.max_residual_uncertainties,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_value(
        cls, value: "AnalysisConsensusPolicy | Mapping[str, Any] | None"
    ) -> "AnalysisConsensusPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisConsensusError("consensus policy must be an object")
        allowed = {
            "schema",
            "version",
            "content_id",
            *cls.__dataclass_fields__,
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisConsensusError(
                "unknown consensus policy fields: " + ", ".join(sorted(unknown))
            )
        if value.get("schema", ANALYSIS_CONSENSUS_POLICY_SCHEMA) != (
            ANALYSIS_CONSENSUS_POLICY_SCHEMA
        ):
            raise AnalysisConsensusError("unsupported consensus policy schema")
        if value.get("version", ANALYSIS_CONSENSUS_VERSION) != (
            ANALYSIS_CONSENSUS_VERSION
        ):
            raise AnalysisConsensusError("unsupported consensus policy version")
        result = cls(
            **{
                name: value[name]
                for name in cls.__dataclass_fields__
                if name in value
            }
        )
        if value.get("content_id") not in (None, result.content_id):
            raise AnalysisConsensusError("consensus policy identity does not match")
        return result


@dataclass(frozen=True)
class AnalysisClaimProvenance:
    """Compact origin identity preserved for every normalized claim."""

    source_id: str
    producer_id: str
    policy_id: str
    capability_id: str
    tree_id: str
    dataset_id: str = ""
    graph_id: str = ""
    chunk_id: str = ""
    model_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "producer_id",
            "policy_id",
            "capability_id",
            "tree_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=2048)
            )
        for name in (
            "dataset_id",
            "graph_id",
            "chunk_id",
            "model_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=False,
                    maximum=2048,
                ),
            )

    @property
    def content_id(self) -> str:
        return _content_id("analysis-claim-provenance", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CONSENSUS_PROVENANCE_SCHEMA,
            "version": ANALYSIS_CONSENSUS_VERSION,
            "source_id": self.source_id,
            "dataset_id": self.dataset_id,
            "graph_id": self.graph_id,
            "chunk_id": self.chunk_id,
            "producer_id": self.producer_id,
            "model_id": self.model_id,
            "policy_id": self.policy_id,
            "capability_id": self.capability_id,
            "tree_id": self.tree_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "content_id": self.content_id}

    @classmethod
    def from_value(
        cls, value: "AnalysisClaimProvenance | Mapping[str, Any]"
    ) -> "AnalysisClaimProvenance":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisConsensusError("claim provenance must be an object")
        aliases = dict(value)
        for alias, canonical in (
            ("source", "source_id"),
            ("dataset", "dataset_id"),
            ("graph", "graph_id"),
            ("chunk", "chunk_id"),
            ("producer", "producer_id"),
            ("model", "model_id"),
            ("policy", "policy_id"),
            ("capability", "capability_id"),
            ("tree", "tree_id"),
        ):
            if canonical not in aliases and alias in aliases:
                aliases[canonical] = aliases.pop(alias)
        allowed = {
            "schema",
            "version",
            "content_id",
            *cls.__dataclass_fields__,
        }
        unknown = set(aliases) - allowed
        if unknown:
            raise AnalysisConsensusError(
                "unknown claim provenance fields: " + ", ".join(sorted(unknown))
            )
        if aliases.get("schema", ANALYSIS_CONSENSUS_PROVENANCE_SCHEMA) != (
            ANALYSIS_CONSENSUS_PROVENANCE_SCHEMA
        ):
            raise AnalysisConsensusError("unsupported claim provenance schema")
        if aliases.get("version", ANALYSIS_CONSENSUS_VERSION) != (
            ANALYSIS_CONSENSUS_VERSION
        ):
            raise AnalysisConsensusError("unsupported claim provenance version")
        result = cls(
            **{
                name: aliases.get(name, "")
                for name in cls.__dataclass_fields__
            }
        )
        if aliases.get("content_id") not in (None, result.content_id):
            raise AnalysisConsensusError("claim provenance identity does not match")
        return result


# Concise public spelling for callers constructing receipts directly.
AnalysisProvenance = AnalysisClaimProvenance


@dataclass(frozen=True)
class AnalysisConsensusClaim:
    """One producer claim normalized independently from its confidence."""

    producer_kind: AnalysisProducerKind
    result_id: str
    verdict: str
    status: AnalysisClaimStatus
    provenance: AnalysisClaimProvenance
    evidence_references: tuple[Mapping[str, Any], ...] = ()
    proposal_only: bool = False
    truncated: bool = False
    confidence_millionths: int = 0
    validates_claim_id: str = ""
    claim_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "producer_kind", AnalysisProducerKind(self.producer_kind)
        )
        object.__setattr__(self, "status", AnalysisClaimStatus(self.status))
        for name in ("result_id", "verdict"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=4096)
            )
        provenance = AnalysisClaimProvenance.from_value(self.provenance)
        object.__setattr__(self, "provenance", provenance)
        if not isinstance(self.proposal_only, bool) or not isinstance(
            self.truncated, bool
        ):
            raise AnalysisConsensusError(
                "proposal_only and truncated must be booleans"
            )
        if (
            isinstance(self.confidence_millionths, bool)
            or not isinstance(self.confidence_millionths, int)
            or not 0 <= self.confidence_millionths <= 1_000_000
        ):
            raise AnalysisConsensusError(
                "confidence_millionths must be from zero through one million"
            )
        validates = _text(
            self.validates_claim_id,
            "validates_claim_id",
            required=False,
            maximum=256,
        )
        if self.producer_kind is AnalysisProducerKind.VALIDATOR:
            if not validates:
                raise AnalysisConsensusError(
                    "validator claims must identify the claim they validate"
                )
        elif validates:
            raise AnalysisConsensusError(
                "only independent validators may validate another claim"
            )
        object.__setattr__(self, "validates_claim_id", validates)
        references = _normalize_references(
            self.evidence_references,
            maximum=DEFAULT_MAX_CONSENSUS_REFERENCES,
            max_reference_bytes=DEFAULT_MAX_CONSENSUS_REFERENCE_BYTES,
        )
        object.__setattr__(self, "evidence_references", references)
        derived = _content_id("analysis-consensus-claim", self._payload())
        if self.claim_id and self.claim_id != derived:
            raise AnalysisConsensusError("claim identity does not match its content")
        object.__setattr__(self, "claim_id", derived)

    @property
    def consensus_eligible(self) -> bool:
        return (
            self.status is AnalysisClaimStatus.CONCLUSIVE
            and not self.truncated
            and self.producer_kind is not AnalysisProducerKind.VALIDATOR
        )

    @property
    def completion_eligible(self) -> bool:
        """Whether the producer state is intrinsically eligible.

        The containing consensus receipt is still non-authoritative and never
        becomes completion evidence.
        """

        return self.consensus_eligible and not self.proposal_only

    @property
    def excluded_from_completion_reasoning(self) -> bool:
        return not self.completion_eligible

    @property
    def semantic_id(self) -> str:
        """Producer-neutral identity used for equality.

        Scores, confidence, producer/model identities, and provenance are not
        inputs.  They cannot manufacture agreement or win a disagreement.
        Evidence references remain attached to the claim for audit, but are
        producer-specific support rather than part of the normalized verdict.
        """

        return _content_id(
            "analysis-consensus-semantic-claim",
            {"verdict": self.verdict},
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CONSENSUS_CLAIM_SCHEMA,
            "version": ANALYSIS_CONSENSUS_VERSION,
            "producer_kind": self.producer_kind.value,
            "result_id": self.result_id,
            "verdict": self.verdict,
            "status": self.status.value,
            "provenance": self.provenance.to_dict(),
            "evidence_references": [
                dict(item) for item in self.evidence_references
            ],
            "proposal_only": self.proposal_only,
            "truncated": self.truncated,
            # Retained only for audit.  It is excluded from semantic identity,
            # resolution, selection, and every authority property.
            "confidence_millionths": self.confidence_millionths,
            "validates_claim_id": self.validates_claim_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "claim_id": self.claim_id}

    @classmethod
    def from_value(
        cls, value: "AnalysisConsensusClaim | Mapping[str, Any]"
    ) -> "AnalysisConsensusClaim":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AnalysisConsensusError("consensus claim must be an object")
        allowed = {
            "schema",
            "version",
            *cls.__dataclass_fields__,
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisConsensusError(
                "unknown consensus claim fields: " + ", ".join(sorted(unknown))
            )
        if value.get("schema", ANALYSIS_CONSENSUS_CLAIM_SCHEMA) != (
            ANALYSIS_CONSENSUS_CLAIM_SCHEMA
        ):
            raise AnalysisConsensusError("unsupported consensus claim schema")
        if value.get("version", ANALYSIS_CONSENSUS_VERSION) != (
            ANALYSIS_CONSENSUS_VERSION
        ):
            raise AnalysisConsensusError("unsupported consensus claim version")
        return cls(
            producer_kind=value.get("producer_kind", ""),
            result_id=value.get("result_id", ""),
            verdict=value.get("verdict", ""),
            status=value.get("status", ""),
            provenance=value.get("provenance") or {},
            evidence_references=tuple(value.get("evidence_references") or ()),
            proposal_only=value.get("proposal_only", False),
            truncated=value.get("truncated", False),
            confidence_millionths=value.get("confidence_millionths", 0),
            validates_claim_id=value.get("validates_claim_id", ""),
            claim_id=value.get("claim_id", ""),
        )


AnalysisClaim = AnalysisConsensusClaim


@dataclass(frozen=True)
class AnalysisConsensusReceipt:
    """One bounded receipt covering agreement, conflict, fallback, or partials."""

    repository_id: str
    tree_id: str
    objective_revision: str
    operation: str
    policy: AnalysisConsensusPolicy
    outcome: AnalysisConsensusOutcome
    resolution: AnalysisConsensusResolution
    claims: tuple[AnalysisConsensusClaim, ...]
    selected_claim_id: str = ""
    fallback_reason_code: str = ""
    residual_uncertainty: tuple[str, ...] = ()
    fallback_explicit: bool = False
    receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_revision",
            "operation",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, maximum=2048)
            )
        policy = AnalysisConsensusPolicy.from_value(self.policy)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(
            self, "outcome", AnalysisConsensusOutcome(self.outcome)
        )
        object.__setattr__(
            self, "resolution", AnalysisConsensusResolution(self.resolution)
        )
        if isinstance(self.claims, (str, bytes)) or not isinstance(
            self.claims, Sequence
        ):
            raise AnalysisConsensusError("claims must be a sequence")
        claims = tuple(AnalysisConsensusClaim.from_value(item) for item in self.claims)
        if not claims or len(claims) > policy.max_claims:
            raise AnalysisConsensusError("claims violate consensus policy bounds")
        by_id = {item.claim_id: item for item in claims}
        if len(by_id) != len(claims):
            raise AnalysisConsensusError("claims must be unique")
        claims = tuple(by_id[key] for key in sorted(by_id))
        object.__setattr__(self, "claims", claims)
        producer_kinds = tuple(item.producer_kind for item in claims)
        if (
            AnalysisProducerKind.LOCAL not in producer_kinds
            or len(set(producer_kinds)) != len(producer_kinds)
        ):
            raise AnalysisConsensusError(
                "receipt requires one local claim and at most one claim per "
                "producer kind"
            )
        for claim in claims:
            if claim.provenance.tree_id != self.tree_id:
                raise AnalysisConsensusError(
                    "claim provenance tree_id is detached from the receipt"
                )
            if len(claim.evidence_references) > policy.max_references_per_claim:
                raise AnalysisConsensusError(
                    "claim references exceed consensus policy bound"
                )
            for reference in claim.evidence_references:
                if (
                    len(canonical_consensus_json(reference).encode("utf-8"))
                    > policy.max_reference_bytes
                ):
                    raise AnalysisConsensusError(
                        "claim reference exceeds consensus policy byte bound"
                    )
                reference_tree = reference.get("tree_id")
                if reference_tree and reference_tree != self.tree_id:
                    raise AnalysisConsensusError(
                        "claim reference tree_id is detached from the receipt"
                    )
        selected = _text(
            self.selected_claim_id,
            "selected_claim_id",
            required=False,
            maximum=256,
        )
        if selected and selected not in by_id:
            raise AnalysisConsensusError(
                "selected_claim_id is not embedded in the receipt"
            )
        object.__setattr__(self, "selected_claim_id", selected)
        reason = _text(
            self.fallback_reason_code,
            "fallback_reason_code",
            required=False,
            maximum=1024,
        )
        object.__setattr__(self, "fallback_reason_code", reason)
        if not isinstance(self.fallback_explicit, bool):
            raise AnalysisConsensusError("fallback_explicit must be a boolean")
        if isinstance(self.residual_uncertainty, str) or not isinstance(
            self.residual_uncertainty, Sequence
        ):
            raise AnalysisConsensusError("residual_uncertainty must be a sequence")
        uncertainty = tuple(
            sorted(
                {
                    _text(item, "residual uncertainty", maximum=2048)
                    for item in self.residual_uncertainty
                }
            )
        )
        if len(uncertainty) > policy.max_residual_uncertainties:
            raise AnalysisConsensusError(
                "residual uncertainty exceeds consensus policy bound"
            )
        object.__setattr__(self, "residual_uncertainty", uncertainty)
        self._validate_outcome(by_id)
        derived = _content_id("analysis-consensus-receipt", self._payload())
        if self.receipt_id and self.receipt_id != derived:
            raise AnalysisConsensusError("receipt identity does not match content")
        object.__setattr__(self, "receipt_id", derived)
        if self.serialized_byte_count > policy.max_receipt_bytes:
            raise AnalysisConsensusError(
                "analysis consensus receipt exceeds max_receipt_bytes"
            )

    def _validate_outcome(
        self, by_id: Mapping[str, AnalysisConsensusClaim]
    ) -> None:
        ordinary = tuple(
            item
            for item in self.claims
            if item.producer_kind is not AnalysisProducerKind.VALIDATOR
        )
        eligible = tuple(item for item in ordinary if item.consensus_eligible)
        validator = tuple(
            item
            for item in self.claims
            if item.producer_kind is AnalysisProducerKind.VALIDATOR
        )
        if self.outcome is AnalysisConsensusOutcome.AGREEMENT:
            if (
                self.resolution is not AnalysisConsensusResolution.AGREEMENT
                or len(eligible) < 2
                or len({item.semantic_id for item in eligible}) != 1
                or self.selected_claim_id
                or self.fallback_explicit
                or self.fallback_reason_code
                or self.residual_uncertainty
                or len({item.provenance.producer_id for item in eligible}) < 2
            ):
                raise AnalysisConsensusError(
                    "agreement requires matching conclusive producer claims"
                )
        elif self.outcome is AnalysisConsensusOutcome.DISAGREEMENT:
            if len(eligible) < 2 or len(
                {item.semantic_id for item in eligible}
            ) < 2:
                raise AnalysisConsensusError(
                    "disagreement requires distinct conclusive claims"
                )
            if self.resolution is AnalysisConsensusResolution.EXPLICIT_UNCERTAINTY:
                if self.selected_claim_id or not self.residual_uncertainty:
                    raise AnalysisConsensusError(
                        "unresolved disagreement must retain explicit uncertainty"
                    )
            elif self.resolution is AnalysisConsensusResolution.DETERMINISTIC_POLICY:
                expected = _deterministic_selection(eligible, self.policy)
                if (
                    expected is None
                    or self.selected_claim_id != expected.claim_id
                    or not self.residual_uncertainty
                ):
                    raise AnalysisConsensusError(
                        "deterministic resolution must match the declared policy"
                    )
            else:
                raise AnalysisConsensusError(
                    "disagreement can only use uncertainty or deterministic policy"
                )
            if self.fallback_explicit or self.fallback_reason_code:
                raise AnalysisConsensusError(
                    "disagreement cannot claim degraded fallback"
                )
        elif self.outcome is AnalysisConsensusOutcome.DEGRADED_FALLBACK:
            selected = by_id.get(self.selected_claim_id)
            datasets = next(
                (
                    item
                    for item in ordinary
                    if item.producer_kind is AnalysisProducerKind.DATASETS
                ),
                None,
            )
            if (
                self.resolution is not AnalysisConsensusResolution.LOCAL_FALLBACK
                or selected is None
                or selected.producer_kind is not AnalysisProducerKind.LOCAL
                or not selected.consensus_eligible
                or (
                    datasets is not None
                    and datasets.status is not AnalysisClaimStatus.FAILED
                )
                or not self.fallback_explicit
                or not self.fallback_reason_code
            ):
                raise AnalysisConsensusError(
                    "degraded fallback requires an explicit eligible local claim"
                )
        elif self.outcome is AnalysisConsensusOutcome.PARTIAL_RESULT:
            if (
                self.resolution is not AnalysisConsensusResolution.PARTIAL_ONLY
                or self.selected_claim_id
                or not self.residual_uncertainty
                or self.fallback_explicit
                or self.fallback_reason_code
                or all(item.consensus_eligible for item in ordinary)
            ):
                raise AnalysisConsensusError(
                    "partial result must retain uncertainty and select no claim"
                )
        elif self.outcome is AnalysisConsensusOutcome.INDEPENDENT_VALIDATION:
            selected = by_id.get(self.selected_claim_id)
            matching = tuple(
                item
                for item in validator
                if item.validates_claim_id == self.selected_claim_id
                and item.status is AnalysisClaimStatus.CONCLUSIVE
                and not item.truncated
            )
            producers = {item.provenance.producer_id for item in ordinary}
            if (
                self.resolution
                is not AnalysisConsensusResolution.INDEPENDENT_VALIDATOR
                or selected is None
                or not selected.consensus_eligible
                or len(matching) != 1
                or matching[0].provenance.producer_id in producers
                or self.fallback_explicit
                or self.fallback_reason_code
            ):
                raise AnalysisConsensusError(
                    "independent validation requires a third-party validator"
                )

    @property
    def content_id(self) -> str:
        return self.receipt_id

    @property
    def result_id(self) -> str:
        return self.receipt_id

    @property
    def selected_claim(self) -> AnalysisConsensusClaim | None:
        return next(
            (
                item
                for item in self.claims
                if item.claim_id == self.selected_claim_id
            ),
            None,
        )

    @property
    def excluded_claim_ids(self) -> tuple[str, ...]:
        return tuple(
            item.claim_id
            for item in self.claims
            if item.excluded_from_completion_reasoning
        )

    @property
    def accepted_claim_ids(self) -> tuple[str, ...]:
        """Claims accepted for downstream *analysis*, never completion."""

        if self.outcome is AnalysisConsensusOutcome.AGREEMENT:
            return tuple(
                item.claim_id for item in self.claims if item.consensus_eligible
            )
        return (self.selected_claim_id,) if self.selected_claim_id else ()

    @property
    def completion_eligible_claim_ids(self) -> tuple[str, ...]:
        # Conflict policy: normalizing consensus never creates completion
        # evidence, even if an embedded local claim has a conclusive state.
        return ()

    @property
    def completion_authority(self) -> bool:
        return False

    @property
    def is_completion_evidence(self) -> bool:
        return False

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return False

    @property
    def non_authoritative(self) -> bool:
        return True

    @property
    def serialized_byte_count(self) -> int:
        return len(canonical_consensus_json(self.to_dict()).encode("utf-8"))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": ANALYSIS_CONSENSUS_RECEIPT_SCHEMA,
            "version": ANALYSIS_CONSENSUS_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "operation": self.operation,
            "policy": self.policy.to_dict(),
            "outcome": self.outcome.value,
            "resolution": self.resolution.value,
            "claims": [item.to_dict() for item in self.claims],
            "selected_claim_id": self.selected_claim_id,
            "accepted_claim_ids": list(self.accepted_claim_ids),
            "excluded_claim_ids": list(self.excluded_claim_ids),
            "completion_eligible_claim_ids": [],
            "fallback_reason_code": self.fallback_reason_code,
            "fallback_explicit": self.fallback_explicit,
            "residual_uncertainty": list(self.residual_uncertainty),
            "non_authoritative": True,
            "completion_authority": False,
            "is_completion_evidence": False,
            "safe_for_completion_reasoning": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnalysisConsensusReceipt":
        if not isinstance(value, Mapping):
            raise AnalysisConsensusError("consensus receipt must be an object")
        allowed = {
            "schema",
            "version",
            "repository_id",
            "tree_id",
            "objective_revision",
            "operation",
            "policy",
            "outcome",
            "resolution",
            "claims",
            "selected_claim_id",
            "accepted_claim_ids",
            "excluded_claim_ids",
            "completion_eligible_claim_ids",
            "fallback_reason_code",
            "fallback_explicit",
            "residual_uncertainty",
            "non_authoritative",
            "completion_authority",
            "is_completion_evidence",
            "safe_for_completion_reasoning",
            "receipt_id",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AnalysisConsensusError(
                "unknown consensus receipt fields: " + ", ".join(sorted(unknown))
            )
        if value.get("schema") != ANALYSIS_CONSENSUS_RECEIPT_SCHEMA:
            raise AnalysisConsensusError("unsupported consensus receipt schema")
        if value.get("version") != ANALYSIS_CONSENSUS_VERSION:
            raise AnalysisConsensusError("unsupported consensus receipt version")
        for name, expected in (
            ("non_authoritative", True),
            ("completion_authority", False),
            ("is_completion_evidence", False),
            ("safe_for_completion_reasoning", False),
        ):
            if value.get(name) is not expected:
                raise AnalysisConsensusError(f"receipt {name} claim does not match")
        result = cls(
            repository_id=value.get("repository_id", ""),
            tree_id=value.get("tree_id", ""),
            objective_revision=value.get("objective_revision", ""),
            operation=value.get("operation", ""),
            policy=value.get("policy") or {},
            outcome=value.get("outcome", ""),
            resolution=value.get("resolution", ""),
            claims=tuple(value.get("claims") or ()),
            selected_claim_id=value.get("selected_claim_id", ""),
            fallback_reason_code=value.get("fallback_reason_code", ""),
            residual_uncertainty=tuple(
                value.get("residual_uncertainty") or ()
            ),
            fallback_explicit=value.get("fallback_explicit", False),
            receipt_id=value.get("receipt_id", ""),
        )
        expected_derived = {
            "accepted_claim_ids": list(result.accepted_claim_ids),
            "excluded_claim_ids": list(result.excluded_claim_ids),
            "completion_eligible_claim_ids": [],
        }
        for name, expected in expected_derived.items():
            if value.get(name) != expected:
                raise AnalysisConsensusError(
                    f"receipt derived field {name} does not match"
                )
        return result

    def equivalent_to(self, other: Any) -> bool:
        """Cold/warm equivalence is exact canonical receipt identity."""

        return bool(
            isinstance(other, AnalysisConsensusReceipt)
            and self.receipt_id == other.receipt_id
            and self.to_dict() == other.to_dict()
        )


def _deterministic_selection(
    claims: Sequence[AnalysisConsensusClaim],
    policy: AnalysisConsensusPolicy,
) -> AnalysisConsensusClaim | None:
    eligible = tuple(item for item in claims if item.consensus_eligible)
    if policy.disagreement_policy is (
        DeterministicDisagreementPolicy.EXPLICIT_UNCERTAINTY
    ):
        return None
    if policy.disagreement_policy is DeterministicDisagreementPolicy.PREFER_LOCAL:
        kind = AnalysisProducerKind.LOCAL
        candidates = tuple(item for item in eligible if item.producer_kind is kind)
    elif policy.disagreement_policy is (
        DeterministicDisagreementPolicy.PREFER_DATASETS
    ):
        kind = AnalysisProducerKind.DATASETS
        candidates = tuple(item for item in eligible if item.producer_kind is kind)
    else:
        candidates = eligible
    # Semantic identity excludes confidence, score, producer/model provenance,
    # and result IDs.  Even the legacy "claim ID" spelling therefore cannot
    # turn one of those audit attributes into a deciding signal.
    return min(candidates, key=lambda item: item.semantic_id) if candidates else None


def build_analysis_consensus_receipt(
    *,
    repository_id: str,
    tree_id: str,
    objective_revision: str,
    operation: str,
    local_claim: AnalysisConsensusClaim | Mapping[str, Any],
    datasets_claim: AnalysisConsensusClaim | Mapping[str, Any] | None = None,
    validator_claim: AnalysisConsensusClaim | Mapping[str, Any] | None = None,
    policy: AnalysisConsensusPolicy | Mapping[str, Any] | None = None,
    fallback_reason_code: str = "",
) -> AnalysisConsensusReceipt:
    """Normalize producer outcomes into one deterministic compact receipt."""

    normalized_policy = AnalysisConsensusPolicy.from_value(policy)
    local = AnalysisConsensusClaim.from_value(local_claim)
    if local.producer_kind is not AnalysisProducerKind.LOCAL:
        raise AnalysisConsensusError("local_claim must have producer_kind=local")
    datasets = (
        AnalysisConsensusClaim.from_value(datasets_claim)
        if datasets_claim is not None
        else None
    )
    if datasets is not None and datasets.producer_kind is not (
        AnalysisProducerKind.DATASETS
    ):
        raise AnalysisConsensusError(
            "datasets_claim must have producer_kind=datasets"
        )
    validator = (
        AnalysisConsensusClaim.from_value(validator_claim)
        if validator_claim is not None
        else None
    )
    if validator is not None and validator.producer_kind is not (
        AnalysisProducerKind.VALIDATOR
    ):
        raise AnalysisConsensusError(
            "validator_claim must have producer_kind=validator"
        )
    claims = tuple(
        item for item in (local, datasets, validator) if item is not None
    )
    if validator is not None:
        selected = next(
            (
                item
                for item in (local, datasets)
                if item is not None
                and item.claim_id == validator.validates_claim_id
                and item.consensus_eligible
            ),
            None,
        )
        producer_ids = {
            item.provenance.producer_id
            for item in (local, datasets)
            if item is not None
        }
        if (
            selected is not None
            and validator.status is AnalysisClaimStatus.CONCLUSIVE
            and not validator.truncated
            and validator.provenance.producer_id not in producer_ids
        ):
            return AnalysisConsensusReceipt(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_revision=objective_revision,
                operation=operation,
                policy=normalized_policy,
                outcome=AnalysisConsensusOutcome.INDEPENDENT_VALIDATION,
                resolution=AnalysisConsensusResolution.INDEPENDENT_VALIDATOR,
                claims=claims,
                selected_claim_id=selected.claim_id,
                residual_uncertainty=(),
            )
    if datasets is None or datasets.status is AnalysisClaimStatus.FAILED:
        if local.consensus_eligible:
            reason = fallback_reason_code or (
                "datasets_result_missing"
                if datasets is None
                else "datasets_result_failed"
            )
            return AnalysisConsensusReceipt(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_revision=objective_revision,
                operation=operation,
                policy=normalized_policy,
                outcome=AnalysisConsensusOutcome.DEGRADED_FALLBACK,
                resolution=AnalysisConsensusResolution.LOCAL_FALLBACK,
                claims=claims,
                selected_claim_id=local.claim_id,
                fallback_reason_code=reason,
                fallback_explicit=True,
                residual_uncertainty=("optional datasets result unavailable",),
            )
    ordinary = tuple(item for item in (local, datasets) if item is not None)
    eligible = tuple(item for item in ordinary if item.consensus_eligible)
    if len(eligible) >= 2 and len({item.semantic_id for item in eligible}) == 1:
        return AnalysisConsensusReceipt(
            repository_id=repository_id,
            tree_id=tree_id,
            objective_revision=objective_revision,
            operation=operation,
            policy=normalized_policy,
            outcome=AnalysisConsensusOutcome.AGREEMENT,
            resolution=AnalysisConsensusResolution.AGREEMENT,
            claims=claims,
        )
    if len(eligible) >= 2:
        selected = _deterministic_selection(eligible, normalized_policy)
        if selected is not None:
            return AnalysisConsensusReceipt(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_revision=objective_revision,
                operation=operation,
                policy=normalized_policy,
                outcome=AnalysisConsensusOutcome.DISAGREEMENT,
                resolution=AnalysisConsensusResolution.DETERMINISTIC_POLICY,
                claims=claims,
                selected_claim_id=selected.claim_id,
                residual_uncertainty=(
                    "producer claims disagree; deterministic policy selected "
                    "one diagnostic claim",
                ),
            )
        return AnalysisConsensusReceipt(
            repository_id=repository_id,
            tree_id=tree_id,
            objective_revision=objective_revision,
            operation=operation,
            policy=normalized_policy,
            outcome=AnalysisConsensusOutcome.DISAGREEMENT,
            resolution=AnalysisConsensusResolution.EXPLICIT_UNCERTAINTY,
            claims=claims,
            residual_uncertainty=(
                "local and datasets claims disagree without an independent "
                "validator or selecting policy",
            ),
        )
    reason = "analysis result is partial, stale, inconclusive, or proposal-only"
    if datasets is not None and datasets.status is AnalysisClaimStatus.STALE:
        reason = "datasets result is stale"
    return AnalysisConsensusReceipt(
        repository_id=repository_id,
        tree_id=tree_id,
        objective_revision=objective_revision,
        operation=operation,
        policy=normalized_policy,
        outcome=AnalysisConsensusOutcome.PARTIAL_RESULT,
        resolution=AnalysisConsensusResolution.PARTIAL_ONLY,
        claims=claims,
        residual_uncertainty=(reason,),
    )


normalize_analysis_consensus = build_analysis_consensus_receipt
reconcile_analysis_results = build_analysis_consensus_receipt
AnalysisOutcomeReceipt = AnalysisConsensusReceipt
AnalysisDisagreementResolution = AnalysisConsensusResolution


__all__ = [
    "ABSOLUTE_MAX_CONSENSUS_RECEIPT_BYTES",
    "ANALYSIS_CONSENSUS_CLAIM_SCHEMA",
    "ANALYSIS_CONSENSUS_POLICY_SCHEMA",
    "ANALYSIS_CONSENSUS_PROVENANCE_SCHEMA",
    "ANALYSIS_CONSENSUS_RECEIPT_SCHEMA",
    "ANALYSIS_CONSENSUS_VERSION",
    "AnalysisClaim",
    "AnalysisClaimProvenance",
    "AnalysisClaimStatus",
    "AnalysisConsensusClaim",
    "AnalysisConsensusError",
    "AnalysisConsensusOutcome",
    "AnalysisConsensusPolicy",
    "AnalysisConsensusReceipt",
    "AnalysisConsensusResolution",
    "AnalysisDisagreementResolution",
    "AnalysisOutcomeReceipt",
    "AnalysisProducerKind",
    "AnalysisProvenance",
    "DEFAULT_MAX_CONSENSUS_CLAIMS",
    "DEFAULT_MAX_CONSENSUS_RECEIPT_BYTES",
    "DEFAULT_MAX_CONSENSUS_REFERENCES",
    "DEFAULT_MAX_CONSENSUS_REFERENCE_BYTES",
    "DEFAULT_MAX_RESIDUAL_UNCERTAINTIES",
    "DeterministicDisagreementPolicy",
    "build_analysis_consensus_receipt",
    "canonical_consensus_json",
    "normalize_analysis_consensus",
    "reconcile_analysis_results",
]
