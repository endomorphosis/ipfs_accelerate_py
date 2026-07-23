"""Immutable, bounded contracts for supervisor analysis evidence.

Analysis stages routinely inspect data that is far too large or too sensitive
to cross the scheduler boundary: repository source, AST datasets, proof
objects, prompts, and model responses.  The contracts in this module carry
only compact conclusions and content-addressed references to those durable
artifacts.

The completion boundary is deliberately fail closed.  Completion eligibility
is derived from typed state and cannot be supplied by a producer.  A partial,
failed, timed-out, stale, truncated, coverage-incomplete, or negatively cached
receipt cannot be constructed with a conclusive outcome.  An evidence packet
is conclusive only when it contains at least one independently eligible stage
receipt.

Confidence, novelty, and relative cost are serialized as integer millionths.
This follows the supervisor's Profile G convention and permits reuse of the
strict DAG-JSON/CID implementation in ``formal_verification_contracts``
without introducing floating-point identity ambiguity.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)


ANALYSIS_CONTRACT_VERSION = 1
CONTRACT_VERSION = ANALYSIS_CONTRACT_VERSION
SCHEMA_VERSION = ANALYSIS_CONTRACT_VERSION

ANALYSIS_LIMITS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/analysis-limits@1"
ANALYSIS_ARTIFACT_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/analysis-artifact-reference@1"
)
ANALYSIS_PROVENANCE_REFERENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/analysis-provenance-reference@1"
)
ANALYSIS_COST_SCHEMA = "ipfs_accelerate_py/agent-supervisor/analysis-cost@1"
ANALYSIS_CANDIDATE_PROPOSAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/analysis-candidate-proposal@1"
)
ANALYSIS_STAGE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/analysis-stage-receipt@1"
)
ANALYSIS_EVIDENCE_PACKET_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/analysis-evidence-packet@1"
)

ARTIFACT_REFERENCE_SCHEMA = ANALYSIS_ARTIFACT_REFERENCE_SCHEMA
PROVENANCE_REFERENCE_SCHEMA = ANALYSIS_PROVENANCE_REFERENCE_SCHEMA
CANDIDATE_PROPOSAL_SCHEMA = ANALYSIS_CANDIDATE_PROPOSAL_SCHEMA
STAGE_RECEIPT_SCHEMA = ANALYSIS_STAGE_RECEIPT_SCHEMA
EVIDENCE_PACKET_SCHEMA = ANALYSIS_EVIDENCE_PACKET_SCHEMA

# Standalone records are bounded even before they are placed in a packet.
# Packet-specific limits are intentionally much smaller by default.
ABSOLUTE_MAX_TEXT_BYTES = 65_536
ABSOLUTE_MAX_RECORD_BYTES = 1_048_576
MILLION = 1_000_000


AnalysisContractValidationError = ContractValidationError


class AnalysisOutcome(str, Enum):
    """Whether analysis reached a trustworthy bounded conclusion."""

    CONCLUSIVE = "conclusive"
    INCONCLUSIVE = "inconclusive"


class AnalysisStageStatus(str, Enum):
    """Execution status of one analysis stage."""

    COMPLETED = "completed"
    SUCCEEDED = "succeeded"
    SUCCESS = "success"
    SUCCESSFUL = "successful"
    SATISFIED = "satisfied"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"

    @property
    def successful(self) -> bool:
        return self in {
            AnalysisStageStatus.COMPLETED,
            AnalysisStageStatus.SUCCEEDED,
            AnalysisStageStatus.SUCCESS,
            AnalysisStageStatus.SUCCESSFUL,
            AnalysisStageStatus.SATISFIED,
        }


class AnalysisFreshness(str, Enum):
    """Freshness state bound into a receipt at the consumption boundary."""

    FRESH = "fresh"
    CURRENT = "current"
    STALE = "stale"

    @property
    def usable(self) -> bool:
        return self in {AnalysisFreshness.FRESH, AnalysisFreshness.CURRENT}


class AnalysisCacheDisposition(str, Enum):
    """How a stage result was obtained from the analysis cache."""

    NOT_CACHED = "not_cached"
    MISS = "miss"
    POSITIVE_HIT = "positive_hit"
    POSITIVE = "positive"
    NEGATIVE_HIT = "negative_hit"
    NEGATIVE = "negative"

    @property
    def negative(self) -> bool:
        return self in {
            AnalysisCacheDisposition.NEGATIVE_HIT,
            AnalysisCacheDisposition.NEGATIVE,
        }


class ProvenanceKind(str, Enum):
    """Bounded vocabulary for analysis evidence origins."""

    ARTIFACT = "artifact"
    AST_RECORD = "ast_record"
    CODE_EVIDENCE = "code_evidence"
    GRAPH_RETRIEVAL = "graph_retrieval"
    OBJECTIVE = "objective"
    SCAN_RECEIPT = "scan_receipt"
    STAGE_RECEIPT = "stage_receipt"
    TOOL = "tool"
    OTHER = "other"


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        normalized = ""
    elif not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    else:
        normalized = value.strip()
    if required and not normalized:
        raise ContractValidationError(f"{field_name} is required")
    if "\x00" in normalized:
        raise ContractValidationError(f"{field_name} must not contain NUL bytes")
    if len(normalized.encode("utf-8")) > max_bytes:
        raise ContractValidationError(
            f"{field_name} exceeds the maximum of {max_bytes} UTF-8 bytes"
        )
    return normalized


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(str(raw))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContractValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: Any, *, field_name: str) -> int:
    result = _nonnegative_int(value, field_name=field_name)
    if result < 1:
        raise ContractValidationError(f"{field_name} must be at least 1")
    return result


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        source: Iterable[Any] = ()
    elif isinstance(values, str):
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractValidationError(f"{field_name} must be a sequence of strings")
    result: list[str] = []
    for index, item in enumerate(source):
        normalized = _text(
            item, field_name=f"{field_name}[{index}]", required=True
        )
        if normalized not in result:
            result.append(normalized)
    if required and not result:
        raise ContractValidationError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _repo_paths(values: Any, *, field_name: str) -> tuple[str, ...]:
    result = _strings(values, field_name=field_name)
    for path in result:
        normalized = path.replace("\\", "/")
        candidate = PurePosixPath(normalized)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ContractValidationError(
                f"{field_name} must contain repository-relative paths"
            )
    return tuple(sorted(path.replace("\\", "/") for path in result))


def _millionths(
    value: Any,
    *,
    field_name: str,
    maximum: int | None,
    already_millionths: bool = False,
) -> int:
    if already_millionths:
        result = _nonnegative_int(value, field_name=field_name)
    else:
        if isinstance(value, bool):
            raise ContractValidationError(f"{field_name} must be numeric")
        try:
            decimal = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ContractValidationError(f"{field_name} must be numeric") from exc
        if not decimal.is_finite() or decimal < 0:
            raise ContractValidationError(
                f"{field_name} must be finite and non-negative"
            )
        result = int(
            (decimal * MILLION).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
    if maximum is not None and result > maximum:
        raise ContractValidationError(
            f"{field_name} must be between 0 and {maximum} millionths"
        )
    return result


def _timestamp(value: Any, *, field_name: str, required: bool = False) -> str:
    if value in (None, ""):
        if required:
            raise ContractValidationError(f"{field_name} is required")
        return ""
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ContractValidationError(
                f"{field_name} must be an ISO-8601 timestamp"
            ) from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise ContractValidationError(
            f"{field_name} must be a datetime or ISO-8601 string"
        )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _schema_and_version(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractValidationError("analysis contract payload must be an object")
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        raise ContractValidationError(
            f"unsupported schema {schema!r}; expected {expected_schema}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, ANALYSIS_CONTRACT_VERSION):
        raise ContractValidationError(
            "unsupported analysis contract version; rebuild with the current contract"
        )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], *, artifact_name: str
) -> None:
    if set(payload).difference(allowed):
        raise ContractValidationError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )


def _claimed_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ContractValidationError(
                f"{artifact_name} content identity does not match payload"
            )


def _bounded_record(value: CanonicalContract, *, artifact_name: str) -> None:
    size = len(value.canonical_bytes())
    if size > ABSOLUTE_MAX_RECORD_BYTES:
        raise ContractValidationError(
            f"{artifact_name} exceeds the absolute record bound of "
            f"{ABSOLUTE_MAX_RECORD_BYTES} bytes"
        )


def _coerce_tuple(
    values: Any,
    expected_type: type,
    decoder: Any,
    *,
    field_name: str,
) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not isinstance(
        values, Sequence
    ):
        raise ContractValidationError(f"{field_name} must be a sequence")
    result: list[Any] = []
    for item in values:
        result.append(item if isinstance(item, expected_type) else decoder(item))
    return tuple(result)


def _unique_sorted(
    values: Sequence[Any], *, identity_name: str, field_name: str
) -> tuple[Any, ...]:
    by_identity: dict[str, Any] = {}
    for item in values:
        identity = str(getattr(item, identity_name))
        previous = by_identity.get(identity)
        if previous is not None:
            if previous != item:
                raise ContractValidationError(
                    f"{field_name} contains conflicting records for {identity}"
                )
            raise ContractValidationError(
                f"{field_name} must not contain duplicate records"
            )
        by_identity[identity] = item
    return tuple(by_identity[key] for key in sorted(by_identity))


@dataclass(frozen=True, init=False)
class AnalysisLimits(CanonicalContract):
    """Count and serialized-byte limits bound into an evidence packet."""

    SCHEMA: ClassVar[str] = ANALYSIS_LIMITS_SCHEMA

    max_stage_receipts: int = 16
    max_candidate_proposals: int = 64
    max_provenance_references: int = 256
    max_artifact_references: int = 128
    max_objective_terms_per_proposal: int = 64
    max_text_bytes: int = 4_096
    max_record_bytes: int = 32_768
    max_serialized_bytes: int = 262_144

    def __init__(
        self,
        max_stage_receipts: int = 16,
        max_candidate_proposals: int = 64,
        max_provenance_references: int = 256,
        max_artifact_references: int = 128,
        max_objective_terms_per_proposal: int = 64,
        max_text_bytes: int = 4_096,
        max_record_bytes: int = 32_768,
        max_serialized_bytes: int = 262_144,
        *,
        max_receipts: int | None = None,
        max_proposals: int | None = None,
        max_provenance: int | None = None,
        max_artifacts: int | None = None,
        max_objective_terms: int | None = None,
        max_packet_bytes: int | None = None,
        max_total_bytes: int | None = None,
    ) -> None:
        values = {
            "max_stage_receipts": (
                max_stage_receipts if max_receipts is None else max_receipts
            ),
            "max_candidate_proposals": (
                max_candidate_proposals
                if max_proposals is None
                else max_proposals
            ),
            "max_provenance_references": (
                max_provenance_references
                if max_provenance is None
                else max_provenance
            ),
            "max_artifact_references": (
                max_artifact_references if max_artifacts is None else max_artifacts
            ),
            "max_objective_terms_per_proposal": (
                max_objective_terms_per_proposal
                if max_objective_terms is None
                else max_objective_terms
            ),
            "max_text_bytes": max_text_bytes,
            "max_record_bytes": max_record_bytes,
            "max_serialized_bytes": (
                max_total_bytes
                if max_total_bytes is not None
                else (
                    max_serialized_bytes
                    if max_packet_bytes is None
                    else max_packet_bytes
                )
            ),
        }
        for name, value in values.items():
            object.__setattr__(
                self,
                name,
                _positive_int(value, field_name=name),
            )
        if self.max_text_bytes > self.max_record_bytes:
            raise ContractValidationError(
                "max_text_bytes cannot exceed max_record_bytes"
            )

    @property
    def max_receipts(self) -> int:
        return self.max_stage_receipts

    @property
    def max_proposals(self) -> int:
        return self.max_candidate_proposals

    @property
    def max_provenance(self) -> int:
        return self.max_provenance_references

    @property
    def max_artifacts(self) -> int:
        return self.max_artifact_references

    @property
    def max_packet_bytes(self) -> int:
        return self.max_serialized_bytes

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "max_stage_receipts": self.max_stage_receipts,
            "max_candidate_proposals": self.max_candidate_proposals,
            "max_provenance_references": self.max_provenance_references,
            "max_artifact_references": self.max_artifact_references,
            "max_objective_terms_per_proposal": self.max_objective_terms_per_proposal,
            "max_text_bytes": self.max_text_bytes,
            "max_record_bytes": self.max_record_bytes,
            "max_serialized_bytes": self.max_serialized_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisLimits":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "max_stage_receipts",
                "max_receipts",
                "max_candidate_proposals",
                "max_proposals",
                "max_provenance_references",
                "max_provenance",
                "max_artifact_references",
                "max_artifacts",
                "max_objective_terms_per_proposal",
                "max_objective_terms",
                "max_text_bytes",
                "max_record_bytes",
                "max_serialized_bytes",
                "max_packet_bytes",
                "max_total_bytes",
                "content_id",
            },
            artifact_name="analysis limits",
        )
        defaults = cls()
        result = cls(
            max_stage_receipts=payload.get(
                "max_stage_receipts",
                payload.get("max_receipts", defaults.max_stage_receipts),
            ),
            max_candidate_proposals=payload.get(
                "max_candidate_proposals",
                payload.get("max_proposals", defaults.max_candidate_proposals),
            ),
            max_provenance_references=payload.get(
                "max_provenance_references",
                payload.get("max_provenance", defaults.max_provenance_references),
            ),
            max_artifact_references=payload.get(
                "max_artifact_references",
                payload.get("max_artifacts", defaults.max_artifact_references),
            ),
            max_objective_terms_per_proposal=payload.get(
                "max_objective_terms_per_proposal",
                payload.get(
                    "max_objective_terms",
                    defaults.max_objective_terms_per_proposal,
                ),
            ),
            max_text_bytes=payload.get("max_text_bytes", defaults.max_text_bytes),
            max_record_bytes=payload.get(
                "max_record_bytes", defaults.max_record_bytes
            ),
            max_serialized_bytes=payload.get(
                "max_serialized_bytes",
                payload.get(
                    "max_packet_bytes",
                    payload.get("max_total_bytes", defaults.max_serialized_bytes),
                ),
            ),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id",),
            artifact_name="analysis limits",
        )
        return result


AnalysisEvidenceLimits = AnalysisLimits
AnalysisContractLimits = AnalysisLimits


@dataclass(frozen=True, init=False)
class ArtifactReference(CanonicalContract):
    """Compact pointer to a durable artifact; never an embedded artifact body."""

    SCHEMA: ClassVar[str] = ANALYSIS_ARTIFACT_REFERENCE_SCHEMA

    artifact_id: str
    kind: str
    uri: str
    artifact_content_id: str
    sha256: str
    media_type: str
    byte_count: int
    record_count: int

    def __init__(
        self,
        artifact_id: str,
        kind: str = "analysis",
        uri: str = "",
        content_id: str = "",
        sha256: str = "",
        media_type: str = "application/json",
        byte_count: int = 0,
        record_count: int = 0,
        *,
        artifact_content_id: str = "",
        reference_content_id: str = "",
    ) -> None:
        values = {
            "artifact_id": artifact_id,
            "kind": kind,
            "uri": uri,
            "artifact_content_id": artifact_content_id or content_id,
            "sha256": sha256,
            "media_type": media_type,
        }
        for name, required in (
            ("artifact_id", True),
            ("kind", True),
            ("uri", False),
            ("artifact_content_id", False),
            ("sha256", False),
            ("media_type", False),
        ):
            object.__setattr__(
                self,
                name,
                _text(values[name], field_name=name, required=required),
            )
        for name, value in (
            ("byte_count", byte_count),
            ("record_count", record_count),
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(value, field_name=name),
            )
        digest = self.sha256.removeprefix("sha256:")
        if digest and (
            len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
        ):
            raise ContractValidationError(
                "sha256 must be a 64-character hexadecimal digest"
            )
        if digest:
            object.__setattr__(self, "sha256", digest.lower())
        _bounded_record(self, artifact_name="artifact reference")
        if reference_content_id and reference_content_id != self.content_id:
            raise ContractValidationError(
                "artifact reference content identity does not match payload"
            )

    @property
    def cid(self) -> str:
        return self.artifact_content_id

    @property
    def path(self) -> str:
        return self.uri

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "uri": self.uri,
            "artifact_content_id": self.artifact_content_id,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "byte_count": self.byte_count,
            "record_count": self.record_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactReference":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "artifact_id",
                "kind",
                "artifact_kind",
                "uri",
                "path",
                "artifact_content_id",
                "content_id",
                "cid",
                "sha256",
                "media_type",
                "byte_count",
                "record_count",
                "reference_content_id",
            },
            artifact_name="artifact reference",
        )
        serialized_contract = bool(payload.get("schema") or payload.get("contract_version"))
        result = cls(
            artifact_id=payload.get("artifact_id", ""),
            kind=payload.get("kind", payload.get("artifact_kind", "analysis")),
            uri=payload.get("uri", payload.get("path", "")),
            artifact_content_id=payload.get(
                "artifact_content_id",
                payload.get(
                    "cid",
                    "" if serialized_contract else payload.get("content_id", ""),
                ),
            ),
            sha256=payload.get("sha256", ""),
            media_type=payload.get("media_type", "application/json"),
            byte_count=payload.get("byte_count", 0),
            record_count=payload.get("record_count", 0),
            reference_content_id=str(
                payload.get("reference_content_id")
                or (payload.get("content_id") if serialized_contract else "")
                or ""
            ),
        )
        return result


AnalysisArtifactReference = ArtifactReference
ArtifactRef = ArtifactReference


@dataclass(frozen=True)
class ProvenanceReference(CanonicalContract):
    """One compact, typed link from a claim to its origin."""

    SCHEMA: ClassVar[str] = ANALYSIS_PROVENANCE_REFERENCE_SCHEMA

    reference_id: str
    kind: ProvenanceKind = ProvenanceKind.OTHER
    artifact: ArtifactReference | None = None
    repository_id: str = ""
    tree_id: str = ""
    path: str = ""
    symbol: str = ""
    record_id: str = ""
    relationship: str = "supports"
    line_start: int = 0
    line_end: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_id",
            _text(self.reference_id, field_name="reference_id", required=True),
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ProvenanceKind, field_name="kind")
        )
        artifact = self.artifact
        if artifact is not None and not isinstance(artifact, ArtifactReference):
            if not isinstance(artifact, Mapping):
                raise ContractValidationError("artifact must be an artifact reference")
            artifact = ArtifactReference.from_dict(artifact)
        object.__setattr__(self, "artifact", artifact)
        for name in (
            "repository_id",
            "tree_id",
            "path",
            "symbol",
            "record_id",
            "relationship",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=name == "relationship",
                ),
            )
        if self.path:
            normalized = self.path.replace("\\", "/")
            candidate = PurePosixPath(normalized)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ContractValidationError("path must be repository-relative")
            object.__setattr__(self, "path", normalized)
        for name in ("line_start", "line_end"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), field_name=name),
            )
        if self.line_end and not self.line_start:
            raise ContractValidationError("line_end requires line_start")
        if self.line_end and self.line_end < self.line_start:
            raise ContractValidationError("line_end cannot precede line_start")
        _bounded_record(self, artifact_name="provenance reference")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "reference_id": self.reference_id,
            "kind": self.kind,
            "artifact": self.artifact.to_record() if self.artifact else None,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "path": self.path,
            "symbol": self.symbol,
            "record_id": self.record_id,
            "relationship": self.relationship,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProvenanceReference":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "reference_id",
                "kind",
                "source_kind",
                "artifact",
                "repository_id",
                "tree_id",
                "path",
                "symbol",
                "record_id",
                "relationship",
                "line_start",
                "line_end",
                "content_id",
            },
            artifact_name="provenance reference",
        )
        result = cls(
            reference_id=payload.get("reference_id", ""),
            kind=payload.get("kind", payload.get("source_kind", ProvenanceKind.OTHER)),
            artifact=payload.get("artifact"),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            path=payload.get("path", ""),
            symbol=payload.get("symbol", ""),
            record_id=payload.get("record_id", ""),
            relationship=payload.get("relationship", "supports"),
            line_start=payload.get("line_start", 0),
            line_end=payload.get("line_end", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id",),
            artifact_name="provenance reference",
        )
        return result


AnalysisProvenanceReference = ProvenanceReference
ProvenanceRef = ProvenanceReference


@dataclass(frozen=True)
class AnalysisCost(CanonicalContract):
    """Integer resource accounting for one bounded analysis operation."""

    SCHEMA: ClassVar[str] = ANALYSIS_COST_SCHEMA

    wall_time_ms: int = 0
    cpu_time_ms: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    records_examined: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
            "wall_time_ms",
            "cpu_time_ms",
            "input_bytes",
            "output_bytes",
            "records_examined",
            "input_tokens",
            "output_tokens",
            "model_calls",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), field_name=name),
            )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "wall_time_ms": self.wall_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "records_examined": self.records_examined,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model_calls": self.model_calls,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisCost":
        _schema_and_version(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "content_id",
            *cls.__dataclass_fields__,
        }
        _reject_unknown(payload, allowed, artifact_name="analysis cost")
        result = cls(
            **{
                name: payload.get(name, 0)
                for name in cls.__dataclass_fields__
                if name != "SCHEMA"
            }
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id",),
            artifact_name="analysis cost",
        )
        return result


@dataclass(frozen=True, init=False)
class CandidateProposal(CanonicalContract):
    """A bounded task candidate with deterministic evidence-backed scoring."""

    SCHEMA: ClassVar[str] = ANALYSIS_CANDIDATE_PROPOSAL_SCHEMA

    summary: str
    objective_terms: tuple[str, ...]
    predicted_files: tuple[str, ...]
    predicted_symbols: tuple[str, ...]
    validation_commands: tuple[str, ...]
    confidence_millionths: int
    novelty_millionths: int
    cost_millionths: int
    provenance: tuple[ProvenanceReference, ...]
    artifacts: tuple[ArtifactReference, ...]
    source_stage: str

    def __init__(
        self,
        summary: str,
        objective_terms: Sequence[str] = (),
        predicted_files: Sequence[str] = (),
        predicted_symbols: Sequence[str] = (),
        validation_commands: Sequence[str] = (),
        *,
        confidence: Any = 0,
        novelty: Any = 0,
        cost: Any = 0,
        confidence_millionths: int | None = None,
        novelty_millionths: int | None = None,
        cost_millionths: int | None = None,
        estimated_cost: Any | None = None,
        estimated_cost_millionths: int | None = None,
        provenance: Sequence[ProvenanceReference | Mapping[str, Any]] = (),
        artifacts: Sequence[ArtifactReference | Mapping[str, Any]] = (),
        source_stage: str = "",
        proposal_id: str = "",
    ) -> None:
        object.__setattr__(
            self, "summary", _text(summary, field_name="summary", required=True)
        )
        object.__setattr__(
            self,
            "objective_terms",
            _strings(objective_terms, field_name="objective_terms", required=True),
        )
        object.__setattr__(
            self,
            "predicted_files",
            _repo_paths(predicted_files, field_name="predicted_files"),
        )
        object.__setattr__(
            self,
            "predicted_symbols",
            _strings(predicted_symbols, field_name="predicted_symbols"),
        )
        object.__setattr__(
            self,
            "validation_commands",
            _strings(
                validation_commands,
                field_name="validation_commands",
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self,
            "confidence_millionths",
            _millionths(
                confidence_millionths if confidence_millionths is not None else confidence,
                field_name="confidence",
                maximum=MILLION,
                already_millionths=confidence_millionths is not None,
            ),
        )
        object.__setattr__(
            self,
            "novelty_millionths",
            _millionths(
                novelty_millionths if novelty_millionths is not None else novelty,
                field_name="novelty",
                maximum=MILLION,
                already_millionths=novelty_millionths is not None,
            ),
        )
        object.__setattr__(
            self,
            "cost_millionths",
            _millionths(
                (
                    cost_millionths
                    if cost_millionths is not None
                    else (
                        estimated_cost_millionths
                        if estimated_cost_millionths is not None
                        else (cost if estimated_cost is None else estimated_cost)
                    )
                ),
                field_name="cost",
                maximum=None,
                already_millionths=(
                    cost_millionths is not None
                    or estimated_cost_millionths is not None
                ),
            ),
        )
        provenance_items = _coerce_tuple(
            provenance,
            ProvenanceReference,
            ProvenanceReference.from_dict,
            field_name="provenance",
        )
        artifact_items = _coerce_tuple(
            artifacts,
            ArtifactReference,
            ArtifactReference.from_dict,
            field_name="artifacts",
        )
        object.__setattr__(
            self,
            "provenance",
            _unique_sorted(
                provenance_items,
                identity_name="reference_id",
                field_name="provenance",
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _unique_sorted(
                artifact_items,
                identity_name="artifact_id",
                field_name="artifacts",
            ),
        )
        object.__setattr__(
            self,
            "source_stage",
            _text(source_stage, field_name="source_stage"),
        )
        _bounded_record(self, artifact_name="candidate proposal")
        if proposal_id and proposal_id != self.proposal_id:
            raise ContractValidationError(
                "candidate proposal content identity does not match payload"
            )

    @property
    def confidence(self) -> float:
        return self.confidence_millionths / MILLION

    @property
    def novelty(self) -> float:
        return self.novelty_millionths / MILLION

    @property
    def cost(self) -> float:
        return self.cost_millionths / MILLION

    @property
    def proposal_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "summary": self.summary,
            "objective_terms": self.objective_terms,
            "predicted_files": self.predicted_files,
            "predicted_symbols": self.predicted_symbols,
            "validation_commands": self.validation_commands,
            "confidence_millionths": self.confidence_millionths,
            "novelty_millionths": self.novelty_millionths,
            "cost_millionths": self.cost_millionths,
            "provenance": tuple(item.to_record() for item in self.provenance),
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "source_stage": self.source_stage,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "proposal_id": self.proposal_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateProposal":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "summary",
                "objective_terms",
                "predicted_files",
                "predicted_symbols",
                "validation_commands",
                "confidence",
                "confidence_millionths",
                "novelty",
                "novelty_millionths",
                "cost",
                "cost_millionths",
                "estimated_cost",
                "estimated_cost_millionths",
                "provenance",
                "artifacts",
                "source_stage",
                "proposal_id",
                "content_id",
            },
            artifact_name="candidate proposal",
        )
        result = cls(
            summary=payload.get("summary", ""),
            objective_terms=tuple(payload.get("objective_terms") or ()),
            predicted_files=tuple(payload.get("predicted_files") or ()),
            predicted_symbols=tuple(payload.get("predicted_symbols") or ()),
            validation_commands=tuple(payload.get("validation_commands") or ()),
            confidence=payload.get("confidence", 0),
            novelty=payload.get("novelty", 0),
            cost=payload.get("cost", payload.get("estimated_cost", 0)),
            confidence_millionths=payload.get("confidence_millionths"),
            novelty_millionths=payload.get("novelty_millionths"),
            cost_millionths=payload.get(
                "cost_millionths", payload.get("estimated_cost_millionths")
            ),
            provenance=tuple(payload.get("provenance") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
            source_stage=payload.get("source_stage", ""),
            proposal_id=str(
                payload.get("proposal_id") or payload.get("content_id") or ""
            ),
        )
        return result


AnalysisCandidateProposal = CandidateProposal
CandidateAnalysisProposal = CandidateProposal


@dataclass(frozen=True, init=False)
class AnalysisStageReceipt(CanonicalContract):
    """Immutable result of one bounded analysis stage."""

    SCHEMA: ClassVar[str] = ANALYSIS_STAGE_RECEIPT_SCHEMA

    stage: str
    status: AnalysisStageStatus
    outcome: AnalysisOutcome
    analyzer_id: str
    analyzer_version: str
    repository_id: str
    tree_id: str
    objective_revision: str
    configuration_digest: str
    query_digest: str
    policy_digest: str
    freshness: AnalysisFreshness
    cache_disposition: AnalysisCacheDisposition
    coverage_complete: bool
    truncated: bool
    reason_code: str
    error_code: str
    started_at: str
    finished_at: str
    cache_expires_at: str
    confidence_millionths: int
    novelty_millionths: int
    cost: AnalysisCost
    proposals: tuple[CandidateProposal, ...]
    provenance: tuple[ProvenanceReference, ...]
    artifacts: tuple[ArtifactReference, ...]

    def __init__(
        self,
        stage: str,
        status: AnalysisStageStatus | str,
        outcome: AnalysisOutcome | str,
        analyzer_id: str,
        analyzer_version: str,
        repository_id: str,
        tree_id: str,
        objective_revision: str,
        *,
        configuration_digest: str = "",
        query_digest: str = "",
        policy_digest: str = "",
        freshness: AnalysisFreshness | str = AnalysisFreshness.FRESH,
        cache_disposition: AnalysisCacheDisposition | str = (
            AnalysisCacheDisposition.NOT_CACHED
        ),
        coverage_complete: bool = True,
        truncated: bool = False,
        reason_code: str = "",
        error_code: str = "",
        started_at: datetime | str = "",
        finished_at: datetime | str = "",
        cache_expires_at: datetime | str = "",
        confidence: Any = 0,
        novelty: Any = 0,
        confidence_millionths: int | None = None,
        novelty_millionths: int | None = None,
        cost: AnalysisCost | Mapping[str, Any] | None = None,
        proposals: Sequence[CandidateProposal | Mapping[str, Any]] = (),
        provenance: Sequence[ProvenanceReference | Mapping[str, Any]] = (),
        artifacts: Sequence[ArtifactReference | Mapping[str, Any]] = (),
        receipt_id: str = "",
    ) -> None:
        for name, value in (
            ("stage", stage),
            ("analyzer_id", analyzer_id),
            ("analyzer_version", analyzer_version),
            ("repository_id", repository_id),
            ("tree_id", tree_id),
            ("objective_revision", objective_revision),
        ):
            object.__setattr__(
                self, name, _text(value, field_name=name, required=True)
            )
        for name, value in (
            ("configuration_digest", configuration_digest),
            ("query_digest", query_digest),
            ("policy_digest", policy_digest),
            ("reason_code", reason_code),
            ("error_code", error_code),
        ):
            object.__setattr__(self, name, _text(value, field_name=name))
        object.__setattr__(
            self,
            "status",
            _enum(status, AnalysisStageStatus, field_name="status"),
        )
        object.__setattr__(
            self,
            "outcome",
            _enum(outcome, AnalysisOutcome, field_name="outcome"),
        )
        object.__setattr__(
            self,
            "freshness",
            _enum(freshness, AnalysisFreshness, field_name="freshness"),
        )
        object.__setattr__(
            self,
            "cache_disposition",
            _enum(
                cache_disposition,
                AnalysisCacheDisposition,
                field_name="cache_disposition",
            ),
        )
        if not isinstance(coverage_complete, bool):
            raise ContractValidationError("coverage_complete must be a boolean")
        if not isinstance(truncated, bool):
            raise ContractValidationError("truncated must be a boolean")
        object.__setattr__(self, "coverage_complete", coverage_complete)
        object.__setattr__(self, "truncated", truncated)
        normalized_started = _timestamp(started_at, field_name="started_at")
        normalized_finished = _timestamp(finished_at, field_name="finished_at")
        if bool(normalized_started) != bool(normalized_finished):
            raise ContractValidationError(
                "started_at and finished_at must either both be set or both be empty"
            )
        if normalized_started and normalized_finished < normalized_started:
            raise ContractValidationError(
                "finished_at must not be earlier than started_at"
            )
        object.__setattr__(self, "started_at", normalized_started)
        object.__setattr__(self, "finished_at", normalized_finished)
        object.__setattr__(
            self,
            "cache_expires_at",
            _timestamp(cache_expires_at, field_name="cache_expires_at"),
        )
        object.__setattr__(
            self,
            "confidence_millionths",
            _millionths(
                confidence_millionths if confidence_millionths is not None else confidence,
                field_name="confidence",
                maximum=MILLION,
                already_millionths=confidence_millionths is not None,
            ),
        )
        object.__setattr__(
            self,
            "novelty_millionths",
            _millionths(
                novelty_millionths if novelty_millionths is not None else novelty,
                field_name="novelty",
                maximum=MILLION,
                already_millionths=novelty_millionths is not None,
            ),
        )
        if cost is None:
            normalized_cost = AnalysisCost()
        elif isinstance(cost, AnalysisCost):
            normalized_cost = cost
        elif isinstance(cost, Mapping):
            normalized_cost = AnalysisCost.from_dict(cost)
        else:
            raise ContractValidationError("cost must be an AnalysisCost or mapping")
        object.__setattr__(self, "cost", normalized_cost)
        proposal_items = _coerce_tuple(
            proposals,
            CandidateProposal,
            CandidateProposal.from_dict,
            field_name="proposals",
        )
        provenance_items = _coerce_tuple(
            provenance,
            ProvenanceReference,
            ProvenanceReference.from_dict,
            field_name="provenance",
        )
        artifact_items = _coerce_tuple(
            artifacts,
            ArtifactReference,
            ArtifactReference.from_dict,
            field_name="artifacts",
        )
        object.__setattr__(
            self,
            "proposals",
            _unique_sorted(
                proposal_items,
                identity_name="proposal_id",
                field_name="proposals",
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _unique_sorted(
                provenance_items,
                identity_name="reference_id",
                field_name="provenance",
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _unique_sorted(
                artifact_items,
                identity_name="artifact_id",
                field_name="artifacts",
            ),
        )
        if self.status in {
            AnalysisStageStatus.FAILED,
            AnalysisStageStatus.TIMED_OUT,
        } and not (self.error_code or self.reason_code):
            raise ContractValidationError(
                f"{self.status.value} stage receipts require an error_code or reason_code"
            )
        if self.status in {
            AnalysisStageStatus.FAILED,
            AnalysisStageStatus.TIMED_OUT,
            AnalysisStageStatus.SKIPPED,
        } and self.proposals:
            raise ContractValidationError(
                f"{self.status.value} stage receipts cannot contain candidate proposals"
            )
        if self.outcome is AnalysisOutcome.CONCLUSIVE and not self._completion_gate:
            raise ContractValidationError(
                "conclusive stage receipts must be successful, fresh, coverage-complete, "
                "non-truncated, error-free, and not negative-cache hits"
            )
        _bounded_record(self, artifact_name="analysis stage receipt")
        if receipt_id and receipt_id != self.receipt_id:
            raise ContractValidationError(
                "analysis stage receipt content identity does not match payload"
            )

    @property
    def confidence(self) -> float:
        return self.confidence_millionths / MILLION

    @property
    def novelty(self) -> float:
        return self.novelty_millionths / MILLION

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def _completion_gate(self) -> bool:
        return (
            self.status.successful
            and self.freshness.usable
            and not self.cache_disposition.negative
            and self.coverage_complete
            and not self.truncated
            and not self.error_code
        )

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.outcome is AnalysisOutcome.CONCLUSIVE and self._completion_gate

    @property
    def is_completion_evidence(self) -> bool:
        return self.safe_for_completion_reasoning

    @property
    def conclusive(self) -> bool:
        return self.outcome is AnalysisOutcome.CONCLUSIVE

    @property
    def inconclusive(self) -> bool:
        return self.outcome is AnalysisOutcome.INCONCLUSIVE

    def __bool__(self) -> bool:
        raise TypeError(
            "AnalysisStageReceipt has no truth value; inspect outcome and "
            "safe_for_completion_reasoning explicitly"
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "stage": self.stage,
            "status": self.status,
            "outcome": self.outcome,
            "analyzer_id": self.analyzer_id,
            "analyzer_version": self.analyzer_version,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "policy_digest": self.policy_digest,
            "freshness": self.freshness,
            "cache_disposition": self.cache_disposition,
            "coverage_complete": self.coverage_complete,
            "truncated": self.truncated,
            "reason_code": self.reason_code,
            "error_code": self.error_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cache_expires_at": self.cache_expires_at,
            "confidence_millionths": self.confidence_millionths,
            "novelty_millionths": self.novelty_millionths,
            "cost": self.cost.to_record(),
            "proposals": tuple(item.to_record() for item in self.proposals),
            "provenance": tuple(item.to_record() for item in self.provenance),
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisStageReceipt":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "stage",
                "status",
                "outcome",
                "analyzer_id",
                "analyzer_version",
                "repository_id",
                "tree_id",
                "objective_revision",
                "configuration_digest",
                "query_digest",
                "policy_digest",
                "freshness",
                "cache_disposition",
                "coverage_complete",
                "truncated",
                "reason_code",
                "error_code",
                "started_at",
                "finished_at",
                "cache_expires_at",
                "confidence",
                "confidence_millionths",
                "novelty",
                "novelty_millionths",
                "cost",
                "proposals",
                "provenance",
                "artifacts",
                "safe_for_completion_reasoning",
                "is_completion_evidence",
                "receipt_id",
                "content_id",
            },
            artifact_name="analysis stage receipt",
        )
        result = cls(
            stage=payload.get("stage", ""),
            status=payload.get("status", ""),
            outcome=payload.get("outcome", AnalysisOutcome.INCONCLUSIVE),
            analyzer_id=payload.get("analyzer_id", ""),
            analyzer_version=payload.get("analyzer_version", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            configuration_digest=payload.get("configuration_digest", ""),
            query_digest=payload.get("query_digest", ""),
            policy_digest=payload.get("policy_digest", ""),
            freshness=payload.get("freshness", AnalysisFreshness.FRESH),
            cache_disposition=payload.get(
                "cache_disposition", AnalysisCacheDisposition.NOT_CACHED
            ),
            coverage_complete=payload.get("coverage_complete", True),
            truncated=payload.get("truncated", False),
            reason_code=payload.get("reason_code", ""),
            error_code=payload.get("error_code", ""),
            started_at=payload.get("started_at", ""),
            finished_at=payload.get("finished_at", ""),
            cache_expires_at=payload.get("cache_expires_at", ""),
            confidence=payload.get("confidence", 0),
            novelty=payload.get("novelty", 0),
            confidence_millionths=payload.get("confidence_millionths"),
            novelty_millionths=payload.get("novelty_millionths"),
            cost=payload.get("cost") or {},
            proposals=tuple(payload.get("proposals") or ()),
            provenance=tuple(payload.get("provenance") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
            receipt_id=str(
                payload.get("receipt_id") or payload.get("content_id") or ""
            ),
        )
        for claim in ("safe_for_completion_reasoning", "is_completion_evidence"):
            if claim in payload and bool(payload[claim]) != result.is_completion_evidence:
                raise ContractValidationError(
                    "stage receipt completion-evidence claim does not match derived state"
                )
        return result


StageReceipt = AnalysisStageReceipt
AnalysisReceipt = AnalysisStageReceipt


@dataclass(frozen=True, init=False)
class AnalysisEvidencePacket(CanonicalContract):
    """Bounded aggregate exchanged between analyzers and the supervisor."""

    SCHEMA: ClassVar[str] = ANALYSIS_EVIDENCE_PACKET_SCHEMA

    repository_id: str
    tree_id: str
    objective_revision: str
    outcome: AnalysisOutcome
    conclusion_code: str
    coverage_complete: bool
    truncated: bool
    stage_receipts: tuple[AnalysisStageReceipt, ...]
    candidate_proposals: tuple[CandidateProposal, ...]
    provenance: tuple[ProvenanceReference, ...]
    artifacts: tuple[ArtifactReference, ...]
    limits: AnalysisLimits

    def __init__(
        self,
        repository_id: str,
        tree_id: str,
        objective_revision: str,
        outcome: AnalysisOutcome | str,
        *,
        conclusion_code: str = "",
        coverage_complete: bool = True,
        truncated: bool = False,
        stage_receipts: Sequence[AnalysisStageReceipt | Mapping[str, Any]] = (),
        candidate_proposals: Sequence[CandidateProposal | Mapping[str, Any]] = (),
        provenance: Sequence[ProvenanceReference | Mapping[str, Any]] = (),
        artifacts: Sequence[ArtifactReference | Mapping[str, Any]] = (),
        limits: AnalysisLimits | Mapping[str, Any] | None = None,
        packet_id: str = "",
        receipts: Sequence[AnalysisStageReceipt | Mapping[str, Any]] | None = None,
        proposals: Sequence[CandidateProposal | Mapping[str, Any]] | None = None,
        bounds: AnalysisLimits | Mapping[str, Any] | None = None,
    ) -> None:
        for name, value in (
            ("repository_id", repository_id),
            ("tree_id", tree_id),
            ("objective_revision", objective_revision),
        ):
            object.__setattr__(
                self, name, _text(value, field_name=name, required=True)
            )
        object.__setattr__(
            self,
            "outcome",
            _enum(outcome, AnalysisOutcome, field_name="outcome"),
        )
        object.__setattr__(
            self,
            "conclusion_code",
            _text(conclusion_code, field_name="conclusion_code"),
        )
        if not isinstance(coverage_complete, bool):
            raise ContractValidationError("coverage_complete must be a boolean")
        if not isinstance(truncated, bool):
            raise ContractValidationError("truncated must be a boolean")
        object.__setattr__(self, "coverage_complete", coverage_complete)
        object.__setattr__(self, "truncated", truncated)
        if limits is not None and bounds is not None:
            raise ContractValidationError("use either limits or bounds, not both")
        selected_limits = limits if bounds is None else bounds
        if selected_limits is None:
            normalized_limits = AnalysisLimits()
        elif isinstance(selected_limits, AnalysisLimits):
            normalized_limits = selected_limits
        elif isinstance(selected_limits, Mapping):
            normalized_limits = AnalysisLimits.from_dict(selected_limits)
        else:
            raise ContractValidationError("limits must be AnalysisLimits or a mapping")
        object.__setattr__(self, "limits", normalized_limits)
        if stage_receipts and receipts is not None:
            raise ContractValidationError(
                "use either stage_receipts or receipts, not both"
            )
        if candidate_proposals and proposals is not None:
            raise ContractValidationError(
                "use either candidate_proposals or proposals, not both"
            )
        selected_receipts = stage_receipts if receipts is None else receipts
        selected_proposals = (
            candidate_proposals if proposals is None else proposals
        )
        receipt_items = _coerce_tuple(
            selected_receipts,
            AnalysisStageReceipt,
            AnalysisStageReceipt.from_dict,
            field_name="stage_receipts",
        )
        proposal_items = _coerce_tuple(
            selected_proposals,
            CandidateProposal,
            CandidateProposal.from_dict,
            field_name="candidate_proposals",
        )
        provenance_items = _coerce_tuple(
            provenance,
            ProvenanceReference,
            ProvenanceReference.from_dict,
            field_name="provenance",
        )
        artifact_items = _coerce_tuple(
            artifacts,
            ArtifactReference,
            ArtifactReference.from_dict,
            field_name="artifacts",
        )
        object.__setattr__(
            self,
            "stage_receipts",
            _unique_sorted(
                receipt_items,
                identity_name="receipt_id",
                field_name="stage_receipts",
            ),
        )
        object.__setattr__(
            self,
            "candidate_proposals",
            _unique_sorted(
                proposal_items,
                identity_name="proposal_id",
                field_name="candidate_proposals",
            ),
        )
        object.__setattr__(
            self,
            "provenance",
            _unique_sorted(
                provenance_items,
                identity_name="reference_id",
                field_name="provenance",
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _unique_sorted(
                artifact_items,
                identity_name="artifact_id",
                field_name="artifacts",
            ),
        )
        for receipt in self.stage_receipts:
            for name in ("repository_id", "tree_id", "objective_revision"):
                if getattr(receipt, name) != getattr(self, name):
                    raise ContractValidationError(
                        f"stage receipt {name} does not match its evidence packet"
                    )
        if self.outcome is AnalysisOutcome.CONCLUSIVE and not self._completion_gate:
            raise ContractValidationError(
                "conclusive evidence packets require complete, non-truncated coverage "
                "and at least one completion-eligible stage receipt"
            )
        self._validate_bounds()
        if packet_id and packet_id != self.packet_id:
            raise ContractValidationError(
                "analysis evidence packet content identity does not match payload"
            )

    @property
    def packet_id(self) -> str:
        return self.content_id

    @property
    def proposals(self) -> tuple[CandidateProposal, ...]:
        return self.candidate_proposals

    @property
    def receipts(self) -> tuple[AnalysisStageReceipt, ...]:
        return self.stage_receipts

    @property
    def completion_evidence_receipts(self) -> tuple[AnalysisStageReceipt, ...]:
        return tuple(
            receipt
            for receipt in self.stage_receipts
            if receipt.safe_for_completion_reasoning
        )

    @property
    def completion_evidence_receipt_ids(self) -> tuple[str, ...]:
        return tuple(
            receipt.receipt_id for receipt in self.completion_evidence_receipts
        )

    @property
    def _completion_gate(self) -> bool:
        return (
            self.coverage_complete
            and not self.truncated
            and bool(self.completion_evidence_receipts)
        )

    @property
    def safe_for_completion_reasoning(self) -> bool:
        return self.outcome is AnalysisOutcome.CONCLUSIVE and self._completion_gate

    @property
    def is_completion_evidence(self) -> bool:
        return self.safe_for_completion_reasoning

    @property
    def conclusive(self) -> bool:
        return self.outcome is AnalysisOutcome.CONCLUSIVE

    @property
    def inconclusive(self) -> bool:
        return self.outcome is AnalysisOutcome.INCONCLUSIVE

    @property
    def serialized_byte_count(self) -> int:
        return len(self.canonical_bytes())

    def __bool__(self) -> bool:
        raise TypeError(
            "AnalysisEvidencePacket has no truth value; inspect outcome and "
            "safe_for_completion_reasoning explicitly"
        )

    def require_completion_evidence(self) -> None:
        if not self.safe_for_completion_reasoning:
            raise ContractValidationError(
                "analysis evidence packet is not safe for completion reasoning"
            )

    def _all_proposals(self) -> tuple[CandidateProposal, ...]:
        return self.candidate_proposals + tuple(
            proposal
            for receipt in self.stage_receipts
            for proposal in receipt.proposals
        )

    def _all_provenance(self) -> tuple[ProvenanceReference, ...]:
        proposals = self._all_proposals()
        return (
            self.provenance
            + tuple(
                reference
                for receipt in self.stage_receipts
                for reference in receipt.provenance
            )
            + tuple(
                reference for proposal in proposals for reference in proposal.provenance
            )
        )

    def _all_artifacts(self) -> tuple[ArtifactReference, ...]:
        proposals = self._all_proposals()
        provenance = self._all_provenance()
        return (
            self.artifacts
            + tuple(
                artifact
                for receipt in self.stage_receipts
                for artifact in receipt.artifacts
            )
            + tuple(
                artifact for proposal in proposals for artifact in proposal.artifacts
            )
            + tuple(
                reference.artifact
                for reference in provenance
                if reference.artifact is not None
            )
        )

    def _validate_bounds(self) -> None:
        limits = self.limits
        collections = (
            (
                "stage_receipts",
                len(self.stage_receipts),
                limits.max_stage_receipts,
            ),
            (
                "candidate_proposals",
                len(self._all_proposals()),
                limits.max_candidate_proposals,
            ),
            (
                "provenance_references",
                len(self._all_provenance()),
                limits.max_provenance_references,
            ),
            (
                "artifact_references",
                len(self._all_artifacts()),
                limits.max_artifact_references,
            ),
        )
        for name, actual, maximum in collections:
            if actual > maximum:
                raise ContractValidationError(
                    f"{name} count {actual} exceeds configured limit {maximum}"
                )
        for proposal in self._all_proposals():
            if len(proposal.objective_terms) > limits.max_objective_terms_per_proposal:
                raise ContractValidationError(
                    "proposal objective_terms exceed the configured count limit"
                )
        text_values: list[tuple[str, str]] = [
            ("conclusion_code", self.conclusion_code)
        ]
        for receipt in self.stage_receipts:
            text_values.extend(
                (
                    ("stage", receipt.stage),
                    ("reason_code", receipt.reason_code),
                    ("error_code", receipt.error_code),
                )
            )
        for proposal in self._all_proposals():
            text_values.append(("proposal summary", proposal.summary))
            text_values.extend(
                ("objective term", value) for value in proposal.objective_terms
            )
        for field_name, value in text_values:
            if len(value.encode("utf-8")) > limits.max_text_bytes:
                raise ContractValidationError(
                    f"{field_name} exceeds configured max_text_bytes"
                )
        records: tuple[CanonicalContract, ...] = (
            *self.stage_receipts,
            *self._all_proposals(),
            *self._all_provenance(),
            *self._all_artifacts(),
        )
        for record in records:
            if len(record.canonical_bytes()) > limits.max_record_bytes:
                raise ContractValidationError(
                    f"{type(record).__name__} exceeds configured max_record_bytes"
                )
        if len(self.canonical_bytes()) > limits.max_serialized_bytes:
            raise ContractValidationError(
                "analysis evidence packet exceeds configured max_serialized_bytes"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": ANALYSIS_CONTRACT_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_revision": self.objective_revision,
            "outcome": self.outcome,
            "conclusion_code": self.conclusion_code,
            "coverage_complete": self.coverage_complete,
            "truncated": self.truncated,
            "stage_receipts": tuple(
                receipt.to_record() for receipt in self.stage_receipts
            ),
            "candidate_proposals": tuple(
                proposal.to_record() for proposal in self.candidate_proposals
            ),
            "provenance": tuple(item.to_record() for item in self.provenance),
            "artifacts": tuple(item.to_record() for item in self.artifacts),
            "limits": self.limits.to_record(),
            "completion_evidence_receipt_ids": (
                self.completion_evidence_receipt_ids
            ),
            "safe_for_completion_reasoning": self.safe_for_completion_reasoning,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "packet_id": self.packet_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisEvidencePacket":
        _schema_and_version(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "repository_id",
                "tree_id",
                "objective_revision",
                "outcome",
                "conclusion_code",
                "coverage_complete",
                "truncated",
                "stage_receipts",
                "receipts",
                "candidate_proposals",
                "proposals",
                "provenance",
                "artifacts",
                "limits",
                "completion_evidence_receipt_ids",
                "safe_for_completion_reasoning",
                "is_completion_evidence",
                "packet_id",
                "content_id",
            },
            artifact_name="analysis evidence packet",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            outcome=payload.get("outcome", AnalysisOutcome.INCONCLUSIVE),
            conclusion_code=payload.get("conclusion_code", ""),
            coverage_complete=payload.get("coverage_complete", True),
            truncated=payload.get("truncated", False),
            stage_receipts=tuple(
                payload.get("stage_receipts", payload.get("receipts")) or ()
            ),
            candidate_proposals=tuple(
                payload.get(
                    "candidate_proposals", payload.get("proposals")
                )
                or ()
            ),
            provenance=tuple(payload.get("provenance") or ()),
            artifacts=tuple(payload.get("artifacts") or ()),
            limits=payload.get("limits") or {},
            packet_id=str(payload.get("packet_id") or payload.get("content_id") or ""),
        )
        for claim in ("safe_for_completion_reasoning", "is_completion_evidence"):
            if claim in payload and bool(payload[claim]) != result.is_completion_evidence:
                raise ContractValidationError(
                    "packet completion-evidence claim does not match derived state"
                )
        if "completion_evidence_receipt_ids" in payload:
            supplied = tuple(payload.get("completion_evidence_receipt_ids") or ())
            if supplied != result.completion_evidence_receipt_ids:
                raise ContractValidationError(
                    "packet completion receipt IDs do not match derived state"
                )
        return result


EvidencePacket = AnalysisEvidencePacket
BoundedAnalysisEvidencePacket = AnalysisEvidencePacket
AnalysisEvidenceOutcome = AnalysisOutcome
AnalysisStageOutcome = AnalysisOutcome
AnalysisReceiptStatus = AnalysisStageStatus
StageStatus = AnalysisStageStatus
CacheDisposition = AnalysisCacheDisposition
Freshness = AnalysisFreshness
EvidenceLimits = AnalysisLimits


def analysis_content_identity(value: CanonicalContract | Mapping[str, Any]) -> str:
    """Return the established CIDv1 content identity for an analysis value."""

    if isinstance(value, CanonicalContract):
        return value.content_id
    from .formal_verification_contracts import content_identity

    return content_identity(value)


def canonical_analysis_json_bytes(
    value: CanonicalContract | Mapping[str, Any],
) -> bytes:
    """Return deterministic canonical bytes for an analysis value."""

    return canonical_json_bytes(value.to_dict() if isinstance(value, CanonicalContract) else value)


__all__ = [
    "ABSOLUTE_MAX_RECORD_BYTES",
    "ABSOLUTE_MAX_TEXT_BYTES",
    "ANALYSIS_ARTIFACT_REFERENCE_SCHEMA",
    "ANALYSIS_CANDIDATE_PROPOSAL_SCHEMA",
    "ANALYSIS_CONTRACT_VERSION",
    "ANALYSIS_COST_SCHEMA",
    "ANALYSIS_EVIDENCE_PACKET_SCHEMA",
    "ANALYSIS_LIMITS_SCHEMA",
    "ANALYSIS_PROVENANCE_REFERENCE_SCHEMA",
    "ANALYSIS_STAGE_RECEIPT_SCHEMA",
    "ARTIFACT_REFERENCE_SCHEMA",
    "AnalysisArtifactReference",
    "AnalysisCacheDisposition",
    "AnalysisCandidateProposal",
    "AnalysisContractLimits",
    "AnalysisContractValidationError",
    "AnalysisCost",
    "AnalysisEvidenceLimits",
    "AnalysisEvidenceOutcome",
    "AnalysisEvidencePacket",
    "AnalysisFreshness",
    "AnalysisLimits",
    "AnalysisOutcome",
    "AnalysisProvenanceReference",
    "AnalysisReceipt",
    "AnalysisReceiptStatus",
    "AnalysisStageReceipt",
    "AnalysisStageOutcome",
    "AnalysisStageStatus",
    "ArtifactRef",
    "ArtifactReference",
    "BoundedAnalysisEvidencePacket",
    "CONTRACT_VERSION",
    "CANDIDATE_PROPOSAL_SCHEMA",
    "CacheDisposition",
    "CandidateAnalysisProposal",
    "CandidateProposal",
    "EvidencePacket",
    "EvidenceLimits",
    "EVIDENCE_PACKET_SCHEMA",
    "Freshness",
    "MILLION",
    "ProvenanceKind",
    "PROVENANCE_REFERENCE_SCHEMA",
    "ProvenanceRef",
    "ProvenanceReference",
    "SCHEMA_VERSION",
    "StageReceipt",
    "STAGE_RECEIPT_SCHEMA",
    "StageStatus",
    "analysis_content_identity",
    "canonical_analysis_json_bytes",
]
