"""Closed contracts shared by the Verified Residual Intelligence Foundry.

The foundry is deliberately a candidate producer.  These contracts do not
grant authority, accept proofs, promote checkpoints, or mark work complete.
They provide bounded, content-addressed records that existing supervisor
authorities may validate and consume.
"""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import content_identity

PROGRAM_ID: Final = "agent-supervisor-verified-residual-intelligence-foundry-v1"
CONTRACT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-intelligence-contracts@1"
MAX_TEXT_BYTES: Final = 16_384
MAX_SEQUENCE_ITEMS: Final = 1_024
MAX_MAPPING_ITEMS: Final = 1_024
MAX_NESTING_DEPTH: Final = 8


class ResidualIntelligenceError(ValueError):
    """A residual-intelligence record violates its closed contract."""


class UnknownFieldError(ResidualIntelligenceError):
    """A strict record contains an undeclared field."""


class AuthorityViolationError(ResidualIntelligenceError):
    """A candidate attempted to create authority or completion."""


class ResidualTaskFamily(str, Enum):
    TASK_CLASSIFICATION = "TASK_CLASSIFICATION"
    RISK_CLASSIFICATION = "RISK_CLASSIFICATION"
    EFFECT_CLASSIFICATION = "EFFECT_CLASSIFICATION"
    AUTHORITY_REQUIREMENT_CLASSIFICATION = "AUTHORITY_REQUIREMENT_CLASSIFICATION"
    CONTEXT_SUFFICIENCY = "CONTEXT_SUFFICIENCY"
    EVIDENCE_RANKING = "EVIDENCE_RANKING"
    PROCEDURE_MATCHING = "PROCEDURE_MATCHING"
    PLAN_BRANCH_RANKING = "PLAN_BRANCH_RANKING"
    TEST_SELECTION = "TEST_SELECTION"
    PROOF_SELECTION = "PROOF_SELECTION"
    FAILURE_ATTRIBUTION = "FAILURE_ATTRIBUTION"
    RETRY_OR_ESCALATE = "RETRY_OR_ESCALATE"
    CACHE_REUSE_CLASSIFICATION = "CACHE_REUSE_CLASSIFICATION"
    MERGE_CONFLICT_CLASSIFICATION = "MERGE_CONFLICT_CLASSIFICATION"
    PATCH_TEMPLATE_SELECTION = "PATCH_TEMPLATE_SELECTION"
    PROCEDURE_HOLE_FILLING = "PROCEDURE_HOLE_FILLING"
    PATCH_SKETCH_GENERATION = "PATCH_SKETCH_GENERATION"
    LEMMA_SUGGESTION = "LEMMA_SUGGESTION"
    TACTIC_SUGGESTION = "TACTIC_SUGGESTION"
    COUNTEREXAMPLE_EXPLANATION = "COUNTEREXAMPLE_EXPLANATION"
    GOAL_REFINEMENT_CANDIDATE = "GOAL_REFINEMENT_CANDIDATE"
    DOCUMENTATION_CLAIM_CLASSIFICATION = "DOCUMENTATION_CLAIM_CLASSIFICATION"
    HUMAN_ESCALATION_CLASSIFICATION = "HUMAN_ESCALATION_CLASSIFICATION"
    NOVEL_UNBOUNDED_REASONING = "NOVEL_UNBOUNDED_REASONING"


class RiskClass(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"


class PrivacyClass(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    REPOSITORY_PRIVATE = "repository_private"
    TENANT_PRIVATE = "tenant_private"
    MATTER_CONFIDENTIAL = "matter_confidential"
    CREDENTIAL = "credential"
    PERSONAL_DATA = "personal_data"
    HEALTH_DATA = "health_data"
    LEGAL_PRIVILEGED = "legal_privileged"
    PROOF_WITNESS = "proof_witness"


class ExpertDisposition(str, Enum):
    ACCEPT = "ACCEPT"
    ABSTAIN = "ABSTAIN"
    REJECT_INPUT = "REJECT_INPUT"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    VALIDATION_REQUIRED = "VALIDATION_REQUIRED"


class PrerequisiteStatus(str, Enum):
    AVAILABLE = "available"
    AVAILABLE_WITH_CAVEATS = "available_with_caveats"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    MISSING = "missing"


class TrainingAvailability(str, Enum):
    ADMITTED = "admitted"
    TRAINING_UNAVAILABLE = "training_unavailable"


class EvidenceAnswer(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


_FORBIDDEN_AUTHORITY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "authority",
        "authorized",
        "authorization",
        "policy_permission",
        "confirmation",
        "proof_accepted",
        "proof_acceptance",
        "completion",
        "completed",
        "mark_complete",
        "promotion",
        "promoted",
        "promotion_pointer",
        "validation_accepted",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(?:^|_)(?:password|passwd|secret|credential|api_key|access_token|private_key)(?:$|_)",
    re.IGNORECASE,
)


def required_text(value: Any, name: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    """Return one bounded, single-line, non-empty text value."""

    if not isinstance(value, str) or not value.strip():
        raise ResidualIntelligenceError(f"{name} must be a non-empty string")
    result = value.strip()
    if "\x00" in result or "\r" in result:
        raise ResidualIntelligenceError(f"{name} contains a forbidden control character")
    if len(result.encode("utf-8")) > max_bytes:
        raise ResidualIntelligenceError(f"{name} exceeds {max_bytes} bytes")
    return result


def optional_text(value: Any, name: str, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if value in (None, ""):
        return ""
    return required_text(value, name, max_bytes=max_bytes)


def bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResidualIntelligenceError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ResidualIntelligenceError(f"{name} must be between {minimum} and {maximum}")
    return int(value)


def text_tuple(
    values: Any,
    name: str,
    *,
    allow_empty: bool = True,
    max_items: int = MAX_SEQUENCE_ITEMS,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ResidualIntelligenceError(f"{name} must be a sequence of strings")
    if len(values) > max_items:
        raise ResidualIntelligenceError(f"{name} exceeds {max_items} items")
    result = tuple(required_text(item, f"{name} item") for item in values)
    if not allow_empty and not result:
        raise ResidualIntelligenceError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ResidualIntelligenceError(f"{name} contains duplicate values")
    return result


def strict_fields(
    payload: Mapping[str, Any],
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
    noun: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ResidualIntelligenceError(f"{noun} must be an object")
    unknown = sorted(str(key) for key in payload if key not in allowed)
    if unknown:
        raise UnknownFieldError(f"{noun} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise ResidualIntelligenceError(f"{noun} is missing required fields: {', '.join(missing)}")


def _validate_json_value(value: Any, *, path: str, depth: int) -> Any:
    if depth > MAX_NESTING_DEPTH:
        raise ResidualIntelligenceError(f"{path} exceeds nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            return optional_text(value, path)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResidualIntelligenceError(f"{path} contains a non-finite number")
        raise ResidualIntelligenceError(f"{path} must encode decimal values as integers")
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            raise ResidualIntelligenceError(f"{path} exceeds mapping bound")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            name = required_text(key, f"{path} key", max_bytes=256)
            if name in normalized:
                raise ResidualIntelligenceError(f"{path} contains duplicate keys")
            normalized[name] = _validate_json_value(item, path=f"{path}.{name}", depth=depth + 1)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            raise ResidualIntelligenceError(f"{path} exceeds sequence bound")
        return [
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise ResidualIntelligenceError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def bounded_json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError(f"{name} must be an object")
    normalized = _validate_json_value(value, path=name, depth=0)
    assert isinstance(normalized, dict)
    try:
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResidualIntelligenceError(f"{name} is not canonical JSON") from exc
    return normalized


def reject_secret_material(value: Mapping[str, Any], *, noun: str) -> None:
    """Reject obvious credential-bearing fields without claiming DLP completeness."""

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                text = str(key)
                if _SECRET_KEY_RE.search(text):
                    raise ResidualIntelligenceError(
                        f"{noun} contains credential-shaped field {path}{text}"
                    )
                visit(child, f"{path}{text}.")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                visit(child, f"{path}{index}.")

    visit(value, "")


def reject_candidate_authority(payload: Mapping[str, Any]) -> None:
    """Forbid fields whose meaning would turn a model candidate into authority."""

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).strip().casefold().replace("-", "_")
                if normalized in _FORBIDDEN_AUTHORITY_KEYS:
                    raise AuthorityViolationError(
                        f"candidate output cannot carry authority field {key!r}"
                    )
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child)

    visit(payload)


def canonical_id(payload: Mapping[str, Any]) -> str:
    return content_identity(dict(payload))


@dataclass(frozen=True)
class PrerequisiteFinding:
    """Exact-tree qualification observation for one pre-existing authority."""

    name: str
    status: PrerequisiteStatus
    source_revision: str
    source_paths: tuple[str, ...]
    evidence_paths: tuple[str, ...]
    schema_versions: tuple[str, ...]
    environment_id: str
    caveats: tuple[str, ...] = ()
    required: bool = False
    schema: str = "ipfs_accelerate_py/agent-supervisor/residual-prerequisite-finding@1"

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "finding_id",
            "name",
            "status",
            "source_revision",
            "source_paths",
            "evidence_paths",
            "schema_versions",
            "environment_id",
            "caveats",
            "required",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", required_text(self.name, "name", max_bytes=256))
        object.__setattr__(self, "status", PrerequisiteStatus(self.status))
        object.__setattr__(
            self, "source_revision", required_text(self.source_revision, "source_revision")
        )
        object.__setattr__(
            self,
            "source_paths",
            text_tuple(self.source_paths, "source_paths", allow_empty=False),
        )
        object.__setattr__(
            self, "evidence_paths", text_tuple(self.evidence_paths, "evidence_paths")
        )
        object.__setattr__(
            self, "schema_versions", text_tuple(self.schema_versions, "schema_versions")
        )
        object.__setattr__(
            self, "environment_id", required_text(self.environment_id, "environment_id")
        )
        object.__setattr__(self, "caveats", text_tuple(self.caveats, "caveats"))
        if type(self.required) is not bool:
            raise ResidualIntelligenceError("required must be boolean")
        if self.schema != ("ipfs_accelerate_py/agent-supervisor/residual-prerequisite-finding@1"):
            raise ResidualIntelligenceError("unsupported prerequisite finding schema")

    @property
    def finding_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    @property
    def blocks_required_work(self) -> bool:
        return self.required and self.status in {
            PrerequisiteStatus.MISSING,
            PrerequisiteStatus.INCOMPATIBLE,
            PrerequisiteStatus.STALE,
        }

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "name": self.name,
            "status": self.status.value,
            "source_revision": self.source_revision,
            "source_paths": list(self.source_paths),
            "evidence_paths": list(self.evidence_paths),
            "schema_versions": list(self.schema_versions),
            "environment_id": self.environment_id,
            "caveats": list(self.caveats),
            "required": self.required,
        }
        if include_id:
            result["finding_id"] = self.finding_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrerequisiteFinding:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"finding_id"},
            noun="prerequisite finding",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            name=str(payload.get("name") or ""),
            status=PrerequisiteStatus(str(payload.get("status") or "")),
            source_revision=str(payload.get("source_revision") or ""),
            source_paths=tuple(payload.get("source_paths") or ()),
            evidence_paths=tuple(payload.get("evidence_paths") or ()),
            schema_versions=tuple(payload.get("schema_versions") or ()),
            environment_id=str(payload.get("environment_id") or ""),
            caveats=tuple(payload.get("caveats") or ()),
            required=payload.get("required"),
        )
        claimed = str(payload.get("finding_id") or "")
        if claimed and claimed != result.finding_id:
            raise ResidualIntelligenceError("prerequisite finding identity mismatch")
        return result


@dataclass(frozen=True)
class TypedBlocker:
    """A scoped continuation blocker; never a whole-program status inference."""

    blocker_code: str
    task_ids: tuple[str, ...]
    prerequisite_ids: tuple[str, ...]
    continuation: str
    retryable: bool
    schema: str = "ipfs_accelerate_py/agent-supervisor/residual-typed-blocker@1"

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "blocker_id",
            "blocker_code",
            "task_ids",
            "prerequisite_ids",
            "continuation",
            "retryable",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocker_code", required_text(self.blocker_code, "blocker_code"))
        object.__setattr__(
            self, "task_ids", text_tuple(self.task_ids, "task_ids", allow_empty=False)
        )
        object.__setattr__(
            self,
            "prerequisite_ids",
            text_tuple(self.prerequisite_ids, "prerequisite_ids"),
        )
        object.__setattr__(self, "continuation", required_text(self.continuation, "continuation"))
        if type(self.retryable) is not bool:
            raise ResidualIntelligenceError("retryable must be boolean")
        if self.schema != ("ipfs_accelerate_py/agent-supervisor/residual-typed-blocker@1"):
            raise ResidualIntelligenceError("unsupported typed blocker schema")

    @property
    def blocker_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "blocker_code": self.blocker_code,
            "task_ids": list(self.task_ids),
            "prerequisite_ids": list(self.prerequisite_ids),
            "continuation": self.continuation,
            "retryable": self.retryable,
        }
        if include_id:
            result["blocker_id"] = self.blocker_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TypedBlocker:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"blocker_id"},
            noun="typed blocker",
        )
        result = cls(
            schema=str(payload.get("schema") or ""),
            blocker_code=str(payload.get("blocker_code") or ""),
            task_ids=tuple(payload.get("task_ids") or ()),
            prerequisite_ids=tuple(payload.get("prerequisite_ids") or ()),
            continuation=str(payload.get("continuation") or ""),
            retryable=payload.get("retryable"),
        )
        claimed = str(payload.get("blocker_id") or "")
        if claimed and claimed != result.blocker_id:
            raise ResidualIntelligenceError("typed blocker identity mismatch")
        return result


__all__ = (
    "AuthorityViolationError",
    "CONTRACT_SCHEMA",
    "EvidenceAnswer",
    "ExpertDisposition",
    "PrerequisiteFinding",
    "PrerequisiteStatus",
    "PrivacyClass",
    "PROGRAM_ID",
    "ResidualIntelligenceError",
    "ResidualTaskFamily",
    "RiskClass",
    "TrainingAvailability",
    "TypedBlocker",
    "UnknownFieldError",
    "bounded_int",
    "bounded_json_mapping",
    "canonical_id",
    "optional_text",
    "reject_candidate_authority",
    "reject_secret_material",
    "required_text",
    "strict_fields",
    "text_tuple",
)
