"""Residual model-call inventory with closed task-family boundaries."""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .contracts import (
    EvidenceAnswer,
    ResidualIntelligenceError,
    ResidualTaskFamily,
    RiskClass,
    bounded_int,
    canonical_id,
    required_text,
    strict_fields,
    text_tuple,
)

MODEL_INVOCATION_OBSERVATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-model-invocation-observation@1"
)
RESIDUAL_FAMILY_BOUNDARY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-family-boundary@1"
)
RESIDUAL_REASONING_INVENTORY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-reasoning-inventory@1"
)


class TrajectoryOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ABSTAINED = "abstained"
    ESCALATED = "escalated"
    FAILED_VALIDATION = "failed_validation"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ResidualFamilyBoundary:
    """Semantic boundary required before observations share an expert family."""

    task_family: ResidualTaskFamily
    input_semantics: str
    output_semantics: str
    risk_class: RiskClass
    authority_class: str
    validation_contract: str
    error_behavior: str
    abstention_behavior: str
    schema: str = RESIDUAL_FAMILY_BOUNDARY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDUAL_FAMILY_BOUNDARY_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual family boundary schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "risk_class", RiskClass(self.risk_class))
        for field in (
            "input_semantics",
            "output_semantics",
            "authority_class",
            "validation_contract",
            "error_behavior",
            "abstention_behavior",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        if self.authority_class.casefold() != "candidate_only":
            raise ResidualIntelligenceError(
                "residual expert family authority_class must be candidate_only"
            )

    @property
    def boundary_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "input_semantics": self.input_semantics,
            "output_semantics": self.output_semantics,
            "risk_class": self.risk_class.value,
            "authority_class": self.authority_class,
            "validation_contract": self.validation_contract,
            "error_behavior": self.error_behavior,
            "abstention_behavior": self.abstention_behavior,
        }
        if include_id:
            result["boundary_id"] = self.boundary_id
        return result


@dataclass(frozen=True)
class ModelInvocationObservation:
    """One accepted or rejected supervisor trajectory model invocation."""

    invocation_id: str
    trajectory_id: str
    repository_state_cid: str
    stage: str
    task_family: ResidualTaskFamily
    question_type: str
    input_contract: str
    output_contract: str
    context_size_bytes: int
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_microunits: int
    validation_references: tuple[str, ...]
    terminal_outcome: TrajectoryOutcome
    deterministic_answer_possible: EvidenceAnswer
    verified_procedure_answer_possible: EvidenceAnswer
    smaller_model_answer_possible: EvidenceAnswer
    affected_decision: EvidenceAnswer
    authoritative: bool
    task_risk: RiskClass
    family_boundary_id: str
    schema: str = MODEL_INVOCATION_OBSERVATION_SCHEMA

    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "observation_id",
            "invocation_id",
            "trajectory_id",
            "repository_state_cid",
            "stage",
            "task_family",
            "question_type",
            "input_contract",
            "output_contract",
            "context_size_bytes",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "latency_ms",
            "cost_microunits",
            "validation_references",
            "terminal_outcome",
            "deterministic_answer_possible",
            "verified_procedure_answer_possible",
            "smaller_model_answer_possible",
            "affected_decision",
            "authoritative",
            "task_risk",
            "family_boundary_id",
        }
    )

    def __post_init__(self) -> None:
        if self.schema != MODEL_INVOCATION_OBSERVATION_SCHEMA:
            raise ResidualIntelligenceError("unsupported model invocation schema")
        for field in (
            "invocation_id",
            "trajectory_id",
            "repository_state_cid",
            "stage",
            "question_type",
            "input_contract",
            "output_contract",
            "provider",
            "model",
            "family_boundary_id",
        ):
            object.__setattr__(self, field, required_text(getattr(self, field), field))
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(self, "task_risk", RiskClass(self.task_risk))
        for field, maximum in (
            ("context_size_bytes", 1_000_000_000),
            ("input_tokens", 100_000_000),
            ("output_tokens", 100_000_000),
            ("latency_ms", 7 * 24 * 60 * 60 * 1000),
            ("cost_microunits", 1_000_000_000_000),
        ):
            object.__setattr__(
                self,
                field,
                bounded_int(getattr(self, field), field, minimum=0, maximum=maximum),
            )
        object.__setattr__(
            self,
            "validation_references",
            text_tuple(self.validation_references, "validation_references"),
        )
        object.__setattr__(self, "terminal_outcome", TrajectoryOutcome(self.terminal_outcome))
        for field in (
            "deterministic_answer_possible",
            "verified_procedure_answer_possible",
            "smaller_model_answer_possible",
            "affected_decision",
        ):
            object.__setattr__(self, field, EvidenceAnswer(getattr(self, field)))
        if type(self.authoritative) is not bool:
            raise ResidualIntelligenceError("authoritative must be boolean")

    @property
    def observation_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "invocation_id": self.invocation_id,
            "trajectory_id": self.trajectory_id,
            "repository_state_cid": self.repository_state_cid,
            "stage": self.stage,
            "task_family": self.task_family.value,
            "question_type": self.question_type,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "context_size_bytes": self.context_size_bytes,
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "cost_microunits": self.cost_microunits,
            "validation_references": list(self.validation_references),
            "terminal_outcome": self.terminal_outcome.value,
            "deterministic_answer_possible": self.deterministic_answer_possible.value,
            "verified_procedure_answer_possible": (self.verified_procedure_answer_possible.value),
            "smaller_model_answer_possible": self.smaller_model_answer_possible.value,
            "affected_decision": self.affected_decision.value,
            "authoritative": self.authoritative,
            "task_risk": self.task_risk.value,
            "family_boundary_id": self.family_boundary_id,
        }
        if include_id:
            result["observation_id"] = self.observation_id
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelInvocationObservation:
        strict_fields(
            payload,
            allowed=cls._FIELDS,
            required=cls._FIELDS - {"observation_id"},
            noun="model invocation observation",
        )
        kwargs = {key: payload.get(key) for key in cls._FIELDS if key not in {"observation_id"}}
        kwargs["validation_references"] = tuple(payload.get("validation_references") or ())
        result = cls(**kwargs)  # type: ignore[arg-type]
        claimed = str(payload.get("observation_id") or "")
        if claimed and claimed != result.observation_id:
            raise ResidualIntelligenceError("model invocation observation identity mismatch")
        return result


@dataclass(frozen=True)
class ResidualReasoningInventory:
    """Content-addressed inventory with semantic-boundary enforcement."""

    repository_revision: str
    environment_id: str
    boundaries: tuple[ResidualFamilyBoundary, ...]
    observations: tuple[ModelInvocationObservation, ...]
    schema: str = RESIDUAL_REASONING_INVENTORY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RESIDUAL_REASONING_INVENTORY_SCHEMA:
            raise ResidualIntelligenceError("unsupported residual reasoning inventory schema")
        object.__setattr__(
            self,
            "repository_revision",
            required_text(self.repository_revision, "repository_revision"),
        )
        object.__setattr__(
            self, "environment_id", required_text(self.environment_id, "environment_id")
        )
        boundaries = tuple(self.boundaries)
        observations = tuple(self.observations)
        if any(not isinstance(item, ResidualFamilyBoundary) for item in boundaries):
            raise ResidualIntelligenceError("boundaries must contain typed records")
        if any(not isinstance(item, ModelInvocationObservation) for item in observations):
            raise ResidualIntelligenceError("observations must contain typed records")
        boundary_by_family = {item.task_family: item for item in boundaries}
        if len(boundary_by_family) != len(boundaries):
            raise ResidualIntelligenceError("a task family has multiple semantic boundaries")
        invocation_ids = [item.invocation_id for item in observations]
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ResidualIntelligenceError("duplicate invocation identity in inventory")
        for item in observations:
            boundary = boundary_by_family.get(item.task_family)
            if boundary is None:
                raise ResidualIntelligenceError(
                    f"missing semantic boundary for {item.task_family.value}"
                )
            if item.family_boundary_id != boundary.boundary_id:
                raise ResidualIntelligenceError("observation family boundary identity mismatch")
            if item.task_risk != boundary.risk_class:
                raise ResidualIntelligenceError("observation risk differs from family boundary")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "observations", observations)

    @property
    def inventory_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def summary(self) -> dict[str, Any]:
        families = Counter(item.task_family.value for item in self.observations)
        outcomes = Counter(item.terminal_outcome.value for item in self.observations)
        return {
            "observation_count": len(self.observations),
            "family_counts": dict(sorted(families.items())),
            "outcome_counts": dict(sorted(outcomes.items())),
            "authoritative_invocation_count": sum(
                1 for item in self.observations if item.authoritative
            ),
            "input_tokens": sum(item.input_tokens for item in self.observations),
            "output_tokens": sum(item.output_tokens for item in self.observations),
            "cost_microunits": sum(item.cost_microunits for item in self.observations),
        }

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "repository_revision": self.repository_revision,
            "environment_id": self.environment_id,
            "boundaries": [item.to_dict() for item in self.boundaries],
            "observations": [item.to_dict() for item in self.observations],
            "summary": self.summary(),
        }
        if include_id:
            result["inventory_id"] = self.inventory_id
        return result


def build_inventory(
    *,
    repository_revision: str,
    environment_id: str,
    boundaries: Sequence[ResidualFamilyBoundary],
    rows: Sequence[Mapping[str, Any]],
) -> ResidualReasoningInventory:
    """Strictly ingest every supplied trajectory row; no row is silently skipped."""

    observations = tuple(ModelInvocationObservation.from_dict(row) for row in rows)
    return ResidualReasoningInventory(
        repository_revision=repository_revision,
        environment_id=environment_id,
        boundaries=tuple(boundaries),
        observations=observations,
    )


__all__ = (
    "MODEL_INVOCATION_OBSERVATION_SCHEMA",
    "RESIDUAL_FAMILY_BOUNDARY_SCHEMA",
    "RESIDUAL_REASONING_INVENTORY_SCHEMA",
    "ModelInvocationObservation",
    "ResidualFamilyBoundary",
    "ResidualReasoningInventory",
    "TrajectoryOutcome",
    "build_inventory",
)
