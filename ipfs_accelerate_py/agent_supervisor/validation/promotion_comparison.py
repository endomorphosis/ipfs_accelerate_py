"""Deterministic M2 promotion comparison.

Compares a candidate checkpoint against a baseline and a closed policy.
Every M2 gate is represented and non-compensable: a loss improvement, a
significant gain on another metric, or a higher composite score cannot
clear a failed, missing, or inconclusive required gate.

The comparison is a pure function of evidence plus policy.  Identical
inputs produce an identical decision and content-addressed receipt.
Models, evaluators, and candidates cannot self-promote through this
surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


PROMOTION_COMPARISON_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-comparison@1"
)
PROMOTION_COMPARISON_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-comparison-policy@1"
)
PROMOTION_COMPARISON_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-comparison-receipt@1"
)
PROMOTION_GATE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/promotion-gate-result@1"
)

# Closed M2 population from PGIR-G090 / IRPromotionManifest@1.
M2_GATES: Final[tuple[str, ...]] = (
    "lineage",
    "syntax",
    "type",
    "semantic",
    "proof",
    "calibration",
    "family",
    "jurisdiction",
    "source_span",
    "latency",
    "resource",
)
IDENTITY_GATES: Final[frozenset[str]] = frozenset(
    {"lineage", "family", "jurisdiction", "source_span"}
)
QUALITY_GATES: Final[frozenset[str]] = frozenset(
    {"syntax", "type", "semantic", "proof", "calibration"}
)
RESOURCE_GATES: Final[frozenset[str]] = frozenset({"latency", "resource"})
REQUIRED_PROMOTE_GATES: Final[frozenset[str]] = frozenset(
    {"lineage", "semantic", "proof"}
)

DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS: Final = 950_000
DEFAULT_PROOF_MINIMUM_MILLIONTHS: Final = 1_000_000
DEFAULT_QUALITY_MINIMUM_MILLIONTHS: Final = 0
DEFAULT_NONINFERIORITY_MARGIN_MILLIONTHS: Final = 0
DEFAULT_LATENCY_CEILING_MILLIONTHS: Final = 10_000_000
DEFAULT_RESOURCE_CEILING_MILLIONTHS: Final = 10_000_000

FORBIDDEN_PROMOTION_ROLES: Final[frozenset[str]] = frozenset(
    {"model", "evaluator", "candidate", "self"}
)


class PromotionComparisonError(ValueError):
    """Malformed comparison evidence or policy."""


class PromotionDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    REGRESSED = "regressed"
    INCONCLUSIVE = "inconclusive"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise PromotionComparisonError(f"{name} must be a string")
    else:
        text = value.strip()
    if "\x00" in text:
        raise PromotionComparisonError(f"{name} must not contain NUL")
    if required and not text:
        raise PromotionComparisonError(f"{name} must be a non-empty string")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PromotionComparisonError(f"{name} must be a boolean")
    return value


def _optional_int(value: Any, name: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionComparisonError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise PromotionComparisonError(f"{name} must be at least {minimum}")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    result = _optional_int(value, name, minimum=minimum)
    if result is None:
        raise PromotionComparisonError(f"{name} must be an integer")
    return result


def _gate_id(value: Any) -> str:
    gate = _text(value, "gate_id")
    if gate not in M2_GATES:
        raise PromotionComparisonError(f"unknown M2 gate: {gate!r}")
    return gate


@dataclass(frozen=True)
class PromotionComparisonPolicy:
    """Closed minima and identity bindings for one comparison."""

    policy_id: str
    policy_revision: str
    required_gates: tuple[str, ...] = M2_GATES
    semantic_minimum_millionths: int = DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS
    proof_minimum_millionths: int = DEFAULT_PROOF_MINIMUM_MILLIONTHS
    quality_minimum_millionths: int = DEFAULT_QUALITY_MINIMUM_MILLIONTHS
    noninferiority_margin_millionths: int = DEFAULT_NONINFERIORITY_MARGIN_MILLIONTHS
    latency_ceiling_millionths: int = DEFAULT_LATENCY_CEILING_MILLIONTHS
    resource_ceiling_millionths: int = DEFAULT_RESOURCE_CEILING_MILLIONTHS
    require_fresh_proof: bool = True
    require_human_approval: bool = False
    required_lineage_identity: str = ""
    schema: str = PROMOTION_COMPARISON_POLICY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(
            self, "policy_revision", _text(self.policy_revision, "policy_revision")
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_COMPARISON_POLICY_SCHEMA:
            raise PromotionComparisonError("unsupported promotion comparison policy schema")
        gates = tuple(_gate_id(item) for item in self.required_gates)
        if len(gates) != len(set(gates)):
            raise PromotionComparisonError("required_gates values must be unique")
        if tuple(sorted(gates)) != tuple(sorted(M2_GATES)):
            raise PromotionComparisonError(
                "required_gates must be the complete non-compensable M2 population"
            )
        object.__setattr__(self, "required_gates", tuple(M2_GATES))
        object.__setattr__(
            self,
            "semantic_minimum_millionths",
            _int(self.semantic_minimum_millionths, "semantic_minimum_millionths"),
        )
        object.__setattr__(
            self,
            "proof_minimum_millionths",
            _int(self.proof_minimum_millionths, "proof_minimum_millionths"),
        )
        object.__setattr__(
            self,
            "quality_minimum_millionths",
            _int(self.quality_minimum_millionths, "quality_minimum_millionths"),
        )
        object.__setattr__(
            self,
            "noninferiority_margin_millionths",
            _int(self.noninferiority_margin_millionths, "noninferiority_margin_millionths"),
        )
        object.__setattr__(
            self,
            "latency_ceiling_millionths",
            _int(self.latency_ceiling_millionths, "latency_ceiling_millionths", minimum=1),
        )
        object.__setattr__(
            self,
            "resource_ceiling_millionths",
            _int(self.resource_ceiling_millionths, "resource_ceiling_millionths", minimum=1),
        )
        object.__setattr__(
            self, "require_fresh_proof", _bool(self.require_fresh_proof, "require_fresh_proof")
        )
        object.__setattr__(
            self,
            "require_human_approval",
            _bool(self.require_human_approval, "require_human_approval"),
        )
        object.__setattr__(
            self,
            "required_lineage_identity",
            _text(self.required_lineage_identity, "required_lineage_identity", required=False),
        )
        if self.semantic_minimum_millionths < DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS:
            raise PromotionComparisonError("semantic minimum cannot be lowered")
        if self.proof_minimum_millionths < DEFAULT_PROOF_MINIMUM_MILLIONTHS:
            raise PromotionComparisonError("proof minimum cannot be lowered")

    @property
    def policy_identity(self) -> str:
        return content_identity(self.to_dict())

    def minimum_for(self, gate_id: str) -> int:
        if gate_id == "semantic":
            return self.semantic_minimum_millionths
        if gate_id == "proof":
            return self.proof_minimum_millionths
        return self.quality_minimum_millionths

    def ceiling_for(self, gate_id: str) -> int:
        if gate_id == "latency":
            return self.latency_ceiling_millionths
        return self.resource_ceiling_millionths

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ceiling_millionths": self.latency_ceiling_millionths,
            "noninferiority_margin_millionths": self.noninferiority_margin_millionths,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "proof_minimum_millionths": self.proof_minimum_millionths,
            "quality_minimum_millionths": self.quality_minimum_millionths,
            "require_fresh_proof": self.require_fresh_proof,
            "require_human_approval": self.require_human_approval,
            "required_gates": list(self.required_gates),
            "required_lineage_identity": self.required_lineage_identity,
            "resource_ceiling_millionths": self.resource_ceiling_millionths,
            "schema": self.schema,
            "semantic_minimum_millionths": self.semantic_minimum_millionths,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionComparisonPolicy":
        if not isinstance(payload, Mapping):
            raise PromotionComparisonError("promotion comparison policy must be an object")
        return cls(
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            required_gates=tuple(payload.get("required_gates") or M2_GATES),
            semantic_minimum_millionths=payload.get(
                "semantic_minimum_millionths", DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS
            ),
            proof_minimum_millionths=payload.get(
                "proof_minimum_millionths", DEFAULT_PROOF_MINIMUM_MILLIONTHS
            ),
            quality_minimum_millionths=payload.get(
                "quality_minimum_millionths", DEFAULT_QUALITY_MINIMUM_MILLIONTHS
            ),
            noninferiority_margin_millionths=payload.get(
                "noninferiority_margin_millionths",
                DEFAULT_NONINFERIORITY_MARGIN_MILLIONTHS,
            ),
            latency_ceiling_millionths=payload.get(
                "latency_ceiling_millionths", DEFAULT_LATENCY_CEILING_MILLIONTHS
            ),
            resource_ceiling_millionths=payload.get(
                "resource_ceiling_millionths", DEFAULT_RESOURCE_CEILING_MILLIONTHS
            ),
            require_fresh_proof=payload.get("require_fresh_proof", True),
            require_human_approval=payload.get("require_human_approval", False),
            required_lineage_identity=payload.get("required_lineage_identity", ""),
            schema=payload.get("schema", PROMOTION_COMPARISON_POLICY_SCHEMA),
        )


@dataclass(frozen=True)
class PromotionGateEvidence:
    """One gate's baseline/candidate observation.  Claims are not authority."""

    gate_id: str
    available: bool = False
    baseline_identity: str = ""
    candidate_identity: str = ""
    baseline_millionths: int | None = None
    candidate_millionths: int | None = None
    noninferiority_passed: bool | None = None
    significant_improvement: bool = False
    evidence_identity: str = ""
    reason: str = ""
    compensable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _gate_id(self.gate_id))
        object.__setattr__(self, "available", _bool(self.available, "available"))
        object.__setattr__(
            self,
            "baseline_identity",
            _text(self.baseline_identity, "baseline_identity", required=False),
        )
        object.__setattr__(
            self,
            "candidate_identity",
            _text(self.candidate_identity, "candidate_identity", required=False),
        )
        object.__setattr__(
            self,
            "baseline_millionths",
            _optional_int(self.baseline_millionths, "baseline_millionths"),
        )
        object.__setattr__(
            self,
            "candidate_millionths",
            _optional_int(self.candidate_millionths, "candidate_millionths"),
        )
        if self.noninferiority_passed is not None:
            object.__setattr__(
                self,
                "noninferiority_passed",
                _bool(self.noninferiority_passed, "noninferiority_passed"),
            )
        object.__setattr__(
            self,
            "significant_improvement",
            _bool(self.significant_improvement, "significant_improvement"),
        )
        object.__setattr__(
            self,
            "evidence_identity",
            _text(self.evidence_identity, "evidence_identity", required=False),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=False))
        object.__setattr__(self, "compensable", _bool(self.compensable, "compensable"))
        if self.compensable:
            raise PromotionComparisonError(
                f"{self.gate_id} is a non-compensable M2 gate"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "baseline_identity": self.baseline_identity,
            "baseline_millionths": self.baseline_millionths,
            "candidate_identity": self.candidate_identity,
            "candidate_millionths": self.candidate_millionths,
            "compensable": False,
            "evidence_identity": self.evidence_identity,
            "gate_id": self.gate_id,
            "noninferiority_passed": self.noninferiority_passed,
            "reason": self.reason,
            "significant_improvement": self.significant_improvement,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionGateEvidence":
        if not isinstance(payload, Mapping):
            raise PromotionComparisonError("gate evidence must be an object")
        return cls(
            gate_id=payload.get("gate_id", ""),
            available=payload.get("available", False),
            baseline_identity=payload.get("baseline_identity", ""),
            candidate_identity=payload.get("candidate_identity", ""),
            baseline_millionths=payload.get("baseline_millionths"),
            candidate_millionths=payload.get("candidate_millionths"),
            noninferiority_passed=payload.get("noninferiority_passed"),
            significant_improvement=payload.get("significant_improvement", False),
            evidence_identity=payload.get("evidence_identity", ""),
            reason=payload.get("reason", ""),
            compensable=payload.get("compensable", False),
        )


@dataclass(frozen=True)
class PromotionGateResult:
    """Deterministic per-gate outcome.  Never compensable."""

    gate_id: str
    status: GateStatus
    reason: str
    compensable: bool = False
    schema: str = PROMOTION_GATE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _gate_id(self.gate_id))
        status = (
            self.status
            if isinstance(self.status, GateStatus)
            else GateStatus(str(self.status))
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=False))
        object.__setattr__(self, "compensable", _bool(self.compensable, "compensable"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.compensable:
            raise PromotionComparisonError(f"{self.gate_id} result cannot be compensable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compensable": False,
            "gate_id": self.gate_id,
            "reason": self.reason,
            "schema": self.schema,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class PromotionComparisonRequest:
    """Inputs for one deterministic comparison.  Not a promotion permit."""

    candidate_checkpoint_id: str
    baseline_checkpoint_id: str
    policy: PromotionComparisonPolicy
    evaluation_report_identity: str
    proof_evidence_identity: str
    actor_identity: str
    expected_current_pointer: str = ""
    actor_role: str = "operator"
    proof_fresh: bool = True
    loss_improved: bool = False
    test_set_selected: bool = False
    hidden_labels_used: bool = False
    gates: Mapping[str, PromotionGateEvidence] = MappingProxyType({})
    schema: str = PROMOTION_COMPARISON_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_checkpoint_id",
            _text(self.candidate_checkpoint_id, "candidate_checkpoint_id"),
        )
        object.__setattr__(
            self,
            "baseline_checkpoint_id",
            _text(self.baseline_checkpoint_id, "baseline_checkpoint_id"),
        )
        policy = (
            self.policy
            if isinstance(self.policy, PromotionComparisonPolicy)
            else PromotionComparisonPolicy.from_dict(self.policy)
        )
        object.__setattr__(self, "policy", policy)
        object.__setattr__(
            self,
            "evaluation_report_identity",
            _text(self.evaluation_report_identity, "evaluation_report_identity"),
        )
        object.__setattr__(
            self,
            "proof_evidence_identity",
            _text(self.proof_evidence_identity, "proof_evidence_identity", required=False),
        )
        object.__setattr__(self, "actor_identity", _text(self.actor_identity, "actor_identity"))
        object.__setattr__(
            self,
            "expected_current_pointer",
            _text(self.expected_current_pointer, "expected_current_pointer", required=False),
        )
        object.__setattr__(self, "actor_role", _text(self.actor_role, "actor_role").casefold())
        object.__setattr__(self, "proof_fresh", _bool(self.proof_fresh, "proof_fresh"))
        object.__setattr__(self, "loss_improved", _bool(self.loss_improved, "loss_improved"))
        object.__setattr__(
            self, "test_set_selected", _bool(self.test_set_selected, "test_set_selected")
        )
        object.__setattr__(
            self, "hidden_labels_used", _bool(self.hidden_labels_used, "hidden_labels_used")
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_COMPARISON_SCHEMA:
            raise PromotionComparisonError("unsupported promotion comparison schema")
        normalized: dict[str, PromotionGateEvidence] = {}
        if not isinstance(self.gates, Mapping):
            raise PromotionComparisonError("gates must be a mapping")
        for key, value in self.gates.items():
            evidence = (
                value
                if isinstance(value, PromotionGateEvidence)
                else PromotionGateEvidence.from_dict(value)
            )
            gate_id = _gate_id(key)
            if evidence.gate_id != gate_id:
                raise PromotionComparisonError("gate map key must match gate_id")
            if gate_id in normalized:
                raise PromotionComparisonError(f"duplicate gate evidence: {gate_id}")
            normalized[gate_id] = evidence
        unknown = sorted(set(normalized) - set(M2_GATES))
        if unknown:
            raise PromotionComparisonError("unknown M2 gates: " + ", ".join(unknown))
        object.__setattr__(self, "gates", MappingProxyType(dict(sorted(normalized.items()))))

    @property
    def self_promotion(self) -> bool:
        return self.candidate_checkpoint_id == self.baseline_checkpoint_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_identity": self.actor_identity,
            "actor_role": self.actor_role,
            "baseline_checkpoint_id": self.baseline_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "evaluation_report_identity": self.evaluation_report_identity,
            "expected_current_pointer": self.expected_current_pointer,
            "gates": {key: value.to_dict() for key, value in self.gates.items()},
            "hidden_labels_used": self.hidden_labels_used,
            "loss_improved": self.loss_improved,
            "policy": self.policy.to_dict(),
            "proof_evidence_identity": self.proof_evidence_identity,
            "proof_fresh": self.proof_fresh,
            "schema": self.schema,
            "test_set_selected": self.test_set_selected,
        }


@dataclass(frozen=True)
class PromotionComparisonReceipt:
    """Tamper-evident comparison result.  Not completion or pointer authority."""

    decision: PromotionDecision
    candidate_checkpoint_id: str
    baseline_checkpoint_id: str
    expected_current_pointer: str
    policy_identity: str
    evaluation_report_identity: str
    proof_evidence_identity: str
    actor_identity: str
    gate_results: tuple[PromotionGateResult, ...]
    admitted_gates: tuple[str, ...]
    reasons: tuple[str, ...]
    loss_improved: bool
    require_human_approval: bool
    schema: str = PROMOTION_COMPARISON_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        decision = (
            self.decision
            if isinstance(self.decision, PromotionDecision)
            else PromotionDecision(str(self.decision))
        )
        object.__setattr__(self, "decision", decision)
        object.__setattr__(
            self,
            "candidate_checkpoint_id",
            _text(self.candidate_checkpoint_id, "candidate_checkpoint_id"),
        )
        object.__setattr__(
            self,
            "baseline_checkpoint_id",
            _text(self.baseline_checkpoint_id, "baseline_checkpoint_id"),
        )
        object.__setattr__(
            self,
            "expected_current_pointer",
            _text(self.expected_current_pointer, "expected_current_pointer", required=False),
        )
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity")
        )
        object.__setattr__(
            self,
            "evaluation_report_identity",
            _text(self.evaluation_report_identity, "evaluation_report_identity"),
        )
        object.__setattr__(
            self,
            "proof_evidence_identity",
            _text(self.proof_evidence_identity, "proof_evidence_identity", required=False),
        )
        object.__setattr__(self, "actor_identity", _text(self.actor_identity, "actor_identity"))
        results = tuple(
            item
            if isinstance(item, PromotionGateResult)
            else PromotionGateResult(
                gate_id=item["gate_id"],
                status=item["status"],
                reason=item.get("reason", ""),
            )
            for item in self.gate_results
        )
        if tuple(item.gate_id for item in results) != M2_GATES:
            raise PromotionComparisonError(
                "receipt must report every M2 gate in catalog order"
            )
        object.__setattr__(self, "gate_results", results)
        admitted = tuple(_gate_id(item) for item in self.admitted_gates)
        if len(admitted) != len(set(admitted)):
            raise PromotionComparisonError("admitted_gates values must be unique")
        object.__setattr__(self, "admitted_gates", admitted)
        object.__setattr__(
            self,
            "reasons",
            tuple(_text(item, "reason") for item in self.reasons),
        )
        object.__setattr__(self, "loss_improved", _bool(self.loss_improved, "loss_improved"))
        object.__setattr__(
            self,
            "require_human_approval",
            _bool(self.require_human_approval, "require_human_approval"),
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != PROMOTION_COMPARISON_RECEIPT_SCHEMA:
            raise PromotionComparisonError("unsupported promotion comparison receipt schema")
        passed = {item.gate_id for item in results if item.status is GateStatus.PASS}
        if self.decision is PromotionDecision.PROMOTE:
            if set(self.admitted_gates) != set(M2_GATES) or passed != set(M2_GATES):
                raise PromotionComparisonError(
                    "promote requires every non-compensable M2 gate to pass"
                )
            missing = REQUIRED_PROMOTE_GATES - set(self.admitted_gates)
            if missing:
                raise PromotionComparisonError(
                    "promotion missing required gates: " + ", ".join(sorted(missing))
                )
        elif self.admitted_gates:
            raise PromotionComparisonError(
                "non-promote decisions cannot admit gates"
            )

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict())

    @property
    def gate_map(self) -> Mapping[str, PromotionGateResult]:
        return MappingProxyType({item.gate_id: item for item in self.gate_results})

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_identity": self.actor_identity,
            "admitted_gates": list(self.admitted_gates),
            "baseline_checkpoint_id": self.baseline_checkpoint_id,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "decision": self.decision.value,
            "evaluation_report_identity": self.evaluation_report_identity,
            "expected_current_pointer": self.expected_current_pointer,
            "gate_results": [item.to_dict() for item in self.gate_results],
            "loss_improved": self.loss_improved,
            "policy_identity": self.policy_identity,
            "proof_evidence_identity": self.proof_evidence_identity,
            "reasons": list(self.reasons),
            "require_human_approval": self.require_human_approval,
            "schema": self.schema,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotionComparisonReceipt":
        if not isinstance(payload, Mapping):
            raise PromotionComparisonError("promotion comparison receipt must be an object")
        claimed = payload.get("receipt_id")
        result = cls(
            decision=payload.get("decision", ""),
            candidate_checkpoint_id=payload.get("candidate_checkpoint_id", ""),
            baseline_checkpoint_id=payload.get("baseline_checkpoint_id", ""),
            expected_current_pointer=payload.get("expected_current_pointer", ""),
            policy_identity=payload.get("policy_identity", ""),
            evaluation_report_identity=payload.get("evaluation_report_identity", ""),
            proof_evidence_identity=payload.get("proof_evidence_identity", ""),
            actor_identity=payload.get("actor_identity", ""),
            gate_results=tuple(payload.get("gate_results") or ()),
            admitted_gates=tuple(payload.get("admitted_gates") or ()),
            reasons=tuple(payload.get("reasons") or ()),
            loss_improved=payload.get("loss_improved", False),
            require_human_approval=payload.get("require_human_approval", False),
            schema=payload.get("schema", PROMOTION_COMPARISON_RECEIPT_SCHEMA),
        )
        if claimed is not None and claimed != result.receipt_id:
            raise PromotionComparisonError("forged promotion comparison receipt_id")
        return result


def _identity_result(
    evidence: PromotionGateEvidence,
    *,
    required_identity: str = "",
) -> PromotionGateResult:
    if not evidence.available:
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.INCONCLUSIVE,
            evidence.reason or f"{evidence.gate_id}:evidence_unavailable",
        )
    if not evidence.baseline_identity or not evidence.candidate_identity:
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.INCONCLUSIVE,
            evidence.reason or f"{evidence.gate_id}:identity_missing",
        )
    if evidence.candidate_identity != evidence.baseline_identity:
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.FAIL,
            f"{evidence.gate_id}:identity_mismatch",
        )
    if required_identity and evidence.candidate_identity != required_identity:
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.FAIL,
            f"{evidence.gate_id}:required_identity_mismatch",
        )
    return PromotionGateResult(evidence.gate_id, GateStatus.PASS, f"{evidence.gate_id}:pass")


def _numeric_result(
    evidence: PromotionGateEvidence,
    *,
    higher_is_better: bool,
    minimum: int,
    ceiling: int,
    margin: int,
) -> PromotionGateResult:
    if not evidence.available:
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.INCONCLUSIVE,
            evidence.reason or f"{evidence.gate_id}:evidence_unavailable",
        )
    if evidence.baseline_millionths is None or evidence.candidate_millionths is None:
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.INCONCLUSIVE,
            evidence.reason or f"{evidence.gate_id}:measurement_missing",
        )
    candidate = evidence.candidate_millionths
    baseline = evidence.baseline_millionths
    if higher_is_better:
        if candidate < minimum:
            return PromotionGateResult(
                evidence.gate_id,
                GateStatus.FAIL,
                f"{evidence.gate_id}:below_minimum:{candidate}",
            )
        if candidate + margin < baseline:
            return PromotionGateResult(
                evidence.gate_id,
                GateStatus.FAIL,
                f"{evidence.gate_id}:noninferiority_failed",
            )
        computed_noninferior = candidate + margin >= baseline
    else:
        if candidate > ceiling:
            return PromotionGateResult(
                evidence.gate_id,
                GateStatus.FAIL,
                f"{evidence.gate_id}:above_ceiling:{candidate}",
            )
        if candidate > baseline + margin:
            return PromotionGateResult(
                evidence.gate_id,
                GateStatus.FAIL,
                f"{evidence.gate_id}:noninferiority_failed",
            )
        computed_noninferior = candidate <= baseline + margin
    if evidence.noninferiority_passed is False or (
        evidence.noninferiority_passed is True and not computed_noninferior
    ):
        return PromotionGateResult(
            evidence.gate_id,
            GateStatus.FAIL,
            f"{evidence.gate_id}:noninferiority_failed",
        )
    return PromotionGateResult(evidence.gate_id, GateStatus.PASS, f"{evidence.gate_id}:pass")


def _evaluate_gate(
    gate_id: str,
    request: PromotionComparisonRequest,
) -> PromotionGateResult:
    evidence = request.gates.get(gate_id) or PromotionGateEvidence(gate_id=gate_id)
    if gate_id in IDENTITY_GATES:
        required = (
            request.policy.required_lineage_identity if gate_id == "lineage" else ""
        )
        return _identity_result(evidence, required_identity=required)
    if gate_id in QUALITY_GATES:
        return _numeric_result(
            evidence,
            higher_is_better=True,
            minimum=request.policy.minimum_for(gate_id),
            ceiling=request.policy.ceiling_for("resource"),
            margin=request.policy.noninferiority_margin_millionths,
        )
    return _numeric_result(
        evidence,
        higher_is_better=False,
        minimum=0,
        ceiling=request.policy.ceiling_for(gate_id),
        margin=request.policy.noninferiority_margin_millionths,
    )


def _policy_rejections(request: PromotionComparisonRequest) -> list[str]:
    reasons: list[str] = []
    if request.self_promotion:
        reasons.append("self_promotion_prohibited")
    if request.actor_role in FORBIDDEN_PROMOTION_ROLES:
        reasons.append(f"actor_role_cannot_promote:{request.actor_role}")
    if request.actor_identity in {
        request.candidate_checkpoint_id,
        request.baseline_checkpoint_id,
    }:
        reasons.append("model_self_promotion_prohibited")
    if request.test_set_selected:
        reasons.append("test_set_selection_prohibited")
    if request.hidden_labels_used:
        reasons.append("hidden_label_use_prohibited")
    if request.policy.require_fresh_proof and (
        not request.proof_evidence_identity or not request.proof_fresh
    ):
        reasons.append("fresh_proof_evidence_required")
    return reasons


def compare_promotion(
    request: PromotionComparisonRequest | Mapping[str, Any],
) -> PromotionComparisonReceipt:
    """Return the unique M2 decision for ``request``.

    Failures are non-compensable.  Loss improvement is recorded and never
    used to override a failed, missing, or inconclusive gate.
    """

    comparison = (
        request
        if isinstance(request, PromotionComparisonRequest)
        else PromotionComparisonRequest(
            candidate_checkpoint_id=request.get("candidate_checkpoint_id", ""),
            baseline_checkpoint_id=request.get("baseline_checkpoint_id", ""),
            policy=request.get("policy", {}),
            evaluation_report_identity=request.get("evaluation_report_identity", ""),
            proof_evidence_identity=request.get("proof_evidence_identity", ""),
            actor_identity=request.get("actor_identity", ""),
            expected_current_pointer=request.get("expected_current_pointer", ""),
            actor_role=request.get("actor_role", "operator"),
            proof_fresh=request.get("proof_fresh", True),
            loss_improved=request.get("loss_improved", False),
            test_set_selected=request.get("test_set_selected", False),
            hidden_labels_used=request.get("hidden_labels_used", False),
            gates=request.get("gates", {}),
            schema=request.get("schema", PROMOTION_COMPARISON_SCHEMA),
        )
    )
    gate_results = tuple(_evaluate_gate(gate_id, comparison) for gate_id in M2_GATES)
    policy_reasons = _policy_rejections(comparison)
    failed = tuple(item for item in gate_results if item.status is GateStatus.FAIL)
    inconclusive = tuple(
        item for item in gate_results if item.status is GateStatus.INCONCLUSIVE
    )
    passed = tuple(item for item in gate_results if item.status is GateStatus.PASS)
    identity_failed = tuple(item for item in failed if item.gate_id in IDENTITY_GATES)
    quality_or_resource_failed = tuple(
        item for item in failed if item.gate_id not in IDENTITY_GATES
    )
    notes: list[str] = []
    if comparison.loss_improved and failed:
        notes.append("loss_improvement_cannot_override_failed_gate")
    if comparison.loss_improved and inconclusive:
        notes.append("loss_improvement_cannot_override_inconclusive_gate")
    if policy_reasons or identity_failed:
        decision = PromotionDecision.REJECT
        reasons = [
            *policy_reasons,
            *notes,
            *(item.reason for item in failed),
            *(item.reason for item in inconclusive),
        ]
        admitted: tuple[str, ...] = ()
    elif quality_or_resource_failed:
        decision = PromotionDecision.REGRESSED
        reasons = [*notes, *(item.reason for item in quality_or_resource_failed)]
        admitted = ()
    elif inconclusive:
        decision = PromotionDecision.INCONCLUSIVE
        reasons = [*notes, *(item.reason for item in inconclusive)]
        admitted = ()
    else:
        decision = PromotionDecision.PROMOTE
        reasons = ["all_m2_gates_passed"]
        admitted = tuple(item.gate_id for item in passed)
    unique_reasons = tuple(dict.fromkeys(reasons))
    return PromotionComparisonReceipt(
        decision=decision,
        candidate_checkpoint_id=comparison.candidate_checkpoint_id,
        baseline_checkpoint_id=comparison.baseline_checkpoint_id,
        expected_current_pointer=comparison.expected_current_pointer,
        policy_identity=comparison.policy.policy_identity,
        evaluation_report_identity=comparison.evaluation_report_identity,
        proof_evidence_identity=comparison.proof_evidence_identity,
        actor_identity=comparison.actor_identity,
        gate_results=gate_results,
        admitted_gates=admitted,
        reasons=unique_reasons,
        loss_improved=comparison.loss_improved,
        require_human_approval=comparison.policy.require_human_approval,
    )


def passing_m2_evidence(
    *,
    lineage_identity: str = "lineage:frozen",
    family_identity: str = "family:legal-ir",
    jurisdiction_identity: str = "jurisdiction:us",
    source_span_identity: str = "source-span:heldout",
    quality_millionths: int = 980_000,
    proof_millionths: int = 1_000_000,
    latency_millionths: int = 100_000,
    resource_millionths: int = 200_000,
    evidence_prefix: str = "evidence",
) -> dict[str, PromotionGateEvidence]:
    """Compact passing M2 fixture used by admission and CAS tests."""

    identities = {
        "lineage": lineage_identity,
        "family": family_identity,
        "jurisdiction": jurisdiction_identity,
        "source_span": source_span_identity,
    }
    numbers = {
        "syntax": quality_millionths,
        "type": quality_millionths,
        "semantic": quality_millionths,
        "proof": proof_millionths,
        "calibration": quality_millionths,
        "latency": latency_millionths,
        "resource": resource_millionths,
    }
    gates: dict[str, PromotionGateEvidence] = {}
    for gate_id in M2_GATES:
        if gate_id in identities:
            identity = identities[gate_id]
            gates[gate_id] = PromotionGateEvidence(
                gate_id=gate_id,
                available=True,
                baseline_identity=identity,
                candidate_identity=identity,
                evidence_identity=f"{evidence_prefix}:{gate_id}",
            )
        else:
            value = numbers[gate_id]
            gates[gate_id] = PromotionGateEvidence(
                gate_id=gate_id,
                available=True,
                baseline_millionths=value,
                candidate_millionths=value,
                noninferiority_passed=True,
                evidence_identity=f"{evidence_prefix}:{gate_id}",
            )
    return gates


__all__ = (
    "DEFAULT_PROOF_MINIMUM_MILLIONTHS",
    "DEFAULT_SEMANTIC_MINIMUM_MILLIONTHS",
    "FORBIDDEN_PROMOTION_ROLES",
    "IDENTITY_GATES",
    "M2_GATES",
    "PROMOTION_COMPARISON_POLICY_SCHEMA",
    "PROMOTION_COMPARISON_RECEIPT_SCHEMA",
    "PROMOTION_COMPARISON_SCHEMA",
    "QUALITY_GATES",
    "REQUIRED_PROMOTE_GATES",
    "RESOURCE_GATES",
    "GateStatus",
    "PromotionComparisonError",
    "PromotionComparisonPolicy",
    "PromotionComparisonReceipt",
    "PromotionComparisonRequest",
    "PromotionDecision",
    "PromotionGateEvidence",
    "PromotionGateResult",
    "compare_promotion",
    "passing_m2_evidence",
)
