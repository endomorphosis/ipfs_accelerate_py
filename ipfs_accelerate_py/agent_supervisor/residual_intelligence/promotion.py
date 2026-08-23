"""Conjunctive promotion and exact rollback. Reports cannot promote."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, RiskClass, required_text

PROMOTION_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-promotion-evidence@1"
)
HARD_GATES: Final[tuple[str, ...]] = (
    "rights",
    "lineage",
    "leakage",
    "privacy",
    "safety",
    "quality",
    "efficiency",
    "autonomy",
    "amortization",
)


@dataclass(frozen=True)
class PromotionEvidence:
    gates: dict[str, bool]
    precision_ppm: int
    critical_false_accepts: int
    efficiency: dict[str, int]
    autonomy: dict[str, int]
    risk: RiskClass
    cas_identity: str
    schema: str = PROMOTION_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        missing = [name for name in HARD_GATES if name not in self.gates]
        if missing:
            raise ResidualIntelligenceError(f"promotion evidence missing gates: {missing}")
        if any(type(self.gates[name]) is not bool for name in HARD_GATES):
            raise ResidualIntelligenceError("promotion gates must be boolean")
        if type(self.precision_ppm) is not int or self.precision_ppm < 0:
            raise ResidualIntelligenceError("precision_ppm must be a non-negative integer")
        object.__setattr__(self, "risk", RiskClass(self.risk))
        object.__setattr__(self, "cas_identity", required_text(self.cas_identity, "cas_identity"))


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason_codes: tuple[str, ...]
    cas_identity: str


@dataclass(frozen=True)
class ExpertRollbackReceipt:
    from_identity: str
    to_identity: str
    cas_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_identity": self.from_identity,
            "to_identity": self.to_identity,
            "cas_identity": self.cas_identity,
            "promoted": False,
        }


@dataclass(frozen=True)
class ExpertPromotionGate:
    def decide(self, evidence: PromotionEvidence) -> PromotionDecision:
        reasons: list[str] = []
        if not all(evidence.gates[name] for name in HARD_GATES):
            reasons.append("hard_gate_failed")
        if evidence.precision_ppm < 990_000:
            reasons.append("precision_below_99")
        if evidence.critical_false_accepts != 0:
            reasons.append("critical_false_accept")
        efficiency = evidence.efficiency
        for key, bound in (
            ("token", 45),
            ("latency", 35),
            ("cost", 60),
            ("energy", 50),
            ("break_even", 30),
        ):
            if int(efficiency.get(key, 0)) < bound:
                reasons.append(f"efficiency_{key}")
        autonomy = evidence.autonomy
        for key, bound in (("local", 70), ("no_human", 40), ("no_remote", 25)):
            if int(autonomy.get(key, 0)) < bound:
                reasons.append(f"autonomy_{key}")
        if evidence.risk in {RiskClass.R4, RiskClass.R5}:
            reasons.append("r4_r5_proposal_only")
        promoted = not reasons and evidence.risk not in {RiskClass.R4, RiskClass.R5}
        return PromotionDecision(
            promoted=promoted,
            reason_codes=tuple(reasons),
            cas_identity=evidence.cas_identity,
        )

    def rollback(self, *, from_identity: str, to_identity: str, cas_identity: str) -> ExpertRollbackReceipt:
        return ExpertRollbackReceipt(
            from_identity=required_text(from_identity, "from_identity"),
            to_identity=required_text(to_identity, "to_identity"),
            cas_identity=required_text(cas_identity, "cas_identity"),
        )
