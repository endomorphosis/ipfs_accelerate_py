"""DCR-062 deterministic, non-authoritative repair candidate portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ..autonomous_repair.contracts import RepairAuthorityRoots, repair_evidence_cid
from ..proof.ir_logic_application import IrLogicRequiredGateDisposition, IrLogicRequiredGateResult
from .default_planner_factory import DcrPlannerCompositionResult
from .proof_carrying_repair_dag import (
    DoctorTransformBinding,
    ProofCarryingRepairPlan,
    RepairPlanDagResult,
)


DCR_CANDIDATE_PORTFOLIO_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-candidate-portfolio@1"
)


class CandidatePortfolioDisposition(str, Enum):
    INTEGRATION_PENDING = "integration_pending"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


def _cid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact CID")
    if "synthetic" in value.lower() or "stub" in value.lower():
        raise ValueError(f"{name} may not be synthetic or stub")
    return value


@dataclass(frozen=True)
class CandidateFacts:
    """Closed integer facts; no prose or caller readiness boolean is accepted."""

    candidate_id: str
    node_ids: tuple[str, ...]
    resource_cost: int
    impact_cost: int
    risk_cost: int
    proof_cache_cid: str
    root_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _cid(self.candidate_id, "candidate_id"))
        nodes = tuple(_cid(value, "node_id") for value in self.node_ids)
        if not nodes or len(nodes) != len(set(nodes)):
            raise ValueError("candidate node_ids must be non-empty and unique")
        object.__setattr__(self, "node_ids", tuple(sorted(nodes)))
        for name in ("resource_cost", "impact_cost", "risk_cost"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "proof_cache_cid", _cid(self.proof_cache_cid, "proof_cache_cid"))
        object.__setattr__(self, "root_cid", _cid(self.root_cid, "root_cid"))

    @property
    def score(self) -> tuple[int, int, int]:
        """Specified minimization order: risk, impact, then resource."""
        return (self.risk_cost, self.impact_cost, self.resource_cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "node_ids": list(self.node_ids),
            "resource_cost": self.resource_cost,
            "impact_cost": self.impact_cost,
            "risk_cost": self.risk_cost,
            "proof_cache_cid": self.proof_cache_cid,
            "root_cid": self.root_cid,
        }

    @property
    def content_id(self) -> str:
        return repair_evidence_cid(self.to_dict())


@dataclass(frozen=True)
class CandidatePortfolioRequest:
    transform: DoctorTransformBinding
    plan: ProofCarryingRepairPlan
    plan_result: RepairPlanDagResult
    logic_gate: IrLogicRequiredGateResult
    current_roots: RepairAuthorityRoots
    proof_cache_cid: str
    planner_composition: DcrPlannerCompositionResult
    candidates: tuple[CandidateFacts, ...]


@dataclass(frozen=True)
class CandidatePortfolio:
    disposition: CandidatePortfolioDisposition
    reason_codes: tuple[str, ...]
    portfolio_cid: str = ""
    candidate_cids: tuple[str, ...] = ()
    selected_candidate_cid: str = ""
    execution_authorized: bool = False
    completion_authorized: bool = False
    model_call_count: int = 0
    provider_call_count: int = 0
    network_call_count: int = 0


def build_deterministic_candidate_portfolio(request: Any) -> CandidatePortfolio:
    """Score only closed facts; all live readiness stays explicitly pending."""
    if not isinstance(request, CandidatePortfolioRequest):
        return CandidatePortfolio(
            CandidatePortfolioDisposition.REJECTED, ("typed_request_required",)
        )
    reasons: list[str] = []
    if not isinstance(request.transform, DoctorTransformBinding):
        reasons.append("typed_dcr052_transform_required")
    if not isinstance(request.plan, ProofCarryingRepairPlan):
        reasons.append("typed_dcr061_plan_required")
    if not isinstance(request.plan_result, RepairPlanDagResult) or (
        request.plan_result.plan_cid != request.plan.content_id
    ):
        reasons.append("dcr061_plan_identity_invalid")
    if (
        not isinstance(request.current_roots, RepairAuthorityRoots)
        or request.plan.authority_roots != request.current_roots
    ):
        reasons.append("current_root_binding_invalid")
    if not isinstance(request.logic_gate, IrLogicRequiredGateResult) or (
        request.logic_gate.disposition is not IrLogicRequiredGateDisposition.PASSING
        or request.logic_gate.model_call_count != 0
        or request.logic_gate.provider_call_count != 0
        or request.logic_gate.execution_authorized
        or request.logic_gate.completion_authorized
    ):
        reasons.append("dcr035_logic_gate_invalid")
    try:
        proof_cache_cid = _cid(request.proof_cache_cid, "proof_cache_cid")
        if not isinstance(request.planner_composition, DcrPlannerCompositionResult):
            raise ValueError("typed DCR-060 planner composition result required")
        if (
            request.planner_composition.execution_authorized
            or request.planner_composition.completion_authorized
            or request.planner_composition.model_call_count
            or request.planner_composition.provider_call_count
            or request.planner_composition.network_call_count
        ):
            raise ValueError("DCR-060 composition result must remain non-authoritative")
        composition_cid = repair_evidence_cid(request.planner_composition.to_dict())
    except ValueError:
        reasons.append("proof_cache_or_planner_identity_invalid")
        proof_cache_cid = composition_cid = ""
    candidates = tuple(request.candidates)
    if not candidates or any(not isinstance(item, CandidateFacts) for item in candidates):
        reasons.append("closed_candidate_facts_required")
    else:
        plan_nodes = {node.node_id for node in request.plan.nodes}
        for item in candidates:
            if (
                not set(item.node_ids).issubset(plan_nodes)
                or item.proof_cache_cid != proof_cache_cid
                or item.root_cid != request.current_roots.content_id
            ):
                reasons.append("candidate_fact_identity_or_node_set_invalid")
    if reasons:
        return CandidatePortfolio(
            CandidatePortfolioDisposition.REJECTED, tuple(sorted(set(reasons)))
        )
    ordered = tuple(sorted(candidates, key=lambda item: (item.score, item.content_id)))
    cids = tuple(item.content_id for item in ordered)
    body = {
        "schema": DCR_CANDIDATE_PORTFOLIO_SCHEMA,
        "transform": request.transform.to_dict(),
        "plan_cid": request.plan.content_id,
        "root_cid": request.current_roots.content_id,
        "proof_cache_cid": proof_cache_cid,
        "planner_composition_cid": composition_cid,
        "candidate_cids": list(cids),
    }
    # Equal numeric score is intentionally ambiguous.  CID canonical ordering
    # gives byte-stable output only; it never silently chooses a repair.
    if len(ordered) > 1 and ordered[0].score == ordered[1].score:
        return CandidatePortfolio(
            CandidatePortfolioDisposition.ABSTAINED,
            ("top_candidate_score_ambiguous",),
            repair_evidence_cid(body),
            cids,
        )
    return CandidatePortfolio(
        CandidatePortfolioDisposition.INTEGRATION_PENDING,
        ("integration_pending_dcr052_dcr060_dcr061_live_evidence",),
        repair_evidence_cid(body),
        cids,
    )


__all__ = [
    "DCR_CANDIDATE_PORTFOLIO_SCHEMA",
    "CandidateFacts",
    "CandidatePortfolio",
    "CandidatePortfolioDisposition",
    "CandidatePortfolioRequest",
    "build_deterministic_candidate_portfolio",
]
