from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import RiskClass
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.promotion import (
    ExpertPromotionGate,
    HARD_GATES,
    PromotionEvidence,
)


def evidence(*, risk: RiskClass = RiskClass.R3, precision: int = 995_000, critical: int = 0) -> PromotionEvidence:
    return PromotionEvidence(
        gates={name: True for name in HARD_GATES},
        precision_ppm=precision,
        critical_false_accepts=critical,
        efficiency={"token": 50, "latency": 40, "cost": 70, "energy": 55, "break_even": 35},
        autonomy={"local": 80, "no_human": 50, "no_remote": 30},
        risk=risk,
        cas_identity="cas:operator:1",
    )


def test_conjunctive_gates_and_r4_remain_proposals() -> None:
    gate = ExpertPromotionGate()
    accepted = gate.decide(evidence())
    assert accepted.promoted is True
    r5 = gate.decide(evidence(risk=RiskClass.R5))
    assert r5.promoted is False
    assert "r4_r5_proposal_only" in r5.reason_codes
    failed = gate.decide(evidence(precision=900_000, critical=1))
    assert failed.promoted is False
    rollback = gate.rollback(
        from_identity="expert:new",
        to_identity="expert:old",
        cas_identity="cas:operator:rollback",
    )
    assert rollback.to_dict()["promoted"] is False
