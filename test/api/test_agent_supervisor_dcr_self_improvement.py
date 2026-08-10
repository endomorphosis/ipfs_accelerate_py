"""DCR-084: bounded self-improvement by evidence and invariants."""

from __future__ import annotations

from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.self_improvement import (
    BOUNDED_SELF_IMPROVEMENT_INTERFACE,
    IMPROVEMENT_PROPOSAL_INTERFACE,
    BoundedSelfImprovement,
    ImprovementProposal,
    ProposalDisposition,
    ProposalKind,
    materialize_improvement_proposals,
)


def _proposal(**overrides: object) -> ImprovementProposal:
    base: dict[str, object] = {
        "proposal_id": "proposal:test",
        "kind": ProposalKind.ORDERING,
        "target": "ordering_weight",
        "parameter": "priority_bias",
        "baseline_value": 1.0,
        "candidate_value": 1.5,
        "baseline_metrics": {"utility": 1.0, "safety_floor": 10.0},
        "candidate_metrics": {"utility": 1.4, "safety_floor": 10.0},
        "invariant_ids": ("inv:no-llm", "inv:authority-strict"),
        "approval_class": "review_required",
        "self_admit": False,
    }
    base.update(overrides)
    return ImprovementProposal(**base)  # type: ignore[arg-type]


def test_interfaces() -> None:
    assert BOUNDED_SELF_IMPROVEMENT_INTERFACE == "BoundedSelfImprovement@1"
    assert IMPROVEMENT_PROPOSAL_INTERFACE == "ImprovementProposal@1"
    assert BoundedSelfImprovement.INTERFACE == BOUNDED_SELF_IMPROVEMENT_INTERFACE


def test_improving_proposal_admitted_for_review_only() -> None:
    engine = BoundedSelfImprovement(
        required_invariants=("inv:no-llm", "inv:authority-strict")
    )
    result = engine.evaluate(_proposal())
    assert result.disposition is ProposalDisposition.ADMITTED_FOR_REVIEW
    assert result.improved is True
    assert result.grants_self_admission is False
    assert result.runtime_model_calls == 0
    assert result.creates_new_work is True
    assert "review_required" in result.reason_codes


def test_self_admit_rejected() -> None:
    engine = BoundedSelfImprovement(
        required_invariants=("inv:no-llm", "inv:authority-strict")
    )
    result = engine.evaluate(_proposal(self_admit=True))
    assert result.disposition is ProposalDisposition.REJECTED
    assert "self_admission_forbidden" in result.reason_codes


def test_forbidden_surface_rejected() -> None:
    engine = BoundedSelfImprovement(
        required_invariants=("inv:no-llm", "inv:authority-strict")
    )
    result = engine.evaluate(
        _proposal(target="validator", rewrites_forbidden_surface=True)
    )
    assert result.disposition is ProposalDisposition.REJECTED
    assert "forbidden_surface" in result.reason_codes


def test_safety_floor_cannot_lower() -> None:
    engine = BoundedSelfImprovement(
        required_invariants=("inv:no-llm", "inv:authority-strict")
    )
    result = engine.evaluate(
        _proposal(
            baseline_metrics={"utility": 1.0, "safety_floor": 10.0},
            candidate_metrics={"utility": 9.0, "safety_floor": 5.0},
        )
    )
    assert result.disposition is ProposalDisposition.REJECTED
    assert result.safety_floor_ok is False


def test_unchanged_and_non_improving_converge_to_noop() -> None:
    engine = BoundedSelfImprovement(
        required_invariants=("inv:no-llm", "inv:authority-strict")
    )
    unchanged = engine.evaluate(
        _proposal(
            baseline_value=1.0,
            candidate_value=1.0,
            baseline_metrics={"utility": 1.0, "safety_floor": 10.0},
            candidate_metrics={"utility": 1.0, "safety_floor": 10.0},
        )
    )
    assert unchanged.disposition is ProposalDisposition.NO_OP
    assert unchanged.creates_new_work is False
    non_improve = engine.evaluate(
        _proposal(
            baseline_value=2.0,
            candidate_value=1.0,
            baseline_metrics={"utility": 2.0, "safety_floor": 10.0},
            candidate_metrics={"utility": 1.0, "safety_floor": 10.0},
        )
    )
    assert non_improve.disposition is ProposalDisposition.NO_OP
    assert non_improve.creates_new_work is False


def test_materialize_improvement_proposals(tmp_path: Path) -> None:
    dest = tmp_path / "improvement-proposals.json"
    payload = materialize_improvement_proposals(destination=dest)
    assert dest.is_file()
    assert payload["runtime_model_calls"] == 0
    assert payload["grants_self_admission"] is False
    assert any(
        item["disposition"] == "admitted_for_review"
        for item in payload["evaluations"]
    )
    assert any(item["disposition"] == "rejected" for item in payload["evaluations"])
    assert any(item["disposition"] == "no_op" for item in payload["evaluations"])
