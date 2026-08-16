"""Focused DCR-053 fixed-point evidence tests (no runtime stages)."""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    DeterministicRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    IrLogicRequiredGateDisposition,
    IrLogicRequiredGateResult,
)
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import (
    DoctorFixedPointState,
    DoctorFixedPointStateKind,
    evaluate_deterministic_doctor_fixed_point,
)
from ipfs_accelerate_py.agent_supervisor.sca_doctor_bridge import (
    DoctorFinding,
    DoctorFindingDisposition,
    DoctorTransform,
    DoctorTransformDisposition,
)


def _finding() -> DoctorFinding:
    semantic_key = {
        "package": "accelerate",
        "operation": "logic.cec_prove",
        "direction": "request",
        "schema": "LogicRequest@1",
        "profile": "base",
        "transport": "loopback_mcp",
    }
    finding_id = content_identity(
        {
            "edge_id": "edge.schema",
            "semantic_key": semantic_key,
            "mismatch_class": "schema",
            "forest_id": "sha256:forest",
            "graph_cid": "sha256:graph",
            "epoch_cid": "sha256:epoch",
        }
    )
    return DoctorFinding(
        disposition=DoctorFindingDisposition.FINDING,
        reason_code="earliest_topological_nonpassing_edge",
        finding_id=finding_id,
        edge_id="edge.schema",
        semantic_key=semantic_key,
        mismatch_class="schema",
        forest_id="sha256:forest",
        graph_cid="sha256:graph",
        epoch_cid="sha256:epoch",
        findings_cid="sha256:epoch",
        evidence_cids=("sha256:source", "sha256:epoch", "sha256:edge"),
    )


def _roots(status: str = "current_live") -> dict[str, str]:
    body = {
        "forest_id": "sha256:forest",
        "graph_cid": "sha256:graph",
        "epoch_cid": "sha256:epoch",
        "findings_cid": "sha256:epoch",
    }
    return {**body, "roots_cid": content_identity(body), "status": status}


def _transform(finding: DoctorFinding, roots: dict[str, str]) -> DoctorTransform:
    return DoctorTransform(
        disposition=DoctorTransformDisposition.TRANSFORM,
        reason_code="unique_policy_pinned_registered_operator",
        transform_id="sha256:transform",
        finding_id=finding.finding_id,
        operator_id="doctor.schema-alias",
        descriptor_id="sha256:descriptor",
        registry_cid="sha256:registry",
        policy_pin_cid="sha256:policy",
        applicability_cid="sha256:applicability",
        proof_cid="sha256:proof",
        impact_cid="sha256:impact",
        roots_cid=roots["roots_cid"],
    )


def _gate() -> IrLogicRequiredGateResult:
    return IrLogicRequiredGateResult(
        disposition=IrLogicRequiredGateDisposition.PASSING,
        reason_codes=(),
        required_identity_cids={},
        receipt_ids=(),
    )


def _state(
    kind: DoctorFixedPointStateKind, measure: int, *, suffix: str = ""
) -> DoctorFixedPointState:
    finding = _finding()
    body = {
        "kind": kind.value,
        "progress_measure": measure,
        "finding_id": finding.finding_id,
        "transform_id": "sha256:transform",
        "evidence_cid": "sha256:evidence" + suffix,
    }
    return DoctorFixedPointState(
        state_id=content_identity(body),
        kind=kind,
        progress_measure=measure,
        finding_id=finding.finding_id,
        transform_id="sha256:transform",
        evidence_cid="sha256:evidence" + suffix,
    )


def _evaluate(states: tuple[DoctorFixedPointState, ...], **overrides: Any):
    finding = _finding()
    roots = _roots(overrides.pop("root_status", "current_live"))
    request = {
        "finding": finding,
        "transform": _transform(finding, roots),
        "dcr035_gate": _gate(),
        "roots": roots,
        "states": states,
        "maximum_states": 4,
    }
    request.update(overrides)
    return evaluate_deterministic_doctor_fixed_point(**request)


def test_proved_fixed_point_and_refutation_are_closed_terminal_outcomes() -> None:
    proved = _evaluate((_state(DoctorFixedPointStateKind.PROVED, 0),))
    refuted = _evaluate((_state(DoctorFixedPointStateKind.REFUTED, 1),))

    assert proved.disposition is DeterministicRepairDisposition.PROVED_VALID
    assert refuted.disposition is DeterministicRepairDisposition.REFUTED_REPAIRABLE
    assert proved.to_dict()["completion_authorized"] is False
    assert proved.to_dict()["model_call_count"] == 0


def test_unknown_cycles_no_progress_and_bound_exhaustion_fail_closed() -> None:
    unknown = _evaluate((_state(DoctorFixedPointStateKind.UNKNOWN, 1),))
    repeated = _state(DoctorFixedPointStateKind.OPEN, 2)
    cycle = _evaluate((repeated, repeated))
    no_progress = _evaluate(
        (
            _state(DoctorFixedPointStateKind.OPEN, 2, suffix="a"),
            _state(DoctorFixedPointStateKind.OPEN, 2, suffix="b"),
        )
    )
    exhaustion = _evaluate(
        tuple(
            _state(DoctorFixedPointStateKind.OPEN, 5 - index, suffix=str(index))
            for index in range(5)
        )
    )

    assert unknown.disposition is DeterministicRepairDisposition.ABSTAIN_REVIEW
    assert cycle.reason_code == "repeated_state_cycle"
    assert no_progress.reason_code == "no_progress"
    assert exhaustion.disposition is DeterministicRepairDisposition.DEFER_CAPABILITY


def test_stale_transitional_and_unknown_inputs_defer_or_reject() -> None:
    transitional = _evaluate((), root_status="transitional")
    stale_roots = _roots()
    stale_roots["forest_id"] = "sha256:other"
    finding = _finding()
    stale = evaluate_deterministic_doctor_fixed_point(
        finding=finding,
        transform=_transform(finding, _roots()),
        dcr035_gate=_gate(),
        roots=stale_roots,
        states=(),
        maximum_states=4,
    )
    invalid = _evaluate((), maximum_states=0)

    assert transitional.disposition is DeterministicRepairDisposition.DEFER_CAPABILITY
    assert stale.disposition is DeterministicRepairDisposition.DEFER_CAPABILITY
    assert invalid.disposition is DeterministicRepairDisposition.REJECTED
