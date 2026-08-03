"""PDR-055: Doctor residuals → plan steering and bounded derived refill.

Covers:
* successful fixed point emits no work
* residual dedupe by exact issue/obligation/root/attempt identities
* append-only plan successors when mapped
* bounded ObjectiveWorkProposal otherwise
* unchanged failure backoff
* capability gaps name exact provider/conformance work
* no completion/mutation authority
* minimal files/context targeting
* derived runtime admission gated until PDR-081
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.doctor_plan_refill import (
    DEFAULT_PARENT_GOAL_ID,
    DERIVED_RUNTIME_SOURCE_GATE,
    DOCTOR_PLAN_REFILL_INTERFACE,
    DOCTOR_PLAN_RESIDUAL_INTERFACE,
    PRODUCER_ID,
    REFILL_AUTHORIZES_COMPLETION,
    REFILL_AUTHORIZES_MUTATION,
    REFILL_AUTHORIZES_SEED_BOARD_EDIT,
    DoctorPlanContext,
    DoctorPlanNode,
    DoctorPlanRefill,
    DoctorPlanRefillAuthorityError,
    DoctorPlanRefillDisposition,
    DoctorPlanRefillError,
    DoctorPlanRefillMemory,
    DoctorPlanRefillPolicy,
    DoctorPlanResidual,
    DoctorPlanTargetSource,
    DoctorResidualDisposition,
    DoctorResidualKind,
    build_plan_steer_refill_materials,
    create_doctor_plan_refill,
    dedupe_residuals,
    doctor_residuals_for_steer,
    extract_residuals_from_fixed_point,
    fixed_point_is_successful,
    refill_doctor_plan_residuals,
    residual_fingerprint,
    residual_identity_key,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    PlanDeltaOperation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_residual(**overrides: object) -> DoctorPlanResidual:
    base: dict[str, object] = {
        "issue_id": "issue:contract-mismatch-1",
        "obligation_id": "obligation:proof-1",
        "root_id": "tree:sha256:fixture-root",
        "attempt_id": "attempt:1",
        "kind": DoctorResidualKind.OPEN_OBLIGATION,
        "predicted_files": (
            "ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py",
        ),
        "context_paths": (
            "ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py",
        ),
        "title": "Resolve Doctor residual issue:contract-mismatch-1",
        "rationale": "Open obligation after fixed-point iteration.",
        "validation_commands": (
            "python -m pytest test/api/test_agent_supervisor_doctor_plan_refill.py -q",
        ),
    }
    base.update(overrides)
    return DoctorPlanResidual(**base)  # type: ignore[arg-type]


def make_plan(*, obligation_id: str = "obligation:proof-1") -> DoctorPlanContext:
    return DoctorPlanContext(
        plan_root="plan:fixture-root",
        plan_revision=3,
        nodes=(
            DoctorPlanNode(
                node_cid="task:parent-running",
                kind="task",
                lifecycle="running",
                goal_id=DEFAULT_PARENT_GOAL_ID,
                obligation_ids=(obligation_id,),
                issue_ids=("issue:contract-mismatch-1",),
                predicted_files=(
                    "ipfs_accelerate_py/agent_supervisor/objectives/doctor_plan_refill.py",
                ),
                title="Parent Doctor repair task",
            ),
        ),
        allowed_delta_operations=(
            PlanDeltaOperation.ADD_TASK.value,
            PlanDeltaOperation.ADD_GOAL.value,
        ),
    )


# ---------------------------------------------------------------------------
# Interface / authority surface
# ---------------------------------------------------------------------------


def test_interfaces_and_authority_constants() -> None:
    assert DOCTOR_PLAN_RESIDUAL_INTERFACE == "DoctorPlanResidual@1"
    assert DOCTOR_PLAN_REFILL_INTERFACE == "DoctorPlanRefill@1"
    assert PRODUCER_ID == "doctor-plan-refill@1"
    assert REFILL_AUTHORIZES_COMPLETION is False
    assert REFILL_AUTHORIZES_MUTATION is False
    assert REFILL_AUTHORIZES_SEED_BOARD_EDIT is False
    assert DERIVED_RUNTIME_SOURCE_GATE == "PDR-081"
    service = create_doctor_plan_refill()
    assert service.INTERFACE == DOCTOR_PLAN_REFILL_INTERFACE
    assert service.producer_id == PRODUCER_ID


def test_residual_rejects_completion_authority_metadata() -> None:
    with pytest.raises(DoctorPlanRefillAuthorityError):
        make_residual(metadata={"completion_authority": True})


def test_residual_capability_gap_requires_named_work() -> None:
    with pytest.raises(DoctorPlanRefillError):
        make_residual(kind=DoctorResidualKind.CAPABILITY_GAP)


# ---------------------------------------------------------------------------
# Fixed-point closed path
# ---------------------------------------------------------------------------


def test_successful_fixed_point_emits_no_work() -> None:
    receipt = refill_doctor_plan_residuals(
        fixed_point={
            "complete": True,
            "residual_free": True,
            "residual_finding_ids": [],
            "open_frontier_ids": [],
        }
    )
    assert receipt.disposition is DoctorPlanRefillDisposition.FIXED_POINT_CLOSED
    assert receipt.emits_work is False
    assert receipt.successors == ()
    assert receipt.work_proposals == ()
    assert receipt.fixed_point_complete is True
    assert receipt.completion_authority is False
    assert receipt.mutation_authority is False


def test_fixed_point_is_successful_helper() -> None:
    assert fixed_point_is_successful({"complete": True, "residual_free": True})
    assert not fixed_point_is_successful(
        {"complete": False, "residual_finding_ids": ["finding:1"]}
    )
    assert not fixed_point_is_successful(None)


def test_extract_residuals_from_incomplete_fixed_point() -> None:
    residuals = extract_residuals_from_fixed_point(
        {
            "complete": False,
            "residual_finding_ids": ["finding:a", "finding:b"],
            "open_frontier_ids": ["frontier:x"],
            "capability_gaps": [
                {
                    "required_capability": "solver.z3",
                    "required_provider": "z3",
                    "required_conformance": "native-solver-conformance@1",
                }
            ],
            "reason_codes": ["unchanged_residual"],
        },
        root_id="tree:sha256:fixture",
        attempt_id="attempt:9",
    )
    kinds = {item.kind for item in residuals}
    assert DoctorResidualKind.OPEN_OBLIGATION in kinds
    assert DoctorResidualKind.FRONTIER in kinds
    assert DoctorResidualKind.CAPABILITY_GAP in kinds
    assert all(item.root_id == "tree:sha256:fixture" for item in residuals)
    assert all(item.attempt_id == "attempt:9" for item in residuals)


def test_extract_residuals_empty_on_success() -> None:
    assert (
        extract_residuals_from_fixed_point(
            {"complete": True, "residual_finding_ids": []}
        )
        == ()
    )


# ---------------------------------------------------------------------------
# Identity / dedupe
# ---------------------------------------------------------------------------


def test_residual_identity_is_exact_four_tuple() -> None:
    a = residual_identity_key(
        issue_id="issue:1",
        obligation_id="ob:1",
        root_id="root:1",
        attempt_id="att:1",
    )
    b = residual_identity_key(
        issue_id="issue:1",
        obligation_id="ob:1",
        root_id="root:1",
        attempt_id="att:1",
    )
    c = residual_identity_key(
        issue_id="issue:1",
        obligation_id="ob:1",
        root_id="root:1",
        attempt_id="att:2",
    )
    assert a == b
    assert a != c
    assert a.startswith("doctor-residual:")


def test_dedupe_by_exact_identities() -> None:
    first = make_residual()
    second = make_residual()  # identical four-tuple
    third = make_residual(attempt_id="attempt:2")
    unique, duplicates = dedupe_residuals([first, second, third])
    assert len(unique) == 2
    assert len(duplicates) == 1
    assert duplicates[0] == first.identity_key


def test_refill_collapses_duplicate_residuals() -> None:
    residual = make_residual()
    receipt = refill_doctor_plan_residuals([residual, residual, residual])
    assert len(receipt.residuals) == 1
    assert receipt.duplicate_identity_keys == (residual.identity_key,)


# ---------------------------------------------------------------------------
# Append-only successors
# ---------------------------------------------------------------------------


def test_mapped_residual_emits_append_only_successor() -> None:
    residual = make_residual()
    plan = make_plan()
    receipt = refill_doctor_plan_residuals([residual], plan=plan)
    assert receipt.disposition is DoctorPlanRefillDisposition.APPEND_ONLY_SUCCESSORS
    assert len(receipt.successors) == 1
    assert receipt.work_proposals == ()
    successor = receipt.successors[0]
    assert successor.delta_item.operation is PlanDeltaOperation.ADD_TASK
    assert successor.parent_node_cid == "task:parent-running"
    assert successor.delta_item.provenance["append_only"] is True
    assert successor.delta_item.provenance["completion_authority"] is False
    assert successor.delta_item.provenance["mutation_authority"] is False
    # Minimal file targeting
    assert residual.predicted_files[0] in successor.delta_item.affected_paths


def test_successor_is_deferred_when_parent_is_running() -> None:
    residual = make_residual()
    plan = make_plan()
    receipt = refill_doctor_plan_residuals([residual], plan=plan)
    item = receipt.successors[0].delta_item
    assert item.effect_class.value == "deferred"
    assert any(p.startswith("target-terminal:") for p in item.preconditions)


def test_unmapped_residual_emits_work_proposal() -> None:
    residual = make_residual(obligation_id="obligation:unknown")
    receipt = refill_doctor_plan_residuals(
        [residual],
        plan=DoctorPlanContext(plan_root="plan:empty", nodes=()),
    )
    assert receipt.disposition is DoctorPlanRefillDisposition.WORK_PROPOSALS
    assert len(receipt.work_proposals) == 1
    assert receipt.successors == ()
    proposal = receipt.work_proposals[0]
    assert isinstance(proposal, ObjectiveWorkProposal)
    assert proposal.kind is ObjectiveWorkKind.TASK
    assert proposal.predicted_files
    # Target source remains gated for derived runtime by default.
    decision = receipt.decisions[0]
    assert decision.disposition is DoctorResidualDisposition.WORK_PROPOSAL
    assert decision.target_source is DoctorPlanTargetSource.OBJECTIVE_HEAP


# ---------------------------------------------------------------------------
# Unchanged failure backoff
# ---------------------------------------------------------------------------


def test_unchanged_failure_flag_backs_off() -> None:
    residual = make_residual(unchanged_failure=True)
    receipt = refill_doctor_plan_residuals([residual], plan=make_plan())
    assert receipt.disposition is DoctorPlanRefillDisposition.UNCHANGED_BACKOFF
    assert receipt.emits_work is False
    assert residual.identity_key in receipt.backoff_identity_keys


def test_identical_fingerprint_backs_off_on_replay() -> None:
    residual = make_residual()
    first = refill_doctor_plan_residuals([residual])
    assert first.emits_work is True
    second = refill_doctor_plan_residuals([residual], memory=first.next_memory)
    assert second.disposition is DoctorPlanRefillDisposition.UNCHANGED_BACKOFF
    assert second.emits_work is False
    assert residual.identity_key in second.backoff_identity_keys


def test_fingerprint_stable_for_same_residual() -> None:
    residual = make_residual()
    assert residual_fingerprint(residual) == residual_fingerprint(make_residual())


# ---------------------------------------------------------------------------
# Capability gaps
# ---------------------------------------------------------------------------


def test_capability_gap_names_exact_provider_conformance_work() -> None:
    residual = make_residual(
        issue_id="capability:solver.z3",
        obligation_id="solver.z3",
        kind=DoctorResidualKind.CAPABILITY_GAP,
        required_capability="solver.z3",
        required_provider="z3",
        required_conformance="native-solver-conformance@1",
        title="",
    )
    receipt = refill_doctor_plan_residuals([residual])
    assert receipt.disposition is DoctorPlanRefillDisposition.CAPABILITY_GAP
    assert len(receipt.work_proposals) == 1
    proposal = receipt.work_proposals[0]
    assert "provider=z3" in proposal.title
    assert "capability=solver.z3" in proposal.title
    assert "conformance=native-solver-conformance@1" in proposal.title
    assert any("provider:z3" in term for term in proposal.expected_evidence_delta)
    assert any(
        "conformance:native-solver-conformance@1" in term
        for term in proposal.expected_evidence_delta
    )
    assert residual.residual_id in receipt.capability_gap_ids


# ---------------------------------------------------------------------------
# Derived runtime gate (PDR-081)
# ---------------------------------------------------------------------------


def test_derived_runtime_admission_disabled_by_default() -> None:
    residual = make_residual(obligation_id="obligation:unmapped")
    receipt = refill_doctor_plan_residuals([residual])
    assert receipt.derived_runtime_admitted is False
    assert "derived_runtime_gated_until_pdr_081" in receipt.reason_codes
    assert receipt.to_dict()["derived_runtime_gate"] == "PDR-081"


def test_derived_runtime_enablement_labels_but_policy_gate_is_explicit() -> None:
    residual = make_residual(obligation_id="obligation:unmapped")
    policy = DoctorPlanRefillPolicy(derived_runtime_admission_enabled=True)
    receipt = refill_doctor_plan_residuals([residual], policy=policy)
    assert receipt.derived_runtime_admitted is True
    decision = receipt.decisions[0]
    assert decision.target_source is DoctorPlanTargetSource.DERIVED_RUNTIME


def test_receipt_rejects_derived_admission_without_policy_gate() -> None:
    residual = make_residual(obligation_id="obligation:unmapped")
    receipt = refill_doctor_plan_residuals([residual])
    # Force-constructing a receipt that claims admission without the gate fails.
    with pytest.raises(DoctorPlanRefillAuthorityError):
        type(receipt)(
            disposition=receipt.disposition,
            residuals=receipt.residuals,
            decisions=receipt.decisions,
            successors=receipt.successors,
            work_proposals=receipt.work_proposals,
            policy=DoctorPlanRefillPolicy(derived_runtime_admission_enabled=False),
            derived_runtime_admitted=True,
        )


# ---------------------------------------------------------------------------
# Service / materials packaging
# ---------------------------------------------------------------------------


def test_stateful_refiller_updates_memory() -> None:
    service = DoctorPlanRefill()
    residual = make_residual()
    first = service.refill([residual])
    assert first.emits_work is True
    second = service.refill([residual])
    assert second.disposition is DoctorPlanRefillDisposition.UNCHANGED_BACKOFF


def test_doctor_residuals_for_steer_packages_materials() -> None:
    residual = make_residual()
    package = doctor_residuals_for_steer(
        residuals=[residual],
        plan=make_plan(),
        request={"directive_cid": "request:fixture"},
        live_state={"plan_revision": 3},
    )
    assert package["read_only"] is True
    assert package["completion_authority"] is False
    assert package["mutation_authority"] is False
    assert package["materials"] is not None
    materials = package["materials"]
    assert materials["doctor_plan_refill"]["completion_authority"] is False
    assert materials["doctor_plan_refill"]["mutation_authority"] is False
    assert len(materials["proposed_delta_items"]) == 1


def test_build_plan_steer_refill_materials_from_receipt() -> None:
    residual = make_residual()
    receipt = refill_doctor_plan_residuals([residual], plan=make_plan())
    materials = build_plan_steer_refill_materials(
        receipt, request="req", live_state="live"
    )
    assert materials["request"] == "req"
    assert materials["live_state"] == "live"
    assert materials["doctor_plan_refill"]["emits_work"] is True


def test_receipt_to_dict_is_body_free_and_authority_free() -> None:
    residual = make_residual()
    receipt = refill_doctor_plan_residuals([residual])
    payload = receipt.to_dict()
    assert payload["completion_authority"] is False
    assert payload["mutation_authority"] is False
    assert payload["seed_board_edit"] is False
    assert payload["interface"] == DOCTOR_PLAN_REFILL_INTERFACE
    assert "receipt_id" in payload


def test_residual_round_trip_dict() -> None:
    residual = make_residual()
    restored = DoctorPlanResidual.from_dict(residual.to_dict())
    assert restored.identity_key == residual.identity_key
    assert restored.issue_id == residual.issue_id
    assert restored.predicted_files == residual.predicted_files


def test_empty_input_disposition() -> None:
    receipt = refill_doctor_plan_residuals([])
    assert receipt.disposition is DoctorPlanRefillDisposition.EMPTY_INPUT
    assert receipt.emits_work is False


def test_mixed_successors_and_proposals() -> None:
    mapped = make_residual()
    unmapped = make_residual(
        issue_id="issue:other",
        obligation_id="obligation:other",
        attempt_id="attempt:2",
        # Distinct paths so path-overlap mapping cannot attach to the plan node.
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py",
        ),
        context_paths=(
            "ipfs_accelerate_py/agent_supervisor/prompt/plan_supervisor_service.py",
        ),
    )
    receipt = refill_doctor_plan_residuals([mapped, unmapped], plan=make_plan())
    assert receipt.disposition is DoctorPlanRefillDisposition.MIXED
    assert len(receipt.successors) == 1
    assert len(receipt.work_proposals) == 1


def test_policy_bounds_proposals() -> None:
    residuals = [
        make_residual(
            issue_id=f"issue:{index}",
            obligation_id=f"obligation:{index}",
            attempt_id=f"attempt:{index}",
        )
        for index in range(5)
    ]
    policy = DoctorPlanRefillPolicy(max_proposals=2, max_residuals=10)
    receipt = refill_doctor_plan_residuals(residuals, policy=policy)
    assert len(receipt.work_proposals) == 2
    bound_decisions = [
        item
        for item in receipt.decisions
        if item.disposition is DoctorResidualDisposition.BOUND_REJECTED
    ]
    assert len(bound_decisions) == 3


def test_paths_must_be_repository_relative() -> None:
    with pytest.raises(DoctorPlanRefillError):
        make_residual(predicted_files=("/etc/passwd",))


def test_memory_round_trip() -> None:
    residual = make_residual()
    first = refill_doctor_plan_residuals([residual])
    memory = DoctorPlanRefillMemory.from_dict(first.next_memory.to_dict())
    second = refill_doctor_plan_residuals([residual], memory=memory)
    assert second.disposition is DoctorPlanRefillDisposition.UNCHANGED_BACKOFF
