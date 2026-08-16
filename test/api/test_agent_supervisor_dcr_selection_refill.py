"""Focused DCR-081 typed selection/refill tests."""

from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.contracts import (
    AuthorityStage,
    DeterministicRepairDisposition,
    RepairAuthorityRoots,
    RepairEvidenceEnvelope,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.objectives.deterministic_repair_selection import (
    RepairSelectionDependencyBundle,
    RepairSelectionEvidence,
    SelectionState,
    select_and_refill_repairs,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_candidate_portfolio import (
    CandidatePortfolio,
    CandidatePortfolioDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_failure_memory import (
    FailureAttempt,
    FailureClass,
    decide_replan,
)
from ipfs_accelerate_py.agent_supervisor.planning.proof_carrying_repair_dag import (
    DoctorTransformBinding,
    ProofCarryingRepairPlan,
    RepairPlanDagDisposition,
    RepairPlanDagResult,
    RepairPlanNode,
    RepairPlanNodeKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_resource_scheduler import (
    RepairResourcePolicy,
    schedule_repair_resources,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_repair_composition import (
    DeterministicRepairCompositionDisposition,
    DeterministicRepairCompositionResult,
)


def _roots() -> RepairAuthorityRoots:
    return RepairAuthorityRoots("repo", "forest", "tree", "policy", "rpr-plan", "rpr-packet")


def _envelope(roots: RepairAuthorityRoots, repair_id: str) -> RepairEvidenceEnvelope:
    observed = RepairEvidenceEnvelope(
        repair_id,
        DeterministicRepairDisposition.REFUTED_REPAIRABLE,
        AuthorityStage.OBSERVED,
        roots,
        "observation",
    )
    return RepairEvidenceEnvelope(
        repair_id,
        DeterministicRepairDisposition.REFUTED_REPAIRABLE,
        AuthorityStage.DERIVED,
        roots,
        "observation",
        AuthorityStage.OBSERVED,
        observed.content_id,
        "derivation",
    )


def _descriptor() -> OperatorDescriptor:
    return OperatorDescriptor.from_mapping(
        {
            "operator_id": "selection.op",
            "kind": "replace_exact_bytes",
            "owner_root": "root",
            "write_scope": ["a.py"],
            "before_predicates": ["before"],
            "after_predicates": ["after"],
            "applicability_proofs": ["proof"],
            "input_schema": {
                "type": "object",
                "required": ["source_digest"],
                "properties": {"source_digest": "sha256"},
                "additional_properties": False,
            },
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {
                "kind": "restore_exact_before_bytes",
                "binding": "source_digest",
            },
            "validation_commands": [["python", "-m", "py_compile", "a.py"]],
        }
    )


def _bundle(*, roots: RepairAuthorityRoots | None = None) -> RepairSelectionDependencyBundle:
    current_roots = roots or _roots()
    descriptor = _descriptor()
    registry = OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )
    node = RepairPlanNode(
        "node-a",
        RepairPlanNodeKind.REPAIR,
        "root",
        "a.py",
        "span",
        "before-digest",
        "after-predicate",
        descriptor,
        registry.report()["registry_cid"],
        "proof",
        "logic",
        "impact",
        "noninterference",
        (("python", "-m", "py_compile", "a.py"),),
        "inverse",
        "rollback",
        resource_bounds=(("cpu", 1),),
    )
    plan = ProofCarryingRepairPlan(
        DoctorTransformBinding("dcr051", "dcr052", "doctor"),
        current_roots,
        registry,
        registry.report()["registry_cid"],
        (node,),
    )
    plan_result = RepairPlanDagResult(
        RepairPlanDagDisposition.INTEGRATION_PENDING,
        ("pending",),
        plan.content_id,
        (node.content_id,),
    )
    portfolio = CandidatePortfolio(
        CandidatePortfolioDisposition.INTEGRATION_PENDING,
        ("pending",),
        "portfolio",
        ("candidate",),
    )
    attempt = FailureAttempt(
        portfolio.portfolio_cid,
        "candidate",
        plan.content_id,
        current_roots.content_id,
        FailureClass.VALIDATION,
        ("new-evidence",),
        (1,),
    )
    decision = decide_replan(portfolio, plan_result, current_roots, attempt)
    policy = RepairResourcePolicy(("lane",), {"cpu": 1}, 4, 1, "epoch")
    schedule = schedule_repair_resources(
        plan,
        plan_result,
        decision,
        portfolio=portfolio,
        attempt=attempt,
        history=(),
        current_roots=current_roots,
        policy=policy,
    )
    return RepairSelectionDependencyBundle(
        current_roots,
        plan,
        plan_result,
        portfolio,
        attempt,
        (),
        decision,
        policy,
        schedule,
    )


def _item(
    key: str,
    risk: int,
    state: SelectionState = SelectionState.DERIVED,
    *,
    bundle: RepairSelectionDependencyBundle | None = None,
) -> RepairSelectionEvidence:
    dependencies = bundle or _bundle()
    envelope = _envelope(dependencies.roots, key)
    if state is SelectionState.COMPLETED:
        state = SelectionState.BLOCKED  # excluded-state coverage without forged publication
    return RepairSelectionEvidence(
        key=key,
        state=state,
        envelope=envelope,
        dependencies=dependencies,
        owner_root="root",
        risk=risk,
        capability="available",
        transition=DeterministicRepairCompositionResult(
            DeterministicRepairCompositionDisposition.DEFER_CAPABILITY,
            ("pending",),
            ("dcr050_doctor_reinspection",),
            key,
        ),
    )


def test_selection_replays_typed_dependencies_and_stays_pending() -> None:
    result = select_and_refill_repairs(
        (_item("b", 2), _item("a", 1), _item("blocked", 0, SelectionState.BLOCKED))
    )

    assert result.disposition == "integration_pending"
    assert result.selected_key == "a"
    assert result.refill_keys == ("a", "b")
    assert result.execution_authorized is False
    assert result.completion_authorized is False
    assert result.model_call_count == result.provider_call_count == result.network_call_count == 0


def test_tie_duplicate_and_fixed_point_do_not_create_work() -> None:
    assert select_and_refill_repairs((_item("a", 1), _item("b", 1))).disposition == "abstained"
    duplicate = _item("same", 1)
    assert select_and_refill_repairs((duplicate, duplicate)).disposition == "rejected"
    fixed = select_and_refill_repairs((_item("a", 1),), existing_keys=("a",))
    assert fixed.refill_keys == () and fixed.selected_key == ""


def test_forged_decision_schedule_and_stale_root_bundle_reject() -> None:
    good = _bundle()
    forged_decision = replace(
        good,
        decision=replace(good.decision, receipt_cid="forged"),
    )
    forged_schedule = replace(
        good,
        schedule=replace(good.schedule, schedule_cid="forged"),
    )
    stale_roots = replace(good, roots=replace(good.roots, repository_forest_cid="other"))

    for index, bundle in enumerate((forged_decision, forged_schedule, stale_roots)):
        result = select_and_refill_repairs((_item(f"bad-{index}", 1, bundle=bundle),))
        assert result.disposition == "rejected"
        assert result.execution_authorized is False
