"""Independent current-tree checks for DOEP-084 CEGAR plan refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    CEGAR_PLAN_REFINEMENT_ALGORITHM,
    CEGAR_PLAN_REFINEMENT_ALGORITHM_VERSION,
    CEGAR_PLAN_REFINEMENT_FORBIDDEN_FIELDS,
    CEGAR_PLAN_REFINEMENT_INTERFACE,
    CEGAR_PLAN_REFINEMENT_SCHEMA,
    CegarPlanPredicateRefinement,
    CegarPlanRefinementDecision,
    CegarPlanRefiner,
    CegarPlanStopReason,
    CegarPlanTraceKind,
    DeltaPlan,
    DeltaPlanStep,
    DeltaReplanStopReason,
    FormalDeltaReplanner,
    FormalReplanner,
    ReplannerValidationError,
    refine_cegar_plan,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_failure_memory import (
    BranchFailureKind,
    BranchFailureObservation,
    FailureMemoryScope,
    TypedBranchFailure,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
REPLANNER_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-084.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-084.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py",
    "test/api/doep/test_doep_084_add_cegar_plan_refinement.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-084.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-084.json",
)
TASK_CID = "sha256:f94deb84993556675bfed86280dec154d516fd566c4da44346c28b6d36d6bed3"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _scope() -> FailureMemoryScope:
    return FailureMemoryScope(
        repository_tree_id="tree:doep-084",
        policy_revision="policy:doep-084-v1",
        environment_id="environment:linux-py312",
        planner_version="and-or-planner-v1",
    )


def _plan(scope: FailureMemoryScope | None = None) -> DeltaPlan:
    scope = scope or _scope()
    return DeltaPlan(
        scope=scope,
        steps=(
            DeltaPlanStep(
                step_id="step:base",
                branch_id="branch:base",
                accepted=True,
                evidence_ids=("evidence:base",),
            ),
            DeltaPlanStep(
                step_id="step:target",
                branch_id="branch:target",
                dependency_ids=("step:base",),
                accepted=True,
                evidence_ids=("evidence:target",),
                obligation_ids=("obligation:target",),
                alternative_ids=("alternative:target",),
                constraint_ids=("constraint:scope",),
                validation_signature_ids=("validation:pytest-failed",),
                capability_ids=("capability:gpu",),
                conflict_scope_ids=("scope:target",),
                resource_ids=("resource:gpu-memory",),
            ),
            DeltaPlanStep(
                step_id="step:suffix",
                branch_id="branch:suffix",
                dependency_ids=("step:target",),
                accepted=True,
                evidence_ids=("evidence:suffix",),
            ),
            DeltaPlanStep(
                step_id="step:independent",
                branch_id="branch:independent",
                dependency_ids=("step:base",),
                accepted=True,
                evidence_ids=("evidence:independent",),
            ),
        ),
    )


def _observation(
    *,
    scope: FailureMemoryScope | None = None,
    evidence_id: str = "evidence:failure-v1",
    delivery_id: str = "delivery:one",
) -> BranchFailureObservation:
    return BranchFailureObservation(
        features=TypedBranchFailure(
            scope=scope or _scope(),
            kind=BranchFailureKind.COUNTEREXAMPLE,
            failure_code="failure:counterexample",
            branch_id="branch:target",
            step_ids=("step:target",),
            obligation_ids=("obligation:target",),
            alternative_ids=("alternative:target",),
            constraint_ids=("constraint:scope",),
            validation_signature_ids=("validation:pytest-failed",),
            capability_ids=("capability:gpu",),
            conflict_scope_ids=("scope:target",),
            resource_ids=("resource:gpu-memory",),
        ),
        evidence_id=evidence_id,
        delivery_id=delivery_id,
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_cegar_extends_formal_delta_replanner_without_competing_subsystem() -> None:
    assert CEGAR_PLAN_REFINEMENT_INTERFACE == "CegarPlanRefinement@1"
    assert CEGAR_PLAN_REFINEMENT_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/cegar-plan-refinement@1"
    )
    assert CEGAR_PLAN_REFINEMENT_ALGORITHM == "deterministic_cegar_plan_refinement"
    assert CEGAR_PLAN_REFINEMENT_ALGORITHM_VERSION == "cegar-plan-refinement/1.0.0"
    assert CegarPlanRefiner is FormalDeltaReplanner
    assert FormalDeltaReplanner.CEGAR_INTERFACE == CEGAR_PLAN_REFINEMENT_INTERFACE
    assert FormalDeltaReplanner.CEGAR_SCHEMA == CEGAR_PLAN_REFINEMENT_SCHEMA
    source = REPLANNER_PATH.read_text(encoding="utf-8").lower()
    assert "not a second planner" in source or "not a competing planner" in source
    assert "not a competing cegar subsystem" in source
    assert "smallest dependent suffix" in source
    assert "completion authority" in source
    assert "spurious" in source and "abstraction" in source
    assert "cegar_engine" not in source or "cegar_engine" in CEGAR_PLAN_REFINEMENT_FORBIDDEN_FIELDS
    assert "class cegarplanner" not in source
    assert "class cegarengine" not in source
    assert "authorizes_full_replan" in CEGAR_PLAN_REFINEMENT_FORBIDDEN_FIELDS
    assert "lease_id" in CEGAR_PLAN_REFINEMENT_FORBIDDEN_FIELDS
    payload = refine_cegar_plan(
        _plan(),
        trace_kind=CegarPlanTraceKind.SPURIOUS,
        anchor_step_ids=("step:target",),
        predicate_hints=(
            {
                "predicate_id": "predicate:x-bound",
                "origin": "validated_interpolant",
                "authority": "advisory",
                "statement": "Shared symbol x is bounded.",
                "implicated_step_ids": ("step:target",),
            },
        ),
    ).to_dict()
    assert "authorizes_full_replan" not in json.dumps(payload)
    assert payload["completion_authoritative"] is False
    assert payload["interface"] == CEGAR_PLAN_REFINEMENT_INTERFACE


def test_spurious_trace_refines_abstraction_without_invalidating_suffix() -> None:
    plan = _plan()
    first = refine_cegar_plan(
        plan,
        trace_kind="spurious",
        counterexample_refinement={
            "refinement_id": "refinement:doep-084-core",
            "claims_interpolant": False,
            "completion_authoritative": False,
            "affected_region_ids": ["step:target"],
            "predicate_hints": [
                {
                    "predicate_id": "predicate:invariant",
                    "origin": "unsat_core_member",
                    "authority": "advisory",
                    "statement": "Invariant member remains implicated.",
                }
            ],
            "unsat_core": {
                "original_core": ["assert:bound", "assert:invariant"],
                "refined_core": ["assert:bound", "assert:invariant"],
                "claims_interpolant": False,
            },
        },
        interpolation_assistance={
            "assistance_id": "assistance:doep-084",
            "qualified": True,
            "validated_interpolant": True,
            "completion_authoritative": False,
            "affected_region_ids": ["step:target"],
            "predicate_hints": [
                {
                    "predicate_id": "predicate:x-le-15",
                    "origin": "validated_interpolant",
                    "authority": "advisory",
                    "statement": "x <= 15",
                }
            ],
        },
    )
    second = refine_cegar_plan(
        plan,
        trace_kind="spurious",
        counterexample_refinement={
            "refinement_id": "refinement:doep-084-core",
            "claims_interpolant": False,
            "completion_authoritative": False,
            "affected_region_ids": ["step:target"],
            "predicate_hints": [
                {
                    "predicate_id": "predicate:invariant",
                    "origin": "unsat_core_member",
                    "authority": "advisory",
                    "statement": "Invariant member remains implicated.",
                }
            ],
            "unsat_core": {
                "original_core": ["assert:bound", "assert:invariant"],
                "refined_core": ["assert:bound", "assert:invariant"],
                "claims_interpolant": False,
            },
        },
        interpolation_assistance={
            "assistance_id": "assistance:doep-084",
            "qualified": True,
            "validated_interpolant": True,
            "completion_authoritative": False,
            "affected_region_ids": ["step:target"],
            "predicate_hints": [
                {
                    "predicate_id": "predicate:x-le-15",
                    "origin": "validated_interpolant",
                    "authority": "advisory",
                    "statement": "x <= 15",
                }
            ],
        },
    )
    assert first.stop_reason is CegarPlanStopReason.ABSTRACTION_REFINED
    assert first.changed
    assert first.invalidated_step_ids == ()
    assert first.affected_step_ids == ("step:suffix", "step:target")
    assert "step:base" in first.preserved_step_ids
    assert "step:independent" in first.preserved_step_ids
    assert "predicate:x-le-15" in first.refined_predicate_ids
    assert "predicate:assert:bound" in first.refined_predicate_ids
    result = {item.step_id: item for item in first.resulting_plan.steps}
    assert result["step:target"].accepted
    assert result["step:suffix"].accepted
    assert result["step:target"].evidence_ids == ("evidence:target",)
    assert "predicate:x-le-15" in result["step:target"].constraint_ids
    assert "constraint:scope" in result["step:target"].constraint_ids
    assert result["step:independent"].constraint_ids == ()
    assert first.to_dict() == second.to_dict()
    assert CegarPlanRefinementDecision.from_dict(first.to_dict()).to_dict() == first.to_dict()


def test_real_counterexample_reopens_only_smallest_dependent_suffix() -> None:
    plan = _plan()
    decision = FormalReplanner().refine_cegar_plan(
        plan,
        trace_kind=CegarPlanTraceKind.REAL,
        observation=_observation(),
        observed_at_milliseconds=100,
        required_preserved_guarantee_step_ids=("step:base", "step:independent"),
    )
    assert decision.stop_reason is CegarPlanStopReason.REPLAN_REQUIRED
    assert decision.trace_kind is CegarPlanTraceKind.REAL
    assert decision.invalidated_step_ids == ("step:suffix", "step:target")
    assert decision.preserved_step_ids == ("step:base", "step:independent")
    assert decision.delta_replan is not None
    assert decision.delta_replan.stop_reason is DeltaReplanStopReason.REPLAN_REQUIRED
    result = {item.step_id: item for item in decision.resulting_plan.steps}
    assert result["step:base"].accepted
    assert result["step:independent"].accepted
    assert not result["step:target"].accepted
    assert not result["step:suffix"].accepted
    assert result["step:target"].evidence_ids == ()
    assert result["step:suffix"].evidence_ids == ()


def test_authority_and_nonqualified_paths_fail_closed() -> None:
    plan = _plan()
    with pytest.raises(ReplannerValidationError, match="completion authority"):
        refine_cegar_plan(
            plan,
            trace_kind="spurious",
            anchor_step_ids=("step:target",),
            counterexample_refinement={"completion_authoritative": True},
        )
    with pytest.raises(ReplannerValidationError, match="operational authority field"):
        refine_cegar_plan(
            plan,
            trace_kind="spurious",
            anchor_step_ids=("step:target",),
            counterexample_refinement={"lease_id": "lease:x", "predicate_hints": []},
        )
    with pytest.raises(ReplannerValidationError, match="interpolant"):
        refine_cegar_plan(
            plan,
            trace_kind="spurious",
            anchor_step_ids=("step:target",),
            counterexample_refinement={
                "claims_interpolant": True,
                "predicate_hints": [],
            },
        )
    with pytest.raises(ReplannerValidationError, match="assume-guarantee"):
        refine_cegar_plan(
            plan,
            trace_kind="real",
            anchor_step_ids=("step:target",),
            failure_event_id="event:blocked-guarantee",
            diagnostic_id="diagnostic:blocked-guarantee",
            required_preserved_guarantee_step_ids=("step:target",),
        )
    nonqualified = refine_cegar_plan(
        plan,
        trace_kind="spurious",
        interpolation_assistance={
            "qualified": False,
            "validated_interpolant": False,
            "completion_authoritative": False,
            "predicate_hints": [],
        },
    )
    assert nonqualified.stop_reason is CegarPlanStopReason.NONQUALIFIED_ASSISTANCE
    assert nonqualified.changed is False
    unavailable = refine_cegar_plan(
        plan,
        trace_kind=CegarPlanTraceKind.UNAVAILABLE,
        anchor_step_ids=("step:target",),
    )
    assert unavailable.stop_reason is CegarPlanStopReason.UNAVAILABLE
    with pytest.raises(ReplannerValidationError, match="completion authority"):
        CegarPlanRefinementDecision(
            original_plan_id=plan.plan_id,
            resulting_plan=plan,
            stop_reason=CegarPlanStopReason.NO_PROGRESS,
            trace_kind=CegarPlanTraceKind.SPURIOUS,
            iteration=0,
            refined_predicates=(),
            affected_step_ids=(),
            preserved_step_ids=("step:base",),
            completion_authoritative=True,
        )
    with pytest.raises(ReplannerValidationError, match="authority"):
        CegarPlanPredicateRefinement(
            predicate_id="predicate:bad",
            origin="validated_interpolant",
            authority="proof",
        )


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-084"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "refine_cegar_plan"
    assert manifest["canonical_extension"]["carrier"] == "FormalDeltaReplanner"
    assert manifest["canonical_extension"]["interface"] == CEGAR_PLAN_REFINEMENT_INTERFACE
    assert manifest["canonical_extension"]["schema"] == CEGAR_PLAN_REFINEMENT_SCHEMA
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(REPLANNER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
