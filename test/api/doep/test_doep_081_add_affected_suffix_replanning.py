"""Independent current-tree checks for DOEP-081 affected-suffix replanning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    AFFECTED_SUFFIX_REPLAN_INTERFACE,
    AFFECTED_SUFFIX_REPLAN_SCHEMA,
    AffectedSuffixReplanner,
    DeltaPlan,
    DeltaPlanStep,
    DeltaReplanStopReason,
    FormalDeltaReplanner,
    FormalReplanner,
    ReplannerValidationError,
    compute_affected_plan_suffix,
    preserved_assume_guarantee_step_ids,
    replan_affected_suffix,
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
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-081.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-081.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/formal_replanner.py",
    "test/api/doep/test_doep_081_add_affected_suffix_replanning.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-081.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-081.json",
)
TASK_CID = "sha256:1a5c06ee76329ffb4affdd75210487ed727de84c265349d49f0d8de0c82b4608"
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
        repository_tree_id="tree:doep-081",
        policy_revision="policy:doep-081-v1",
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


def test_affected_suffix_extends_formal_delta_replanner_without_competing_subsystem() -> None:
    assert AFFECTED_SUFFIX_REPLAN_INTERFACE == "AffectedSuffixReplanning@1"
    assert AFFECTED_SUFFIX_REPLAN_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/affected-suffix-replan@1"
    )
    assert AffectedSuffixReplanner is FormalDeltaReplanner
    assert FormalDeltaReplanner.INTERFACE == AFFECTED_SUFFIX_REPLAN_INTERFACE
    assert FormalDeltaReplanner.SCHEMA == AFFECTED_SUFFIX_REPLAN_SCHEMA
    source = REPLANNER_PATH.read_text(encoding="utf-8").lower()
    assert "not a second planner" in source or "not a competing planner" in source
    assert "smallest dependent suffix" in source
    assert "assume-guarantee" in source
    assert "completion authority" in source
    assert "affected_suffix_engine" not in source
    assert "class affectedsuffixplanner" not in source
    assert "authorizes_full_replan" not in json.dumps(
        replan_affected_suffix(
            _plan(),
            anchor_step_ids=("step:target",),
            failure_event_id="event:probe",
            diagnostic_id="diagnostic:probe",
        ).to_dict()
    )


def test_anchor_driven_replan_invalidates_only_smallest_dependent_suffix() -> None:
    plan = _plan()
    assert compute_affected_plan_suffix(plan, ("step:target",)) == (
        "step:suffix",
        "step:target",
    )
    decision = replan_affected_suffix(
        plan,
        anchor_step_ids=("step:target",),
        failure_event_id="event:impact-doep-081",
        diagnostic_id="diagnostic:impact-doep-081",
        required_preserved_guarantee_step_ids=("step:base", "step:independent"),
    )
    assert decision.stop_reason is DeltaReplanStopReason.REPLAN_REQUIRED
    assert decision.direct_failure_step_ids == ("step:target",)
    assert decision.invalidated_step_ids == ("step:suffix", "step:target")
    assert decision.stale_dependency_step_ids == ("step:suffix",)
    assert decision.preserved_step_ids == ("step:base", "step:independent")
    assert decision.preserved_branch_ids == ("branch:base", "branch:independent")
    result = {item.step_id: item for item in decision.resulting_plan.steps}
    assert result["step:base"] == {item.step_id: item for item in plan.steps}["step:base"]
    assert result["step:independent"].accepted
    assert result["step:independent"].evidence_ids == ("evidence:independent",)
    assert not result["step:target"].accepted
    assert not result["step:suffix"].accepted
    assert result["step:target"].evidence_ids == ()
    assert result["step:suffix"].evidence_ids == ()
    assert preserved_assume_guarantee_step_ids(
        decision, ("step:base", "step:target", "step:independent")
    ) == ("step:base", "step:independent")


def test_observation_path_and_assume_guarantee_preservation_fail_closed() -> None:
    plan = _plan()
    via_observation = FormalReplanner().replan_affected_suffix(
        plan,
        _observation(),
        observed_at_milliseconds=100,
        required_preserved_guarantee_step_ids=("step:base",),
    )
    assert via_observation.changed
    assert via_observation.invalidated_step_ids == ("step:suffix", "step:target")

    with pytest.raises(ReplannerValidationError, match="assume-guarantee"):
        replan_affected_suffix(
            plan,
            anchor_step_ids=("step:target",),
            failure_event_id="event:blocked-guarantee",
            diagnostic_id="diagnostic:blocked-guarantee",
            required_preserved_guarantee_step_ids=("step:target",),
        )
    with pytest.raises(ReplannerValidationError, match="exactly one"):
        replan_affected_suffix(plan)
    with pytest.raises(ReplannerValidationError, match="exactly one"):
        replan_affected_suffix(
            plan,
            _observation(),
            anchor_step_ids=("step:target",),
            failure_event_id="event:both",
            diagnostic_id="diagnostic:both",
        )
    with pytest.raises(ReplannerValidationError, match="outside the plan"):
        compute_affected_plan_suffix(plan, ("step:missing",))


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-081"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "replan_affected_suffix"
    assert manifest["canonical_extension"]["carrier"] == "FormalDeltaReplanner"
    assert manifest["canonical_extension"]["interface"] == AFFECTED_SUFFIX_REPLAN_INTERFACE
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
