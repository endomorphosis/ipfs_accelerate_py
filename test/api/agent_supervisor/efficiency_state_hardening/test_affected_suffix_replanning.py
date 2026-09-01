"""ASEH-052: deterministic affected-suffix regeneration contracts."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    AffectedSuffixReplanDecision,
    AffectedSuffixReplanner,
    AffectedSuffixTask,
    AffectedSuffixWorkKind,
    DeltaPlan,
    DeltaPlanStep,
    ReplannerValidationError,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_failure_memory import (
    BranchFailureKind,
    BranchFailureObservation,
    FailureMemoryScope,
    TypedBranchFailure,
)


def _scope() -> FailureMemoryScope:
    return FailureMemoryScope(
        repository_tree_id="tree:aseh-052",
        policy_revision="policy:aseh-052",
        environment_id="environment:py312",
        planner_version="planner:aseh-052",
    )


def _plan() -> DeltaPlan:
    return DeltaPlan(
        scope=_scope(),
        steps=(
            DeltaPlanStep(
                step_id="step:prefix",
                branch_id="branch:prefix",
                accepted=True,
                evidence_ids=("evidence:prefix",),
            ),
            DeltaPlanStep(
                step_id="step:affected",
                branch_id="branch:affected",
                dependency_ids=("step:prefix",),
                accepted=True,
                evidence_ids=("evidence:affected",),
                obligation_ids=("obligation:affected",),
            ),
            DeltaPlanStep(
                step_id="step:suffix",
                branch_id="branch:suffix",
                dependency_ids=("step:affected",),
                accepted=True,
                evidence_ids=("evidence:suffix",),
            ),
            DeltaPlanStep(
                step_id="step:unaffected",
                branch_id="branch:unaffected",
                dependency_ids=("step:prefix",),
                accepted=True,
                evidence_ids=("evidence:unaffected",),
            ),
        ),
    )


def _observation() -> BranchFailureObservation:
    return BranchFailureObservation(
        features=TypedBranchFailure(
            scope=_scope(),
            kind=BranchFailureKind.COUNTEREXAMPLE,
            failure_code="failure:affected",
            branch_id="branch:affected",
            step_ids=("step:affected",),
            obligation_ids=("obligation:affected",),
        ),
        evidence_id="evidence:failure",
    )


def _task(
    task_id: str,
    kind: AffectedSuffixWorkKind,
    semantic_key: str,
    replaces: tuple[str, ...] = (),
    dependencies: tuple[str, ...] = (),
) -> AffectedSuffixTask:
    return AffectedSuffixTask(
        step=DeltaPlanStep(
            step_id=task_id,
            branch_id=f"branch:{task_id.rsplit(':', 1)[-1]}",
            dependency_ids=dependencies,
            accepted=False,
        ),
        work_kind=kind,
        semantic_key=semantic_key,
        replaces_step_ids=replaces,
    )


def _regenerated_tasks() -> tuple[AffectedSuffixTask, ...]:
    return (
        _task(
            "repair:affected",
            AffectedSuffixWorkKind.TASK,
            "semantic:affected-repair",
            ("step:affected",),
            ("step:prefix",),
        ),
        _task(
            "repair:proof-a",
            AffectedSuffixWorkKind.PROOF,
            "semantic:shared-proof",
            ("step:suffix",),
            ("repair:affected",),
        ),
        _task(
            "repair:proof-b",
            AffectedSuffixWorkKind.PROOF,
            "semantic:shared-proof",
            ("step:suffix",),
            ("repair:affected",),
        ),
        _task(
            "validation:final",
            AffectedSuffixWorkKind.FINAL_VALIDATION,
            "semantic:final-validation",
        ),
    )


def test_replans_only_the_affected_suffix_and_coalesces_identical_work() -> None:
    original = _plan()
    decision = AffectedSuffixReplanner().replan(
        original,
        _observation(),
        _regenerated_tasks(),
        observed_at_milliseconds=10,
    )

    assert decision.changed
    assert decision.delta_decision.invalidated_step_ids == ("step:affected", "step:suffix")
    assert decision.replacement_step_ids == (
        ("step:affected", "repair:affected"),
        ("step:suffix", "repair:proof-a"),
    )
    assert decision.coalesced_task_ids == (("repair:proof-b", "repair:proof-a"),)
    assert decision.validation_reserved
    assert decision.final_validation_step_id == "validation:final"

    result = {item.step_id: item for item in decision.resulting_plan.steps}
    original_steps = {item.step_id: item for item in original.steps}
    assert result["step:prefix"] == original_steps["step:prefix"]
    assert result["step:unaffected"] == original_steps["step:unaffected"]
    assert "step:affected" not in result and "step:suffix" not in result
    assert "repair:proof-b" not in result
    assert not result["repair:affected"].accepted
    assert not result["repair:proof-a"].accepted
    assert not result["validation:final"].accepted
    assert result["validation:final"].evidence_ids == ()
    assert result["validation:final"].dependency_ids == (
        "repair:proof-a",
        "step:unaffected",
    )
    # The result is deterministic and has a tamper-evident full plan identity.
    replay = AffectedSuffixReplanner().replan(
        original,
        _observation(),
        tuple(reversed(_regenerated_tasks())),
        observed_at_milliseconds=10,
    )
    assert replay.decision_id == decision.decision_id
    assert AffectedSuffixReplanDecision.from_dict(decision.to_dict()) == decision


def test_final_validation_is_an_unspent_deterministic_leaf() -> None:
    decision = AffectedSuffixReplanner().replan(
        _plan(), _observation(), _regenerated_tasks(), observed_at_milliseconds=10
    )
    by_id = {item.step_id: item for item in decision.resulting_plan.steps}
    final = by_id[decision.final_validation_step_id]
    assert not final.accepted
    assert not final.evidence_ids
    assert not any(final.step_id in item.dependency_ids for item in by_id.values())

    forged = copy.deepcopy(decision.to_dict())
    for item in forged["regenerated_tasks"]:
        if item["step"]["step_id"] == "validation:final":
            item["step"]["accepted"] = True
    with pytest.raises(ReplannerValidationError, match="unaccepted|identity"):
        AffectedSuffixReplanDecision.from_dict(forged)


def test_rejects_regeneration_that_changes_the_preserved_prefix_or_dangles() -> None:
    tasks = list(_regenerated_tasks())
    tasks[0] = _task(
        "step:unaffected",
        AffectedSuffixWorkKind.TASK,
        "semantic:illegal-prefix-replacement",
        ("step:affected",),
        ("step:prefix",),
    )
    with pytest.raises(ReplannerValidationError, match="unaffected accepted step"):
        AffectedSuffixReplanner().replan(
            _plan(), _observation(), tasks, observed_at_milliseconds=10
        )

    dangling = list(_regenerated_tasks())
    dangling[1] = _task(
        "repair:proof-a",
        AffectedSuffixWorkKind.PROOF,
        "semantic:shared-proof",
        ("step:suffix",),
        ("step:suffix",),
    )
    dangling.pop(2)
    with pytest.raises(ReplannerValidationError, match="dangling dependency"):
        AffectedSuffixReplanner().replan(
            _plan(), _observation(), dangling, observed_at_milliseconds=10
        )
