"""Independent current-tree checks for DOEP-056 objective stop conditions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
    FORBIDDEN_OBJECTIVE_STOP_CONDITIONS,
    LEGAL_OBJECTIVE_STOP_CONDITIONS,
    OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_BINDING,
    OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_CONSUMES,
    OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_INTERFACE,
    OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_SCHEMA,
    OBJECTIVE_STOP_DECISION_SCHEMA,
    OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING,
    PLAN_COMPLETENESS_WITNESS_BINDING,
    CompletionEvidence,
    GoalLifecycle,
    GoalState,
    ObjectiveSatisfactionError,
    ObjectiveStopDecision,
    ObjectiveStopDisposition,
    apply_objective_satisfaction_and_stop_conditions,
    evaluate_goal_completion,
    evaluate_objective_satisfaction_and_stop_conditions,
    is_legal_objective_stop_condition,
    objective_satisfaction_and_stop_conditions_are_armed,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/objectives/goal_completion.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-056.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-056.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/objectives/goal_completion.py",
    "test/api/doep/test_doep_056_add_objective_satisfaction_and_stop_conditions.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-056.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-056.json",
)
TASK_CID = "sha256:f6ed95b62f0c7845f16cabd5da92c45cec8aaff4472e46abb4cd1d82632a8a9b"
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
NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
CURRENT_TREE = "sha256:current-repository-tree"
CRITERION = "The public API returns a verified result."


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_evidence(**overrides: object) -> CompletionEvidence:
    values: dict[str, object] = {
        "acceptance_criterion": CRITERION,
        "producing_task_or_scan": "DOEP-056",
        "validation_receipt": "bafy-validation-receipt",
        "repository_tree": CURRENT_TREE,
        "freshness": True,
        "provenance_cid": "bafy-completion-provenance",
        "validation_passed": True,
        "observed_at": NOW - timedelta(minutes=5),
        "contradictory": False,
    }
    values.update(overrides)
    return CompletionEvidence(**values)


def _passing_gate() -> dict[str, object]:
    return {
        "coverage": {
            "verified": True,
            "repository_tree": CURRENT_TREE,
            "evaluated_at": NOW.isoformat(),
            "criteria": [{"criterion": CRITERION, "status": "verified"}],
        },
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
        },
        "exhaustion_quorum": {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
            "members": [
                {
                    "member_id": "normal-member",
                    "evidence_channel": "exhaustive",
                    "receipt_cid": "bafy-normal-scan",
                    "scan_mode": "exhaustive",
                    "analyzer_version": "completion-analyzer/v1",
                    "passed": True,
                    "analyzer_health": {"status": "healthy", "healthy": True},
                    "exhaustive": True,
                    "safe_for_completion_reasoning": True,
                    "conclusive": True,
                    "contradicted": False,
                    "finished_at": (NOW - timedelta(minutes=3)).isoformat(),
                    "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
                },
                {
                    "member_id": "audit-member",
                    "evidence_channel": "audit",
                    "receipt_cid": "bafy-audit-scan",
                    "scan_mode": "audit",
                    "analyzer_version": "completion-analyzer/v1",
                    "passed": True,
                    "analyzer_health": {"status": "healthy", "healthy": True},
                    "exhaustive": True,
                    "safe_for_completion_reasoning": True,
                    "conclusive": True,
                    "contradicted": False,
                    "finished_at": (NOW - timedelta(minutes=2)).isoformat(),
                    "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
                },
            ],
        },
    }


def _evaluate_stop(**overrides: object) -> ObjectiveStopDecision:
    values: dict[str, object] = {
        "current_state": GoalState.ACTIVE,
        "acceptance_criteria": [CRITERION],
        "evidence": (),
        "tasks_complete": False,
        "repository_tree": CURRENT_TREE,
        "now": NOW,
        "freshness_seconds": 3600,
        "goal_id": "DOEP-G060.S3",
    }
    values.update(overrides)
    return evaluate_objective_satisfaction_and_stop_conditions(**values)


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_goal_completion_without_competing_subsystem() -> None:
    assert OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_INTERFACE == (
        "ObjectiveSatisfactionAndStopConditions@1"
    )
    assert (
        OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_BINDING
        == OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_INTERFACE
    )
    assert OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/objective-satisfaction-and-stop-conditions@1"
    )
    assert OBJECTIVE_STOP_DECISION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/objective-stop-decision@1"
    )
    assert OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_CONSUMES == (
        "ipfs_accelerate_py.agent_supervisor.completion_gate.v1",
        PLAN_COMPLETENESS_WITNESS_BINDING,
        AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
        OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING,
    )
    assert GoalLifecycle.SATISFACTION_INTERFACE == (
        OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_INTERFACE
    )
    assert GoalLifecycle.SATISFACTION_BINDING == (
        OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_BINDING
    )
    assert evaluate_objective_satisfaction_and_stop_conditions.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.objectives.goal_completion"
    )
    assert apply_objective_satisfaction_and_stop_conditions.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.objectives.goal_completion"
    )
    assert apply_objective_satisfaction_and_stop_conditions is not (
        evaluate_objective_satisfaction_and_stop_conditions
    )
    assert evaluate_goal_completion.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.objectives.goal_completion"
    )
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "class GoalLifecycle" in source
    assert "def evaluate_goal_completion(" in source
    assert "def evaluate_objective_satisfaction_and_stop_conditions(" in source
    assert "def apply_objective_satisfaction_and_stop_conditions(" in source
    assert "not a second completion owner" in source
    assert "never writes DuckDB" in source
    assert "cannot skip these controls" in source
    assert "An empty queue is not completion" in source
    assert "class CompetingStopEngine" not in source
    assert "class ObjectiveStopBus" not in source
    assert "class CompetingCompletionOwner" not in source
    assert "class StopConditionEngine" not in source
    assert "CREATE TABLE" not in source
    assert "import duckdb" not in source.casefold()
    assert objective_satisfaction_and_stop_conditions_are_armed()
    assert LEGAL_OBJECTIVE_STOP_CONDITIONS == (
        "admitted_evidence_satisfied",
        "human_or_policy_required",
        "budget_exhausted",
        "no_admissible_convergent_plan",
        "cancelled",
    )
    for legal in LEGAL_OBJECTIVE_STOP_CONDITIONS:
        assert is_legal_objective_stop_condition(legal)
    for forbidden in FORBIDDEN_OBJECTIVE_STOP_CONDITIONS:
        assert not is_legal_objective_stop_condition(forbidden)


def test_admitted_evidence_satisfies_and_stops() -> None:
    decision = _evaluate_stop(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        evidence=(_complete_evidence(),),
        tasks_complete=True,
        **_passing_gate(),
    )
    assert decision.disposition is ObjectiveStopDisposition.SATISFIED
    assert decision.stop_condition == "admitted_evidence_satisfied"
    assert decision.reason_code == "admitted_evidence_satisfied"
    assert decision.satisfied is True
    assert decision.stopped is True
    assert decision.success is True
    assert decision.completion is not None
    assert decision.completion.verified is True
    payload = decision.to_dict()
    assert payload["schema"] == OBJECTIVE_STOP_DECISION_SCHEMA
    assert payload["carrier"] == "GoalLifecycle"
    assert payload["binding"] == OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_BINDING
    assert payload["empty_queue_is_completion"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["database_write"] is False
    assert payload["authorizes_completion"] is False
    assert payload["no_competing_subsystem_created"] is True
    via_lifecycle = GoalLifecycle(
        goal_id="DOEP-G060.S3",
        state=GoalState.PROVISIONALLY_COMPLETE,
    ).evaluate_satisfaction_and_stop(
        acceptance_criteria=[CRITERION],
        evidence=(_complete_evidence(),),
        tasks_complete=True,
        repository_tree=CURRENT_TREE,
        now=NOW,
        **_passing_gate(),
    )
    assert via_lifecycle.disposition is ObjectiveStopDisposition.SATISFIED
    assert via_lifecycle.goal_id == "DOEP-G060.S3"


def test_empty_queue_is_not_completion_even_with_worker_assertion() -> None:
    decision = _evaluate_stop(
        queue_empty=True,
        open_task_count=0,
        tasks_complete=True,
        worker_assertion=True,
    )
    assert decision.disposition is ObjectiveStopDisposition.CONTINUE
    assert decision.reason_code == "empty_queue_is_not_completion"
    assert decision.satisfied is False
    assert decision.stopped is False
    assert decision.queue_empty is True
    assert decision.worker_assertion is True
    payload = decision.to_dict()
    assert payload["empty_queue_is_completion"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["completion_authoritative"] is False
    assert payload["success"] is False
    assert '"completion_authoritative": true' not in json.dumps(payload)
    via_count = apply_objective_satisfaction_and_stop_conditions(
        acceptance_criteria=[CRITERION],
        open_task_count=0,
        worker_assertion=True,
        goal_id="DOEP-G060.S3",
        repository_tree=CURRENT_TREE,
        now=NOW,
    )
    assert via_count.disposition is ObjectiveStopDisposition.CONTINUE
    assert via_count.reason_code == "empty_queue_is_not_completion"


def test_worker_assertion_cannot_satisfy_or_skip_typed_terminals() -> None:
    asserted = _evaluate_stop(worker_assertion=True, tasks_complete=True)
    assert asserted.disposition is ObjectiveStopDisposition.CONTINUE
    assert asserted.satisfied is False
    cancelled = _evaluate_stop(
        worker_assertion=True,
        cancelled=True,
        queue_empty=True,
    )
    assert cancelled.disposition is ObjectiveStopDisposition.CANCELLED
    assert cancelled.stop_condition == "cancelled"
    assert cancelled.satisfied is False
    assert cancelled.stopped is True
    payload = cancelled.to_dict()
    assert payload["worker_assertion_is_authority"] is False
    assert payload["empty_queue_is_completion"] is False
    with pytest.raises(ObjectiveSatisfactionError, match="worker_assertion"):
        _evaluate_stop(worker_assertion="yes")  # type: ignore[arg-type]


def test_typed_terminals_stop_without_claiming_satisfaction() -> None:
    policy = _evaluate_stop(human_or_policy_required=True, worker_assertion=True)
    assert policy.disposition is ObjectiveStopDisposition.HUMAN_OR_POLICY_REQUIRED
    assert policy.stop_condition == "human_or_policy_required"
    assert policy.satisfied is False
    budget = apply_objective_satisfaction_and_stop_conditions(
        acceptance_criteria=[CRITERION],
        budget_exhausted=True,
        queue_empty=True,
        repository_tree=CURRENT_TREE,
        now=NOW,
    )
    assert budget.disposition is ObjectiveStopDisposition.BUDGET_EXHAUSTED
    assert budget.stop_condition == "budget_exhausted"
    assert budget.to_dict()["empty_queue_is_completion"] is False
    no_plan = _evaluate_stop(no_admissible_convergent_plan=True)
    assert no_plan.disposition is ObjectiveStopDisposition.NO_ADMISSIBLE_CONVERGENT_PLAN
    quarantined = _evaluate_stop(quarantined_nonconvergent=True, worker_assertion=True)
    assert quarantined.disposition is ObjectiveStopDisposition.NO_ADMISSIBLE_CONVERGENT_PLAN
    assert quarantined.reason_code == "no_admissible_convergent_plan"
    cancelled = _evaluate_stop(
        cancelled=True,
        budget_exhausted=True,
        human_or_policy_required=True,
    )
    assert cancelled.disposition is ObjectiveStopDisposition.CANCELLED


def test_admitted_satisfaction_outranks_typed_terminals_and_empty_queue() -> None:
    decision = _evaluate_stop(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        evidence=(_complete_evidence(),),
        tasks_complete=True,
        queue_empty=True,
        cancelled=True,
        budget_exhausted=True,
        worker_assertion=True,
        **_passing_gate(),
    )
    assert decision.disposition is ObjectiveStopDisposition.SATISFIED
    assert decision.stop_condition == "admitted_evidence_satisfied"
    assert decision.queue_empty is True
    payload = decision.to_dict()
    assert payload["empty_queue_is_completion"] is False
    assert payload["worker_assertion_is_authority"] is False


def test_existing_two_phase_completion_gate_is_still_required() -> None:
    active = _evaluate_stop(
        current_state=GoalState.ACTIVE,
        evidence=(_complete_evidence(),),
        tasks_complete=True,
        **_passing_gate(),
    )
    assert active.disposition is ObjectiveStopDisposition.CONTINUE
    assert active.completion is not None
    assert active.completion.state is GoalState.PROVISIONALLY_COMPLETE
    assert active.completion.verified is False
    missing = _evaluate_stop(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        tasks_complete=True,
        **_passing_gate(),
    )
    assert missing.disposition is ObjectiveStopDisposition.CONTINUE
    assert missing.satisfied is False
    with pytest.raises(ObjectiveSatisfactionError, match="queue_empty contradicts"):
        _evaluate_stop(queue_empty=True, open_task_count=2)


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-056"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == (
        "evaluate_objective_satisfaction_and_stop_conditions"
    )
    assert manifest["canonical_extension"]["carrier"] == "GoalLifecycle"
    assert manifest["canonical_extension"]["binding"] == (
        OBJECTIVE_SATISFACTION_AND_STOP_CONDITIONS_BINDING
    )
    assert manifest["canonical_extension"]["authority"] == (
        "model_free_admitted_evidence_or_typed_terminal_stop"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SOURCE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert ObjectiveStopDecision(
        disposition=ObjectiveStopDisposition.CANCELLED,
    ).to_dict()["model_free"] is True
