from __future__ import annotations

from datetime import datetime, timedelta, timezone
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalCompletionDecision,
    GoalLifecycle,
    GoalState,
    IllegalGoalTransitionError,
    evaluate_goal_completion,
    evaluate_completion_gate,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    completion_tree_identity,
    reconcile_objective_goal_completion,
)


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
CURRENT_TREE = "sha256:current-repository-tree"


def _complete_evidence(**overrides: object) -> CompletionEvidence:
    values: dict[str, object] = {
        "acceptance_criterion": "The public API returns a verified result.",
        "producing_task_or_scan": "REF-206",
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


def _evaluate(
    evidence: list[CompletionEvidence],
    *,
    tasks_complete: bool = True,
    current_state: GoalState = GoalState.ACTIVE,
    gate_overrides: dict[str, object] | None = None,
) -> GoalCompletionDecision:
    gate: dict[str, object] = {
        "coverage": {
            "criteria": [{
                "criterion": "The public API returns a verified result.",
                "status": "verified",
            }]
        },
        "analyzer_health": {"status": "healthy"},
        "exhaustion_quorum": {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
        },
    }
    gate.update(gate_overrides or {})
    return evaluate_goal_completion(
        current_state=current_state,
        acceptance_criteria=["The public API returns a verified result."],
        evidence=evidence,
        tasks_complete=tasks_complete,
        repository_tree=CURRENT_TREE,
        now=NOW,
        freshness_seconds=3600,
        **gate,
    )


def _tracker_gate(identity, criterion: str = "criterion one") -> dict[str, object]:
    return {
        "coverage": {"criteria": [{"criterion": criterion, "status": "verified"}]},
        "analyzer_health": {"status": "healthy"},
        "exhaustion_quorum": {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": {
                "repository_id": identity.repository_id,
                "tree_id": identity.tree_id,
            },
        },
    }


def test_goal_state_contract_names_every_required_lifecycle_state() -> None:
    assert {state.value for state in GoalState} == {
        "active",
        "provisionally_complete",
        "verified_complete",
        "analysis_inconclusive",
        "blocked",
        "reopened",
    }


def test_goal_lifecycle_enforces_legal_transitions_and_records_history() -> None:
    lifecycle = GoalLifecycle()

    assert lifecycle.state is GoalState.ACTIVE
    assert lifecycle.can_transition(GoalState.PROVISIONALLY_COMPLETE)
    transitioned = lifecycle.transition(
        GoalState.PROVISIONALLY_COMPLETE,
        reason="all producing tasks reached a terminal success status",
    )
    assert transitioned is GoalState.PROVISIONALLY_COMPLETE
    assert lifecycle.state is GoalState.PROVISIONALLY_COMPLETE
    first = lifecycle.history[-1]
    assert first.previous_state is GoalState.ACTIVE
    assert first.state is GoalState.PROVISIONALLY_COMPLETE
    assert first.reason

    assert lifecycle.can_transition(GoalState.VERIFIED_COMPLETE)
    transitioned = lifecycle.transition(
        GoalState.VERIFIED_COMPLETE,
        reason="fresh validation receipt covers every acceptance criterion",
        evidence=[_complete_evidence()],
    )
    assert transitioned is GoalState.VERIFIED_COMPLETE
    assert lifecycle.state is GoalState.VERIFIED_COMPLETE
    second = lifecycle.history[-1]
    assert second.previous_state is GoalState.PROVISIONALLY_COMPLETE
    assert second.state is GoalState.VERIFIED_COMPLETE
    assert second.evidence_cids == ("bafy-completion-provenance",)
    assert len(lifecycle.history) == 2

    assert lifecycle.can_transition(GoalState.REOPENED)
    lifecycle.transition(
        GoalState.REOPENED,
        reason="the repository tree changed after verification",
    )
    assert lifecycle.state is GoalState.REOPENED
    assert lifecycle.can_transition(GoalState.ACTIVE)


@pytest.mark.parametrize(
    ("initial_state", "illegal_target"),
    [
        (GoalState.ACTIVE, GoalState.VERIFIED_COMPLETE),
        (GoalState.VERIFIED_COMPLETE, GoalState.ACTIVE),
        (GoalState.BLOCKED, GoalState.VERIFIED_COMPLETE),
        (GoalState.ANALYSIS_INCONCLUSIVE, GoalState.VERIFIED_COMPLETE),
    ],
)
def test_goal_lifecycle_rejects_illegal_shortcuts(
    initial_state: GoalState,
    illegal_target: GoalState,
) -> None:
    lifecycle = GoalLifecycle(state=initial_state)

    assert not lifecycle.can_transition(illegal_target)
    with pytest.raises(IllegalGoalTransitionError, match="illegal goal transition"):
        lifecycle.transition(illegal_target, reason="attempted shortcut")

    assert lifecycle.state is initial_state
    assert lifecycle.history == []


def test_task_completion_alone_is_only_provisional() -> None:
    decision = _evaluate([])

    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.verified is False
    assert decision.actionable_reasons
    assert "evidence" in " ".join(decision.actionable_reasons).lower()


def test_fresh_complete_evidence_verifies_after_provisional_transition() -> None:
    evidence = _complete_evidence()

    provisional = _evaluate([evidence])
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.verified is False

    decision = _evaluate(
        [evidence],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert decision.state is GoalState.VERIFIED_COMPLETE
    assert decision.verified is True
    assert not decision.actionable_reasons
    assert decision.evidence_results[0].evidence == evidence


@pytest.mark.parametrize(
    ("gate_overrides", "reason_code"),
    [
        ({"coverage": {"criteria": [{"criterion": "The public API returns a verified result.", "status": "stale"}]}}, "coverage_unverified"),
        ({"analyzer_health": {"status": "partial"}}, "analyzer_unhealthy"),
        ({"exhaustion_quorum": {"satisfied": False, "members": [], "duplicates": [{"reason": "duplicate_evidence_channel"}]}}, "exhaustion_quorum_unsatisfied"),
        ({"analysis_result": {"terminal_reason": "timed_out", "safe_for_completion_reasoning": False}}, "analysis_not_completion_safe"),
    ],
)
def test_completion_gate_rejects_nonqualifying_analysis_proof(
    gate_overrides: dict[str, object], reason_code: str
) -> None:
    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides=gate_overrides,
    )

    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.verified is False
    assert reason_code in decision.reason_codes
    assert decision.to_dict()["completion_gate"]["evaluated_evidence"]


@pytest.mark.parametrize("child_state", [GoalState.ANALYSIS_INCONCLUSIVE, GoalState.REOPENED])
def test_parent_completion_cannot_hide_nonverified_child(child_state: GoalState) -> None:
    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides={"child_goals": [{"goal_id": "G1.S1", "state": child_state.value, "verified": False}]},
    )

    assert decision.verified is False
    assert "child" in " ".join(decision.reason_codes)


@pytest.mark.parametrize(
    ("override", "reason_code"),
    [
        ({"acceptance_criterion": ""}, "missing_acceptance_criterion"),
        ({"producing_task_or_scan": ""}, "missing_producer"),
        ({"validation_receipt": ""}, "missing_validation_receipt"),
        ({"repository_tree": ""}, "missing_repository_tree"),
        ({"freshness": None}, "missing_freshness"),
        ({"provenance_cid": ""}, "missing_provenance_cid"),
        ({"observed_at": None}, "missing_observed_at"),
    ],
)
def test_missing_required_evidence_fails_closed_with_actionable_reason(
    override: dict[str, object],
    reason_code: str,
) -> None:
    decision = _evaluate([_complete_evidence(**override)])

    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.verified is False
    evidence_result = decision.evidence_results[0]
    assert reason_code in {*decision.reason_codes, *evidence_result.reason_codes}
    assert decision.actionable_reasons
    assert evidence_result.actionable_reasons


def test_stale_evidence_fails_closed() -> None:
    decision = _evaluate(
        [
            _complete_evidence(
                freshness=False,
                observed_at=NOW - timedelta(hours=2),
            )
        ]
    )

    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.verified is False
    assert "stale_evidence" in decision.reason_codes


def test_evidence_from_a_different_repository_tree_fails_closed() -> None:
    decision = _evaluate(
        [_complete_evidence(repository_tree="sha256:previous-repository-tree")]
    )

    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.verified is False
    reasons = " ".join(decision.actionable_reasons).lower()
    assert "tree" in reasons
    assert "current" in reasons or "mismatch" in reasons


def test_failed_validation_receipt_fails_closed() -> None:
    decision = _evaluate([_complete_evidence(validation_passed=False)])

    assert decision.state is GoalState.PROVISIONALLY_COMPLETE
    assert decision.verified is False
    assert "validation" in " ".join(decision.actionable_reasons).lower()


def test_contradictory_evidence_reopens_a_previously_verified_goal() -> None:
    decision = evaluate_goal_completion(
        current_state=GoalState.VERIFIED_COMPLETE,
        acceptance_criteria=["The public API returns a verified result."],
        evidence=[_complete_evidence(contradictory=True)],
        tasks_complete=True,
        repository_tree=CURRENT_TREE,
        now=NOW,
        freshness_seconds=3600,
    )

    assert decision.state is GoalState.REOPENED
    assert decision.verified is False
    assert "contradict" in " ".join(decision.actionable_reasons).lower()


def test_incomplete_tasks_keep_an_active_goal_active_even_with_evidence() -> None:
    decision = _evaluate([_complete_evidence()], tasks_complete=False)

    assert decision.state is GoalState.ACTIVE
    assert decision.verified is False
    assert "task" in " ".join(decision.actionable_reasons).lower()


def test_completion_evidence_and_decision_have_json_safe_round_trips() -> None:
    evidence = _complete_evidence()
    serialized_evidence = evidence.to_dict()

    assert serialized_evidence["observed_at"] == evidence.observed_at.isoformat()
    assert CompletionEvidence.from_dict(serialized_evidence) == evidence

    decision = _evaluate(
        [evidence],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    serialized_decision = decision.to_dict()

    assert serialized_decision["state"] == "verified_complete"
    assert serialized_decision["verified"] is True
    assert serialized_decision["evidence_results"] == [
        {
            "valid": True,
            "reason_codes": [],
            "actionable_reasons": [],
            "evidence": serialized_evidence,
        }
    ]
    assert serialized_decision["actionable_reasons"] == []


def _git(repo, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_objective_tracker_persists_provisional_then_verified_state(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Goals

## G10.S3 Evidence-backed goal

- Status: active
- Evidence: criterion one
- Acceptance: criterion one
- Validation: test -f proof.txt
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Closed task board\n", encoding="utf-8")
    (repo / "proof.txt").write_text("proof\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    identity = completion_tree_identity(repo, objective_path=objective_path)
    evidence = _complete_evidence(
        acceptance_criterion="criterion one",
        repository_tree=identity.tree_id,
        observed_at=datetime.now(timezone.utc),
    )

    provisional = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_evidence_records={"G10.S3": [evidence]},
        completion_gate_records={"G10.S3": _tracker_gate(identity)},
    )

    assert provisional.provisional_goal_ids == ["G10.S3"]
    assert provisional.completed_goal_ids == []
    assert "- Status: provisionally_complete" in objective_path.read_text(encoding="utf-8")

    verified = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_gate_records={"G10.S3": _tracker_gate(identity)},
    )

    assert verified.verified_goal_ids == ["G10.S3"]
    assert verified.completed_goal_ids == ["G10.S3"]
    assert verified.validation_results["G10.S3"]["receipt_cid"].startswith("b")
    assert "- Status: verified_complete" in objective_path.read_text(encoding="utf-8")


def test_objective_tracker_never_verifies_task_drain_without_evidence(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective.md"
    objective_path.write_text(
        """# Goals

## G10.S3 Evidence-backed goal

- Status: active
- Acceptance: criterion one
- Validation: true
""",
        encoding="utf-8",
    )

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
    )

    assert result.provisional_goal_ids == ["G10.S3"]
    assert result.verified_goal_ids == []
    assert result.decisions["G10.S3"]["reason_codes"][0] == "missing_criterion_evidence"
    assert "coverage_missing" in result.decisions["G10.S3"]["reason_codes"]
    assert "exhaustion_quorum_missing" in result.decisions["G10.S3"]["reason_codes"]


def test_objective_tracker_automatically_aggregates_descendant_state(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective.md"
    objective_path.write_text(
        """# Goals

## G1 Parent

- Status: provisionally_complete
- Acceptance: parent criterion
- Validation: true

## G1.S1 Child

- Status: reopened
- Parents: G1
- Acceptance: child criterion
- Validation: true
""",
        encoding="utf-8",
    )
    identity = completion_tree_identity(repo, objective_path=objective_path)
    evidence = _complete_evidence(
        acceptance_criterion="parent criterion",
        repository_tree=identity.tree_id,
        observed_at=datetime.now(timezone.utc),
    )

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        completion_evidence_records={"G1": [evidence]},
        completion_gate_records={"G1": _tracker_gate(identity, "parent criterion")},
    )

    parent = result.decisions["G1"]
    assert parent["verified"] is False
    assert "child_reopened" in parent["reason_codes"]
    assert parent["completion_gate"]["evaluated_evidence"]["child_goals"] == [
        {"goal_id": "G1.S1", "state": "reopened", "verified": False}
    ]
