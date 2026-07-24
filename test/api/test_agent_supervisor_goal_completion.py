from __future__ import annotations

from datetime import datetime, timedelta, timezone
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    ContradictionEvidence,
    GoalCompletionDecision,
    GoalLifecycle,
    GoalState,
    IllegalGoalTransitionError,
    discover_goal_contradictions,
    evaluate_goal_completion,
    evaluate_completion_gate,
    completion_diagnostics,
    is_schedulable_goal_state,
    migrate_legacy_goal_completion,
    reconcile_goal_reopenings,
    reopen_goal_for_contradictions,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    completion_tree_identity,
    migrate_legacy_objective_goals,
    reconcile_objective_goal_completion,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
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
            "verified": True,
            "repository_tree": CURRENT_TREE,
            "evaluated_at": NOW.isoformat(),
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
            "members": [
                {
                    "member_id": "normal-member",
                    "evidence_channel": "exhaustive",
                    "receipt_cid": "bafy-normal-scan",
                    "scan_mode": "exhaustive",
                    "finished_at": (NOW - timedelta(minutes=3)).isoformat(),
                    "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
                },
                {
                    "member_id": "audit-member",
                    "evidence_channel": "audit",
                    "receipt_cid": "bafy-audit-scan",
                    "scan_mode": "audit",
                    "finished_at": (NOW - timedelta(minutes=2)).isoformat(),
                    "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
                },
            ],
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
    evaluated_at = datetime.now(timezone.utc)
    binding = {
        "repository_id": identity.repository_id,
        "tree_id": identity.tree_id,
    }
    return {
        "coverage": {
            "verified": True,
            "repository_tree": identity.tree_id,
            "evaluated_at": evaluated_at.isoformat(),
            "criteria": [{"criterion": criterion, "status": "verified"}],
        },
        "analyzer_health": {"status": "healthy"},
        "exhaustion_quorum": {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": binding,
            "members": [
                {
                    "member_id": "normal-member",
                    "evidence_channel": "exhaustive",
                    "receipt_cid": "bafy-normal-scan",
                    "finished_at": evaluated_at.isoformat(),
                    "binding": binding,
                },
                {
                    "member_id": "audit-member",
                    "evidence_channel": "audit",
                    "receipt_cid": "bafy-audit-scan",
                    "finished_at": evaluated_at.isoformat(),
                    "binding": binding,
                },
            ],
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


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (GoalState.ACTIVE, True),
        (GoalState.PROVISIONALLY_COMPLETE, True),
        (GoalState.ANALYSIS_INCONCLUSIVE, True),
        (GoalState.REOPENED, True),
        (GoalState.VERIFIED_COMPLETE, False),
        (GoalState.BLOCKED, False),
    ],
)
def test_goal_scheduler_keeps_nonterminal_proof_work_actionable(
    state: GoalState,
    expected: bool,
) -> None:
    assert is_schedulable_goal_state(state) is expected


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


def test_completion_gate_rejects_healthy_but_explicitly_unsafe_analyzer() -> None:
    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides={
            "analyzer_health": {
                "status": "healthy",
                "healthy": True,
                "safe_for_completion_reasoning": False,
            }
        },
    )

    assert decision.verified is False
    assert "analyzer_completion_unsafe" in decision.reason_codes


@pytest.mark.parametrize(
    ("coverage_override", "reason_code"),
    [
        ({"verified": False}, "coverage_unverified"),
        ({"reason_codes": ["contradicted_surface"]}, "coverage_unverified"),
        ({"repository_tree": "sha256:old-tree"}, "coverage_tree_mismatch"),
        ({"evaluated_at": (NOW - timedelta(hours=2)).isoformat()}, "coverage_stale"),
        ({"evaluated_at": ["malformed"]}, "coverage_stale"),
    ],
)
def test_completion_gate_rejects_contradictory_stale_or_malformed_coverage(
    coverage_override: dict[str, object], reason_code: str
) -> None:
    coverage: dict[str, object] = {
        "verified": True,
        "repository_tree": CURRENT_TREE,
        "evaluated_at": NOW.isoformat(),
        "criteria": [{
            "acceptance_criterion": "The public API returns a verified result.",
            "status": "verified",
        }],
    }
    coverage.update(coverage_override)

    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides={"coverage": coverage},
    )

    assert decision.verified is False
    assert reason_code in decision.reason_codes


@pytest.mark.parametrize(
    "terminal_status",
    ["partial", "skipped", "failed", "timed_out", "duplicate_only", "unsupported"],
)
def test_failed_terminal_validation_cannot_be_overridden_by_positive_summary(
    terminal_status: str,
) -> None:
    decision = _evaluate([
        _complete_evidence(
            validation_passed=True,
            validation_receipt={"status": terminal_status, "passed": True},
        )
    ])

    assert decision.verified is False
    assert "failed_validation" in decision.reason_codes


@pytest.mark.parametrize(
    "quorum_override",
    [
        {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
        },
        {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "members": [],
            "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
        },
        {
            "satisfied": True,
            "quorum_met": False,
            "required_members": 1,
            "member_count": 1,
            "members": [{
                "member_id": "one",
                "evidence_channel": "audit",
                "receipt_cid": "bafy-one",
                "finished_at": NOW.isoformat(),
                "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
            }],
            "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
        },
    ],
)
def test_forged_or_contradictory_quorum_summary_fails_closed(
    quorum_override: dict[str, object],
) -> None:
    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides={"exhaustion_quorum": quorum_override},
    )

    assert decision.verified is False
    assert any(code.startswith("exhaustion_quorum_") for code in decision.reason_codes)


def test_nested_audit_receipt_is_normalized_without_losing_exact_input() -> None:
    nested_analysis = {
        "receipt": {
            "terminal_reason": "exhausted",
            "safe_for_completion_reasoning": True,
            "receipt_cid": "bafy-audit-analysis",
        },
        "counts": {"novel": 0, "changed": 0},
    }
    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides={"analysis_result": nested_analysis},
    )

    assert decision.verified is True
    gate = decision.to_dict()["completion_gate"]
    assert gate["evaluated_evidence"]["analysis_result"] == nested_analysis
    assert "analysis_completion_safe" in gate["pass_reason_codes"]


def test_parent_gate_recursively_rejects_hidden_reopened_descendant() -> None:
    decision = _evaluate(
        [_complete_evidence()],
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        gate_overrides={
            "child_goals": [{
                "goal_id": "G1.S1",
                "state": "verified_complete",
                "verified": True,
                "completion_gate": {
                    "passed": True,
                    "evaluated_evidence": {
                        "child_goals": [{
                            "goal_id": "G1.S1.S1",
                            "state": "reopened",
                            "verified": False,
                        }]
                    },
                },
            }]
        },
    )

    assert decision.verified is False
    assert "child_reopened" in decision.reason_codes
    children_check = next(
        check for check in decision.gate.checks if check.name == "child_goals"
    )
    assert children_check.evidence["unverified_children"][0]["goal_id"] == "G1.S1.S1"


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


@pytest.mark.parametrize(
    "kind",
    ["mapped_finding", "failed_validation", "changed_surface", "invalidated_receipt"],
)
@pytest.mark.parametrize(
    "completed_state",
    [GoalState.PROVISIONALLY_COMPLETE, GoalState.VERIFIED_COMPLETE],
)
def test_each_completion_contradiction_reopens_and_records_full_provenance(
    kind: str,
    completed_state: GoalState,
) -> None:
    contradiction = ContradictionEvidence(
        goal_id="G10.S3",
        kind=kind,
        summary=f"{kind} invalidated the completion claim",
        impacted_criteria=("criterion two", "criterion one"),
        invalidated_evidence=("bafy-old-audit", "bafy-old-validation"),
        source_receipt={"receipt_cid": f"bafy-{kind}", "tree_id": CURRENT_TREE},
        scheduled_work=({"task_id": "REF-901", "reason": kind},),
        detected_at=NOW,
    )
    historical = [{"goal_id": "G10.S3", "receipt_id": "bafy-completion"}]

    decision = reopen_goal_for_contradictions(
        goal_id="G10.S3",
        current_state=completed_state,
        contradictions=[contradiction],
        historical_completion_receipts=historical,
        now=NOW,
    )
    payload = decision.to_dict()

    assert decision.state is GoalState.REOPENED
    assert decision.reopened is True
    assert decision.idempotent is False
    assert payload["contradiction_ids"] == [contradiction.contradiction_id]
    assert payload["impacted_criteria"] == ["criterion one", "criterion two"]
    assert payload["invalidated_evidence"] == ["bafy-old-audit", "bafy-old-validation"]
    assert payload["source_receipts"] == [contradiction.source_receipt]
    assert payload["newly_scheduled_work"] == [{"task_id": "REF-901", "reason": kind}]
    assert payload["historical_completion_receipts"] == historical
    assert payload["reopening_receipt"]["historical_completion_receipt_ids"] == [
        "bafy-completion"
    ]


def test_unrelated_contradiction_does_not_churn_completed_goal() -> None:
    decision = reopen_goal_for_contradictions(
        goal_id="G10.S3",
        current_state=GoalState.VERIFIED_COMPLETE,
        contradictions=[
            ContradictionEvidence(
                goal_id="G99",
                kind="mapped_finding",
                source_receipt={"receipt_cid": "bafy-unrelated"},
            )
        ],
        now=NOW,
    )

    assert decision.state is GoalState.VERIFIED_COMPLETE
    assert decision.reopened is False
    assert decision.changed is False
    assert decision.contradictions == ()
    assert decision.reopening_receipt == {}


def test_replayed_contradiction_is_idempotent_and_retains_effective_reopened_state() -> None:
    contradiction = ContradictionEvidence(
        goal_id="G10.S3",
        kind="failed_validation",
        impacted_criteria=("criterion one",),
        invalidated_evidence=("bafy-completion",),
        source_receipt={"receipt_cid": "bafy-failed-validation"},
        scheduled_work=({"task_id": "REF-901"},),
        detected_at=NOW,
    )
    first = reopen_goal_for_contradictions(
        goal_id="G10.S3",
        current_state=GoalState.VERIFIED_COMPLETE,
        contradictions=[contradiction],
        historical_completion_receipts=[
            {"goal_id": "G10.S3", "receipt_id": "bafy-completion"}
        ],
        now=NOW,
    )
    replay = reopen_goal_for_contradictions(
        goal_id="G10.S3",
        current_state=GoalState.VERIFIED_COMPLETE,
        contradictions=[contradiction],
        historical_completion_receipts=[
            {"goal_id": "G10.S3", "receipt_id": "bafy-completion"}
        ],
        existing_reopen_receipts=[first.reopening_receipt],
        now=NOW + timedelta(minutes=10),
    )

    assert replay.state is GoalState.REOPENED
    assert replay.reopened is False
    assert replay.idempotent is True
    assert replay.changed is True
    assert replay.reopening_receipt == {}
    assert replay.historical_completion_receipts == first.historical_completion_receipts


def test_reopening_recalculates_parent_and_dependent_without_erasing_history() -> None:
    goals = [
        {"goal_id": "G1", "state": "verified_complete", "acceptance_criteria": ["parent"]},
        {
            "goal_id": "G1.S1",
            "state": "verified_complete",
            "parent_goal_ids": ["G1"],
            "acceptance_criteria": ["child"],
        },
        {
            "goal_id": "G2",
            "state": "provisionally_complete",
            "depends_on": ["G1.S1"],
            "acceptance_criteria": ["dependent"],
        },
        {"goal_id": "G3", "state": "verified_complete"},
    ]
    completion_history = [
        {"goal_id": goal["goal_id"], "receipt_id": f"complete-{goal['goal_id']}"}
        for goal in goals
    ]
    contradiction = ContradictionEvidence(
        goal_id="G1.S1",
        kind="changed_surface",
        impacted_criteria=("child",),
        invalidated_evidence=("complete-G1.S1",),
        source_receipt={"receipt_cid": "bafy-changed-tree", "tree_id": CURRENT_TREE},
        scheduled_work=({"task_id": "REF-901"},),
        detected_at=NOW,
    )

    decisions = reconcile_goal_reopenings(
        goals,
        [contradiction],
        historical_completion_receipts=completion_history,
        now=NOW,
    )

    assert decisions["G1.S1"].state is GoalState.REOPENED
    assert decisions["G1"].state is GoalState.REOPENED
    assert decisions["G2"].state is GoalState.REOPENED
    assert "G3" not in decisions
    assert decisions["G1"].historical_completion_receipts == (
        {"goal_id": "G1", "receipt_id": "complete-G1"},
    )
    assert decisions["G2"].historical_completion_receipts == (
        {"goal_id": "G2", "receipt_id": "complete-G2"},
    )
    assert decisions["G1"].contradictions[0].kind == "child_reopened"
    assert decisions["G2"].contradictions[0].kind == "dependency_reopened"


def test_coverage_discovery_detects_every_reopening_surface_without_cross_goal_churn() -> None:
    previous = {
        "graph_id": "coverage-old",
        "registered_goal_ids": ["G1", "G2"],
        "criteria": [
            {
                "criterion_id": "criterion-G1",
                "goal_id": "G1",
                "criterion": "criterion one",
                "task_ids": ["REF-100"],
                "validation_receipt_ids": ["receipt-old"],
                "provenance_cids": ["bafy-old-proof"],
            }
        ],
        "finding_assignments": [],
    }
    current = {
        "graph_id": "coverage-new",
        "registered_goal_ids": ["G1", "G2"],
        "criteria": [
            {
                "criterion_id": "criterion-G1",
                "goal_id": "G1",
                "criterion": "criterion one",
                "task_ids": ["REF-101"],
                "validation_receipt_ids": ["receipt-failed", "receipt-invalidated"],
                "provenance_cids": ["bafy-old-proof"],
            }
        ],
        "receipts": [
            {
                "receipt_id": "receipt-failed",
                "status": "contradicted",
                "passed": False,
                "provenance_cid": "bafy-failed",
                "raw": {},
            },
            {
                "receipt_id": "receipt-invalidated",
                "status": "contradicted",
                "passed": True,
                "provenance_cid": "bafy-invalidated",
                "raw": {"audit_invalidated": True},
            },
        ],
        "finding_assignments": [
            {
                "finding_id": "finding-G1",
                "goal_id": "G1",
                "finding": {"actionable": True, "criterion": "criterion one"},
            },
            {
                "finding_id": "finding-unmapped",
                "goal_id": "__unmapped__",
                "finding": {"actionable": True},
            },
        ],
    }

    first = discover_goal_contradictions(
        current,
        previous_coverage=previous,
        completed_goal_ids=["G1"],
        scheduled_work="REF-901",
        detected_at=NOW,
    )
    replay = discover_goal_contradictions(
        current,
        previous_coverage=previous,
        completed_goal_ids=["G1"],
        scheduled_work="REF-901",
        detected_at=NOW + timedelta(minutes=5),
    )

    assert {item.kind for item in first} == {
        "mapped_finding",
        "failed_validation",
        "invalidated_receipt",
        "changed_surface",
    }
    assert {item.goal_id for item in first} == {"G1"}
    assert [item.contradiction_id for item in first] == [
        item.contradiction_id for item in replay
    ]
    assert all(item.scheduled_work for item in first)
    assert all(item.scheduled_work[0]["task_id"] == "REF-901" for item in first)


def test_supervisor_projects_only_explicitly_mapped_findings_as_contradictions() -> None:
    receipt = {
        "receipt_cid": "bafy-codebase-scan",
        "tree_id": CURRENT_TREE,
        "terminal_reason": "generated",
    }
    findings = [
        {
            "fingerprint": "finding-mapped",
            "goal_id": "G10.S3",
            "kind": "swallowed_exception",
            "impacted_criteria": ["criterion two", "criterion one"],
            "invalidated_evidence": ["bafy-old-validation"],
            "follow_up_task_id": "REF-901",
        },
        {
            "fingerprint": "finding-unrelated",
            "kind": "annotation",
            "bundle_key": "codebase/unregistered-bucket",
            "follow_up_task_id": "REF-902",
        },
    ]

    contradictions = PortalImplementationSupervisor._mapped_finding_contradictions(
        findings,
        source_receipt=receipt,
    )

    assert len(contradictions) == 1
    contradiction = contradictions[0]
    assert contradiction["kind"] == "mapped_finding"
    assert contradiction["goal_id"] == "G10.S3"
    assert contradiction["finding_id"] == "finding-mapped"
    assert contradiction["impacted_criteria"] == ["criterion two", "criterion one"]
    assert contradiction["invalidated_evidence"] == ["bafy-old-validation"]
    assert contradiction["source_receipt"] == receipt
    assert contradiction["scheduled_work"] == [{"task_id": "REF-901"}]


def test_supervisor_mapped_finding_projection_is_deterministic_and_deduplicated() -> None:
    receipt = {"receipt_cid": "bafy-codebase-scan", "tree_id": CURRENT_TREE}
    finding = {
        "fingerprint": "finding-one",
        "goal_ids": ["G2", "G1", "G2"],
        "newly_scheduled_work": ["REF-901"],
    }

    first = PortalImplementationSupervisor._mapped_finding_contradictions(
        [finding, dict(finding)],
        source_receipt=receipt,
    )
    second = PortalImplementationSupervisor._mapped_finding_contradictions(
        [dict(finding), finding],
        source_receipt=receipt,
    )

    assert first == second
    assert [item["goal_id"] for item in first] == ["G1", "G2"]
    assert len({item["contradiction_id"] for item in first}) == 2

    later_scan = PortalImplementationSupervisor._mapped_finding_contradictions(
        [finding],
        source_receipt={"receipt_cid": "bafy-later-scan", "tree_id": "later-tree"},
    )
    assert [item["contradiction_id"] for item in later_scan] == [
        item["contradiction_id"] for item in first
    ]
    assert later_scan[0]["source_receipt"]["receipt_cid"] == "bafy-later-scan"

    distinct_finding = PortalImplementationSupervisor._mapped_finding_contradictions(
        [finding, {**finding, "fingerprint": "finding-two"}],
        source_receipt=receipt,
    )
    assert len(distinct_finding) == 4
    assert len({item["contradiction_id"] for item in distinct_finding}) == 4


def test_supervisor_deterministically_maps_new_surface_finding_to_goal_coverage() -> None:
    goals = [{
        "goal_id": "G10.S3",
        "title": "Meta glasses API",
        "fields": {
            "status": "verified_complete",
            "acceptance": "The Meta glasses API remains operational.",
            "outputs": "swissknife/meta_glasses.py",
        },
    }]
    finding = {
        "fingerprint": "finding-new-surface",
        "source": "swissknife/meta_glasses.py:42",
        "kind": "swallowed_exception",
        "follow_up_task_id": "REF-901",
    }

    contradictions = PortalImplementationSupervisor._mapped_finding_contradictions(
        [finding],
        source_receipt={"receipt_cid": "bafy-codebase-scan"},
        goals=goals,
    )

    assert len(contradictions) == 1
    assert contradictions[0]["goal_id"] == "G10.S3"
    assert contradictions[0]["impacted_criteria"] == [
        "The Meta glasses API remains operational."
    ]
    mapping = contradictions[0]["source_receipt"]["finding_mapping"]
    assert mapping["inferred"] is True
    assert mapping["confidence"] >= 0.2


def test_supervisor_materializes_reopening_without_erasing_completion_history(tmp_path) -> None:
    objective_path = tmp_path / "objective.md"
    objective_path.write_text(
        """# Goals

## G10.S3 Completed goal

- Status: verified_complete
- Acceptance: criterion one
- Completed at: 2026-07-22T10:00:00+00:00
- Completion validation: bafy-original-validation
- Completion evidence records: [{"provenance_cid":"bafy-original-evidence"}]
""",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    config = PortalSupervisorConfig(
        todo_path=tmp_path / "todo.md",
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        objective_path=objective_path,
        repo_root=tmp_path,
    )
    supervisor = PortalImplementationSupervisor(config)
    receipt = {
        "receipt_id": "reopen-one",
        "goal_id": "G10.S3",
        "previous_state": "verified_complete",
        "state": "reopened",
        "reopened_at": NOW.isoformat(),
        "contradiction_ids": ["contradiction-one"],
        "impacted_criteria": ["criterion one"],
        "invalidated_evidence": ["bafy-original-evidence"],
        "source_receipts": [{"receipt_cid": "bafy-regression"}],
        "newly_scheduled_work": [{"task_id": "REF-901"}],
    }
    result = {
        "effective_reopened_goal_ids": ["G10.S3"],
        "goal_reopening_receipts": [receipt],
    }

    materialized = supervisor._materialize_objective_goal_reopenings(
        objective_path,
        result,
    )
    first_text = objective_path.read_text(encoding="utf-8")
    replay = supervisor._materialize_objective_goal_reopenings(objective_path, result)

    assert materialized == {"changed": True, "goal_ids": ["G10.S3"]}
    assert replay == {
        "changed": False,
        "goal_ids": ["G10.S3"],
        "reason": "already_materialized",
    }
    assert objective_path.read_text(encoding="utf-8") == first_text
    assert "- Status: reopened" in first_text
    assert "- Completion validation: bafy-original-validation" in first_text
    assert 'bafy-original-evidence"}]' in first_text
    assert "- Goal reopening receipts:" in first_text
    assert "contradiction-one" in first_text
    assert "bafy-regression" in first_text
    assert "REF-901" in first_text


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


def test_objective_tracker_does_not_treat_an_empty_configured_board_as_task_drain(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Goals

## G10.S3 Unstarted evidence-backed goal

- Status: active
- Acceptance: missing criterion
- Validation: true
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Empty task board\n", encoding="utf-8")

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
    )

    assert result.provisional_goal_ids == []
    assert result.decisions["G10.S3"]["state"] == "active"
    assert "- Status: active" in objective_path.read_text(encoding="utf-8")


def test_open_validation_gate_does_not_reopen_implementation_stage(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Goals

## G10.S3 Implemented goal awaiting proof

- Status: active
- Acceptance: criterion one
- Validation: true
""",
        encoding="utf-8",
    )
    todo_path.write_text(
        """# Task board

## PORTAL-001 Produce completion evidence

- Status: todo
- Goal id: G10.S3
- Candidate kind: validation_gate
- Merge role: validation_gate
- Missing evidence: objective validation repair
""",
        encoding="utf-8",
    )

    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
    )

    assert result.provisional_goal_ids == ["G10.S3"]
    assert result.decisions["G10.S3"]["state"] == "provisionally_complete"
    assert "- Status: provisionally_complete" in objective_path.read_text(
        encoding="utf-8"
    )


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


def test_legacy_completed_goal_migration_fails_closed_and_is_replay_stable() -> None:
    first = migrate_legacy_goal_completion(
        goal_id="G-legacy",
        legacy_state="completed",
        acceptance_criteria="criterion one",
        now=NOW,
        repository_tree=CURRENT_TREE,
    )
    replay = migrate_legacy_goal_completion(
        goal_id="G-legacy",
        legacy_state="completed",
        acceptance_criteria="criterion one",
        now=NOW + timedelta(minutes=1),
        repository_tree=CURRENT_TREE,
    )

    assert first.state is GoalState.PROVISIONALLY_COMPLETE
    assert first.verified is False
    assert first.migration_id == replay.migration_id
    assert "legacy_completion_unverified" in first.reason_codes
    assert first.to_dict()["schema"].endswith("goal_completion_migration@1")


def test_completion_compatibility_readers_accept_legacy_event_and_evidence_shapes() -> None:
    lifecycle = GoalLifecycle.from_dict({"goal_id": "G-old", "status": "completed", "events": []})
    evidence = CompletionEvidence.from_dict({
        "version": 0,
        "criterion": "criterion one",
        "task_id": "REF-old",
        "tree_identity": CURRENT_TREE,
        "receipt_cid": "bafy-old",
    })

    assert lifecycle.state is GoalState.VERIFIED_COMPLETE
    assert evidence.acceptance_criterion == "criterion one"
    assert evidence.producing_task_or_scan == "REF-old"
    assert evidence.repository_tree == CURRENT_TREE
    assert evidence.schema_version == 1
    assert evidence.metadata["source_schema_version"] == 0


def test_legacy_completed_goal_only_migrates_verified_with_full_gate() -> None:
    evidence = _complete_evidence()
    migration = migrate_legacy_goal_completion(
        goal_id="G-legacy",
        legacy_state="complete",
        acceptance_criteria="The public API returns a verified result.",
        evidence=[evidence],
        repository_tree=CURRENT_TREE,
        now=NOW,
        coverage={
            "verified": True,
            "repository_tree": CURRENT_TREE,
            "evaluated_at": NOW.isoformat(),
            "criteria": [{
                "criterion": "The public API returns a verified result.",
                "status": "verified",
            }],
        },
        analyzer_health={"status": "healthy"},
        exhaustion_quorum={
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
            "members": [
                {
                    "member_id": "normal",
                    "evidence_channel": "exhaustive",
                    "receipt_cid": "bafy-normal",
                    "finished_at": NOW.isoformat(),
                    "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
                },
                {
                    "member_id": "audit",
                    "evidence_channel": "audit",
                    "receipt_cid": "bafy-audit",
                    "finished_at": NOW.isoformat(),
                    "binding": {"tree_id": CURRENT_TREE, "repository_id": ""},
                },
            ],
        },
    )

    assert migration.state is GoalState.VERIFIED_COMPLETE
    assert migration.verified is True
    assert migration.to_dict()["diagnostics"]["confidence"] == 1.0


def test_completion_diagnostics_exposes_stale_health_quorum_and_uncovered_proof() -> None:
    decision = _evaluate([
        _complete_evidence(observed_at=NOW - timedelta(hours=2))
    ], current_state=GoalState.PROVISIONALLY_COMPLETE)
    diagnostics = completion_diagnostics(decision)

    assert diagnostics["lifecycle_state"] == "provisionally_complete"
    assert diagnostics["confidence"] < 1.0
    assert diagnostics["uncovered_criteria"]
    assert diagnostics["stale_evidence"][0]["reason_codes"] == ["stale_evidence"]
    assert diagnostics["analyzer_health"]["status"] == "healthy"
    assert diagnostics["exhaustion_quorum"]["satisfied"] is True
    assert diagnostics["reopen_reasons"] == []


def test_objective_goal_migration_can_preview_resume_and_replay(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective.md"
    original = """# Goals

## G1 Old completion one

- Status: completed
- Acceptance: criterion one

## G2 Old completion two

- Status: complete
- Acceptance: criterion two
"""
    objective_path.write_text(original, encoding="utf-8")

    preview = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        preview=True,
        max_goals=1,
        now=NOW.isoformat(),
    )
    assert preview.changed is False
    assert preview.migrated_goal_ids == ["G1"]
    assert preview.remaining_goal_ids == ["G2"]
    assert objective_path.read_text(encoding="utf-8") == original

    first = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        max_goals=1,
        now=NOW.isoformat(),
    )
    assert first.changed is True
    assert first.provisional_goal_ids == ["G1"]
    after_first = objective_path.read_text(encoding="utf-8")
    assert "- Status: provisionally_complete" in after_first
    assert "- Status: complete" in after_first
    assert "- Completion migration id: goal-migration-" in after_first

    resumed = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        max_goals=1,
        now=NOW.isoformat(),
    )
    assert resumed.migrated_goal_ids == ["G2"]
    completed_text = objective_path.read_text(encoding="utf-8")
    assert completed_text.count("- Status: provisionally_complete") == 2

    replay = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        now=NOW.isoformat(),
    )
    assert replay.changed is False
    assert replay.candidate_goal_ids == []
    assert objective_path.read_text(encoding="utf-8") == completed_text
