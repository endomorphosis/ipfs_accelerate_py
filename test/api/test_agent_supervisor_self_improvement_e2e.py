from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

import ipfs_accelerate_py.agent_supervisor as supervisor_api
from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_completion import (
    SELF_IMPROVEMENT_ROOT_ACCEPTANCE_CRITERIA,
    SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS,
    SELF_IMPROVEMENT_ROOT_OBJECTIVE_ID,
    SELF_IMPROVEMENT_ROOT_OBJECTIVE_REVISION,
    SELF_IMPROVEMENT_ROOT_PRODUCING_TASK_IDS,
    SELF_IMPROVEMENT_ROOT_REQUIRED_EXHAUSTIVE_RECEIPTS,
    evaluate_self_improvement_root_completion,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement_rollout import (
    PAIRED_EFFICIENCY_REQUIREMENT_ID,
    PAIRED_ROLLOUT_ACCEPTANCE_CRITERIA,
    PAIRED_ROLLOUT_CHILD_GOAL_IDS,
    PAIRED_ROLLOUT_COMPLETION_ANALYZER_VERSION,
    PAIRED_ROLLOUT_COMPLETION_CONFIGURATION_REVISION,
    PAIRED_ROLLOUT_OBJECTIVE_ID,
    PAIRED_ROLLOUT_OBJECTIVE_REVISION,
    PAIRED_ROLLOUT_PRODUCING_TASK_IDS,
    PAIRED_ROLLOUT_REQUIRED_EXHAUSTIVE_RECEIPTS,
    SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
    PairedFixtureKind,
    PairedRolloutFixture,
    PairedRolloutRequirementEvidence,
    PairedRolloutReport,
    PairedRolloutReportStore,
    PairedRolloutValidationError,
    REQUIRED_PAIRED_FIXTURE_KINDS,
    RolloutBehaviorMeasurement,
    SelfImprovementRolloutMode,
    evaluate_paired_rollout_completion,
    evaluate_paired_self_improvement_rollout,
)


NOW = datetime(2026, 7, 24, 22, 30, tzinfo=timezone.utc)
REPOSITORY_ID = "ipfs-accelerate-py"
REPOSITORY_TREE = "sha256:asi-082-current-tree"
ANALYZER_VERSION = "objective-analyzer@asi-082"
CONFIGURATION_REVISION = "self-improvement-v1"
ROLLOUT_EVIDENCE_TREE = "sha256:" + "e" * 64


def _binding() -> dict[str, str]:
    return {
        "repository_id": REPOSITORY_ID,
        "tree_id": REPOSITORY_TREE,
        "objective_id": SELF_IMPROVEMENT_ROOT_OBJECTIVE_ID,
        "objective_revision": SELF_IMPROVEMENT_ROOT_OBJECTIVE_REVISION,
        "analyzer_version": ANALYZER_VERSION,
        "configuration_revision": CONFIGURATION_REVISION,
    }


def _evidence() -> list[CompletionEvidence]:
    return [
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan=f"test:asi-082:{index}",
            validation_receipt=f"bafy-validation-{index}",
            repository_id=REPOSITORY_ID,
            repository_tree=REPOSITORY_TREE,
            freshness=True,
            provenance_cid=f"bafy-root-criterion-{index}",
            validation_passed=True,
            observed_at=NOW - timedelta(minutes=5),
            contradictory=False,
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "reason_codes": [],
                }
            },
        )
        for index, criterion in enumerate(
            SELF_IMPROVEMENT_ROOT_ACCEPTANCE_CRITERIA,
            start=1,
        )
    ]


def _coverage(
    evidence: list[CompletionEvidence],
) -> dict[str, object]:
    receipt_by_criterion = {
        item.acceptance_criterion: item.provenance_cid for item in evidence
    }
    return {
        "verified": True,
        "repository_tree": REPOSITORY_TREE,
        "evaluated_at": (NOW - timedelta(minutes=2)).isoformat(),
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "implementation": [
                    "ipfs_accelerate_py/agent_supervisor",
                    "test/api/test_agent_supervisor_self_improvement_e2e.py",
                ],
                "validation_receipt_id": receipt_by_criterion[criterion],
            }
            for criterion in SELF_IMPROVEMENT_ROOT_ACCEPTANCE_CRITERIA
        ],
    }


def _health() -> dict[str, object]:
    return {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "binding": _binding(),
    }


def _quorum() -> dict[str, object]:
    binding = _binding()
    return {
        "satisfied": True,
        "required_members": (
            SELF_IMPROVEMENT_ROOT_REQUIRED_EXHAUSTIVE_RECEIPTS
        ),
        "member_count": 2,
        "binding": binding,
        "members": [
            {
                "member_id": "deterministic-objective-scan",
                "evidence_channel": "objective-graph",
                "receipt_cid": "bafy-root-objective-scan",
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": (NOW - timedelta(minutes=4)).isoformat(),
                "binding": binding,
            },
            {
                "member_id": "independent-backlog-audit",
                "evidence_channel": "todo-proof-audit",
                "receipt_cid": "bafy-root-backlog-audit",
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": (NOW - timedelta(minutes=3)).isoformat(),
                "binding": binding,
            },
        ],
    }


def _proof_requirement(goal_id: str) -> dict[str, object]:
    return {
        "goal_id": goal_id,
        "acceptance_criterion": f"{goal_id} remains proved",
        "obligation_id": f"obligation:{goal_id}",
        "proof_receipt_id": f"proof:{goal_id}",
        "required_assurance": "candidate",
        "authoritative_assurance": "candidate",
        "proof_verdict": "proved",
        "freshness": "current",
        "repository_tree": REPOSITORY_TREE,
        "provenance_id": f"bafy-proof-{goal_id}",
        "assurance_satisfied": True,
        "contradicted": False,
        "reason_codes": [],
    }


def _children() -> list[dict[str, object]]:
    return [
        {
            "goal_id": goal_id,
            "state": "verified_complete",
            "verified": True,
            "proof_requirements": [_proof_requirement(goal_id)],
            "completion_gate": {
                "passed": True,
                "evaluated_evidence": {
                    "repository_id": REPOSITORY_ID,
                    "repository_tree": REPOSITORY_TREE,
                    "evaluated_at": (
                        NOW - timedelta(minutes=6)
                    ).isoformat(),
                    "validation_evidence": [
                        {
                            "valid": True,
                            "reason_codes": [],
                            "evidence": {
                                "repository_id": REPOSITORY_ID,
                                "repository_tree": REPOSITORY_TREE,
                                "provenance_cid": f"bafy-child-{goal_id}",
                            },
                        }
                    ],
                    "proof_requirements": [_proof_requirement(goal_id)],
                    "child_goals": [],
                },
            },
        }
        for goal_id in SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS
    ]


def _tasks() -> list[dict[str, str]]:
    return [
        {"task_id": task_id, "status": "completed"}
        for task_id in SELF_IMPROVEMENT_ROOT_PRODUCING_TASK_IDS
    ]


def _inputs() -> dict[str, object]:
    evidence = _evidence()
    return {
        "repository_id": REPOSITORY_ID,
        "repository_tree": REPOSITORY_TREE,
        "producing_tasks": _tasks(),
        "child_goals": _children(),
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": _coverage(evidence),
        "analyzer_health": _health(),
        "exhaustion_quorum": _quorum(),
        "now": NOW,
        "freshness_seconds": 3600,
    }


def test_root_contract_fixes_every_non_narrowable_population() -> None:
    assert SELF_IMPROVEMENT_ROOT_OBJECTIVE_ID == "ASI-G000"
    assert len(SELF_IMPROVEMENT_ROOT_ACCEPTANCE_CRITERIA) == 4
    assert SELF_IMPROVEMENT_ROOT_PRODUCING_TASK_IDS == tuple(
        f"ASI-{index:03d}" for index in range(1, 25)
    )
    assert SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS == tuple(
        f"ASI-G{index:03d}" for index in range(10, 100, 10)
    )
    assert SELF_IMPROVEMENT_ROOT_REQUIRED_EXHAUSTIVE_RECEIPTS == 2
    assert (
        supervisor_api.evaluate_self_improvement_root_completion
        is evaluate_self_improvement_root_completion
    )


def test_fresh_closed_root_packet_requires_two_lifecycle_transitions() -> None:
    values = _inputs()
    provisional = evaluate_self_improvement_root_completion(**values)

    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.verified is False
    assert provisional.gate is not None and provisional.gate.passed
    assert "provisional_transition_required" in provisional.reason_codes

    verified = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified is True
    assert verified.gate is not None and verified.gate.passed


@pytest.mark.parametrize(
    "mutation",
    [
        lambda tasks: tasks.pop(),
        lambda tasks: tasks.append(deepcopy(tasks[0])),
        lambda tasks: tasks[0].update(status="todo"),
        lambda tasks: tasks[0].update(task_id="ASI-999"),
    ],
    ids=["missing", "duplicate", "incomplete", "foreign"],
)
def test_root_rejects_narrowed_or_incomplete_producer_population(
    mutation,
) -> None:
    values = _inputs()
    tasks = values["producing_tasks"]
    assert isinstance(tasks, list)
    mutation(tasks)

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert "tasks_incomplete" in decision.reason_codes


def test_root_rejects_false_task_completion_even_with_complete_population() -> None:
    values = _inputs()
    values["tasks_complete"] = False

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert "tasks_incomplete" in decision.reason_codes


def test_root_rejects_a_caller_lowered_exhaustion_configuration() -> None:
    with pytest.raises(ValueError, match="configured ASI-G000 count"):
        evaluate_self_improvement_root_completion(
            **_inputs(),
            required_exhaustive_receipts=1,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].pop("implementation"),
        lambda rows: rows[0].update(validation_receipt_id="bafy-detached"),
        lambda rows: rows.pop(),
        lambda rows: rows.append(deepcopy(rows[0])),
    ],
    ids=[
        "missing-implementation",
        "detached-validation",
        "missing-criterion",
        "duplicate-criterion",
    ],
)
def test_root_requires_exact_implementation_and_validation_coverage(
    mutation,
) -> None:
    values = _inputs()
    coverage = values["coverage"]
    assert isinstance(coverage, dict)
    rows = coverage["criteria"]
    assert isinstance(rows, list)
    mutation(rows)

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert any(
        code in decision.reason_codes
        for code in ("coverage_missing", "coverage_unverified")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda health: health.pop("safe_for_completion_reasoning"),
        lambda health: health.update(healthy=False),
        lambda health: health["binding"].update(tree_id="sha256:old"),
        lambda health: health["binding"].pop("configuration_revision"),
    ],
    ids=["implicit-safety", "unhealthy", "foreign-tree", "incomplete-binding"],
)
def test_root_requires_explicit_fully_bound_completion_safe_analyzer(
    mutation,
) -> None:
    values = _inputs()
    health = values["analyzer_health"]
    assert isinstance(health, dict)
    mutation(health)

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert any(
        code in decision.reason_codes
        for code in ("analyzer_unhealthy", "analyzer_completion_unsafe")
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda quorum: (
            quorum["members"].pop(),
            quorum.update(member_count=1),
        ),
        lambda quorum: quorum["members"][1].update(
            receipt_cid=quorum["members"][0]["receipt_cid"]
        ),
        lambda quorum: quorum["members"][1].update(healthy=False),
        lambda quorum: quorum["members"][1].update(
            safe_for_completion_reasoning=False
        ),
        lambda quorum: quorum["members"][1].update(scan_mode="incremental"),
        lambda quorum: quorum["members"][1].update(
            finished_at=(NOW - timedelta(hours=2)).isoformat()
        ),
        lambda quorum: quorum["members"][1]["binding"].update(
            tree_id="sha256:foreign"
        ),
    ],
    ids=[
        "insufficient",
        "duplicate-receipt",
        "unhealthy",
        "unsafe",
        "non-exhaustive",
        "stale",
        "foreign-bound",
    ],
)
def test_root_requires_independent_fresh_healthy_exhaustive_receipts(
    mutation,
) -> None:
    values = _inputs()
    quorum = values["exhaustion_quorum"]
    assert isinstance(quorum, dict)
    mutation(quorum)

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert any(
        code.startswith("exhaustion_quorum_")
        for code in decision.reason_codes
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda children: children.pop(),
        lambda children: children.append(deepcopy(children[0])),
        lambda children: children[0].update(state="reopened", verified=False),
        lambda children: children[0].pop("completion_gate"),
        lambda children: children[0]["completion_gate"][
            "evaluated_evidence"
        ].update(repository_tree="sha256:old"),
        lambda children: children[0]["completion_gate"][
            "evaluated_evidence"
        ].update(evaluated_at=(NOW - timedelta(hours=2)).isoformat()),
        lambda children: children[0]["completion_gate"][
            "evaluated_evidence"
        ]["validation_evidence"][0].update(valid=False),
    ],
    ids=[
        "missing",
        "duplicate",
        "reopened",
        "missing-gate",
        "foreign-tree",
        "stale",
        "failed-validation",
    ],
)
def test_root_requires_exact_fresh_tree_bound_child_population(
    mutation,
) -> None:
    values = _inputs()
    children = values["child_goals"]
    assert isinstance(children, list)
    mutation(children)

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert any(code.startswith("child_") for code in decision.reason_codes)


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    [
        ("freshness", "stale", "child_proof_stale"),
        ("proof_verdict", "inconclusive", "child_proof_inconclusive"),
        ("proof_verdict", "disproved", "child_proof_contradicted"),
        (
            "assurance_satisfied",
            False,
            "child_required_assurance_not_satisfied",
        ),
    ],
)
def test_root_revalidates_every_descendant_proof_requirement(
    field: str,
    value: object,
    reason_code: str,
) -> None:
    values = _inputs()
    children = values["child_goals"]
    assert isinstance(children, list)
    requirements = children[0]["proof_requirements"]
    assert isinstance(requirements, list)
    requirements[0][field] = value
    if field == "proof_verdict" and value == "disproved":
        requirements[0]["contradicted"] = True

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert reason_code in decision.reason_codes


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence: evidence.pop(),
        lambda evidence: evidence.__setitem__(
            0, replace(evidence[0], validation_passed=False)
        ),
        lambda evidence: evidence.__setitem__(
            0,
            replace(
                evidence[0],
                observed_at=NOW - timedelta(hours=2),
            ),
        ),
        lambda evidence: evidence.append(
            CompletionEvidence(
                acceptance_criterion=evidence[0].acceptance_criterion,
                producing_task_or_scan="test:failed-sibling",
                validation_receipt="bafy-failed-sibling",
                repository_id=REPOSITORY_ID,
                repository_tree=REPOSITORY_TREE,
                freshness=True,
                provenance_cid="bafy-failed-sibling",
                validation_passed=False,
                observed_at=NOW - timedelta(minutes=1),
                metadata={
                    "evidence_source_policy": {
                        "satisfies": True,
                        "reason_codes": [],
                    }
                },
            )
        ),
    ],
    ids=["missing", "failed", "stale", "invalid-sibling"],
)
def test_every_submitted_validation_must_be_fresh_passing_and_covered(
    mutation,
) -> None:
    values = _inputs()
    evidence = values["evidence"]
    assert isinstance(evidence, list)
    mutation(evidence)

    decision = evaluate_self_improvement_root_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified
    assert any(
        code in decision.reason_codes
        for code in (
            "missing_criterion_evidence",
            "failed_validation",
            "stale_evidence",
            "validation_evidence_incomplete",
        )
    )


def test_checked_in_root_remains_actionable_until_live_proof_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    objective_text = (
        repo_root
        / "docs/architecture/agent_supervisor_self_improvement.objectives.md"
    ).read_text(encoding="utf-8")
    todo_text = (
        repo_root
        / "docs/architecture/agent_supervisor_self_improvement.todo.md"
    ).read_text(encoding="utf-8")
    root_block = objective_text.split(
        "## ASI-G000 Efficient and trustworthy supervisor control loop",
        1,
    )[1].split("\n## ", 1)[0]
    task_states = dict(
        re.findall(
            r"^## (ASI-\d{3}) .+?\n\n- Status: ([^\n]+)",
            todo_text,
            flags=re.MULTILINE,
        )
    )

    assert "- Status: provisionally_complete" in root_block
    assert "- ASI-082 root completion gate:" in root_block
    assert task_states["ASI-082"] in {"todo", "completed"}
    assert set(SELF_IMPROVEMENT_ROOT_PRODUCING_TASK_IDS) <= task_states.keys()
    for child_id in SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS:
        child_block = objective_text.split(f"## {child_id} ", 1)[1].split(
            "\n## ",
            1,
        )[0]
        assert "- Status: verified_complete" not in child_block


def test_checked_in_rollout_gap_routes_to_canonical_heap_child() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    objective_text = (
        repo_root
        / "docs/architecture/agent_supervisor_self_improvement.objectives.md"
    ).read_text(encoding="utf-8")
    efficiency_block = objective_text.split(
        "## ASI-G113 Prove "
        "146189916032404266364029134505159070240",
        1,
    )[1].split("\n## ", 1)[0]
    lazy_export_block = objective_text.split(
        "## ASI-G114 Prove "
        "300500866741873729474343907613893393545",
        1,
    )[1].split("\n## ", 1)[0]
    strict_validation_block = objective_text.split(
        "## ASI-G100 Prove "
        "314133036252270790078901745919131980427",
        1,
    )[1].split("\n## ", 1)[0]
    planning_block = objective_text.split("## ASI-G115 ", 1)[1].split(
        "\n## ",
        1,
    )[0]

    assert "## ASI-G116 " not in objective_text
    assert "- Parent: ASI-G090" in efficiency_block
    assert (
        "- Evidence: 146189916032404266364029134505159070240"
        in efficiency_block
    )
    assert "`PAIRED_EFFICIENCY_REQUIREMENT_ID`" in efficiency_block
    assert "`PairedRolloutRequirementEvidence`" in efficiency_block
    assert "ASI-041 resolves discovery fingerprint" in efficiency_block
    assert "scan's ASI-G116 allocation is stale" in efficiency_block
    assert "- Parent: ASI-G090" in lazy_export_block
    assert (
        "- Evidence: 300500866741873729474343907613893393545"
        in lazy_export_block
    )
    assert (
        "`PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID`"
        in lazy_export_block
    )
    assert "`PAIRED_ROLLOUT_STABLE_EXPORTS`" in lazy_export_block
    assert (
        "test_stable_rollout_exports_remain_lazy_without_optional_providers"
        in lazy_export_block
    )
    assert "ASI-053 resolves discovery fingerprint" in lazy_export_block
    assert "scan's ASI-G100 allocation is stale" in lazy_export_block
    assert "314133036252270790078901745919131980427" in (
        strict_validation_block
    )
    assert "300500866741873729474343907613893393545" not in (
        strict_validation_block
    )
    assert "312819945606360295782005228058369235550" in planning_block
    assert "146189916032404266364029134505159070240" not in planning_block


def _rollout_measurement(
    kind: PairedFixtureKind,
    *,
    candidate: bool,
) -> RolloutBehaviorMeasurement:
    rejected = {
        PairedFixtureKind.CONTRADICTORY,
        PairedFixtureKind.MALFORMED_OUTPUT,
        PairedFixtureKind.FAILED_VALIDATION,
    }
    accepted = (
        4
        if kind is PairedFixtureKind.INDEPENDENT_PARALLEL
        else 2
        if kind is PairedFixtureKind.CONFLICTING_PARALLEL
        else 0
        if kind
        in rejected
        | {
            PairedFixtureKind.PROVIDER_UNAVAILABLE,
            PairedFixtureKind.DRAINED_REFILL,
        }
        else 1
    )
    outcome = (
        "rejected"
        if kind in rejected
        else "degraded"
        if kind is PairedFixtureKind.PROVIDER_UNAVAILABLE
        else "exhausted"
        if kind is PairedFixtureKind.DRAINED_REFILL
        else "accepted"
    )
    repeated = kind in {
        PairedFixtureKind.WARM,
        PairedFixtureKind.RESTART,
    }
    state = "sha256:" + "d" * 64
    seeded = 1 if kind is PairedFixtureKind.FAILED_VALIDATION else 0
    return RolloutBehaviorMeasurement(
        input_tokens=620 if candidate else 1_000,
        cache_lookups=10 if repeated else 1,
        cache_hits=8 if candidate and repeated else 2 if repeated else 0,
        false_completions=0,
        authority_violations=0,
        stale_authoritative_hits=0,
        artifact_count=1,
        artifact_bytes=512,
        elapsed_ms=(
            1_900
            if candidate and kind is PairedFixtureKind.INDEPENDENT_PARALLEL
            else 4_000
            if kind is PairedFixtureKind.INDEPENDENT_PARALLEL
            else 900
            if candidate
            else 1_000
        ),
        completed_work=max(1, accepted),
        accepted_work=accepted,
        evidence_coverage_bps=9_200 if candidate else 9_000,
        quality_score_bps=9_200 if candidate else 9_000,
        invalid_plan_branches=4 if candidate else 5,
        seeded_defects=seeded,
        detected_defects=seeded,
        escaped_defects=0,
        false_rejections=0,
        merge_conflicts=(
            1 if kind is PairedFixtureKind.CONFLICTING_PARALLEL else 0
        ),
        duplicate_executions=0,
        unauthorized_mutations=0,
        terminal_outcome=outcome,
        state_digest_before=(
            state if kind is PairedFixtureKind.RESTART else ""
        ),
        state_digest_after=(
            state if kind is PairedFixtureKind.RESTART else ""
        ),
    )


def _rollout_fixtures() -> tuple[PairedRolloutFixture, ...]:
    return tuple(
        PairedRolloutFixture(
            fixture_id=f"e2e:{kind.value}",
            fixture_kind=kind,
            fixture_revision="asi-023-e2e@1",
            input_digest="sha256:" + f"{index + 100:064x}",
            baseline=_rollout_measurement(kind, candidate=False),
            candidate=_rollout_measurement(kind, candidate=True),
        )
        for index, kind in enumerate(REQUIRED_PAIRED_FIXTURE_KINDS)
    )


def _rollout_evidence(
    report: PairedRolloutReport,
    requirement_id: str,
) -> PairedRolloutRequirementEvidence:
    return report.evidence_for(
        requirement_id,
        repository_id=REPOSITORY_ID,
        repository_tree=ROLLOUT_EVIDENCE_TREE,
    )


def test_paired_rollout_gate_survives_process_restart_without_state_drift(
    tmp_path,
) -> None:
    report = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    store = PairedRolloutReportStore(tmp_path / "paired-rollout")
    store.persist(report)

    restarted = PairedRolloutReportStore(tmp_path / "paired-rollout")
    recovered = restarted.load(report.report_id)

    assert recovered.report_id == report.report_id
    assert recovered.promotion_allowed
    assert recovered.effective_mode is SelfImprovementRolloutMode.AUTOMATIC
    assert recovered["metrics"]["candidate_false_completions"] == 0
    assert recovered["metrics"]["candidate_authority_violations"] == 0
    assert recovered["metrics"]["candidate_stale_authoritative_hits"] == 0
    fixtures = {
        item["fixture_kind"]: item for item in recovered["fixtures"]
    }
    assert fixtures["malformed_output"]["candidate"][
        "terminal_outcome"
    ] == "rejected"
    assert fixtures["provider_unavailable"]["candidate"][
        "terminal_outcome"
    ] == "degraded"
    assert fixtures["drained_refill"]["candidate"][
        "duplicate_executions"
    ] == 0


def test_any_end_to_end_authority_failure_keeps_candidate_in_shadow() -> None:
    fixtures = tuple(
        replace(
            item,
            candidate=replace(
                item.candidate,
                stale_authoritative_hits=1,
            ),
        )
        if item.fixture_kind is PairedFixtureKind.STALE_CACHE
        else item
        for item in _rollout_fixtures()
    )

    decision = evaluate_paired_self_improvement_rollout(
        fixtures,
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )

    assert not decision.promotion_allowed
    assert decision.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert "candidate_stale_authoritative_hit" in decision.reason_codes
    assert not decision["nonnegotiable_gate_passed"]


def test_rollout_requirement_evidence_is_typed_bound_and_content_addressed() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )

    safety = _rollout_evidence(
        report, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
    )
    efficiency = _rollout_evidence(report, PAIRED_EFFICIENCY_REQUIREMENT_ID)

    assert SHADOW_FALSE_COMPLETION_REQUIREMENT_ID == (
        "109590900757783560279417463762322084165"
    )
    assert PAIRED_EFFICIENCY_REQUIREMENT_ID == (
        "146189916032404266364029134505159070240"
    )
    assert safety.requirement_id == SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
    assert efficiency.requirement_id == PAIRED_EFFICIENCY_REQUIREMENT_ID
    assert safety.goal_id == "ASI-G112"
    assert efficiency.goal_id == "ASI-G113"
    assert safety.repository_id == REPOSITORY_ID
    assert safety.repository_tree == ROLLOUT_EVIDENCE_TREE
    assert safety.requirement_satisfied
    assert efficiency.requirement_satisfied
    assert safety.report_id == report.report_id
    assert efficiency.report_id == report.report_id
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", safety.evidence_id)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", efficiency.evidence_id)
    assert safety.evidence_id != efficiency.evidence_id


@pytest.mark.parametrize("kind", REQUIRED_PAIRED_FIXTURE_KINDS)
def test_seeded_false_completion_proves_shadow_blocking_only_for_closed_population(
    kind: PairedFixtureKind,
) -> None:
    fixtures = tuple(
        replace(
            item,
            candidate=replace(item.candidate, false_completions=1),
        )
        if item.fixture_kind is kind
        else item
        for item in _rollout_fixtures()
    )
    blocked = evaluate_paired_self_improvement_rollout(
        fixtures,
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )

    evidence = _rollout_evidence(
        blocked, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
    )
    assert blocked.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert "candidate_false_completion" in blocked.reason_codes
    assert evidence.requirement_satisfied
    assert evidence.report_id == blocked.report_id

    incomplete = evaluate_paired_self_improvement_rollout(
        fixtures[:-1],
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    incomplete_evidence = _rollout_evidence(
        incomplete,
        SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
    )
    assert incomplete.effective_mode is SelfImprovementRolloutMode.SHADOW
    assert not incomplete_evidence.requirement_satisfied
    assert "required_fixture_missing:drained_refill" in (
        incomplete_evidence.reason_codes
    )


def test_rollout_requirement_evidence_restoration_rejects_tampering() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    evidence = _rollout_evidence(report, PAIRED_EFFICIENCY_REQUIREMENT_ID)

    assert PairedRolloutRequirementEvidence.from_dict(
        evidence.to_dict(),
        report=report,
    ) == evidence

    tampered = evidence.to_dict()
    tampered["requirement_satisfied"] = False
    with pytest.raises(
        PairedRolloutValidationError,
        match="evidence|identity|derived",
    ):
        PairedRolloutRequirementEvidence.from_dict(tampered, report=report)

    detached = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures()[:-1],
        desired_mode=SelfImprovementRolloutMode.AUTOMATIC,
        evaluated_at=NOW,
    )
    with pytest.raises(
        PairedRolloutValidationError,
        match="report|detached|identity",
    ):
        PairedRolloutRequirementEvidence.from_dict(
            evidence.to_dict(),
            report=detached,
        )


def _g090_binding() -> dict[str, str]:
    return {
        "repository_id": REPOSITORY_ID,
        "tree_id": ROLLOUT_EVIDENCE_TREE,
        "objective_id": PAIRED_ROLLOUT_OBJECTIVE_ID,
        "objective_revision": PAIRED_ROLLOUT_OBJECTIVE_REVISION,
        "analyzer_version": PAIRED_ROLLOUT_COMPLETION_ANALYZER_VERSION,
        "configuration_revision": (
            PAIRED_ROLLOUT_COMPLETION_CONFIGURATION_REVISION
        ),
    }


def _g090_completion_evidence() -> list[CompletionEvidence]:
    return [
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan=f"test:asi-090:{index}",
            validation_receipt=f"bafy-g090-validation-{index}",
            repository_id=REPOSITORY_ID,
            repository_tree=ROLLOUT_EVIDENCE_TREE,
            freshness=True,
            provenance_cid=f"bafy-g090-criterion-{index}",
            validation_passed=True,
            observed_at=NOW - timedelta(minutes=5),
            contradictory=False,
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "reason_codes": [],
                }
            },
        )
        for index, criterion in enumerate(
            PAIRED_ROLLOUT_ACCEPTANCE_CRITERIA,
            start=1,
        )
    ]


def _g090_coverage(
    evidence: list[CompletionEvidence],
) -> dict[str, object]:
    receipts = {
        item.acceptance_criterion: item.provenance_cid for item in evidence
    }
    return {
        "verified": True,
        "repository_tree": ROLLOUT_EVIDENCE_TREE,
        "evaluated_at": (NOW - timedelta(minutes=2)).isoformat(),
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "implementation": [
                    "ipfs_accelerate_py/agent_supervisor/"
                    "self_improvement_rollout.py",
                    "ipfs_accelerate_py/agent_supervisor/__init__.py",
                    "docs/guides/AGENT_SUPERVISOR_GUIDE.md",
                ],
                "validation_receipt_id": receipts[criterion],
            }
            for criterion in PAIRED_ROLLOUT_ACCEPTANCE_CRITERIA
        ],
    }


def _g090_children() -> list[dict[str, object]]:
    return [
        {
            "goal_id": goal_id,
            "state": "verified_complete",
            "verified": True,
            "proof_requirements": [
                {
                    "goal_id": goal_id,
                    "acceptance_criterion": f"{goal_id} remains proved",
                    "obligation_id": f"obligation:{goal_id}",
                    "proof_receipt_id": f"proof:{goal_id}",
                    "required_assurance": "candidate",
                    "authoritative_assurance": "candidate",
                    "proof_verdict": "proved",
                    "freshness": "current",
                    "repository_tree": ROLLOUT_EVIDENCE_TREE,
                    "provenance_id": f"bafy-g090-proof-{goal_id}",
                    "assurance_satisfied": True,
                    "contradicted": False,
                    "reason_codes": [],
                }
            ],
            "completion_gate": {
                "passed": True,
                "evaluated_evidence": {
                    "repository_id": REPOSITORY_ID,
                    "repository_tree": ROLLOUT_EVIDENCE_TREE,
                    "evaluated_at": (
                        NOW - timedelta(minutes=6)
                    ).isoformat(),
                    "validation_evidence": [
                        {
                            "valid": True,
                            "reason_codes": [],
                            "evidence": {
                                "repository_id": REPOSITORY_ID,
                                "repository_tree": ROLLOUT_EVIDENCE_TREE,
                                "provenance_cid": (
                                    f"bafy-g090-child-{goal_id}"
                                ),
                            },
                        }
                    ],
                },
            },
        }
        for goal_id in PAIRED_ROLLOUT_CHILD_GOAL_IDS
    ]


def _g090_quorum() -> dict[str, object]:
    binding = _g090_binding()
    return {
        "satisfied": True,
        "required_members": PAIRED_ROLLOUT_REQUIRED_EXHAUSTIVE_RECEIPTS,
        "member_count": PAIRED_ROLLOUT_REQUIRED_EXHAUSTIVE_RECEIPTS,
        "binding": binding,
        "members": [
            {
                "member_id": "g090-objective-scan",
                "evidence_channel": "paired-rollout",
                "receipt_cid": "bafy-g090-objective-scan",
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": (NOW - timedelta(minutes=4)).isoformat(),
                "binding": binding,
            },
            {
                "member_id": "g090-backlog-audit",
                "evidence_channel": "stable-export-adoption",
                "receipt_cid": "bafy-g090-backlog-audit",
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": (NOW - timedelta(minutes=3)).isoformat(),
                "binding": binding,
            },
        ],
    }


def _g090_inputs(report: PairedRolloutReport) -> dict[str, object]:
    evidence = _g090_completion_evidence()
    return {
        "repository_id": REPOSITORY_ID,
        "repository_tree": ROLLOUT_EVIDENCE_TREE,
        "requirement_evidence": [
            _rollout_evidence(
                report, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
            ),
            _rollout_evidence(report, PAIRED_EFFICIENCY_REQUIREMENT_ID),
        ],
        "producing_tasks": [
            {"task_id": task_id, "status": "completed"}
            for task_id in PAIRED_ROLLOUT_PRODUCING_TASK_IDS
        ],
        "child_goals": _g090_children(),
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": _g090_coverage(evidence),
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "binding": _g090_binding(),
        },
        "exhaustion_quorum": _g090_quorum(),
        "now": NOW,
        "freshness_seconds": 3600,
    }


def test_g090_completion_requires_closed_current_tree_packet_and_two_phases() -> None:
    report = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        evaluated_at=NOW,
    )
    values = _g090_inputs(report)

    assert PAIRED_ROLLOUT_OBJECTIVE_ID == "ASI-G090"
    assert PAIRED_ROLLOUT_PRODUCING_TASK_IDS == ("ASI-023", "ASI-024")
    assert PAIRED_ROLLOUT_CHILD_GOAL_IDS == (
        "ASI-G112",
        "ASI-G113",
        "ASI-G114",
    )
    assert len(PAIRED_ROLLOUT_ACCEPTANCE_CRITERIA) == 5
    assert PAIRED_ROLLOUT_REQUIRED_EXHAUSTIVE_RECEIPTS == 2
    assert (
        supervisor_api.evaluate_paired_rollout_completion
        is evaluate_paired_rollout_completion
    )

    provisional = report.evaluate_objective_completion(**values)
    assert provisional.gate.passed
    assert provisional.next_state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert "provisional_transition_required" in provisional.reason_codes

    verified = evaluate_paired_rollout_completion(
        report,
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    assert verified.gate.passed
    assert verified.next_state is GoalState.VERIFIED_COMPLETE
    assert verified.verified


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        (
            "producer",
            lambda values: values["producing_tasks"].pop(),
        ),
        (
            "child",
            lambda values: values["child_goals"][0].update(
                state="active"
            ),
        ),
        (
            "criterion",
            lambda values: values["evidence"].pop(),
        ),
        (
            "failed criterion receipt",
            lambda values: values["evidence"].__setitem__(
                0,
                replace(
                    values["evidence"][0],
                    validation_passed=False,
                ),
            ),
        ),
        (
            "stale criterion receipt",
            lambda values: values["evidence"].__setitem__(
                0,
                replace(
                    values["evidence"][0],
                    observed_at=NOW - timedelta(hours=2),
                ),
            ),
        ),
        (
            "foreign-tree criterion receipt",
            lambda values: values["evidence"].__setitem__(
                0,
                replace(
                    values["evidence"][0],
                    repository_tree="sha256:" + "f" * 64,
                ),
            ),
        ),
        (
            "coverage",
            lambda values: values["coverage"]["criteria"][0].update(
                validation_receipt_id="bafy-detached"
            ),
        ),
        (
            "analyzer",
            lambda values: values["analyzer_health"].update(
                safe_for_completion_reasoning=False
            ),
        ),
        (
            "analyzer binding",
            lambda values: values["analyzer_health"]["binding"].update(
                tree_id="sha256:" + "f" * 64
            ),
        ),
        (
            "quorum",
            lambda values: values["exhaustion_quorum"]["members"][1].update(
                receipt_cid="bafy-g090-objective-scan"
            ),
        ),
        (
            "stale quorum receipt",
            lambda values: values["exhaustion_quorum"]["members"][1].update(
                finished_at=(NOW - timedelta(hours=2)).isoformat()
            ),
        ),
        (
            "non-exhaustive quorum receipt",
            lambda values: values["exhaustion_quorum"]["members"][1].update(
                scan_mode="sampled"
            ),
        ),
        (
            "requirement",
            lambda values: values["requirement_evidence"].pop(),
        ),
    ],
)
def test_g090_rejects_open_or_unbound_completion_packet(
    field: str,
    mutate,
) -> None:
    report = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        evaluated_at=NOW,
    )
    values = _g090_inputs(report)
    mutate(values)

    decision = report.evaluate_objective_completion(
        **values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )

    assert not decision.verified, field
    assert decision.next_state is not GoalState.VERIFIED_COMPLETE, field


def test_g090_rejects_failed_or_stale_rollout_and_lowered_quorum() -> None:
    failed = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures()[:-1],
        evaluated_at=NOW,
    )
    failed_decision = failed.evaluate_objective_completion(
        **_g090_inputs(failed),
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    assert not failed_decision.verified
    assert failed_decision.next_state is not GoalState.VERIFIED_COMPLETE

    report = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        evaluated_at=NOW - timedelta(hours=2),
    )
    stale_values = _g090_inputs(report)
    stale_values["requirement_evidence"] = [
        _rollout_evidence(
            report, SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
        ),
        _rollout_evidence(report, PAIRED_EFFICIENCY_REQUIREMENT_ID),
    ]
    stale_decision = report.evaluate_objective_completion(
        **stale_values,
        current_state=GoalState.PROVISIONALLY_COMPLETE,
    )
    assert not stale_decision.verified

    fresh = evaluate_paired_self_improvement_rollout(
        _rollout_fixtures(),
        evaluated_at=NOW,
    )
    with pytest.raises(ValueError, match="configured ASI-G090 count"):
        fresh.evaluate_objective_completion(
            **_g090_inputs(fresh),
            required_exhaustive_receipts=1,
        )


def test_operator_profiles_document_the_g090_completion_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    guide = (
        repo_root / "docs/guides/AGENT_SUPERVISOR_GUIDE.md"
    ).read_text(encoding="utf-8")
    architecture = (
        repo_root / "docs/architecture/AGENT_SUPERVISOR_ARCHITECTURE.md"
    ).read_text(encoding="utf-8")
    plan = (
        repo_root
        / "docs/architecture/AGENT_SUPERVISOR_SELF_IMPROVEMENT_PLAN.md"
    ).read_text(encoding="utf-8")
    discovery = (
        repo_root
        / "data/agent_supervisor/discovery/"
        "2026-07-25-asi-090-completion-gate-evidence.md"
    ).read_text(encoding="utf-8")

    assert "### Requesting ASI-G090 completion" in guide
    assert "test_agent_supervisor_self_improvement_e2e.py" in guide
    assert "test_agent_supervisor_self_improvement_benchmark.py" in guide
    assert "`safe_for_completion_reasoning: true`" in guide
    assert "`paired-rollout-completion@1`" in guide
    assert "exactly two fresh members" in guide
    assert "separate later evaluation" in guide
    assert "Completion authority remains separate." in architecture
    assert "### ASI-G090 parent completion gate" in plan
    assert "- Producing tasks: ASI-023, ASI-024" in discovery
    assert "- Child goals: ASI-G112, ASI-G113, ASI-G114" in discovery
    assert (
        "This file is an audit and provenance index, not a completion receipt."
        in discovery
    )


def test_stable_rollout_exports_remain_lazy_without_optional_providers() -> None:
    module = "ipfs_accelerate_py.agent_supervisor"
    rollout_module = f"{module}.self_improvement_rollout"
    program = f"""
import importlib
import json
import sys

import {module} as api

requirement_id = api.PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID
goal_id = api.PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID
stable_exports = api.PAIRED_ROLLOUT_STABLE_EXPORTS
optional_modules = tuple(
    f"{{api.__name__}}.{{name}}" for name in api._LAZY_PROVIDER_EXPORTS
) + (
    f"{{api.__name__}}.ipfs_datasets_analysis_provider",
    f"{{api.__name__}}.ipfs_datasets_logic_provider",
)
before = {{
    "rollout_loaded": {rollout_module!r} in sys.modules,
    "optional_loaded": [
        name for name in optional_modules if name in sys.modules
    ],
    "requirement_id": requirement_id,
    "goal_id": goal_id,
    "requirement_in_all": (
        "PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID" in api.__all__
    ),
    "goal_in_all": "PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID" in api.__all__,
    "manifest_in_all": "PAIRED_ROLLOUT_STABLE_EXPORTS" in api.__all__,
    "exports_in_all": all(name in api.__all__ for name in stable_exports),
}}
root_values = {{name: getattr(api, name) for name in stable_exports}}
rollout = importlib.import_module({rollout_module!r})
identical = all(
    value is getattr(rollout, name) for name, value in root_values.items()
)
module_public = tuple(rollout.__all__)
manifest_complete = (
    set(stable_exports) == set(module_public)
    and len(stable_exports) == len(set(stable_exports))
)
report_version = api.PAIRED_ROLLOUT_REPORT_VERSION
evidence_version = api.PAIRED_ROLLOUT_REQUIREMENT_EVIDENCE_VERSION
rollout_type = api.PairedRolloutReport
evidence_type = api.PairedRolloutRequirementEvidence
after = {{
    "rollout_loaded": {rollout_module!r} in sys.modules,
    "optional_loaded": [
        name for name in optional_modules if name in sys.modules
    ],
    "requirement_id": api.PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID,
    "goal_id": api.PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID,
    "safety_id": api.SHADOW_FALSE_COMPLETION_REQUIREMENT_ID,
    "efficiency_id": api.PAIRED_EFFICIENCY_REQUIREMENT_ID,
    "report_version": report_version,
    "evidence_version": evidence_version,
    "identical": identical,
    "manifest_complete": manifest_complete,
    "rollout_type_module": rollout_type.__module__,
    "evidence_type_module": evidence_type.__module__,
}}
print(json.dumps({{"before": before, "after": after}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["before"] == {
        "exports_in_all": True,
        "goal_id": "ASI-G114",
        "goal_in_all": True,
        "manifest_in_all": True,
        "optional_loaded": [],
        "requirement_id": (
            "300500866741873729474343907613893393545"
        ),
        "requirement_in_all": True,
        "rollout_loaded": False,
    }
    assert result["after"]["rollout_loaded"] is True
    assert result["after"]["optional_loaded"] == []
    assert supervisor_api.PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID == (
        "300500866741873729474343907613893393545"
    )
    assert result["after"]["requirement_id"] == (
        supervisor_api.PAIRED_ROLLOUT_LAZY_EXPORT_REQUIREMENT_ID
    )
    assert supervisor_api.PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID == "ASI-G114"
    assert result["after"]["goal_id"] == (
        supervisor_api.PAIRED_ROLLOUT_LAZY_EXPORT_GOAL_ID
    )
    assert result["after"]["safety_id"] == (
        SHADOW_FALSE_COMPLETION_REQUIREMENT_ID
    )
    assert result["after"]["efficiency_id"] == (
        PAIRED_EFFICIENCY_REQUIREMENT_ID
    )
    assert result["after"]["report_version"] == 2
    assert result["after"]["evidence_version"] == 1
    assert result["after"]["identical"] is True
    assert result["after"]["manifest_complete"] is True
    assert result["after"]["rollout_type_module"] == rollout_module
    assert result["after"]["evidence_type_module"] == rollout_module
