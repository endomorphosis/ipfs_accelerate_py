from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re

import pytest

import ipfs_accelerate_py.agent_supervisor as supervisor_api
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement_completion import (
    SELF_IMPROVEMENT_ROOT_ACCEPTANCE_CRITERIA,
    SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS,
    SELF_IMPROVEMENT_ROOT_OBJECTIVE_ID,
    SELF_IMPROVEMENT_ROOT_OBJECTIVE_REVISION,
    SELF_IMPROVEMENT_ROOT_PRODUCING_TASK_IDS,
    SELF_IMPROVEMENT_ROOT_REQUIRED_EXHAUSTIVE_RECEIPTS,
    evaluate_self_improvement_root_completion,
)


NOW = datetime(2026, 7, 24, 22, 30, tzinfo=timezone.utc)
REPOSITORY_ID = "ipfs-accelerate-py"
REPOSITORY_TREE = "sha256:asi-082-current-tree"
ANALYZER_VERSION = "objective-analyzer@asi-082"
CONFIGURATION_REVISION = "self-improvement-v1"


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
    assert task_states["ASI-082"] == "todo"
    assert any(
        task_states[task_id] != "completed"
        for task_id in SELF_IMPROVEMENT_ROOT_PRODUCING_TASK_IDS
    )
    for child_id in SELF_IMPROVEMENT_ROOT_CHILD_GOAL_IDS:
        child_block = objective_text.split(f"## {child_id} ", 1)[1].split(
            "\n## ",
            1,
        )[0]
        assert "- Status: verified_complete" not in child_block
