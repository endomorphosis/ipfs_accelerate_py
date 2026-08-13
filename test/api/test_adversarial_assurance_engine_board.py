"""Fail-closed tests for the operator-owned AAE supervisor controls."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts import validate_adversarial_assurance_engine_board as validator


CONTROL_PATHS = (
    validator.PLAN_REL,
    validator.OBJECTIVES_REL,
    validator.TODO_REL,
    validator.SCHEDULER_REL,
    validator.PREREQUISITES_REL,
    validator.LAUNCHER_REL,
)


def _copy_controls(tmp_path: Path) -> Path:
    for relative in CONTROL_PATHS:
        source = validator.REPO_ROOT / relative
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return tmp_path


def _rewrite(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _json_mutation(path: Path, callback) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    callback(payload)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_current_controls_are_valid_and_exactly_projected() -> None:
    report = validator.validate()
    assert report["valid"] is True, report["errors"]
    assert report["task_count"] == 64
    assert report["goal_count"] == 10
    assert report["initial_completed_task_ids"] == ["AAE-000"]
    assert report["initial_ready_task_ids"] == [
        "AAE-001",
        "AAE-002",
        "AAE-003",
        "AAE-004",
    ]
    assert report["initial_blocked_task_ids"] == ["AAE-006"]
    assert report["terminal_task_id"] == "AAE-063"


def test_blocked_prerequisite_is_truthful_but_not_release_authority() -> None:
    board = validator.validate(check_repository=False)
    release = validator.validate_prerequisites(check_repository=False)
    assert board["valid"] is True
    assert board["operator_gate"] == {
        "task_id": "AAE-006",
        "receipt_status": "blocked",
        "release_valid": False,
    }
    assert release["valid"] is False
    assert release["runtime_and_sealing_authorized"] is False


def test_monotonic_completed_progress_remains_valid(tmp_path: Path) -> None:
    root = _copy_controls(tmp_path)
    _rewrite(
        root / validator.TODO_REL,
        "## AAE-001 Inventory accelerate execution, verification, policy, state-machine, and ZK surfaces\n\n- Status: todo",
        "## AAE-001 Inventory accelerate execution, verification, policy, state-machine, and ZK surfaces\n\n- Status: completed",
    )
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is True, report["errors"]


def test_out_of_order_completion_fails_closed(tmp_path: Path) -> None:
    root = _copy_controls(tmp_path)
    _rewrite(
        root / validator.TODO_REL,
        "## AAE-005 Reconcile authority matrix, manifests, blind spots, and focused baselines\n\n- Status: todo",
        "## AAE-005 Reconcile authority matrix, manifests, blind spots, and focused baselines\n\n- Status: completed",
    )
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any("AAE-005 is completed before dependencies" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("relative", "old", "new", "expected"),
    (
        (
            validator.TODO_REL,
            "## AAE-063 Publish trust model",
            "## AAE-064 Publish trust model",
            "task IDs/order differ",
        ),
        (
            validator.TODO_REL,
            "- Depends on: AAE-056, AAE-057, AAE-058, AAE-061, AAE-062\n- Goal id: AAE-G090",
            "- Depends on: AAE-056\n- Goal id: AAE-G090",
            "dependencies differ",
        ),
        (
            validator.TODO_REL,
            "- Status: blocked\n- Blocked reason: SCG and IncrementalProofSealer",
            "- Status: todo\n- Blocked reason: SCG and IncrementalProofSealer",
            "AAE-006 status",
        ),
        (
            validator.TODO_REL,
            "- Is schedulable: false\n- Review only: false\n- Priority: P0\n- Track: prerequisite-release",
            "- Is schedulable: true\n- Review only: false\n- Priority: P0\n- Track: prerequisite-release",
            "AAE-006 schedulability differs",
        ),
        (
            validator.PLAN_REL,
            "The system used semantically targeted counterfactual mutations",
            "The system guessed",
            "prescribed bounded final claim",
        ),
    ),
)
def test_markdown_control_mutations_fail_closed(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    expected: str,
) -> None:
    root = _copy_controls(tmp_path)
    _rewrite(root / relative, old, new)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(expected in error for error in report["errors"]), report["errors"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda payload: payload["protected_paths"].pop(),
            "protected_paths differ",
        ),
        (
            lambda payload: payload["source_binding"].update(
                {"ipfs_datasets_planning_revision": "0" * 40}
            ),
            "source_binding differs",
        ),
        (
            lambda payload: payload["authority_policy"].update(
                {"mutation_score_proves_correctness": True}
            ),
            "authority doctrine differs",
        ),
        (
            lambda payload: payload["lanes"][0]["initial_task_ids"].append("AAE-001"),
            "wrong strict shard",
        ),
    ),
)
def test_scheduler_mutations_fail_closed(tmp_path: Path, mutation, expected: str) -> None:
    root = _copy_controls(tmp_path)
    _json_mutation(root / validator.SCHEDULER_REL, mutation)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(expected in error for error in report["errors"]), report["errors"]


def test_forged_completed_prerequisite_fails_closed(tmp_path: Path) -> None:
    root = _copy_controls(tmp_path)

    def forge(payload: dict[str, object]) -> None:
        payload["status"] = "completed"
        payload["runtime_and_sealing_authorized"] = True

    _json_mutation(root / validator.PREREQUISITES_REL, forge)
    report = validator.validate(root, check_repository=False)
    assert report["valid"] is False
    assert any(
        "completed prerequisite receipt" in error for error in report["errors"]
    ), report["errors"]
