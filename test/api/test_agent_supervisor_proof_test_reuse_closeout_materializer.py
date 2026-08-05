"""Tests for closeout materializer merge projection and git recovery."""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_closeout_materializer import (
    CloseoutMaterializerIdentity,
    materialize_task_evidence,
    project_managed_merge_queue_record,
    project_managed_merge_queue_records,
    recover_managed_merge_receipts_from_git,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_task_evidence import (
    ProofTestReuseTaskEvidenceCollector,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_cached_test_validation import (
    validation_command_identity,
)

NOW = 1_786_000_000.0
NOW_MS = int(NOW * 1_000)
COMMAND = (
    "IPFS_TEST_PROOF_REUSE_MODE=off "
    "python3 -m pytest test/proof/test_one.py -q"
)
REPOSITORY_ID = "repository:sha256:current"
STATE_CID = "baguqeera-state-current"
COMMIT = "f" * 40
TREE = "e" * 40
GITLINKS = "baguqeera-gitlinks-current"
FOREST = "baguqeera-forest-current"
OVERLAY = "baguqeera-overlay-clean"


def _task(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "goal_id": f"{task_id}-GOAL",
        "canonical_task_cid": f"baguqeera-task-{task_id.lower()}",
        "board_namespace": "proof-backed-test-reuse-v1",
        "validation": [COMMAND],
    }


def _board(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "valid": True,
        "board_namespace": "proof-backed-test-reuse-v1",
        "task_count": len(tasks),
        "task_ids": [task["task_id"] for task in tasks],
        "task_cids": {
            task["task_id"]: task["canonical_task_cid"] for task in tasks
        },
    }


def _seal(record: dict[str, Any], field: str) -> dict[str, Any]:
    return {**record, field: content_identity(record)}


def _validation(task: dict[str, Any]) -> dict[str, Any]:
    record = {
        "task_id": task["task_id"],
        "goal_id": task["goal_id"],
        "task_cid": task["canonical_task_cid"],
        "validation_command": COMMAND,
        "validation_command_cid": validation_command_identity(COMMAND),
        "repository_id": REPOSITORY_ID,
        "repository_state_cid": STATE_CID,
        "git_commit_id": COMMIT,
        "git_tree_id": TREE,
        "gitlink_state_cid": GITLINKS,
        "repository_forest_cid": FOREST,
        "dirty": False,
        "dirty_overlay_cid": OVERLAY,
        "proof_reuse_mode": "off",
        "disposition": "executed",
        "status": "passed",
        "passed": True,
        "exit_code": 0,
        "skipped_count": 0,
        "observed_at_ms": NOW_MS - 1_000,
        "fresh_until_ms": NOW_MS + 30_000,
    }
    return _seal(record, "validation_receipt_cid")


def _identity() -> CloseoutMaterializerIdentity:
    return CloseoutMaterializerIdentity(
        repository_id=REPOSITORY_ID,
        repository_state_cid=STATE_CID,
        git_commit_id=COMMIT,
        git_tree_id=TREE,
        gitlink_state_cid=GITLINKS,
        repository_forest_cid=FOREST,
        dirty=False,
        dirty_overlay_cid=OVERLAY,
    )


def test_project_managed_merge_queue_record_strips_floats_and_seals() -> None:
    raw = {
        "task_id": "PTR-012",
        "status": "completed",
        "canonical_task_id": "baguqeera-task-ptr-012",
        "commit_sha": "a" * 40,
        "enqueued_at": 1785552741.843,
        "claimed_at": 0.0,
        "metadata": {"nested": {"score": 1.5}},
    }
    projected = project_managed_merge_queue_record(raw)
    assert projected is not None
    assert projected["task_id"] == "PTR-012"
    assert projected["canonical_task_cid"] == "baguqeera-task-ptr-012"
    assert projected["status"] == "completed"
    assert projected["commit_sha"] == "a" * 40
    assert "merge_receipt_cid" in projected
    assert "enqueued_at" not in projected
    # Sealed body must content-identity without floats.
    body = {
        key: value
        for key, value in projected.items()
        if key != "merge_receipt_cid"
    }
    assert content_identity(body) == projected["merge_receipt_cid"]


def test_collector_accepts_raw_daemon_merge_rows_with_floats() -> None:
    task = _task("PTR-012")
    raw_merge = {
        "task_id": task["task_id"],
        "status": "completed",
        "canonical_task_id": task["canonical_task_cid"],
        "commit_sha": "b" * 40,
        "enqueued_at": 1785552741.843,
        "failure_count": 0,
        "metadata": {"x": 1.25},
    }
    collector = ProofTestReuseTaskEvidenceCollector(
        repository_id=REPOSITORY_ID,
        repository_state_cid=STATE_CID,
        git_commit_id=COMMIT,
        git_tree_id=TREE,
        gitlink_state_cid=GITLINKS,
        repository_forest_cid=FOREST,
        dirty=False,
        dirty_overlay_cid=OVERLAY,
        freshness_seconds=300.0,
        ancestry_verifier=lambda ancestor, target: bool(ancestor),
        clock=lambda: NOW,
    )
    result = collector.collect(
        _board([task]),
        task_records=[task],
        merge_queue_records=[raw_merge],
        validation_receipts=[_validation(task)],
    )
    assert len(result.evidence) == 1
    assert result.evidence[0].task_id == "PTR-012"
    assert not result.gaps


def test_materialize_task_evidence_reports_gaps_and_next_actions() -> None:
    task = _task("PTR-012")
    report = materialize_task_evidence(
        identity=_identity(),
        validated_board=_board([task]),
        task_records=[task],
        merge_queue_records=[],
        validation_receipts=[_validation(task)],
        freshness_seconds=300.0,
        ancestry_verifier=lambda ancestor, target: True,
        clock=lambda: NOW,
    )
    assert report.authority is False
    assert report.validation_receipt_count == 1
    assert report.evidence_count == 0
    assert "PTR-012" in report.completion_missing_task_ids
    assert any("managed-merge" in action for action in report.next_actions)


def test_recover_managed_merge_receipts_from_git_uses_ancestor_commits() -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        calls.append((tuple(args), kwargs))

        class Result:
            def __init__(self, returncode: int = 0, stdout: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""

        if args[:3] == ["git", "log", "--oneline"]:
            return Result(
                0,
                "abc1234 PTR-010: mark todo completed\n"
                "def5678 PTR-010: Implement core locator\n",
            )
        if args[:2] == ["git", "rev-parse"]:
            return Result(0, "a" * 40 + "\n")
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return Result(0, "")
        return Result(1, "")

    recovered = recover_managed_merge_receipts_from_git(
        repo_root="/tmp",
        task_ids=["PTR-010"],
        task_cids={"PTR-010": "baguqeera-task-ptr-010"},
        head_commit="f" * 40,
        git_runner=fake_run,
    )
    assert len(recovered) == 1
    assert recovered[0]["task_id"] == "PTR-010"
    assert recovered[0]["commit_sha"] == "a" * 40
    assert recovered[0]["recovery_source"] == "git_ancestry"
    assert project_managed_merge_queue_records(recovered)


def test_materialize_with_git_recovery_closes_completion_gap() -> None:
    task = _task("PTR-010")

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        class Result:
            def __init__(self, returncode: int = 0, stdout: str = "") -> None:
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = ""

        if args[:3] == ["git", "log", "--oneline"]:
            return Result(0, "abc1234 PTR-010: mark todo completed\n")
        if args[:2] == ["git", "rev-parse"]:
            return Result(0, "c" * 40 + "\n")
        if args[:3] == ["git", "merge-base", "--is-ancestor"]:
            return Result(0, "")
        return Result(1, "")

    # Patch recovery by injecting recovered merge rows via empty queue + custom recovery
    # through recover function used inside materialize — we call recover then materialize.
    recovered = recover_managed_merge_receipts_from_git(
        repo_root="/tmp",
        task_ids=["PTR-010"],
        task_cids={"PTR-010": task["canonical_task_cid"]},
        head_commit=COMMIT,
        git_runner=fake_run,
    )
    report = materialize_task_evidence(
        identity=_identity(),
        validated_board=_board([task]),
        task_records=[task],
        merge_queue_records=recovered,
        validation_receipts=[_validation(task)],
        freshness_seconds=300.0,
        ancestry_verifier=lambda ancestor, target: True,
        clock=lambda: NOW,
    )
    assert report.evidence_count == 1
    assert report.gap_count == 0
    assert report.evidence_task_ids == ("PTR-010",)
