from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import (
    MergeQueue,
    MergeQueueFenceError,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import MergeTrain


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repo_with_candidate(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Merge Callback Test")
    _git(
        repo,
        "config",
        "user.email",
        "merge-callback@example.invalid",
    )
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    _git(repo, "switch", "-c", "candidate/callback")
    (repo / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt")
    _git(repo, "commit", "-m", "candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    return repo, base, candidate


@pytest.mark.parametrize(
    ("validation_case", "expected_validation_reason"),
    (
        ("missing", "callback_post_merge_validation_missing"),
        ("failed", "seeded_post_merge_failure"),
        ("mismatched", "callback_post_merge_validation_unbound"),
    ),
)
def test_callback_landed_mutation_settles_terminally_with_acceptance_pending(
    tmp_path: Path,
    validation_case: str,
    expected_validation_reason: str,
) -> None:
    repo, base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue", max_attempts=3)
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-PENDING",
        canonical_task_id="canonical-callback-pending",
        commit_sha=candidate,
    )
    callback_calls: list[str] = []
    validator_calls: list[str] = []

    def merge_callback(claimed) -> dict[str, Any]:
        callback_calls.append(claimed.request_id)
        _git(repo, "merge", "--ff-only", candidate)
        result: dict[str, Any] = {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
        }
        if validation_case == "failed":
            result["post_merge_validation"] = {
                "passed": False,
                "reason": "seeded_post_merge_failure",
                "validated_commit": candidate,
            }
        elif validation_case == "mismatched":
            result["post_merge_validation"] = {
                "passed": True,
                "validated_commit": base,
            }
        return result

    def post_merge_validator(*_args, **_kwargs):
        validator_calls.append("called")
        return {"passed": False, "reason": "must_not_be_reinvoked"}

    train = MergeTrain(
        repo,
        queue,
        merge_callback=merge_callback,
        post_merge_validation=post_merge_validator,
    )
    result = train.run_once()

    assert result is not None
    assert result["status"] == "integrated_pending_validation"
    assert result["merged"] is True
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["acceptance_pending"] is True
    assert result["completion_authoritative"] is False
    assert result["integration_terminal"] is True
    assert result["reason"] == "post_merge_validation_pending"
    assert result["validation_reason"] == expected_validation_reason
    assert result["target_commit"] == candidate
    assert result["merge_commit"] == candidate
    assert result["queue_settlement"] == {
        "status": "completed",
        "terminal": True,
    }
    assert result["post_merge_validation"]["passed"] is False
    assert callback_calls == [request.request_id]
    assert validator_calls == []
    assert _git(repo, "rev-parse", "refs/heads/main") == candidate

    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    completion = stored.metadata["completion"]
    assert completion["integrated"] is True
    assert completion["accepted"] is False
    assert completion["acceptance_pending"] is True
    assert completion["validation_reason"] == expected_validation_reason
    assert queue.pending_count() == 0
    assert train.run_once() is None
    assert callback_calls == [request.request_id]

    receipts = list(train.receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "integrated_pending_validation"
    assert receipt["integrated"] is True
    assert receipt["accepted"] is False


def test_successful_callback_lifts_validation_into_runtime_completion(
    tmp_path: Path,
) -> None:
    repo, _base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-ACCEPTED",
        canonical_task_id="canonical-callback-accepted",
        commit_sha=candidate,
    )
    validation = {
        "passed": True,
        "validated_commit": candidate,
        "receipt_id": "validation:callback-target",
    }

    def merge_callback(_claimed) -> dict[str, Any]:
        _git(repo, "merge", "--ff-only", candidate)
        return {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
            "post_merge_validation": validation,
        }

    train = MergeTrain(
        repo,
        queue,
        merge_callback=merge_callback,
        post_merge_validation=lambda *_args, **_kwargs: pytest.fail(
            "callback-supplied evidence must be preserved"
        ),
    )
    runtime_completions: list[dict[str, Any]] = []

    def observe_runtime_completion(
        claimed,
        *,
        target_commit: str,
        status: str,
        evidence: dict[str, Any],
    ) -> None:
        runtime_completions.append(
            {
                "request_id": claimed.request_id,
                "target_commit": target_commit,
                "status": status,
                "evidence": dict(evidence),
            }
        )

    train._runtime_completion = observe_runtime_completion  # type: ignore[method-assign]
    result = train.run_once()

    assert result is not None
    assert result["status"] == "merged"
    assert result["merged"] is True
    assert result["integrated"] is True
    assert result["accepted"] is True
    assert result["acceptance_pending"] is False
    assert result["post_merge_validation"]["passed"] is True
    assert result["post_merge_validation"]["validated_commit"] == candidate
    assert result["merge_result"]["post_merge_validation"] == (
        result["post_merge_validation"]
    )
    assert runtime_completions == [
        {
            "request_id": request.request_id,
            "target_commit": candidate,
            "status": "merged",
            "evidence": result["post_merge_validation"],
        }
    ]
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"


def test_callback_cannot_label_unrelated_child_as_completion_publication(
    tmp_path: Path,
) -> None:
    repo, _base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-SPOOFED-PUBLICATION",
        canonical_task_id="canonical-callback-spoofed-publication",
        commit_sha=candidate,
    )

    def merge_callback(_claimed) -> dict[str, Any]:
        _git(repo, "merge", "--ff-only", candidate)
        (repo / "unrelated.txt").write_text("unvalidated\n", encoding="utf-8")
        _git(repo, "add", "unrelated.txt")
        _git(repo, "commit", "-m", "unrelated immediate child")
        completion_commit = _git(repo, "rev-parse", "HEAD")
        return {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
            "post_merge_validation": {
                "passed": True,
                "validated_commit": candidate,
                "receipt_id": "validation:before-unrelated-child",
            },
            "todo_update_result": {
                "commit_result": {
                    "committed": True,
                    "commit": completion_commit,
                    "repo": str(repo),
                    "path": str(repo / "base.txt"),
                }
            },
        }

    train = MergeTrain(
        repo,
        queue,
        merge_callback=merge_callback,
        post_merge_validation=lambda *_args, **_kwargs: pytest.fail(
            "callback-supplied evidence must be inspected"
        ),
    )
    result = train.run_once()

    assert result is not None
    assert result["status"] == "integrated_pending_validation"
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["validation_reason"] == (
        "callback_post_merge_validation_unbound"
    )
    binding = result["post_merge_validation"]
    assert binding["completion_publication_trusted"] is False
    assert binding["completion_publication_path"] == "base.txt"
    assert binding["completion_publication_changed_paths"] == [
        "unrelated.txt"
    ]
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"


def test_callback_board_child_has_integrity_bound_acceptance_receipt(
    tmp_path: Path,
) -> None:
    repo, _base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-BOARD-PUBLICATION",
        canonical_task_id="canonical-callback-board-publication",
        commit_sha=candidate,
    )
    completion_commit = ""

    def merge_callback(_claimed) -> dict[str, Any]:
        nonlocal completion_commit
        _git(repo, "merge", "--ff-only", candidate)
        (repo / "base.txt").write_text("completed board\n", encoding="utf-8")
        _git(repo, "add", "base.txt")
        _git(repo, "commit", "-m", "publish board completion")
        completion_commit = _git(repo, "rev-parse", "HEAD")
        return {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
            "post_merge_validation": {
                "passed": True,
                "validated_commit": candidate,
                "receipt_id": "validation:before-board-child",
            },
            "todo_update_result": {
                "commit_result": {
                    "committed": True,
                    "commit": completion_commit,
                    "repo": str(repo),
                    "path": str(repo / "base.txt"),
                }
            },
        }

    train = MergeTrain(
        repo,
        queue,
        merge_callback=merge_callback,
        post_merge_validation=lambda *_args, **_kwargs: pytest.fail(
            "callback-supplied evidence must be inspected"
        ),
    )
    result = train.run_once()

    assert result is not None
    assert result["status"] == "merged"
    assert result["accepted"] is True
    assert result["target_commit"] == completion_commit
    [receipt] = train.acceptance_evidence_receipts()
    assert receipt.verify_integrity() is True
    assert receipt.target_commit == completion_commit
    assert receipt.post_merge_validation["validated_commit"] == candidate
    assert receipt.proved_requirement_ids_for(completion_commit) == (
        receipt.requirement_id,
    )
    binding = receipt.completion_publication_binding
    assert binding["completion_publication_trusted"] is True
    assert binding["completion_publication_parent"] == candidate
    assert binding["completion_publication_changed_paths"] == ["base.txt"]
    assert queue.get(request.request_id).status == "completed"


def test_preflight_only_callback_missing_validation_is_not_requeued(
    tmp_path: Path,
) -> None:
    repo, _base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-PREFLIGHT-ONLY",
        canonical_task_id="canonical-callback-preflight-only",
        commit_sha=candidate,
    )
    callback_calls: list[str] = []

    def merge_callback(claimed) -> dict[str, Any]:
        callback_calls.append(claimed.request_id)
        _git(repo, "merge", "--ff-only", candidate)
        return {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
        }

    train = MergeTrain(
        repo,
        queue,
        merge_callback=merge_callback,
        preflight_callback=lambda *_args, **_kwargs: {
            "passed": True,
            "target_sensitive": False,
        },
    )
    result = train.run_once()

    assert result is not None
    assert result["status"] == "integrated_pending_validation"
    assert result["merged"] is True
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["acceptance_pending"] is True
    assert result["validation_reason"] == (
        "post_merge_validation_receipt_missing"
    )
    assert callback_calls == [request.request_id]
    assert _git(repo, "rev-parse", "refs/heads/main") == candidate
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert stored.failure_count == 0
    assert train.run_once() is None
    assert callback_calls == [request.request_id]


def test_callback_post_acceptance_freshness_failure_is_not_requeued(
    tmp_path: Path,
) -> None:
    repo, base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-FRESHNESS",
        canonical_task_id="canonical-callback-freshness",
        commit_sha=candidate,
    )
    callback_calls: list[str] = []

    def merge_callback(claimed) -> dict[str, Any]:
        callback_calls.append(claimed.request_id)
        _git(repo, "merge", "--ff-only", candidate)
        return {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
            "post_merge_validation": {
                "passed": True,
                "validated_commit": candidate,
                "receipt_id": "validation:before-target-change",
            },
        }

    train = MergeTrain(
        repo,
        queue,
        merge_callback=merge_callback,
        post_merge_validation=lambda *_args, **_kwargs: pytest.fail(
            "callback-supplied evidence must be inspected"
        ),
    )
    live_target = train._target_commit
    candidate_reads = 0

    def target_changes_after_callback_receipt() -> str:
        nonlocal candidate_reads
        observed = live_target()
        if observed == candidate:
            candidate_reads += 1
            if candidate_reads > 1:
                return base
        return observed

    train._target_commit = (  # type: ignore[method-assign]
        target_changes_after_callback_receipt
    )
    result = train.run_once()

    assert result is not None
    assert result["status"] == "integrated_pending_validation"
    assert result["merged"] is True
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["acceptance_pending"] is True
    assert result["validation_reason"] == "post_merge_target_changed"
    assert result["post_merge_validation"]["passed"] is False
    assert result["post_merge_validation"]["validated_commit"] == candidate
    assert result["post_merge_validation"]["current_target_commit"] == base
    assert callback_calls == [request.request_id]
    assert _git(repo, "rev-parse", "refs/heads/main") == candidate
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert stored.failure_count == 0
    assert train.run_once() is None
    assert callback_calls == [request.request_id]


@pytest.mark.parametrize("post_merge_gate", (False, True))
def test_callback_queue_fence_preserves_landed_integration_without_retry(
    tmp_path: Path,
    post_merge_gate: bool,
) -> None:
    repo, _base, candidate = _repo_with_candidate(tmp_path)
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/callback",
        task_id="CALLBACK-FENCED",
        canonical_task_id="canonical-callback-fenced",
        commit_sha=candidate,
    )
    callback_calls: list[str] = []

    def merge_callback(claimed) -> dict[str, Any]:
        callback_calls.append(claimed.request_id)
        _git(repo, "merge", "--ff-only", candidate)
        result: dict[str, Any] = {
            "merged": True,
            "target_commit": candidate,
            "merge_commit": candidate,
        }
        if post_merge_gate:
            result["post_merge_validation"] = {
                "passed": True,
                "validated_commit": candidate,
                "receipt_id": "validation:callback-fenced",
            }
        return result

    train_kwargs: dict[str, Any] = {"merge_callback": merge_callback}
    if post_merge_gate:
        train_kwargs["post_merge_validation"] = (
            lambda *_args, **_kwargs: pytest.fail(
                "callback-supplied evidence must be used"
            )
        )
    train = MergeTrain(repo, queue, **train_kwargs)
    train._runtime_completion = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: None
    )

    def reject_stale_settlement(*_args, **_kwargs) -> None:
        raise MergeQueueFenceError("seeded newer queue owner")

    complete_claim = queue.complete
    queue.complete = reject_stale_settlement  # type: ignore[method-assign]
    result = train.run_once()

    assert result is not None
    assert result["status"] == "integrated_pending_acceptance"
    assert result["merged"] is True
    assert result["integrated"] is True
    assert result["accepted"] is False
    assert result["acceptance_pending"] is True
    assert result["completion_authoritative"] is False
    assert result["integration_terminal"] is True
    assert result["reason"] == "merge_queue_claim_fenced"
    assert result["queue_settlement"]["status"] == "fenced_out"
    assert result["queue_settlement"]["terminal"] is False
    assert callback_calls == [request.request_id]
    assert _git(repo, "rev-parse", "refs/heads/main") == candidate

    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "processing"
    assert stored.failure_count == 0
    queue.complete = complete_claim  # type: ignore[method-assign]
    recovered = train.run_once()
    assert recovered is not None
    assert recovered["callback_recovered"] is True
    assert recovered["callback_reinvoked"] is False
    assert recovered["integrated"] is True
    assert recovered["accepted"] is False
    assert recovered["acceptance_pending"] is True
    assert recovered["queue_settlement"] == {
        "status": "completed",
        "terminal": True,
    }
    assert callback_calls == [request.request_id]
    stored = queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert train.run_once() is None
