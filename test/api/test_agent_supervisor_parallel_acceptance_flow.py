from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge_train import (
    PARALLEL_ACCEPTANCE_EVIDENCE_ID,
    MergeTrain,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    ValidationScheduler,
)


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


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Acceptance Flow Test")
    _git(repo, "config", "user.email", "acceptance@example.invalid")
    (repo / "base.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _candidate(repo: Path, base: str, branch: str, path: str) -> str:
    _git(repo, "switch", "-C", branch, base)
    (repo / path).write_text(f"{branch}\n", encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", branch)
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")
    return commit


def test_validation_lane_reports_measured_parallel_throughput(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        barrier.wait(timeout=5)
        time.sleep(0.04)
        return {"returncode": 0, "output": spec.command}

    cache_dir = tmp_path / "validation-cache"
    commands = ["pytest tests/test_alpha.py", "pytest tests/test_beta.py"]
    report = ValidationScheduler(
        max_workers=2,
        resource_budget=2,
        runner=runner,
        cache_dir=cache_dir,
    ).run(
        commands,
        workspace_path=tmp_path,
        changed_files=(),
        target_commit="fixture",
        dependency_state="fixture",
    )

    assert report["passed"] is True
    assert report["throughput"]["lane"] == "validation"
    assert report["throughput"]["peak_parallelism"] == 2
    assert report["throughput"]["completed_count"] == 2
    assert report["throughput"]["parallel_speedup"] > 1.4
    assert report["stages"][0]["throughput"]["completed_count"] == 2

    def unexpected_runner(**_kwargs):
        raise AssertionError("exact successful validation receipts must be reused")

    reused = ValidationScheduler(
        max_workers=2,
        resource_budget=2,
        runner=unexpected_runner,
        cache_dir=cache_dir,
    ).run(
        commands,
        workspace_path=tmp_path,
        changed_files=(),
        target_commit="fixture",
        dependency_state="fixture",
    )

    assert sorted(calls) == sorted(commands)
    assert reused["passed"] is True
    assert reused["cache_hits"] == 2
    assert all(item["cache_hit"] is True for item in reused["results"])
    assert [item["cache_key"] for item in reused["results"]] == [
        item["cache_key"] for item in report["results"]
    ]
    assert [item["output"] for item in reused["results"]] == commands


def test_parallel_acceptance_validates_synthesized_tree_before_target_cas(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    candidate = _candidate(repo, base, "candidate/fails", "bad.txt")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/fails",
        task_id="FAILS",
        canonical_task_id="canonical-fails",
        commit_sha=candidate,
    )
    observed: list[tuple[str, bool]] = []

    def reject(_request, *, workspace, synthesized_commit, **_kwargs):
        observed.append(
            (
                _git(workspace, "rev-parse", "HEAD"),
                (workspace / "bad.txt").exists(),
            )
        )
        return {
            "passed": False,
            "reason": "seeded_post_merge_failure",
            "validated_commit": synthesized_commit,
        }

    result = MergeTrain(
        repo,
        queue,
        preflight_workers=2,
        post_merge_validation=reject,
    ).drain_parallel()[0]

    assert result["reason"] == "seeded_post_merge_failure"
    assert observed == [(candidate, True)]
    assert _git(repo, "rev-parse", "refs/heads/main") == base
    assert queue.get(request.request_id).status == "quarantined"  # type: ignore[union-attr]
    assert queue.status()["completed"] == 0


def test_parallel_preflights_keep_mutation_serial_and_gate_every_completion(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    first = _candidate(repo, base, "candidate/one", "one.txt")
    second = _candidate(repo, base, "candidate/two", "two.txt")
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    requests = [
        queue.enqueue(
            branch_name="candidate/one",
            task_id="ONE",
            canonical_task_id="canonical-one",
            commit_sha=first,
        ),
        queue.enqueue(
            branch_name="candidate/two",
            task_id="TWO",
            canonical_task_id="canonical-two",
            commit_sha=second,
        ),
    ]
    barrier = threading.Barrier(2)
    active = 0
    peak = 0
    lock = threading.Lock()
    validated: list[str] = []

    def preflight(_request, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=5)
        time.sleep(0.04)
        with lock:
            active -= 1
        return {"passed": True, "target_sensitive": False}

    def validate(_request, *, workspace, synthesized_commit, **_kwargs):
        assert _git(workspace, "rev-parse", "HEAD") == synthesized_commit
        validated.append(synthesized_commit)
        return {"passed": True, "validated_commit": synthesized_commit}

    train = MergeTrain(
        repo,
        queue,
        preflight_callback=preflight,
        post_merge_validation=validate,
        preflight_workers=2,
    )
    results = train.drain()

    assert peak == 2
    assert len(results) == 2
    assert all(result["accepted"] is True for result in results)
    assert len(validated) == 2
    assert all(
        queue.get(request.request_id).status == "completed"  # type: ignore[union-attr]
        for request in requests
    )
    for result in results:
        receipt = result["acceptance_receipt"]
        assert receipt["requirement_id"] == PARALLEL_ACCEPTANCE_EVIDENCE_ID
        assert receipt["receipt_id"].startswith("sha256:")
        assert receipt["accepted"] is True
        assert receipt["sequence"] == (
            "parallel_preflight",
            "synthesized_merged_tree",
            "post_merge_validation",
            "serialized_target_mutation",
            "queue_completion_authorized",
        )
        assert (
            result["post_merge_validation"]["validated_commit"]
            == result["target_commit"]
        )
    throughput = train.status()["throughput"]
    assert throughput["lane"] == "validation-merge-acceptance"
    assert throughput["accepted_count"] == 2
    assert throughput["peak_preflight_parallelism"] == 2
    assert throughput["mutation_parallelism"] == 1
    assert throughput["requirement_id"] == PARALLEL_ACCEPTANCE_EVIDENCE_ID


def test_target_sensitive_preflight_is_revalidated_after_each_accepted_merge(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    first = _candidate(repo, base, "candidate/first", "first.txt")
    second = _candidate(repo, base, "candidate/second", "second.txt")
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    requests = [
        queue.enqueue(
            branch_name="candidate/first",
            task_id="FIRST",
            canonical_task_id="canonical-first",
            commit_sha=first,
        ),
        queue.enqueue(
            branch_name="candidate/second",
            task_id="SECOND",
            canonical_task_id="canonical-second",
            commit_sha=second,
        ),
    ]
    barrier = threading.Barrier(2)
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def target_sensitive_preflight(request, *, target_commit, **_kwargs):
        with calls_lock:
            calls.append((request.task_id, target_commit))
            initial_call = len(calls) <= 2
        if initial_call:
            barrier.wait(timeout=5)
        return {
            "passed": True,
            "target_sensitive": True,
            "validated_target": target_commit,
        }

    def validate(_request, *, synthesized_commit, **_kwargs):
        return {"passed": True, "validated_commit": synthesized_commit}

    train = MergeTrain(
        repo,
        queue,
        preflight_callback=target_sensitive_preflight,
        post_merge_validation=validate,
        preflight_workers=2,
    )
    results = train.drain_parallel(max_items=2)

    assert len(results) == 2
    assert all(item["accepted"] is True for item in results)
    assert set(calls[:2]) == {("FIRST", base), ("SECOND", base)}
    assert calls[2:] == [("SECOND", results[0]["target_commit"])]
    assert results[1]["preflight"]["stale_preflight_replaced"] is True
    assert (
        results[1]["preflight"]["validated_target"]
        == results[0]["target_commit"]
    )
    assert train.status()["throughput"]["stale_preflight_count"] == 1
    assert all(
        queue.get(request.request_id).status == "completed"  # type: ignore[union-attr]
        for request in requests
    )
    assert list(train.worktree_dir.iterdir()) == []


def test_conflicting_parallel_preflight_cannot_mutate_past_queue_order(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    first = _candidate(repo, base, "candidate/first", "base.txt")
    second = _candidate(repo, base, "candidate/second", "base.txt")
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    first_request = queue.enqueue(
        branch_name="candidate/first",
        task_id="FIRST",
        canonical_task_id="canonical-first",
        commit_sha=first,
    )
    second_request = queue.enqueue(
        branch_name="candidate/second",
        task_id="SECOND",
        canonical_task_id="canonical-second",
        commit_sha=second,
    )

    def validate(_request, *, synthesized_commit, **_kwargs):
        return {"passed": True, "validated_commit": synthesized_commit}

    results = MergeTrain(
        repo,
        queue,
        post_merge_validation=validate,
        preflight_workers=2,
    ).drain_parallel(max_items=2)

    assert len(results) == 2
    assert results[0]["accepted"] is True
    assert results[0]["request_id"] == first_request.request_id
    assert results[1]["accepted"] is False
    assert results[1]["reason"] == "preflight_failed"
    assert results[1]["preflight"]["stale_preflight_replaced"] is True
    assert _git(repo, "show", "refs/heads/main:base.txt") == "candidate/first"
    assert queue.get(first_request.request_id).status == "completed"  # type: ignore[union-attr]
    assert queue.get(second_request.request_id).status == "pending"  # type: ignore[union-attr]


def test_validation_receipt_for_another_commit_cannot_authorize_completion(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    candidate = _candidate(repo, base, "candidate/mismatch", "mismatch.txt")
    queue = MergeQueue(tmp_path / "queue")
    request = queue.enqueue(
        branch_name="candidate/mismatch",
        task_id="MISMATCH",
        canonical_task_id="canonical-mismatch",
        commit_sha=candidate,
    )

    def wrong_commit_receipt(_request, **_kwargs):
        return {"passed": True, "validated_commit": base}

    result = MergeTrain(
        repo,
        queue,
        preflight_workers=2,
        post_merge_validation=wrong_commit_receipt,
    ).drain_parallel()[0]

    assert result["accepted"] is False
    assert result["reason"] == "post_merge_validation_target_mismatch"
    validation = result["post_merge_validation"]
    assert validation["passed"] is False
    assert validation["validated_commit"] == base
    assert validation["synthesized_commit"] != base
    assert _git(repo, "rev-parse", "refs/heads/main") == base
    assert queue.get(request.request_id).status == "quarantined"  # type: ignore[union-attr]
    assert queue.status()["completed"] == 0


def test_restart_recovers_abandoned_claim_and_reapplies_post_merge_gate(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    candidate = _candidate(repo, base, "candidate/restart", "restart.txt")
    queue_dir = tmp_path / "queue"
    queue = MergeQueue(queue_dir, max_attempts=3)
    request = queue.enqueue(
        branch_name="candidate/restart",
        task_id="RESTART",
        canonical_task_id="canonical-restart",
        commit_sha=candidate,
    )
    abandoned = queue.dequeue(consumer_id="merge-train:999999:crashed")
    assert abandoned is not None and abandoned.status == "processing"
    validated: list[str] = []

    def validate(_request, *, synthesized_commit, **_kwargs):
        validated.append(synthesized_commit)
        return {"passed": True, "validated_commit": synthesized_commit}

    restarted_queue = MergeQueue(queue_dir, max_attempts=3)
    result = MergeTrain(
        repo,
        restarted_queue,
        owner_id="merge-train:replacement",
        preflight_workers=2,
        post_merge_validation=validate,
    ).drain_parallel()[0]

    assert result["accepted"] is True
    assert validated == [result["target_commit"]]
    stored = restarted_queue.get(request.request_id)
    assert stored is not None and stored.status == "completed"
    assert stored.attempt == 2
    assert stored.failure_count == 1
    assert _git(repo, "rev-parse", "refs/heads/main") == result["target_commit"]


def test_merge_queue_batch_claims_respect_merge_debt_bound(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "queue", max_processing=2)
    for ordinal in range(4):
        queue.enqueue(
            branch_name=f"candidate/{ordinal}",
            task_id=f"TASK-{ordinal}",
            commit_sha=str(ordinal + 1) * 40,
        )

    claimed = queue.dequeue_many(4, consumer_id="parallel-train")

    assert len(claimed) == 2
    assert queue.dequeue_many(1, consumer_id="other-train") == ()
    status = queue.status()
    assert status["merge_debt"] == 2
    assert status["backpressure"] is True
    assert status["throughput"]["lane"] == "merge-queue-persistence"
