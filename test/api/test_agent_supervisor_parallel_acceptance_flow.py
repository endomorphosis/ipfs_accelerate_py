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

    def runner(*, spec, **_kwargs):
        barrier.wait(timeout=5)
        time.sleep(0.04)
        return {"returncode": 0, "output": spec.command}

    report = ValidationScheduler(
        max_workers=2, resource_budget=2, runner=runner
    ).run(
        ["pytest tests/test_alpha.py", "pytest tests/test_beta.py"],
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
