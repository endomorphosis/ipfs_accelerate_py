from __future__ import annotations

import copy
import subprocess
import threading
import time
from concurrent.futures import CancelledError
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.merge.merge_train import (
    PARALLEL_ACCEPTANCE_EVIDENCE_ID,
    PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA,
    PARALLEL_EXECUTION_COMPLETION_ANALYZER_VERSION,
    PARALLEL_EXECUTION_COMPLETION_CONFIGURATION_REVISION,
    PARALLEL_EXECUTION_OBJECTIVE_ID,
    PARALLEL_EXECUTION_OBJECTIVE_REVISION,
    PARALLEL_EXECUTION_PRODUCING_TASK_IDS,
    PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS,
    MergeTrain,
    ParallelAcceptanceReceipt,
    evaluate_parallel_execution_completion,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchRequest,
    ProviderBatchScheduler,
    ProviderBatchSchedulerConfig,
    ProviderBatchStatus,
)
from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    AdaptiveThroughputRun,
    ResourcePolicy,
    evaluate_adaptive_throughput_benchmark,
)
from ipfs_accelerate_py.agent_supervisor.validation.validation_scheduler import (
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


def _g060_binding(repository_tree: str) -> dict[str, str]:
    return {
        "repository_id": "repository:ipfs-accelerate-py",
        "tree_id": repository_tree,
        "objective_id": PARALLEL_EXECUTION_OBJECTIVE_ID,
        "objective_revision": PARALLEL_EXECUTION_OBJECTIVE_REVISION,
        "analyzer_version": (
            PARALLEL_EXECUTION_COMPLETION_ANALYZER_VERSION
        ),
        "configuration_revision": (
            PARALLEL_EXECUTION_COMPLETION_CONFIGURATION_REVISION
        ),
    }


def _provider_cancellation_receipt():
    entered = threading.Event()
    release = threading.Event()

    def dispatch(requests):
        members = tuple(requests)
        entered.set()
        assert release.wait(5)
        return [f"accepted:{item.request_id}" for item in members]

    scheduler = ProviderBatchScheduler(
        dispatch,
        config=ProviderBatchSchedulerConfig(
            max_batch_size=2,
            min_batch_size=2,
            batch_window_ms=5,
            max_parallel_batches=1,
        ),
    )
    try:
        cancelled = scheduler.submit(
            ProviderBatchRequest(
                request_id="cancel-me",
                payload="cancel",
                token_budget=100,
            )
        )
        sibling = scheduler.submit(
            ProviderBatchRequest(
                request_id="keep-me",
                payload="keep",
                token_budget=200,
            )
        )
        assert entered.wait(5)
        assert scheduler.cancel("cancel-me") is True
        release.set()
        assert sibling.result(timeout=5).status is ProviderBatchStatus.SUCCEEDED
        with pytest.raises(CancelledError):
            cancelled.result(timeout=5)
        assert scheduler.flush(5)
        receipts = scheduler.partial_cancellation_evidence()
        assert len(receipts) == 1
        return receipts[0]
    finally:
        release.set()
        scheduler.shutdown(wait=True)


def _g060_completion_packet(tmp_path: Path) -> dict[str, object]:
    repo, base = _repo(tmp_path)
    candidate = _candidate(
        repo, base, "candidate/completion", "completion.txt"
    )
    queue = MergeQueue(tmp_path / "completion-queue")

    def validate(_request, *, synthesized_commit, **_kwargs):
        return {
            "passed": True,
            "validated_commit": synthesized_commit,
            "receipt_id": "validation:merged-tree",
        }

    queue.enqueue(
        branch_name="candidate/completion",
        task_id="COMPLETION",
        canonical_task_id="canonical-completion",
        commit_sha=candidate,
    )
    train = MergeTrain(
        repo,
        queue,
        post_merge_validation=validate,
        preflight_workers=2,
    )
    accepted = train.drain_parallel()[0]
    live_acceptance_receipts = train.acceptance_evidence_receipts()
    assert len(live_acceptance_receipts) == 1
    acceptance_receipt = live_acceptance_receipts[0]
    assert isinstance(acceptance_receipt, ParallelAcceptanceReceipt)
    restored_acceptance = ParallelAcceptanceReceipt.from_dict(
        accepted["acceptance_receipt"]
    )
    assert restored_acceptance.verify_integrity() is False
    repository_tree = accepted["target_commit"]

    # G060's acceptance criterion is an explicit >=2x adaptive-throughput
    # proof. Keep the fixture bound to that criterion even though the global
    # policy default is stricter for ordinary production runs.
    policy = ResourcePolicy(
        max_lanes=2,
        adaptive_enabled=True,
        adaptive_minimum_throughput_multiplier=2,
    )
    fixture_ids = ("analysis", "validation")
    adaptive_receipt = evaluate_adaptive_throughput_benchmark(
        AdaptiveThroughputRun(
            fixture_ids=fixture_ids,
            executed_fixture_ids=fixture_ids,
            accepted_fixture_ids=fixture_ids,
            duration_ms=200,
            peak_concurrency=1,
        ),
        AdaptiveThroughputRun(
            fixture_ids=fixture_ids,
            executed_fixture_ids=fixture_ids,
            accepted_fixture_ids=fixture_ids,
            duration_ms=90,
            peak_concurrency=2,
        ),
        policy=policy,
        repository_tree_id=repository_tree,
    )
    provider_receipt = _provider_cancellation_receipt()

    now = datetime.now(timezone.utc)
    validation_command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_adaptive_resources.py "
        "test/api/test_agent_supervisor_provider_batch_scheduler.py "
        "test/api/test_agent_supervisor_parallel_acceptance_flow.py -q"
    )
    criterion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-083",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": repository_tree,
                "command": validation_command,
            },
            validation_passed=True,
            repository_id="repository:ipfs-accelerate-py",
            repository_tree=repository_tree,
            freshness={"fresh": True},
            observed_at=now - timedelta(minutes=2),
            provenance_cid=f"validation:asi-083:{index}",
        )
        for index, criterion in enumerate(
            PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = {
        "verified": True,
        "repository_id": "repository:ipfs-accelerate-py",
        "repository_tree": repository_tree,
        "evaluated_at": (now - timedelta(minutes=1)).isoformat(),
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    + (
                        "resource_scheduler.py"
                        if index == 1
                        else (
                            "provider_batch_scheduler.py"
                            if index == 2
                            else "merge_train.py"
                        )
                    )
                ),
                "validation": validation_command,
                "validation_receipt_ids": [
                    f"validation:asi-083:{index}"
                ],
            }
            for index, criterion in enumerate(
                PARALLEL_EXECUTION_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    binding = _g060_binding(repository_tree)
    members = [
        {
            "member_id": "asi-083-implementation",
            "evidence_channel": "implementation-validation",
            "receipt_cid": "scan:asi-083:implementation",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "analyzer_version": "asi-083/implementation-v1",
            "passed": True,
            "healthy": True,
            "exhaustive": True,
            "safe_for_completion_reasoning": True,
            "conclusive": True,
            "contradicted": False,
            "finished_at": (now - timedelta(minutes=4)).isoformat(),
        },
        {
            "member_id": "asi-083-independent-audit",
            "evidence_channel": "independent-audit",
            "receipt_cid": "scan:asi-083:audit",
            "binding": dict(binding),
            "scan_mode": "exhaustive",
            "analyzer_version": "asi-083/independent-audit-v1",
            "passed": True,
            "healthy": True,
            "exhaustive": True,
            "safe_for_completion_reasoning": True,
            "conclusive": True,
            "contradicted": False,
            "finished_at": (now - timedelta(minutes=3)).isoformat(),
        },
    ]
    return {
        "repository_id": "repository:ipfs-accelerate-py",
        "repository_tree": repository_tree,
        "resource_policy": policy,
        "operational_evidence": (
            adaptive_receipt,
            provider_receipt,
            acceptance_receipt,
        ),
        "producing_tasks": [
            {"task_id": task_id, "status": "completed"}
            for task_id in PARALLEL_EXECUTION_PRODUCING_TASK_IDS
        ],
        "evidence": criterion_evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "exhaustive": True,
            "safe_for_completion_reasoning": True,
            "binding": dict(binding),
        },
        "exhaustion_quorum": {
            "required_members": (
                PARALLEL_EXECUTION_REQUIRED_EXHAUSTIVE_RECEIPTS
            ),
            "member_count": len(members),
            "satisfied": True,
            "quorum_met": True,
            "binding": dict(binding),
            "members": members,
        },
        "now": now,
        "freshness_seconds": 3600,
    }


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


def test_g060_completion_requires_live_typed_current_tree_proof_packet(
    tmp_path: Path,
) -> None:
    packet = _g060_completion_packet(tmp_path)
    provisional = evaluate_parallel_execution_completion(
        current_state=GoalState.ACTIVE,
        **packet,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.verified is False

    verified = evaluate_parallel_execution_completion(
        current_state=provisional.state,
        **packet,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified is True
    evaluated = verified.gate.to_dict()["evaluated_evidence"]
    assert evaluated["repository_tree"] == packet["repository_tree"]
    assert evaluated["coverage"]["verified"] is True
    assert evaluated["analyzer_health"]["healthy"] is True
    assert evaluated["analyzer_health"][
        "safe_for_completion_reasoning"
    ] is True
    assert evaluated["exhaustion_quorum"]["member_count"] == 2

    acceptance = packet["operational_evidence"][2]  # type: ignore[index]
    assert isinstance(acceptance, ParallelAcceptanceReceipt)
    assert acceptance.verify_integrity()
    assert acceptance.proved_requirement_ids_for(
        str(packet["repository_tree"])
    ) == (PARALLEL_ACCEPTANCE_EVIDENCE_ID,)
    assert acceptance.proved_requirement_ids_for("foreign-tree") == ()
    tampered = acceptance.to_dict()
    tampered["target_commit"] = "foreign-tree"
    with pytest.raises(ValueError, match="identity mismatch"):
        ParallelAcceptanceReceipt.from_dict(tampered)


def test_g060_completion_fails_closed_for_each_missing_gate(
    tmp_path: Path,
) -> None:
    packet = _g060_completion_packet(tmp_path)

    def rejected(**overrides: object) -> None:
        candidate = {**packet, **overrides}
        decision = evaluate_parallel_execution_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **candidate,
        )
        assert decision.verified is False
        assert decision.state is not GoalState.VERIFIED_COMPLETE
        assert decision.reason_codes

    rejected(
        producing_tasks=list(packet["producing_tasks"])[:-1],  # type: ignore[arg-type]
    )
    for producer_failure in ("wrong", "duplicate", "nonterminal"):
        producers = copy.deepcopy(packet["producing_tasks"])
        if producer_failure == "wrong":
            producers[-1]["task_id"] = "ASI-083"
        elif producer_failure == "duplicate":
            producers[-1]["task_id"] = producers[0]["task_id"]
        else:
            producers[-1]["status"] = "active"
        rejected(producing_tasks=producers)

    rejected(
        operational_evidence=tuple(packet["operational_evidence"])[:-1],  # type: ignore[arg-type]
    )
    forged_operational = list(packet["operational_evidence"])  # type: ignore[arg-type]
    forged_operational[2] = replace(
        forged_operational[2],
        _producer_seal=None,
    )
    rejected(operational_evidence=tuple(forged_operational))
    rejected(resource_policy={"adaptive_enabled": "invalid"})

    submitted = list(packet["evidence"])  # type: ignore[arg-type]
    failed_evidence = list(submitted)
    failed_evidence[0] = replace(
        failed_evidence[0],
        validation_passed=False,
        validation_receipt={
            "status": "failed",
            "tree_id": packet["repository_tree"],
        },
    )
    rejected(evidence=tuple(failed_evidence))
    rejected(evidence=tuple(submitted[:-1]))
    duplicate_evidence = list(submitted)
    duplicate_evidence[-1] = replace(
        duplicate_evidence[-1],
        acceptance_criterion=duplicate_evidence[0].acceptance_criterion,
        provenance_cid=duplicate_evidence[0].provenance_cid,
    )
    rejected(evidence=tuple(duplicate_evidence))
    stale_evidence = list(submitted)
    stale_evidence[0] = replace(
        stale_evidence[0],
        freshness={"fresh": False},
        observed_at=packet["now"] - timedelta(hours=2),  # type: ignore[operator]
    )
    rejected(evidence=tuple(stale_evidence))
    foreign_evidence = list(submitted)
    foreign_evidence[0] = replace(
        foreign_evidence[0],
        repository_id="repository:foreign",
        repository_tree="foreign-tree",
    )
    rejected(evidence=tuple(foreign_evidence))
    rejected(
        evidence=(
            *submitted,
            replace(
                submitted[0],
                acceptance_criterion="caller-selected extra criterion",
                provenance_cid="validation:extra",
            ),
        )
    )

    coverage = copy.deepcopy(packet["coverage"])
    coverage["criteria"][0]["validation_receipt_ids"] = [  # type: ignore[index]
        "validation:foreign"
    ]
    rejected(coverage=coverage)
    missing_coverage = copy.deepcopy(packet["coverage"])
    missing_coverage["criteria"].pop()  # type: ignore[union-attr]
    rejected(coverage=missing_coverage)
    duplicate_coverage = copy.deepcopy(packet["coverage"])
    duplicate_coverage["criteria"][-1] = copy.deepcopy(  # type: ignore[index]
        duplicate_coverage["criteria"][0]  # type: ignore[index]
    )
    rejected(coverage=duplicate_coverage)
    unimplemented_coverage = copy.deepcopy(packet["coverage"])
    unimplemented_coverage["criteria"][0]["implementation"] = ""  # type: ignore[index]
    rejected(coverage=unimplemented_coverage)

    analyzer = copy.deepcopy(packet["analyzer_health"])
    analyzer["safe_for_completion_reasoning"] = False
    rejected(analyzer_health=analyzer)
    rejected(analyzer_health={})
    unhealthy_analyzer = copy.deepcopy(packet["analyzer_health"])
    unhealthy_analyzer.update({"status": "degraded", "healthy": False})
    rejected(analyzer_health=unhealthy_analyzer)
    foreign_analyzer = copy.deepcopy(packet["analyzer_health"])
    foreign_analyzer["binding"]["configuration_revision"] = "foreign"  # type: ignore[index]
    rejected(analyzer_health=foreign_analyzer)

    quorum = copy.deepcopy(packet["exhaustion_quorum"])
    quorum["members"][1]["evidence_channel"] = (  # type: ignore[index]
        quorum["members"][0]["evidence_channel"]  # type: ignore[index]
    )
    rejected(exhaustion_quorum=quorum)

    stale_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    stale_quorum["members"][1]["finished_at"] = (  # type: ignore[index]
        packet["now"] - timedelta(hours=2)  # type: ignore[operator]
    ).isoformat()
    rejected(exhaustion_quorum=stale_quorum)
    insufficient_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    insufficient_quorum["members"].pop()  # type: ignore[union-attr]
    insufficient_quorum["member_count"] = 1
    rejected(exhaustion_quorum=insufficient_quorum)
    duplicate_member_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    duplicate_member_quorum["members"][1]["member_id"] = (  # type: ignore[index]
        duplicate_member_quorum["members"][0]["member_id"]  # type: ignore[index]
    )
    rejected(exhaustion_quorum=duplicate_member_quorum)
    duplicate_receipt_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    duplicate_receipt_quorum["members"][1]["receipt_cid"] = (  # type: ignore[index]
        duplicate_receipt_quorum["members"][0]["receipt_cid"]  # type: ignore[index]
    )
    rejected(exhaustion_quorum=duplicate_receipt_quorum)
    unhealthy_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    unhealthy_quorum["members"][1]["healthy"] = False  # type: ignore[index]
    rejected(exhaustion_quorum=unhealthy_quorum)
    unsafe_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    unsafe_quorum["members"][1][  # type: ignore[index]
        "safe_for_completion_reasoning"
    ] = False
    rejected(exhaustion_quorum=unsafe_quorum)
    partial_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    partial_quorum["members"][1]["scan_mode"] = "partial"  # type: ignore[index]
    rejected(exhaustion_quorum=partial_quorum)
    foreign_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    foreign_quorum["binding"]["tree_id"] = "foreign-tree"  # type: ignore[index]
    rejected(exhaustion_quorum=foreign_quorum)
    foreign_member_quorum = copy.deepcopy(packet["exhaustion_quorum"])
    foreign_member_quorum["members"][1]["binding"][  # type: ignore[index]
        "tree_id"
    ] = "foreign-tree"
    rejected(exhaustion_quorum=foreign_member_quorum)

    with pytest.raises(
        ValueError, match="must equal the configured ASI-G060 count"
    ):
        evaluate_parallel_execution_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            required_exhaustive_receipts=1,
            **{
                key: value
                for key, value in packet.items()
                if key != "required_exhaustive_receipts"
            },
        )
