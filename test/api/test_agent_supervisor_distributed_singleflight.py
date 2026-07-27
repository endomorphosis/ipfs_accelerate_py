from __future__ import annotations

import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.cache_coordinator import (
    CacheAuthority,
    CacheNamespace,
    NamespaceCacheCoordinator,
    build_namespace_semantic_key,
    build_semantic_cache_key,
)
from ipfs_accelerate_py.agent_supervisor.lease_coordination import (
    DistributedSingleFlightCancelled,
    DistributedSingleFlightCoordinator,
    DistributedSingleFlightExecutionError,
    DistributedSingleFlightTimeout,
    StaleSingleFlightLeaseError,
)


_COLLAPSED_NAMESPACES = (
    CacheNamespace.ANALYSIS,
    CacheNamespace.CONTEXT,
    CacheNamespace.PLANNING,
    CacheNamespace.PROVIDER,
    CacheNamespace.PROOF,
    CacheNamespace.VALIDATION,
    CacheNamespace.MERGE,
)


def _key(namespace: CacheNamespace, ordinal: int = 1):
    return build_semantic_cache_key(
        namespace,
        {
            "fixture": "distributed-single-flight",
            "semantic_revision": ordinal,
        },
    )


def _process_cache_worker(
    cache_path: str,
    barrier: multiprocessing.synchronize.Barrier,
    invocations: multiprocessing.sharedctypes.Synchronized,
    output: multiprocessing.queues.Queue,
) -> None:
    coordinator = NamespaceCacheCoordinator(cache_path, lease_seconds=1)
    barrier.wait(10)

    def produce() -> dict[str, object]:
        with invocations.get_lock():
            invocations.value += 1
        time.sleep(0.2)
        return {"receipt_id": "process-shared"}

    try:
        result = coordinator.get_or_compute(
            _key(CacheNamespace.ANALYSIS),
            produce,
            authority=CacheAuthority.AUTHORITATIVE,
            require_completion_evidence=True,
        )
        output.put(
            {
                "value": result.value,
                "completion": result.is_completion_evidence,
                "produced": result.produced,
                "shared": result.shared,
                "fencing_token": result.fencing_token,
                "attestation_id": (
                    result.attestation.attestation_id
                    if result.attestation is not None
                    else ""
                ),
            }
        )
    except BaseException as exc:  # pragma: no cover - parent reports child error
        output.put({"error": f"{type(exc).__name__}: {exc}"})


def test_all_expensive_namespaces_collapse_paired_thread_misses(
    tmp_path: Path,
) -> None:
    coordinator = NamespaceCacheCoordinator(tmp_path, lease_seconds=1)
    members_per_key = 10
    barrier = threading.Barrier(
        len(_COLLAPSED_NAMESPACES) * members_per_key
    )
    calls = {namespace: 0 for namespace in _COLLAPSED_NAMESPACES}
    lock = threading.Lock()

    def run(namespace: CacheNamespace):
        barrier.wait(10)

        def produce() -> dict[str, str]:
            with lock:
                calls[namespace] += 1
            # Keep the miss cohort open long enough for every thread to join.
            time.sleep(0.1)
            return {"namespace": namespace.value, "receipt_id": "shared"}

        return coordinator.get_or_compute(
            _key(namespace),
            produce,
            authority=CacheAuthority.AUTHORITATIVE,
            require_completion_evidence=True,
        )

    work = [
        namespace
        for namespace in _COLLAPSED_NAMESPACES
        for _ in range(members_per_key)
    ]
    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        results = list(pool.map(run, work))

    request_count = len(work)
    producer_count = sum(calls.values())
    ideal_count = len(_COLLAPSED_NAMESPACES)
    collapse_ratio = (request_count - producer_count) / request_count
    duplicate_compute_ratio = (
        (producer_count - ideal_count) / (request_count - ideal_count)
    )

    assert calls == {
        namespace: 1 for namespace in _COLLAPSED_NAMESPACES
    }
    assert collapse_ratio >= 0.60
    assert duplicate_compute_ratio < 0.05
    assert sum(result.produced for result in results) == ideal_count
    assert all(result.is_completion_evidence for result in results)
    assert coordinator.metrics().stale_authoritative_hits == 0

    for namespace in _COLLAPSED_NAMESPACES:
        cohort = [
            result
            for requested, result in zip(work, results)
            if requested is namespace and result.attestation is not None
        ]
        assert cohort
        assert len(
            {result.attestation.attestation_id for result in cohort}
        ) == 1
        assert len({result.fencing_token for result in cohort}) == 1
        assert all(result.attested for result in cohort)


def test_provider_namespace_has_a_complete_semantic_key_contract() -> None:
    key = build_namespace_semantic_key(
        CacheNamespace.PROVIDER,
        operation="infer",
        request_digest="sha256:request",
        provider_id="provider:test",
        provider_version="provider@1",
        capability_revision="capability@1",
        protocol_version="protocol@1",
        configuration_digest="sha256:configuration",
        policy_digest="sha256:policy",
        resource_budget_digest="sha256:budget",
    )
    assert key.namespace is CacheNamespace.PROVIDER
    assert key.key_id.startswith("supervisor-cache:provider:sha256:")


def test_heartbeat_expiry_takeover_and_publication_are_strictly_fenced(
    tmp_path: Path,
) -> None:
    now = [1_000]
    coordinator = DistributedSingleFlightCoordinator(
        tmp_path / "flights.sqlite3",
        lease_seconds=0.1,
        clock_ms=lambda: now[0],
    )
    key = {"key_id": "semantic:analysis:heartbeat", "namespace": "analysis"}
    first = coordinator.acquire(key, owner_id="host-a")
    follower = coordinator.acquire(key, owner_id="host-b")
    assert first.acquired
    assert not follower.acquired
    assert follower.lease_id == ""
    assert follower.fencing_token == first.fencing_token

    now[0] = 1_050
    renewed = coordinator.heartbeat(first)
    assert renewed.expires_at_ms == 1_150
    now[0] = 1_151
    replacement = coordinator.acquire(key, owner_id="host-b")
    assert replacement.acquired
    assert replacement.fencing_token == first.fencing_token + 1

    with pytest.raises(StaleSingleFlightLeaseError):
        coordinator.publish(first, {"stale": True})
    with pytest.raises(StaleSingleFlightLeaseError):
        coordinator.publish(
            replace(replacement, lease_id="foreign-lease"),
            {"foreign": True},
        )

    outcome = coordinator.publish(replacement, {"receipt_id": "replacement"})
    assert outcome.fencing_token == replacement.fencing_token
    assert coordinator.verify_outcome(outcome)
    assert coordinator.release(replacement)
    observed = coordinator.coordinate(
        key,
        lambda: pytest.fail("completed outcome executed twice"),
    )
    assert observed.shared
    assert observed.outcome == outcome
    assert observed.attestation.attestation_id == outcome.attestation.attestation_id


def test_owner_error_is_one_bounded_fail_closed_follower_outcome(
    tmp_path: Path,
) -> None:
    coordinator = DistributedSingleFlightCoordinator(
        tmp_path / "failure.sqlite3",
        lease_seconds=1,
        max_outcome_bytes=2_048,
    )
    key = {"key_id": "semantic:proof:failure", "namespace": "proof"}
    entered = threading.Event()
    release = threading.Event()
    secret = "provider-secret-must-not-persist"
    owner_errors: list[BaseException] = []
    follower_errors: list[BaseException] = []

    def fail() -> None:
        entered.set()
        assert release.wait(5)
        raise RuntimeError(secret)

    owner = threading.Thread(
        target=lambda: _capture_error(
            owner_errors, lambda: coordinator.coordinate(key, fail)
        )
    )
    owner.start()
    assert entered.wait(5)
    follower = threading.Thread(
        target=lambda: _capture_error(
            follower_errors,
            lambda: coordinator.coordinate(
                key,
                lambda: pytest.fail("follower executed"),
            ),
        )
    )
    follower.start()
    time.sleep(0.05)
    release.set()
    owner.join(5)
    follower.join(5)

    assert len(owner_errors) == len(follower_errors) == 1
    assert isinstance(owner_errors[0], RuntimeError)
    assert isinstance(
        follower_errors[0], DistributedSingleFlightExecutionError
    )
    follower_error = follower_errors[0]
    assert isinstance(follower_error, DistributedSingleFlightExecutionError)
    assert follower_error.reason_code == "single_flight_execution_failed"
    assert follower_error.outcome is not None
    assert coordinator.verify_outcome(follower_error.outcome)
    assert (
        len(json.dumps(follower_error.outcome.to_dict()).encode("utf-8"))
        < coordinator.max_outcome_bytes
    )
    assert secret not in (tmp_path / "failure.sqlite3").read_bytes().decode(
        "utf-8", errors="ignore"
    )


def _capture_error(
    destination: list[BaseException],
    operation,
) -> None:
    try:
        operation()
    except BaseException as exc:
        destination.append(exc)


def test_follower_cancellation_and_deadline_do_not_cancel_owner(
    tmp_path: Path,
) -> None:
    coordinator = DistributedSingleFlightCoordinator(
        tmp_path / "members.sqlite3",
        lease_seconds=1,
    )
    key = {"key_id": "semantic:validation:members", "namespace": "validation"}
    entered = threading.Event()
    release = threading.Event()
    owner_results = []

    def execute():
        entered.set()
        assert release.wait(5)
        return {"receipt_id": "owner-completed"}

    owner = threading.Thread(
        target=lambda: owner_results.append(
            coordinator.coordinate(key, execute)
        )
    )
    owner.start()
    assert entered.wait(5)

    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(DistributedSingleFlightCancelled):
        coordinator.coordinate(
            key,
            lambda: pytest.fail("cancelled follower executed"),
            cancel_event=cancelled,
        )
    with pytest.raises(DistributedSingleFlightTimeout):
        coordinator.coordinate(
            key,
            lambda: pytest.fail("expired follower executed"),
            timeout_seconds=0.05,
        )

    release.set()
    owner.join(5)
    assert not owner.is_alive()
    assert len(owner_results) == 1
    shared = coordinator.coordinate(
        key,
        lambda: pytest.fail("late follower executed"),
    )
    assert shared.shared
    assert shared.value == {"receipt_id": "owner-completed"}
    assert shared.outcome == owner_results[0].outcome


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="cross-process single-flight fixture requires fork",
)
def test_processes_share_one_attested_bounded_outcome(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    process_count = 6
    barrier = context.Barrier(process_count)
    invocations = context.Value("i", 0)
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_cache_worker,
            args=(str(tmp_path), barrier, invocations, output),
        )
        for _ in range(process_count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)

    assert all(process.exitcode == 0 for process in processes)
    results = [output.get(timeout=2) for _ in processes]
    assert not [result for result in results if "error" in result]
    assert invocations.value == 1
    assert sum(result["produced"] for result in results) == 1
    assert all(result["completion"] for result in results)
    attested = {
        result["attestation_id"]
        for result in results
        if result["attestation_id"]
    }
    assert len(attested) == 1
    assert len(
        {
            result["fencing_token"]
            for result in results
            if result["fencing_token"]
        }
    ) == 1


def test_stale_authoritative_record_cannot_hide_behind_live_flight_outcome(
    tmp_path: Path,
) -> None:
    now = [1_000.0]
    coordinator = NamespaceCacheCoordinator(
        tmp_path,
        clock=lambda: now[0],
        outcome_ttl_seconds=30,
    )
    key = _key(CacheNamespace.MERGE)
    calls = 0

    def produce():
        nonlocal calls
        calls += 1
        return {"receipt_id": f"merge-{calls}"}

    first = coordinator.get_or_compute(
        key,
        produce,
        authority=CacheAuthority.AUTHORITATIVE,
        ttl_seconds=1,
        require_completion_evidence=True,
    )
    now[0] += 2
    refreshed = coordinator.get_or_compute(
        key,
        produce,
        authority=CacheAuthority.AUTHORITATIVE,
        ttl_seconds=1,
        require_completion_evidence=True,
    )

    assert calls == 2
    assert first.value == {"receipt_id": "merge-1"}
    assert refreshed.value == {"receipt_id": "merge-2"}
    assert refreshed.fencing_token == first.fencing_token + 1
    assert refreshed.is_completion_evidence
    assert coordinator.metrics().stale_authoritative_hits == 0
