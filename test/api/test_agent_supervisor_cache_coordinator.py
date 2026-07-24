from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_cache import (
    AnalysisCache,
    AnalysisCacheKey,
    AnalysisCacheReason,
)
from ipfs_accelerate_py.agent_supervisor.cache_coordinator import (
    AnalysisCacheCoordinator,
    CacheCoordinationStatus,
    CachePublication,
)


def _key(**changes: object) -> AnalysisCacheKey:
    values: dict[str, object] = {
        "repository_tree_identity": "tree:sha256:111",
        "objective_revision": "objective@1",
        "analyzer_version": "analyzer@1",
        "schema_version": "schema@1",
        "configuration_digest": "sha256:config-1",
        "query_digest": "sha256:query-1",
        "policy_digest": "sha256:policy-1",
    }
    values.update(changes)
    return AnalysisCacheKey(**values)


def _receipt(status: str = "successful", ordinal: int = 1):
    return {
        "status": status,
        "receipt_id": f"receipt-{ordinal}",
        "summary": {"ordinal": ordinal},
        "artifact_refs": [{"artifact_id": f"artifact-{ordinal}"}],
    }


def test_exact_cache_hit_avoids_producer_and_stale_key_does_not(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(tmp_path)
    cache.put(_key(), _receipt())
    coordinator = AnalysisCacheCoordinator(cache)
    calls = 0

    def producer():
        nonlocal calls
        calls += 1
        return _receipt(ordinal=2)

    exact = coordinator.get_or_compute(_key(), producer)
    changed = coordinator.get_or_compute(
        _key(repository_tree_identity="tree:sha256:222"), producer
    )

    assert exact.status is CacheCoordinationStatus.CACHE_HIT
    assert exact.is_completion_evidence
    assert changed.status is CacheCoordinationStatus.PRODUCED
    assert AnalysisCacheReason.EXACT_KEY_HIT.value in changed.reason_codes
    assert calls == 1


def test_threads_with_identical_key_execute_one_producer(
    tmp_path: Path,
) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    calls = 0

    def producer():
        nonlocal calls
        with lock:
            calls += 1
        entered.set()
        assert release.wait(5)
        return _receipt()

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(coordinator.get_or_compute, _key(), producer)
            for _ in range(16)
        ]
        assert entered.wait(5)
        release.set()
        results = [future.result(timeout=10) for future in futures]

    assert calls == 1
    assert sum(item.status is CacheCoordinationStatus.PRODUCED for item in results) == 1
    assert all(item.is_completion_evidence for item in results)
    metrics = coordinator.metrics()
    assert metrics.followers + metrics.cache_hits >= 15
    assert metrics.active_flights == 0


def test_unrelated_keys_are_not_globally_serialized(tmp_path: Path) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    barrier = threading.Barrier(2)

    def producer(ordinal: int):
        barrier.wait(timeout=5)
        return _receipt(ordinal=ordinal)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            coordinator.get_or_compute,
            _key(query_digest="sha256:q1"),
            lambda: producer(1),
        )
        second = executor.submit(
            coordinator.get_or_compute,
            _key(query_digest="sha256:q2"),
            lambda: producer(2),
        )
        assert first.result(timeout=10).produced
        assert second.result(timeout=10).produced


def test_failure_fans_out_cleans_flight_and_next_call_retries(
    tmp_path: Path,
) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    entered = threading.Event()
    all_begun = threading.Event()
    begin_lock = threading.Lock()
    begin_count = 0
    calls = 0
    original_begin = coordinator._begin

    def observed_begin(key):
        nonlocal begin_count
        result = original_begin(key)
        with begin_lock:
            begin_count += 1
            if begin_count == 4:
                all_begun.set()
        return result

    coordinator._begin = observed_begin

    def broken():
        nonlocal calls
        calls += 1
        entered.set()
        assert all_begun.wait(5)
        raise RuntimeError("fixture failure")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(coordinator.get_or_compute, _key(), broken)
            for _ in range(4)
        ]
        assert entered.wait(5)
        for future in futures:
            with pytest.raises(RuntimeError, match="fixture failure"):
                future.result(timeout=10)

    recovered = coordinator.get_or_compute(_key(), lambda: _receipt())
    assert recovered.is_completion_evidence
    assert calls == 1
    assert coordinator.metrics().active_flights == 0


def test_negative_cache_record_cannot_bypass_producer(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    cache.put(_key(), _receipt("partial"))
    coordinator = AnalysisCacheCoordinator(cache)
    calls = 0

    def producer():
        nonlocal calls
        calls += 1
        return _receipt()

    result = coordinator.get_or_compute(_key(), producer)

    assert calls == 1
    assert result.status is CacheCoordinationStatus.PRODUCED
    assert result.is_completion_evidence


def test_outer_artifact_validator_turns_compact_hit_into_keyed_miss(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(tmp_path)
    cache.put(_key(), _receipt())
    coordinator = AnalysisCacheCoordinator(cache)
    calls = 0

    def producer():
        nonlocal calls
        calls += 1
        return _receipt(ordinal=2)

    result = coordinator.get_or_compute(
        _key(),
        producer,
        completion_validator=lambda lookup: (
            lookup.receipt is not None
            and lookup.receipt.get("receipt_id") == "receipt-2"
        ),
    )

    assert result.status is CacheCoordinationStatus.PRODUCED
    assert result.is_completion_evidence
    assert result.receipt is not None
    assert result.receipt["receipt_id"] == "receipt-2"
    assert calls == 1
    assert coordinator.metrics().cache_validation_rejections == 2


def test_completion_validator_must_return_literal_boolean(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(tmp_path)
    cache.put(_key(), _receipt())
    coordinator = AnalysisCacheCoordinator(cache)

    with pytest.raises(
        RuntimeError, match="completion_validator must return a boolean"
    ):
        coordinator.get_or_compute(
            _key(),
            lambda: _receipt(ordinal=2),
            completion_validator=lambda lookup: "yes",
        )


def test_publication_inherits_or_overrides_call_ttl(tmp_path: Path) -> None:
    now = 1_000.0
    cache = AnalysisCache(tmp_path, clock=lambda: now)
    coordinator = AnalysisCacheCoordinator(cache)

    inherited = coordinator.get_or_compute(
        _key(query_digest="sha256:inherit"),
        lambda: CachePublication(_receipt()),
        ttl_seconds=7,
    )
    overridden = coordinator.get_or_compute(
        _key(query_digest="sha256:override"),
        lambda: CachePublication(_receipt(ordinal=2), ttl_seconds=11),
        ttl_seconds=7,
    )

    assert inherited.entry is not None
    assert overridden.entry is not None
    assert inherited.entry.expires_at_ms - inherited.entry.created_at_ms == 7_000
    assert overridden.entry.expires_at_ms - overridden.entry.created_at_ms == 11_000


def test_async_identical_misses_share_the_sync_safe_flight(
    tmp_path: Path,
) -> None:
    async def scenario():
        coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def producer():
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return _receipt()

        tasks = [
            asyncio.create_task(
                coordinator.async_get_or_compute(_key(), producer)
            )
            for _ in range(8)
        ]
        await asyncio.wait_for(entered.wait(), timeout=5)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
        return coordinator, calls, results

    coordinator, calls, results = asyncio.run(scenario())
    assert calls == 1
    assert all(result.is_completion_evidence for result in results)
    assert sum(result.shared for result in results) == 7
    assert coordinator.metrics().active_flights == 0


def test_sync_leader_and_async_followers_share_one_cross_facade_flight(
    tmp_path: Path,
) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def sync_producer():
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(5)
        return _receipt()

    async def join_from_async_facade():
        async def duplicate_producer():
            raise AssertionError("async follower unexpectedly became producer")

        tasks = [
            asyncio.create_task(
                coordinator.async_get_or_compute(_key(), duplicate_producer)
            )
            for _ in range(4)
        ]
        for _ in range(10):
            if coordinator.metrics().followers == 4:
                break
            await asyncio.sleep(0)
        assert coordinator.metrics().followers == 4
        release.set()
        return await asyncio.gather(*tasks)

    with ThreadPoolExecutor(max_workers=1) as executor:
        leader = executor.submit(
            coordinator.get_or_compute, _key(), sync_producer
        )
        assert entered.wait(5)
        followers = asyncio.run(join_from_async_facade())
        produced = leader.result(timeout=10)

    assert calls == 1
    assert produced.status is CacheCoordinationStatus.PRODUCED
    assert all(item.status is CacheCoordinationStatus.SHARED for item in followers)
    assert all(item.is_completion_evidence for item in followers)
    assert coordinator.metrics().active_flights == 0
