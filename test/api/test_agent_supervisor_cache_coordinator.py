from __future__ import annotations

import asyncio
import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_cache import (
    AnalysisCache,
    AnalysisCacheKey,
    AnalysisCacheReason,
)
from ipfs_accelerate_py.agent_supervisor.analysis.cache_coordinator import (
    INTEGRATED_ANALYSIS_CACHE_ACCEPTANCE_CRITERIA,
    SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID,
    AnalysisCacheCoordinator,
    BoundedArtifactReference,
    CacheAuthority,
    CacheCoordinationError,
    CacheCoordinationStatus,
    CacheNamespace,
    CachePublication,
    CacheQuotaPolicy,
    CacheRecordOutcome,
    NamespaceCacheCASAdapter,
    NamespaceCacheCoordinator,
    NamespaceLookupStatus,
    SingleFlightCollapseEvidence,
    build_namespace_semantic_key,
    namespace_metadata,
)
from ipfs_accelerate_py.agent_supervisor.runtime.runtime_cas import RuntimeCAS
from ipfs_accelerate_py.agent_supervisor.self_improvement.self_improvement.supervisor_v2_contracts import (
    ResultBinding,
    SemanticDependencyIdentity,
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


def test_g020_cache_criteria_keep_runtime_authority_separate_from_completion() -> None:
    assert INTEGRATED_ANALYSIS_CACHE_ACCEPTANCE_CRITERIA == (
        "expensive identical misses collapse across lanes",
        "stale or negative records never become completion evidence",
        (
            "repeated fixtures achieve at least 70 percent cache reuse with "
            "zero stale authoritative hits."
        ),
    )
    # A cache key is caller-selected runtime identity. It carries no
    # criterion, analyzer-health, validation-freshness, or quorum authority.
    key_payload = _key().to_dict()
    assert "acceptance_criterion" not in key_payload
    assert "analyzer_health" not in key_payload
    assert "exhaustion_quorum" not in key_payload


def _common_analysis_key(**changes: object):
    dimensions: dict[str, object] = {
        "repository_tree_identity": "tree:sha256:111",
        "objective_revision": "objective@1",
        "analyzer_version": "analyzer@1",
        "schema_version": "schema@1",
        "configuration_digest": "sha256:config-1",
        "query_digest": "sha256:query-1",
        "policy_digest": "sha256:policy-1",
    }
    dimensions.update(changes)
    return build_namespace_semantic_key(CacheNamespace.ANALYSIS, dimensions)


def _runtime_binding() -> ResultBinding:
    dependency = SemanticDependencyIdentity(
        namespace="legacy-cache",
        key="analysis-key",
        revision="legacy@1",
        digest="sha256:" + "1" * 64,
    )
    return ResultBinding(
        repository_id="repository:test",
        tree_id="tree:sha256:111",
        objective_id="objective:test",
        objective_revision="objective@1",
        task_id="task:test",
        task_revision="task@1",
        policy_id="policy:test",
        policy_revision="policy@1",
        producer_id="producer:test",
        producer_revision="producer@1",
        capability_id="capability:test",
        capability_revision="capability@1",
        environment_id="environment:test",
        environment_revision="environment@1",
        semantic_dependencies=(dependency,),
    )


def _cross_process_common_cache_worker(
    cache_path: str,
    marker_path: str,
    ready: multiprocessing.synchronize.Barrier,
    output: multiprocessing.queues.Queue,
) -> None:
    coordinator = NamespaceCacheCoordinator(cache_path)
    # Do not let process-spawn latency turn an intended concurrent miss into
    # a post-publication cache hit.  Every worker must be ready to perform its
    # first lookup before any worker may acquire the cross-process lease.
    ready.wait(10)

    def produce():
        with open(marker_path, "a", encoding="utf-8") as stream:
            stream.write("produced\n")
            stream.flush()
        time.sleep(0.15)
        return {"receipt_id": "shared"}

    result = coordinator.get_or_compute(
        _common_analysis_key(),
        produce,
        authority=CacheAuthority.AUTHORITATIVE,
        require_completion_evidence=True,
        payload_validator=lambda value: (
            isinstance(value, dict) and value.get("receipt_id") == "shared"
        ),
    )
    output.put(
        {
            "completion": result.is_completion_evidence,
            "produced": result.produced,
            "shared": result.shared,
        }
    )


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
        deadline = time.monotonic() + 5
        while (
            coordinator.metrics().followers < 15
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert coordinator.metrics().followers == 15
        release.set()
        results = [future.result(timeout=10) for future in futures]

    assert calls == 1
    assert sum(item.status is CacheCoordinationStatus.PRODUCED for item in results) == 1
    assert sum(item.status is CacheCoordinationStatus.SHARED for item in results) == 15
    assert all(item.is_completion_evidence for item in results)
    evidence = {
        item.single_flight_collapse_evidence.evidence_id
        for item in results
        if item.single_flight_collapse_evidence is not None
    }
    assert len(evidence) == 1
    assert all(
        item.proved_requirement_ids_for(_key())
        == (SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID,)
        for item in results
    )
    witness = results[0].single_flight_collapse_evidence
    assert witness is not None
    assert witness.producer_invocation_count == 1
    assert witness.follower_count == 15
    assert witness.participant_count == 16
    assert (
        SingleFlightCollapseEvidence.from_dict(witness.to_dict()) == witness
    )
    metrics = coordinator.metrics()
    assert metrics.followers + metrics.cache_hits >= 15
    assert metrics.active_flights == 0


def test_single_flight_evidence_is_active_key_and_publication_bound(
    tmp_path: Path,
) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    entered = threading.Event()
    release = threading.Event()

    def producer():
        entered.set()
        assert release.wait(5)
        return _receipt()

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(coordinator.get_or_compute, _key(), producer)
        assert entered.wait(5)
        follower = executor.submit(
            coordinator.get_or_compute,
            _key(),
            lambda: pytest.fail("follower ran a second producer"),
        )
        deadline = time.monotonic() + 5
        while (
            coordinator.metrics().followers < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert coordinator.metrics().followers == 1
        release.set()
        results = [leader.result(timeout=10), follower.result(timeout=10)]

    witness = results[0].single_flight_collapse_evidence
    assert witness is not None
    assert all(witness.proves_for(_key(), result) for result in results)
    assert not witness.proves_for(
        _key(query_digest="sha256:other"), results[0]
    )
    assert results[0].proved_requirement_ids == ()

    unattested = replace(
        results[0],
        single_flight_collapse_evidence=None,
        _single_flight_attestation=None,
    )
    assert not witness.proves_for(_key(), unattested)
    with pytest.raises(CacheCoordinationError, match="coordinator-attested"):
        SingleFlightCollapseEvidence.from_result(
            unattested,
            follower_count=1,
        )
    attestation = results[0]._single_flight_attestation
    assert attestation is not None
    forged_attestation = replace(attestation, seal=object())
    with pytest.raises(CacheCoordinationError, match="coordinator-attested"):
        SingleFlightCollapseEvidence.from_result(
            unattested,
            follower_count=1,
            _attestation=forged_attestation,
        )

    forged = replace(
        witness,
        publication_entry_digest="sha256:" + ("0" * 64),
    )
    assert not forged.proves_for(_key(), results[0])
    with pytest.raises(CacheCoordinationError, match="detached"):
        replace(
            results[0],
            single_flight_collapse_evidence=forged,
        )

    malformed = witness.to_dict()
    malformed["participant_count"] = 99
    with pytest.raises(CacheCoordinationError):
        SingleFlightCollapseEvidence.from_dict(malformed)


def test_singleton_hit_and_non_authoritative_flight_cannot_claim_collapse(
    tmp_path: Path,
) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    singleton = coordinator.get_or_compute(_key(), lambda: _receipt())
    cached = coordinator.get_or_compute(
        _key(), lambda: pytest.fail("cache hit executed producer")
    )

    assert singleton.single_flight_collapse_evidence is None
    assert singleton.operational_evidence_claim_references == ()
    assert cached.status is CacheCoordinationStatus.CACHE_HIT
    assert cached.operational_evidence_claim_references == ()

    entered = threading.Event()
    release = threading.Event()

    def inconclusive():
        entered.set()
        assert release.wait(5)
        return CachePublication(_receipt("partial", 2), store=False)

    other_key = _key(query_digest="sha256:negative")
    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(
            coordinator.get_or_compute, other_key, inconclusive
        )
        assert entered.wait(5)
        follower = executor.submit(
            coordinator.get_or_compute,
            other_key,
            lambda: pytest.fail("negative follower ran producer"),
        )
        deadline = time.monotonic() + 5
        while (
            coordinator.metrics().followers < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert coordinator.metrics().followers == 1
        release.set()
        results = [leader.result(timeout=10), follower.result(timeout=10)]

    assert all(not item.is_completion_evidence for item in results)
    assert all(
        item.single_flight_collapse_evidence is None for item in results
    )
    assert all(
        item.proved_requirement_ids_for(other_key) == () for item in results
    )


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


def test_joined_caller_reapplies_its_own_completion_validator(
    tmp_path: Path,
) -> None:
    coordinator = AnalysisCacheCoordinator(AnalysisCache(tmp_path))
    entered = threading.Event()
    release = threading.Event()
    follower_called = False

    def leader_producer():
        entered.set()
        assert release.wait(5)
        return _receipt(ordinal=1)

    def follower_producer():
        nonlocal follower_called
        follower_called = True
        return _receipt(ordinal=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        leader = pool.submit(
            coordinator.get_or_compute,
            _key(),
            leader_producer,
            completion_validator=lambda lookup: (
                lookup.receipt is not None
                and lookup.receipt.get("receipt_id") == "receipt-1"
            ),
        )
        assert entered.wait(5)
        follower = pool.submit(
            coordinator.get_or_compute,
            _key(),
            follower_producer,
            completion_validator=lambda lookup: (
                lookup.receipt is not None
                and lookup.receipt.get("receipt_id") == "receipt-2"
            ),
        )
        deadline = time.monotonic() + 5
        while (
            coordinator.metrics().followers < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        release.set()

        produced = leader.result(timeout=5)
        with pytest.raises(
            CacheCoordinationError,
            match="shared completion result rejected",
        ):
            follower.result(timeout=5)

    assert produced.status is CacheCoordinationStatus.PRODUCED
    assert produced.is_completion_evidence
    assert not follower_called
    assert coordinator.metrics().cache_validation_rejections == 1
    assert coordinator.metrics().active_flights == 0


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
    assert all(
        result.proved_requirement_ids_for(_key())
        == (SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID,)
        for result in results
    )
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
    all_results = (produced, *followers)
    assert len(
        {
            item.single_flight_collapse_evidence.evidence_id
            for item in all_results
            if item.single_flight_collapse_evidence is not None
        }
    ) == 1
    assert all(
        item.proved_requirement_ids_for(_key())
        == (SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID,)
        for item in all_results
    )
    assert coordinator.metrics().active_flights == 0


def test_common_namespace_metadata_and_every_analysis_dimension_are_bound() -> None:
    metadata = namespace_metadata(
        CacheNamespace.ANALYSIS,
        authority=CacheAuthority.AUTHORITATIVE,
    )
    assert metadata.to_dict()["namespace"] == "analysis"
    assert metadata.to_dict()["authority"] == "authoritative"
    assert set(metadata.required_dimensions) == {
        "repository_tree_identity",
        "objective_revision",
        "analyzer_version",
        "schema_version",
        "configuration_digest",
        "query_digest",
        "policy_digest",
    }

    base = _common_analysis_key()
    variants = (
        _common_analysis_key(repository_tree_identity="tree:sha256:222"),
        _common_analysis_key(objective_revision="objective@2"),
        _common_analysis_key(analyzer_version="analyzer@2"),
        _common_analysis_key(schema_version="schema@2"),
        _common_analysis_key(configuration_digest="sha256:config-2"),
        _common_analysis_key(query_digest="sha256:query-2"),
        _common_analysis_key(policy_digest="sha256:policy-2"),
    )
    assert len({base.key_id, *(item.key_id for item in variants)}) == 8
    with pytest.raises(ValueError, match="missing dimensions"):
        build_namespace_semantic_key(
            CacheNamespace.ANALYSIS,
            {"repository_tree_identity": "tree-only"},
        )


def test_namespace_cache_adapter_reuses_exact_authoritative_native_entry(
    tmp_path: Path,
) -> None:
    native = NamespaceCacheCoordinator(tmp_path / "native")
    key = _common_analysis_key()
    assert native.put(
        key,
        {"receipt_id": "legacy-exact"},
        authority=CacheAuthority.AUTHORITATIVE,
    )
    runtime = RuntimeCAS(
        tmp_path / "runtime", current_tree_id="tree:sha256:111"
    )
    adapter = NamespaceCacheCASAdapter(
        native,
        runtime,
        producer_version="producer@1",
        policy_version="policy@1",
        capability_version="capability@1",
    )

    imported = adapter.import_entry(key, binding=_runtime_binding())

    assert imported is not None
    assert imported.payload["payload"] == {"receipt_id": "legacy-exact"}
    assert imported.identity.namespace == CacheNamespace.ANALYSIS.value
    projected = runtime.get_projection(
        key.key_id,
        namespace=CacheNamespace.ANALYSIS.value,
    )
    assert projected is not None
    assert projected.artifact_id == imported.artifact_id
    assert runtime.get(imported.key).artifact_id == imported.artifact_id

    incompatible = replace(
        _runtime_binding(), producer_revision="producer@2"
    )
    with pytest.raises(ValueError, match="producer_revision"):
        adapter.import_entry(key, binding=incompatible)


def test_proof_drafts_have_a_separate_non_authoritative_namespace(
    tmp_path: Path,
) -> None:
    draft_key = build_namespace_semantic_key(
        CacheNamespace.PROOF_DRAFT,
        goal_digest="goal",
        repository_tree_digest="tree",
        vocabulary_digest="vocabulary",
        compiler_digest="compiler",
        model_route_digest="route",
        model_version="model",
        assumptions_digest="assumptions",
        bounds_digest="bounds",
        policy_digest="policy",
    )
    proof_key = build_namespace_semantic_key(
        CacheNamespace.PROOF,
        obligation="obligation",
        premises=[],
        translator="translator",
        solver="solver",
        kernel="kernel",
        toolchain="toolchain",
        theorem_registry="registry",
        policy="policy",
        resource_budget="budget",
        candidate_tree="tree",
    )
    assert draft_key.namespace is CacheNamespace.PROOF_DRAFT
    assert proof_key.namespace is CacheNamespace.PROOF
    assert draft_key.key_id != proof_key.key_id

    coordinator = NamespaceCacheCoordinator(tmp_path)
    with pytest.raises(ValueError, match="proof drafts"):
        coordinator.put(
            draft_key,
            {"candidate": "untrusted"},
            authority=CacheAuthority.AUTHORITATIVE,
        )
    draft = coordinator.put(
        draft_key,
        {"candidate": "untrusted"},
        authority=CacheAuthority.DRAFT,
        outcome=CacheRecordOutcome.INCONCLUSIVE,
    )
    assert draft is not None
    assert draft.expires_at_ms is not None
    assert not draft.is_completion_evidence


def test_common_exact_reuse_negative_ttl_and_zero_stale_authority(
    tmp_path: Path,
) -> None:
    now = 1_000.0
    coordinator = NamespaceCacheCoordinator(
        tmp_path, clock=lambda: now
    )
    key = _common_analysis_key()
    calls = 0

    def successful():
        nonlocal calls
        calls += 1
        return {"receipt_id": "authoritative"}

    first = coordinator.get_or_compute(
        key,
        successful,
        authority=CacheAuthority.AUTHORITATIVE,
        ttl_seconds=2,
        require_completion_evidence=True,
    )
    reused = coordinator.get_or_compute(
        key,
        lambda: pytest.fail("exact completion hit executed producer"),
        authority=CacheAuthority.AUTHORITATIVE,
        require_completion_evidence=True,
    )
    assert first.produced and first.is_completion_evidence
    assert reused.cache_hit and reused.is_completion_evidence
    assert calls == 1

    negative_key = _common_analysis_key(query_digest="sha256:negative")
    negative = coordinator.put(
        negative_key,
        {"reason": "provider unavailable"},
        outcome=CacheRecordOutcome.INCONCLUSIVE,
        authority=CacheAuthority.AUTHORITATIVE,
    )
    assert negative is not None
    assert negative.expires_at_ms is not None
    assert not negative.is_completion_evidence
    rejected = coordinator.lookup(
        negative_key, require_completion_evidence=True
    )
    assert rejected.status is NamespaceLookupStatus.REJECTED
    assert not rejected.is_completion_evidence

    now += 3
    stale = coordinator.lookup(key, require_completion_evidence=True)
    assert stale.status is NamespaceLookupStatus.REJECTED
    assert not stale.is_completion_evidence
    metrics = coordinator.metrics()
    assert metrics.stale_rejections == 1


def test_common_corruption_and_poison_are_rejected_then_repaired(
    tmp_path: Path,
) -> None:
    coordinator = NamespaceCacheCoordinator(tmp_path)
    key = _common_analysis_key()
    assert coordinator.put(
        key,
        {"receipt_id": "one"},
        authority=CacheAuthority.AUTHORITATIVE,
    )
    path = coordinator._entry_path(key)
    path.write_text("{broken", encoding="utf-8")
    corrupt = coordinator.lookup(key, require_completion_evidence=True)
    assert corrupt.status is NamespaceLookupStatus.REJECTED
    assert not path.exists()

    repaired = coordinator.get_or_compute(
        key,
        lambda: {"receipt_id": "two"},
        authority=CacheAuthority.AUTHORITATIVE,
        require_completion_evidence=True,
    )
    assert repaired.is_completion_evidence

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authority"] = "draft"
    path.write_text(json.dumps(payload), encoding="utf-8")
    poisoned = coordinator.lookup(key, require_completion_evidence=True)
    assert poisoned.status is NamespaceLookupStatus.REJECTED
    assert coordinator.metrics().poisoned_rejections >= 2


def test_common_quota_gc_and_artifact_reference_bounds(
    tmp_path: Path,
) -> None:
    quota = CacheQuotaPolicy(
        max_entries=2,
        max_bytes=16 * 1024,
        max_entry_bytes=8 * 1024,
        max_artifact_references=1,
        max_artifact_reference_bytes=256,
    )
    coordinator = NamespaceCacheCoordinator(tmp_path, quotas=quota)
    with pytest.raises(ValueError, match="cannot embed"):
        BoundedArtifactReference(
            {"artifact_id": "unsafe", "content": "embedded body"}
        )

    for ordinal in range(3):
        entry = coordinator.put(
            _common_analysis_key(query_digest=f"sha256:q-{ordinal}"),
            {"receipt_id": f"r-{ordinal}"},
            authority=(
                CacheAuthority.DIAGNOSTIC
                if ordinal == 0
                else CacheAuthority.AUTHORITATIVE
            ),
            artifact_references=(
                {"artifact_id": f"a-{ordinal}", "digest": f"sha256:{ordinal}"},
            ),
        )
        assert entry is not None

    stats = coordinator.metrics(CacheNamespace.ANALYSIS)
    assert stats.entries == 2
    assert stats.evictions >= 1
    assert coordinator.lookup(
        _common_analysis_key(query_digest="sha256:q-0")
    ).status is NamespaceLookupStatus.MISS

    rejected = coordinator.put(
        _common_analysis_key(query_digest="sha256:too-many-refs"),
        {"receipt_id": "bounded"},
        artifact_references=(
            {"artifact_id": "a"},
            {"artifact_id": "b"},
        ),
    )
    assert rejected is None


def test_common_cross_process_single_flight_collapses_one_miss(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Barrier(3)
    output = context.Queue()
    marker = tmp_path / "producer-markers.txt"
    processes = [
        context.Process(
            target=_cross_process_common_cache_worker,
            args=(str(tmp_path / "cache"), str(marker), ready, output),
        )
        for _ in range(3)
    ]
    for process in processes:
        process.start()
    results = [output.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    assert marker.read_text(encoding="utf-8").splitlines() == ["produced"]
    assert all(result["completion"] for result in results)
    assert sum(result["produced"] for result in results) == 1
    assert sum(result["shared"] for result in results) == 2
