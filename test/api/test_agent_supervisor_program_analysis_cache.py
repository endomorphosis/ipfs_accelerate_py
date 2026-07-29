"""Tests for the dependency-aware program-analysis cache (VFS-011)."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.analysis_cache import (
    AnalysisOutcome,
)
from ipfs_accelerate_py.agent_supervisor.analysis.cache_coordinator import (
    CacheCoordinationStatus,
)
from ipfs_accelerate_py.agent_supervisor.program_analysis_cache import (
    CACHE_INVALIDATION_PROOF_EVIDENCE,
    DEPENDENCY_CACHE_EVIDENCE,
    ProgramAnalysisAuthority,
    ProgramAnalysisCache,
    ProgramAnalysisCacheKey,
    ProgramAnalysisCacheReason,
    ProgramAnalysisComponentKind,
    ProgramAnalysisLookupStatus,
    build_program_analysis_cache_key,
    compact_program_analysis_receipt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.runtime_cas import (
    RuntimeAuthority,
)


def _key(**changes: object) -> ProgramAnalysisCacheKey:
    values: dict[str, object] = {
        "forest_identity": "forest:sha256:111",
        "objective_revision": "objective-revision-1",
        "policy_revision": "policy-revision-1",
        "analyzer_version": "analyzer@1",
        "schema_version": "schema@1",
        "configuration_digest": "sha256:config-1",
        "query_digest": "sha256:query-1",
        "capability_revision": "capability@1",
        "assumption_digest": "sha256:assumption-1",
        "toolchain_version": "toolchain@1",
        "component_kind": ProgramAnalysisComponentKind.INVENTORY,
        "authority": ProgramAnalysisAuthority.AUTHORITATIVE,
    }
    values.update(changes)
    return ProgramAnalysisCacheKey(**values)


def _receipt(
    status: AnalysisOutcome | str = AnalysisOutcome.SUCCESSFUL,
    *,
    ordinal: int = 1,
    component: str = "inventory",
) -> dict[str, object]:
    return {
        "status": status.value if isinstance(status, AnalysisOutcome) else status,
        "receipt_id": f"{component}-receipt-{ordinal}",
        "summary": {"files_considered": ordinal, "component": component},
        "counts": {"files": ordinal, "findings": max(0, ordinal - 1)},
        "artifact_refs": [
            {
                "artifact_id": f"{component}-{ordinal}",
                "cid": f"bafy-{component}-{ordinal}",
                "digest": f"sha256:{component}{ordinal:0>58}",
            }
        ],
    }


def test_exact_key_hit_is_deterministic_and_survives_restart(tmp_path: Path) -> None:
    now = [1_000.0]
    cache = ProgramAnalysisCache(tmp_path, clock=lambda: now[0])
    miss = cache.lookup(_key())
    assert miss.status is ProgramAnalysisLookupStatus.MISS
    assert miss.reason_codes == (ProgramAnalysisCacheReason.CACHE_MISS.value,)

    stored = cache.put(_key(), _receipt())
    assert stored.stored
    assert stored.entry is not None
    assert stored.runtime_artifact is not None
    assert stored.entry.receipt["program_key"]["component_kind"] == "inventory"
    assert stored.entry.receipt["runtime_artifact_id"] == (
        stored.runtime_artifact.artifact_id
    )

    path_bytes = cache.entry_path(_key()).read_bytes()
    assert b"entry_digest" in path_bytes or b"program_key" in path_bytes
    assert json.loads(path_bytes)["receipt"]["program_key"]["schema"]

    reopened = cache.reopen()
    hit = reopened.lookup(_key())
    assert hit.status is ProgramAnalysisLookupStatus.HIT
    assert hit.reason_codes == (ProgramAnalysisCacheReason.EXACT_KEY_HIT.value,)
    assert hit.is_completion_evidence
    assert hit.runtime_artifact is not None
    assert hit.receipt is not None
    assert hit.receipt["receipt_id"] == "inventory-receipt-1"


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        (
            "forest_identity",
            "forest:sha256:222",
            ProgramAnalysisCacheReason.FOREST_IDENTITY_CHANGED,
        ),
        (
            "objective_revision",
            "objective-revision-2",
            ProgramAnalysisCacheReason.OBJECTIVE_REVISION_CHANGED,
        ),
        (
            "policy_revision",
            "policy-revision-2",
            ProgramAnalysisCacheReason.POLICY_REVISION_CHANGED,
        ),
        (
            "analyzer_version",
            "analyzer@2",
            ProgramAnalysisCacheReason.ANALYZER_VERSION_CHANGED,
        ),
        (
            "schema_version",
            "schema@2",
            ProgramAnalysisCacheReason.SCHEMA_VERSION_CHANGED,
        ),
        (
            "configuration_digest",
            "sha256:config-2",
            ProgramAnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED,
        ),
        (
            "query_digest",
            "sha256:query-2",
            ProgramAnalysisCacheReason.QUERY_DIGEST_CHANGED,
        ),
        (
            "capability_revision",
            "capability@2",
            ProgramAnalysisCacheReason.CAPABILITY_REVISION_CHANGED,
        ),
        (
            "assumption_digest",
            "sha256:assumption-2",
            ProgramAnalysisCacheReason.ASSUMPTION_DIGEST_CHANGED,
        ),
        (
            "toolchain_version",
            "toolchain@2",
            ProgramAnalysisCacheReason.TOOLCHAIN_VERSION_CHANGED,
        ),
        (
            "component_kind",
            ProgramAnalysisComponentKind.AST,
            ProgramAnalysisCacheReason.COMPONENT_KIND_CHANGED,
        ),
        (
            "authority",
            ProgramAnalysisAuthority.DRAFT,
            ProgramAnalysisCacheReason.AUTHORITY_CHANGED,
        ),
    ],
)
def test_every_dependency_dimension_has_an_explicit_invalidation_reason(
    tmp_path: Path,
    field: str,
    replacement: object,
    reason: ProgramAnalysisCacheReason,
) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    assert cache.put(_key(), _receipt())

    lookup = cache.lookup(_key(**{field: replacement}))
    assert lookup.status is ProgramAnalysisLookupStatus.INVALIDATED
    assert reason.value in lookup.reason_codes
    assert not lookup.hit
    assert lookup.receipt is None
    assert not lookup.is_completion_evidence


@pytest.mark.parametrize(
    "component",
    list(ProgramAnalysisComponentKind),
)
def test_all_stage_components_are_independently_keyed(
    tmp_path: Path, component: ProgramAnalysisComponentKind
) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    key = _key(component_kind=component)
    stored = cache.put(key, _receipt(component=component.value))
    assert stored.stored
    hit = cache.lookup(key)
    assert hit.hit
    assert hit.receipt is not None
    assert hit.receipt["component_kind"] == component.value

    # Unrelated sibling components remain independent misses until stored.
    for other in ProgramAnalysisComponentKind:
        if other is component:
            continue
        sibling = cache.lookup(_key(component_kind=other))
        assert not sibling.hit


def test_compact_receipts_reject_heavy_payloads_and_store_blob_refs(
    tmp_path: Path,
) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    rejected = cache.put(
        _key(),
        {**_receipt(), "source_text": "print('not cache material')"},
    )
    assert not rejected.stored
    assert ProgramAnalysisCacheReason.MALFORMED_RECEIPT.value in (
        rejected.reason_codes
    )

    body = b'{"ast":{"type":"Module","body":[]}}'
    stored = cache.put(
        _key(component_kind=ProgramAnalysisComponentKind.AST),
        _receipt(component="ast"),
        blob_bodies=(body,),
    )
    assert stored.stored
    assert len(stored.blob_refs) == 1
    assert stored.blob_refs[0].digest.startswith("sha256:")
    hit = cache.lookup(_key(component_kind=ProgramAnalysisComponentKind.AST))
    assert hit.hit
    assert hit.receipt is not None
    assert hit.receipt["blob_refs"][0]["digest"] == stored.blob_refs[0].digest


@pytest.mark.parametrize(
    "outcome",
    [
        AnalysisOutcome.PARTIAL,
        AnalysisOutcome.FAILED,
        AnalysisOutcome.TIMED_OUT,
        AnalysisOutcome.INCONCLUSIVE,
    ],
)
def test_negative_ttl_never_satisfies_completion(
    tmp_path: Path, outcome: AnalysisOutcome
) -> None:
    now = [2_000.0]
    cache = ProgramAnalysisCache(
        tmp_path,
        default_negative_ttl_seconds=5,
        max_negative_ttl_seconds=10,
        clock=lambda: now[0],
    )
    stored = cache.put(_key(), _receipt(outcome), ttl_seconds=1_000)
    assert stored.stored
    assert stored.entry is not None
    assert stored.entry.expires_at_ms == 2_010_000

    reusable = cache.lookup(_key(), require_completion_evidence=False)
    assert reusable.hit
    assert reusable.outcome is outcome
    assert not reusable.is_completion_evidence

    completion = cache.lookup(_key(), require_completion_evidence=True)
    assert completion.status is ProgramAnalysisLookupStatus.INVALIDATED
    assert (
        ProgramAnalysisCacheReason.NOT_COMPLETION_EVIDENCE.value
        in completion.reason_codes
    )

    now[0] += 11
    stale = cache.lookup(_key(), require_completion_evidence=False)
    assert stale.status is ProgramAnalysisLookupStatus.INVALIDATED
    assert (
        ProgramAnalysisCacheReason.STALE_NEGATIVE_ENTRY.value
        in stale.reason_codes
        or ProgramAnalysisCacheReason.RUNTIME_ARTIFACT_STALE.value
        in stale.reason_codes
        or ProgramAnalysisCacheReason.STALE_ENTRY.value in stale.reason_codes
    )
    assert not stale.is_completion_evidence


def test_authority_namespace_isolation_prevents_draft_upgrade(
    tmp_path: Path,
) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    draft_key = _key(authority=ProgramAnalysisAuthority.DRAFT)
    auth_key = _key(authority=ProgramAnalysisAuthority.AUTHORITATIVE)

    assert cache.put(draft_key, _receipt(ordinal=1))
    assert cache.put(auth_key, _receipt(ordinal=2))

    draft_hit = cache.lookup(draft_key)
    auth_hit = cache.lookup(auth_key)
    assert draft_hit.hit and auth_hit.hit
    assert draft_hit.runtime_artifact is not None
    assert auth_hit.runtime_artifact is not None
    assert (
        draft_hit.runtime_artifact.artifact_id
        != auth_hit.runtime_artifact.artifact_id
    )
    assert draft_hit.runtime_artifact.identity.authority is RuntimeAuthority.DRAFT
    assert (
        auth_hit.runtime_artifact.identity.authority
        is RuntimeAuthority.AUTHORITATIVE
    )
    assert not draft_hit.is_completion_evidence
    assert auth_hit.is_completion_evidence

    # Looking up the authoritative key never returns the draft entry.
    assert cache.lookup(auth_key).receipt is not None
    assert cache.lookup(auth_key).receipt["receipt_id"] == "inventory-receipt-2"


def test_transitive_invalidation_preserves_unrelated_components(
    tmp_path: Path,
) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    inventory_key = _key(component_kind=ProgramAnalysisComponentKind.INVENTORY)
    ast_key = _key(component_kind=ProgramAnalysisComponentKind.AST)
    graph_key = _key(component_kind=ProgramAnalysisComponentKind.GRAPH)
    unrelated_key = _key(
        component_kind=ProgramAnalysisComponentKind.INVENTORY,
        query_digest="sha256:unrelated-query",
    )

    inv = cache.put(inventory_key, _receipt(component="inventory"))
    assert inv.stored and inv.runtime_artifact is not None
    ast = cache.put_component(
        ast_key,
        _receipt(component="ast"),
        upstream=(inventory_key,),
    )
    assert ast.stored and ast.runtime_artifact is not None
    graph = cache.put_component(
        graph_key,
        _receipt(component="graph"),
        upstream=(inventory_key, ast_key),
    )
    assert graph.stored
    unrelated = cache.put(unrelated_key, _receipt(ordinal=9, component="inventory"))
    assert unrelated.stored

    # Invalidate AST only: graph (dependent) goes; inventory and unrelated stay.
    result = cache.invalidate_component(ast_key, include_root=True)
    assert ast.runtime_artifact.artifact_id in set(
        result["invalidated_artifact_ids"]
    )
    assert graph.runtime_artifact is not None
    assert graph.runtime_artifact.artifact_id in set(
        result["invalidated_artifact_ids"]
    )
    assert inv.runtime_artifact.artifact_id not in set(
        result["invalidated_artifact_ids"]
    )

    assert not cache.lookup(ast_key).hit
    assert not cache.lookup(graph_key).hit
    assert cache.lookup(inventory_key).hit
    assert cache.lookup(unrelated_key).hit
    assert cache.lookup(unrelated_key).is_completion_evidence


def test_invalidate_dimension_only_touches_matching_population(
    tmp_path: Path,
) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    a = _key(capability_revision="capability@A")
    b = _key(capability_revision="capability@B")
    c = _key(
        capability_revision="capability@A",
        component_kind=ProgramAnalysisComponentKind.PROOF,
    )
    assert cache.put(a, _receipt(ordinal=1))
    assert cache.put(b, _receipt(ordinal=2))
    assert cache.put(c, _receipt(ordinal=3, component="proof"))

    report = cache.invalidate_dimension(capability_revision="capability@A")
    assert report["removed_entries"] >= 2
    assert not cache.lookup(a).hit
    assert not cache.lookup(c).hit
    assert cache.lookup(b).hit


def test_corruption_recovery_and_atomic_rewrite(tmp_path: Path) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    assert cache.put(_key(), _receipt())
    path = cache.entry_path(_key())
    path.write_text('{"schema":', encoding="utf-8")

    corrupt = cache.lookup(_key())
    assert corrupt.status is ProgramAnalysisLookupStatus.INVALIDATED
    assert (
        ProgramAnalysisCacheReason.CORRUPT_ENTRY.value in corrupt.reason_codes
    )
    assert corrupt.receipt is None

    recovered = cache.put(_key(), _receipt(ordinal=2))
    assert recovered.stored
    hit = cache.lookup(_key())
    assert hit.hit
    assert hit.receipt is not None
    assert hit.receipt["receipt_id"] == "inventory-receipt-2"
    # Entry is valid JSON after recovery.
    json.loads(path.read_text(encoding="utf-8"))


def test_process_single_flight_collapses_identical_misses(tmp_path: Path) -> None:
    cache = ProgramAnalysisCache(tmp_path, wait_timeout_seconds=10.0)
    barrier = threading.Barrier(8)
    results: list[ProgramAnalysisLookupStatus | str] = []
    invocations = {"count": 0}
    lock = threading.Lock()

    def producer() -> dict[str, object]:
        with lock:
            invocations["count"] += 1
        time.sleep(0.05)
        return _receipt(ordinal=invocations["count"])

    def worker() -> None:
        try:
            barrier.wait()
            result = cache.get_or_compute(_key(), producer)
            results.append(result.status)
        except Exception as exc:  # pragma: no cover
            results.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(
        item is ProgramAnalysisLookupStatus.HIT
        or (
            isinstance(item, ProgramAnalysisLookupStatus)
            and item is ProgramAnalysisLookupStatus.HIT
        )
        for item in results
    ), results
    assert invocations["count"] == 1
    # Followers and leader share one durable completion hit.
    assert cache.lookup(_key()).is_completion_evidence
    metrics = cache.coordinator.metrics()
    produced = getattr(metrics, "produced", 0) + getattr(
        metrics, "producer_invocation_count", 0
    )
    assert produced >= 1 or invocations["count"] == 1


def _cross_process_worker(
    cache_path: str,
    start: multiprocessing.synchronize.Barrier,
    output: multiprocessing.queues.Queue,
    worker_id: int,
) -> None:
    try:
        cache = ProgramAnalysisCache(cache_path, wait_timeout_seconds=20.0)
        calls = {"n": 0}

        def producer() -> dict[str, object]:
            calls["n"] += 1
            time.sleep(0.1)
            return _receipt(ordinal=worker_id + 1)

        start.wait()
        result = cache.get_or_compute(_key(), producer)
        output.put(
            {
                "worker": worker_id,
                "hit": result.hit,
                "completion": result.is_completion_evidence,
                "calls": calls["n"],
                "receipt_id": (result.receipt or {}).get("receipt_id"),
            }
        )
    except BaseException as exc:  # pragma: no cover
        output.put({"worker": worker_id, "error": f"{type(exc).__name__}: {exc}"})


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="cross-process single-flight test requires fork",
)
def test_cross_process_single_flight_and_zero_stale_hits(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Barrier(4)
    output = context.Queue()
    processes = [
        context.Process(
            target=_cross_process_worker,
            args=(str(tmp_path), start, output, ordinal),
        )
        for ordinal in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)

    assert all(process.exitcode == 0 for process in processes)
    reports = [output.get(timeout=2) for _ in processes]
    assert all("error" not in report for report in reports), reports
    assert all(report["hit"] and report["completion"] for report in reports)
    # Exactly one process should have invoked the producer.
    assert sum(report["calls"] for report in reports) == 1
    receipt_ids = {report["receipt_id"] for report in reports}
    assert len(receipt_ids) == 1

    restarted = ProgramAnalysisCache(tmp_path)
    hit = restarted.lookup(_key())
    assert hit.is_completion_evidence
    assert hit.reason_codes == (ProgramAnalysisCacheReason.EXACT_KEY_HIT.value,)


def test_concurrent_writers_never_publish_partial_json(tmp_path: Path) -> None:
    cache = ProgramAnalysisCache(tmp_path, max_entries=64)
    barrier = threading.Barrier(12)
    failures: list[BaseException] = []

    def worker(ordinal: int) -> None:
        try:
            barrier.wait()
            key = _key(query_digest=f"sha256:query-{ordinal}")
            assert cache.put(key, _receipt(ordinal=ordinal))
            assert cache.lookup(key).hit
        except BaseException as exc:  # pragma: no cover
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(ordinal,)) for ordinal in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    for path in cache.analysis_cache.entries_dir.glob("*/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "entry_digest" in payload
        assert "program_key" in payload["receipt"]


def test_quotas_and_gc_bound_retained_entries(tmp_path: Path) -> None:
    now = [3_000.0]
    cache = ProgramAnalysisCache(
        tmp_path,
        max_entries=3,
        max_bytes=8_000,
        max_entry_bytes=2_500,
        max_receipt_bytes=1_500,
        clock=lambda: now[0],
    )
    latest = _key()
    for ordinal in range(10):
        now[0] += 1
        latest = _key(query_digest=f"sha256:bounded-{ordinal}")
        assert cache.put(latest, _receipt(ordinal=ordinal))

    pruned = cache.prune()
    assert pruned >= 0
    stats = cache.stats()
    assert stats.entry_count <= 3
    assert stats.total_bytes <= 8_000
    assert cache.lookup(latest).hit


def test_build_key_aliases_and_evidence_constants(tmp_path: Path) -> None:
    key = build_program_analysis_cache_key(
        repository_tree_identity="forest:alias",
        objective_revision="obj",
        policy_digest="pol",
        analyzer_version="an",
        schema_version="sch",
        configuration_digest="cfg",
        query_digest="q",
        capability_digest="cap",
        assumptions_digest="ass",
        toolchain="tc",
        component_kind="zk",
    )
    assert key.forest_identity == "forest:alias"
    assert key.policy_revision == "pol"
    assert key.component_kind is ProgramAnalysisComponentKind.ZK
    assert DEPENDENCY_CACHE_EVIDENCE == "vfs/dependency-cache@1"
    assert CACHE_INVALIDATION_PROOF_EVIDENCE == "vfs/cache-invalidation-proof@1"

    compact = compact_program_analysis_receipt(
        _receipt(component="zk"), key=key
    )
    assert compact["program_key"]["component_kind"] == "zk"
    assert compact["component_kind"] == "zk"


def test_zero_stale_authoritative_hits_under_concurrency_and_restart(
    tmp_path: Path,
) -> None:
    now = [5_000.0]
    cache = ProgramAnalysisCache(
        tmp_path,
        default_success_ttl_seconds=2,
        clock=lambda: now[0],
    )
    assert cache.put(_key(), _receipt())
    assert cache.lookup(_key()).is_completion_evidence

    barrier = threading.Barrier(6)
    outcomes: list[bool] = []

    def worker() -> None:
        barrier.wait()
        local = ProgramAnalysisCache(tmp_path, clock=lambda: now[0])
        # Advance past success TTL for some readers.
        sample = local.lookup(_key())
        outcomes.append(bool(sample.is_completion_evidence))

    now[0] += 3  # all success entries are stale
    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # Zero stale authoritative hits: completion evidence must be false for all.
    assert outcomes == [False] * 6
    restarted = ProgramAnalysisCache(tmp_path, clock=lambda: now[0])
    stale = restarted.lookup(_key())
    assert not stale.is_completion_evidence
    assert stale.status is ProgramAnalysisLookupStatus.INVALIDATED


def test_get_or_compute_reuses_exact_hit_without_producer(tmp_path: Path) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    assert cache.put(_key(), _receipt())
    calls = {"n": 0}

    def producer() -> dict[str, object]:
        calls["n"] += 1
        return _receipt(ordinal=99)

    result = cache.get_or_compute(_key(), producer)
    assert result.hit
    assert result.is_completion_evidence
    assert calls["n"] == 0
    assert result.coordination is not None
    assert result.coordination.status is CacheCoordinationStatus.CACHE_HIT


def test_failed_flight_does_not_poison_future_lookups(tmp_path: Path) -> None:
    cache = ProgramAnalysisCache(tmp_path)
    attempts = {"n": 0}

    def failing_producer() -> dict[str, object]:
        attempts["n"] += 1
        raise RuntimeError("producer boom")

    with pytest.raises(RuntimeError, match="producer boom"):
        cache.get_or_compute(_key(), failing_producer)
    assert attempts["n"] == 1
    assert not cache.lookup(_key()).hit

    def good_producer() -> dict[str, object]:
        attempts["n"] += 1
        return _receipt(ordinal=7)

    recovered = cache.get_or_compute(_key(), good_producer)
    assert recovered.hit
    assert recovered.is_completion_evidence
    assert attempts["n"] == 2
