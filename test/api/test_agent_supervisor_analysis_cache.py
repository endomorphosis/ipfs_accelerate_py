from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_cache import (
    AnalysisCache,
    AnalysisCacheEntry,
    AnalysisCacheKey,
    AnalysisCacheLookupStatus,
    AnalysisCacheReason,
    AnalysisOutcome,
    compact_analysis_receipt,
)


def _key(**changes: object) -> AnalysisCacheKey:
    values: dict[str, object] = {
        "repository_tree_identity": "tree:sha256:111",
        "objective_revision": "objective-revision-1",
        "analyzer_version": "analyzer@1",
        "schema_version": "analysis-schema@1",
        "configuration_digest": "sha256:config-1",
        "query_digest": "sha256:query-1",
        "policy_digest": "sha256:policy-1",
    }
    values.update(changes)
    return AnalysisCacheKey(**values)


def _receipt(
    status: AnalysisOutcome | str = AnalysisOutcome.SUCCESSFUL,
    *,
    ordinal: int = 1,
) -> dict[str, object]:
    return {
        "status": status.value if isinstance(status, AnalysisOutcome) else status,
        "receipt_id": f"receipt-{ordinal}",
        "summary": {"files_considered": ordinal, "finding_count": 2},
        "counts": {"files": ordinal, "findings": 2},
        "artifact_refs": [
            {
                "artifact_id": f"analysis-{ordinal}",
                "cid": f"bafy-analysis-{ordinal}",
                "digest": f"sha256:artifact-{ordinal}",
            }
        ],
    }


def test_exact_key_hit_is_deterministic_and_persistent(tmp_path: Path) -> None:
    now = [1_000.0]
    cache = AnalysisCache(tmp_path, clock=lambda: now[0])
    miss = cache.lookup(_key())
    assert miss.status is AnalysisCacheLookupStatus.MISS
    assert miss.reason_codes == (AnalysisCacheReason.CACHE_MISS.value,)

    stored = cache.put(_key(), _receipt())

    assert stored.stored
    assert stored.entry is not None
    first_bytes = cache.entry_path(_key()).read_bytes()
    assert first_bytes == stored.entry.serialize()
    assert json.loads(first_bytes)["entry_digest"].startswith("sha256:")

    reopened = AnalysisCache(tmp_path, clock=lambda: now[0])
    hit = reopened.lookup(_key())
    assert hit.status is AnalysisCacheLookupStatus.HIT
    assert hit.reason_codes == (AnalysisCacheReason.EXACT_KEY_HIT.value,)
    assert hit.receipt == compact_analysis_receipt(_receipt())
    assert hit.outcome is AnalysisOutcome.SUCCESSFUL
    assert hit.is_completion_evidence

    reconstructed = AnalysisCacheEntry.create(
        _key(),
        compact_analysis_receipt(_receipt()),
        created_at_ms=1_000_000,
        expires_at_ms=None,
    )
    assert reconstructed.serialize() == first_bytes


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        (
            "repository_tree_identity",
            "tree:sha256:222",
            AnalysisCacheReason.REPOSITORY_TREE_IDENTITY_CHANGED,
        ),
        (
            "objective_revision",
            "objective-revision-2",
            AnalysisCacheReason.OBJECTIVE_REVISION_CHANGED,
        ),
        (
            "analyzer_version",
            "analyzer@2",
            AnalysisCacheReason.ANALYZER_VERSION_CHANGED,
        ),
        (
            "schema_version",
            "analysis-schema@2",
            AnalysisCacheReason.SCHEMA_VERSION_CHANGED,
        ),
        (
            "configuration_digest",
            "sha256:config-2",
            AnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED,
        ),
        (
            "query_digest",
            "sha256:query-2",
            AnalysisCacheReason.QUERY_DIGEST_CHANGED,
        ),
        (
            "policy_digest",
            "sha256:policy-2",
            AnalysisCacheReason.POLICY_DIGEST_CHANGED,
        ),
    ],
)
def test_every_key_dimension_has_an_explicit_invalidation_reason(
    tmp_path: Path,
    field: str,
    replacement: str,
    reason: AnalysisCacheReason,
) -> None:
    cache = AnalysisCache(tmp_path)
    assert cache.put(_key(), _receipt())

    lookup = cache.lookup(_key(**{field: replacement}))

    assert lookup.status is AnalysisCacheLookupStatus.INVALIDATED
    assert lookup.reason_codes == (reason.value,)
    assert not lookup.hit
    assert lookup.receipt is None
    assert not lookup.is_completion_evidence


@pytest.mark.parametrize(
    "outcome",
    [
        AnalysisOutcome.PARTIAL,
        AnalysisOutcome.FAILED,
        AnalysisOutcome.TIMED_OUT,
        AnalysisOutcome.INCONCLUSIVE,
    ],
)
def test_non_success_outcomes_have_bounded_ttl_and_never_complete(
    tmp_path: Path, outcome: AnalysisOutcome
) -> None:
    now = [2_000.0]
    cache = AnalysisCache(
        tmp_path,
        default_negative_ttl_seconds=5,
        max_negative_ttl_seconds=10,
        clock=lambda: now[0],
    )
    stored = cache.put(_key(), _receipt(outcome), ttl_seconds=1_000)
    assert stored.stored
    assert stored.entry is not None
    assert stored.entry.expires_at_ms == 2_010_000

    reusable = cache.lookup(_key())
    assert reusable.hit
    assert reusable.outcome is outcome
    assert not reusable.is_completion_evidence

    completion = cache.lookup(_key(), require_completion_evidence=True)
    assert completion.status is AnalysisCacheLookupStatus.INVALIDATED
    assert completion.reason_codes == (
        AnalysisCacheReason.NOT_COMPLETION_EVIDENCE.value,
    )

    now[0] += 11
    stale = cache.lookup(_key())
    assert stale.status is AnalysisCacheLookupStatus.INVALIDATED
    assert stale.reason_codes == (
        AnalysisCacheReason.STALE_NEGATIVE_ENTRY.value,
    )
    assert stale.receipt is None
    assert not stale.is_completion_evidence


def test_compact_receipts_exclude_heavy_analysis_payloads(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    forbidden_receipts = [
        {**_receipt(), "source_text": "print('not cache material')"},
        {**_receipt(), "decoded_model_output": {"large": "payload"}},
        {**_receipt(), "ast_body": {"type": "Module", "body": []}},
        {
            **_receipt(),
            "artifact_graph": {
                "nodes": [{"artifact": "recursive"}],
                "edges": [],
            },
        },
        {
            **_receipt(),
            "artifact_refs": [
                {
                    "artifact_id": "nested",
                    "children": [{"artifact_id": "recursive"}],
                }
            ],
        },
    ]

    for receipt in forbidden_receipts:
        result = cache.put(_key(), receipt)
        assert not result.stored
        assert result.reason_codes == (
            AnalysisCacheReason.MALFORMED_RECEIPT.value,
        )
    assert cache.stats().entry_count == 0


def test_corruption_is_safe_explicit_and_recoverable(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    assert cache.put(_key(), _receipt())
    path = cache.entry_path(_key())
    path.write_text('{"schema":', encoding="utf-8")

    corrupt = cache.lookup(_key())
    assert corrupt.status is AnalysisCacheLookupStatus.INVALIDATED
    assert corrupt.reason_codes == (AnalysisCacheReason.CORRUPT_ENTRY.value,)
    assert corrupt.receipt is None

    recovered = cache.put(_key(), _receipt(ordinal=2))
    assert recovered.stored
    hit = cache.lookup(_key())
    assert hit.hit
    assert hit.receipt is not None
    assert hit.receipt["receipt_id"] == "receipt-2"


def test_concurrent_thread_writers_never_publish_partial_json(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(tmp_path, max_entries=64)
    barrier = threading.Barrier(16)
    failures: list[BaseException] = []

    def worker(ordinal: int) -> None:
        try:
            barrier.wait()
            key = _key(query_digest=f"sha256:query-{ordinal}")
            assert cache.put(key, _receipt(ordinal=ordinal))
            assert cache.lookup(key).hit
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(ordinal,)) for ordinal in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    assert cache.stats().entry_count == 16
    for path in cache.entries_dir.glob("*/*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["entry_digest"]


def test_concurrent_same_key_writers_converge_on_one_valid_entry(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(tmp_path, max_entries=8)
    barrier = threading.Barrier(12)
    failures: list[BaseException] = []

    def worker(ordinal: int) -> None:
        try:
            barrier.wait()
            assert cache.put(_key(), _receipt(ordinal=ordinal))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    threads = [
        threading.Thread(target=worker, args=(ordinal,)) for ordinal in range(12)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not failures
    hit = cache.lookup(_key())
    assert hit.hit
    assert hit.receipt is not None
    assert hit.receipt["receipt_id"] in {
        f"receipt-{ordinal}" for ordinal in range(12)
    }
    assert cache.stats().entry_count == 1


def _process_writer(
    cache_path: str,
    start: multiprocessing.synchronize.Barrier,
    ordinal: int,
    output: multiprocessing.queues.Queue,
) -> None:
    try:
        cache = AnalysisCache(cache_path, max_entries=32)
        start.wait()
        key = _key(query_digest=f"sha256:process-query-{ordinal}")
        output.put((ordinal, bool(cache.put(key, _receipt(ordinal=ordinal)))))
    except BaseException as exc:  # pragma: no cover - parent reports child error
        output.put((ordinal, f"{type(exc).__name__}: {exc}"))


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="cross-process writer test requires fork",
)
def test_concurrent_process_writers_preserve_all_entries(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    start = context.Barrier(6)
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_writer,
            args=(str(tmp_path), start, ordinal, output),
        )
        for ordinal in range(6)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(output.get(timeout=2) for _ in processes) == [
        (ordinal, True) for ordinal in range(6)
    ]
    cache = AnalysisCache(tmp_path, max_entries=32)
    assert cache.stats().entry_count == 6
    for ordinal in range(6):
        assert cache.lookup(
            _key(query_digest=f"sha256:process-query-{ordinal}")
        ).hit


def test_count_and_byte_retention_bounds_are_enforced_on_disk(
    tmp_path: Path,
) -> None:
    now = [3_000.0]
    cache = AnalysisCache(
        tmp_path,
        max_entries=3,
        max_bytes=2_500,
        max_entry_bytes=1_500,
        max_receipt_bytes=900,
        clock=lambda: now[0],
    )
    latest_key = _key()
    for ordinal in range(12):
        now[0] += 1
        latest_key = _key(query_digest=f"sha256:bounded-{ordinal}")
        assert cache.put(latest_key, _receipt(ordinal=ordinal))

    paths = list(cache.entries_dir.glob("*/*.json"))
    persisted_bytes = sum(path.stat().st_size for path in paths)
    stats = cache.stats()
    assert len(paths) == stats.entry_count
    assert stats.entry_count <= 3
    assert persisted_bytes == stats.total_bytes
    assert persisted_bytes <= 2_500
    assert cache.lookup(latest_key).hit


def test_entry_size_limit_rejects_without_disturbing_existing_entry(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(
        tmp_path,
        max_bytes=2_048,
        max_entry_bytes=1_024,
        max_receipt_bytes=900,
    )
    assert cache.put(_key(), _receipt())
    oversized = _receipt(ordinal=2)
    oversized["summary"] = {"compact_but_too_large": "x" * 500}

    rejected = cache.put(
        _key(query_digest="sha256:oversized"),
        oversized,
    )
    assert not rejected.stored
    assert rejected.reason_codes == (
        AnalysisCacheReason.ENTRY_TOO_LARGE.value,
    )
    assert cache.lookup(_key()).hit
