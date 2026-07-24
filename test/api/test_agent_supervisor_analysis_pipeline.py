from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analyzer_health import (
    AnalysisEscalationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.analysis_cache import (
    AnalysisCache,
    AnalysisCacheLookupStatus,
    AnalysisCacheReason,
)
from ipfs_accelerate_py.agent_supervisor.analysis_pipeline import (
    EXACT_TREE_REUSE_ACCEPTANCE_CRITERIA,
    EXACT_TREE_REUSE_REQUIREMENT_ID,
    SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID,
    AnalysisBindingError,
    AnalysisPipeline,
    AnalysisPipelinePolicy,
    AnalysisPipelineRequest,
    AnalysisPipelineResult,
    ExactTreeReuseEvidence,
    PipelineCacheStatus,
    make_analysis_stage_receipt,
)
from ipfs_accelerate_py.agent_supervisor.audit_scanner import (
    run_low_backlog_analysis,
)
from ipfs_accelerate_py.agent_supervisor.conflict_graph import (
    build_python_ast_blob_record,
)
from ipfs_accelerate_py.agent_supervisor.ipfs_datasets_analysis_provider import (
    IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    AnalysisProviderStatus,
    IpfsDatasetsAnalysisProvider,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)


G095_ACCEPTANCE_CRITERIA = (
    "Capability inspection is deterministic and side-effect free",
    "Dispatch lazily imports only a supported requested operation",
    "Provider policy is a non-expandable resource envelope",
    (
        "Capability and result payloads are canonical, bounded, compact, "
        "and identity checked"
    ),
    (
        "Timed-out non-cooperative backends cannot create unbounded "
        "dispatch threads"
    ),
    (
        "Unavailable, unsupported, disabled, unhealthy, timed-out, failed, "
        "malformed, and cancelled results state exact reasons and "
        "deterministic fallback"
    ),
    (
        "No provider result, including a backend-supplied authority claim, "
        "is completion evidence"
    ),
)


def _request(**changes: object) -> AnalysisPipelineRequest:
    values: dict[str, object] = {
        "repository_id": "repo:fixture",
        "tree_id": "tree:sha256:111",
        "objective_revision": "objective@1",
        "query": {"text": "analysis cache authority", "objective_terms": ["cache"]},
        "analyzer_id": "fixture.analyzer",
        "analyzer_version": "fixture-analyzer@1",
        "schema_version": "analysis-schema@1",
        "configuration_digest": "sha256:config-1",
        "query_digest": "sha256:query-1",
        "policy_digest": "sha256:policy-1",
        "retrieval_inputs": {
            "records": [
                {
                    "record_id": "record-1",
                    "title": "analysis cache authority",
                    "task_id": "ASI-027",
                }
            ]
        },
    }
    values.update(changes)
    return AnalysisPipelineRequest(**values)


class _Analyzer:
    def __init__(self) -> None:
        self.calls = 0
        self.contexts = []
        self._lock = threading.Lock()

    def analyze(self, context):
        with self._lock:
            self.calls += 1
            self.contexts.append(context)
        return make_analysis_stage_receipt(
            context.request,
            successful=True,
            reason_code="fixture_complete",
        )


def test_exact_reuse_is_content_bound_and_emits_the_requirement(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    request = _request()

    cold = pipeline.analyze(request)
    warm = pipeline.analyze(request)

    assert cold.cache_status is PipelineCacheStatus.PRODUCED
    assert cold.evidence_claim_references == ()
    assert warm.cache_status is PipelineCacheStatus.EXACT_HIT
    assert warm.packet.packet_id == cold.packet.packet_id
    assert warm.safe_for_completion_reasoning
    assert warm.evidence_claim_references == (EXACT_TREE_REUSE_REQUIREMENT_ID,)
    assert warm.exact_tree_reuse_evidence is not None
    assert (
        warm.exact_tree_reuse_evidence.cache_key_id
        == warm.request.cache_key.key_id
    )
    assert warm.exact_tree_reuse_evidence.tree_id == request.tree_id
    assert warm.cache_lookup is not None
    assert warm.exact_tree_reuse_evidence.proves_for(
        warm.request, warm.packet, warm.cache_lookup
    )
    restored = ExactTreeReuseEvidence.from_dict(
        warm.exact_tree_reuse_evidence.to_dict()
    )
    assert restored == warm.exact_tree_reuse_evidence
    assert restored.proves_for(warm.request, warm.packet, warm.cache_lookup)
    assert analyzer.calls == 1


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        (
            "repository_id",
            "repo:other",
            AnalysisCacheReason.REPOSITORY_TREE_IDENTITY_CHANGED.value,
        ),
        (
            "tree_id",
            "tree:sha256:222",
            AnalysisCacheReason.REPOSITORY_TREE_IDENTITY_CHANGED.value,
        ),
        (
            "objective_revision",
            "objective@2",
            AnalysisCacheReason.OBJECTIVE_REVISION_CHANGED.value,
        ),
        (
            "analyzer_version",
            "fixture-analyzer@2",
            AnalysisCacheReason.ANALYZER_VERSION_CHANGED.value,
        ),
        (
            "schema_version",
            "analysis-schema@2",
            AnalysisCacheReason.SCHEMA_VERSION_CHANGED.value,
        ),
        (
            "configuration_digest",
            "sha256:config-2",
            AnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED.value,
        ),
        (
            "query_digest",
            "sha256:query-2",
            AnalysisCacheReason.QUERY_DIGEST_CHANGED.value,
        ),
        (
            "policy_digest",
            "sha256:policy-2",
            AnalysisCacheReason.POLICY_DIGEST_CHANGED.value,
        ),
    ],
)
def test_every_authority_dimension_rejects_historical_success(
    tmp_path: Path,
    field: str,
    replacement: str,
    reason: str,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    original = _request()
    pipeline.analyze(original)

    changed = _request(**{field: replacement})
    result = pipeline.analyze(changed)

    assert result.cache_status is PipelineCacheStatus.PRODUCED
    assert reason in result.cache_reason_codes
    assert result.evidence_claim_references == ()
    assert result.packet.tree_id == changed.tree_id
    assert result.packet.objective_revision == changed.objective_revision
    assert analyzer.calls == 2
    assert pipeline.metrics.stale_authoritative_hits == 0


def test_expired_success_is_attached_for_diagnostics_but_never_reused(
    tmp_path: Path,
) -> None:
    now = [100.0]
    cache = AnalysisCache(
        tmp_path,
        clock=lambda: now[0],
        default_success_ttl_seconds=1,
    )
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(cache, analyzer)
    request = _request()
    first = pipeline.analyze(request)
    now[0] += 2

    historical = cache.lookup(
        first.request.cache_key, require_completion_evidence=True
    )
    assert historical.invalidated
    assert historical.entry is not None
    assert historical.entry.is_completion_evidence

    result = pipeline.analyze(request)

    assert result.cache_status is PipelineCacheStatus.PRODUCED
    assert AnalysisCacheReason.STALE_ENTRY.value in result.cache_reason_codes
    assert result.evidence_claim_references == ()
    assert analyzer.calls == 2
    assert pipeline.metrics.stale_authoritative_hits == 0


def test_missing_packet_artifact_forces_recompute_instead_of_cache_authority(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    cache = AnalysisCache(tmp_path)
    pipeline = AnalysisPipeline(cache, analyzer)
    request = _request()
    first = pipeline.analyze(request)
    lookup = cache.lookup(
        first.request.cache_key, require_completion_evidence=True
    )
    assert lookup.receipt is not None
    artifact = lookup.receipt["artifact_refs"][0]
    Path(artifact["path"]).unlink()

    repaired = pipeline.analyze(request)

    assert repaired.cache_status is PipelineCacheStatus.PRODUCED
    assert repaired.cache_reason_codes == ("packet_artifact_invalid",)
    assert repaired.evidence_claim_references == ()
    assert analyzer.calls == 2
    assert cache.lookup(
        repaired.request.cache_key, require_completion_evidence=True
    ).is_completion_evidence


def test_concurrent_invalid_artifact_repairs_remain_single_flight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = 8
    analyzer = _Analyzer()
    cache = AnalysisCache(tmp_path)
    pipeline = AnalysisPipeline(cache, analyzer)
    request = _request()
    cold = pipeline.analyze(request)
    lookup = cache.lookup(
        cold.request.cache_key, require_completion_evidence=True
    )
    assert lookup.receipt is not None
    Path(lookup.receipt["artifact_refs"][0]["path"]).unlink()

    # Hold every caller in its optimistic artifact read so all of them observe
    # the same invalid binding before any repair can publish a replacement.
    artifact_barrier = threading.Barrier(workers)
    original_get = pipeline._artifacts.get
    call_lock = threading.Lock()
    invalid_reads = 0

    def synchronized_get(reference):
        nonlocal invalid_reads
        with call_lock:
            invalid_reads += 1
            ordinal = invalid_reads
        if ordinal <= workers:
            artifact_barrier.wait(timeout=10)
            raise FileNotFoundError("fixture removed packet artifact")
        return original_get(reference)

    monkeypatch.setattr(pipeline._artifacts, "get", synchronized_get)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(pipeline.analyze, request) for _ in range(workers)
        ]
        repaired = [future.result(timeout=15) for future in futures]

    assert analyzer.calls == 2
    assert sum(
        item.cache_status is PipelineCacheStatus.PRODUCED for item in repaired
    ) == 1
    assert all(
        item.cache_status
        in {
            PipelineCacheStatus.PRODUCED,
            PipelineCacheStatus.JOINED,
            PipelineCacheStatus.EXACT_HIT,
        }
        for item in repaired
    )
    assert {item.packet.packet_id for item in repaired} == {cold.packet.packet_id}
    assert pipeline.coordinator.metrics().cache_validation_rejections >= 1
    assert pipeline.metrics.stale_authoritative_hits == 0


def test_negative_result_never_short_circuits_a_later_producer(
    tmp_path: Path,
) -> None:
    calls = 0

    def analyzer(context):
        nonlocal calls
        calls += 1
        return make_analysis_stage_receipt(
            context.request,
            successful=calls > 1,
            coverage_complete=calls > 1,
            reason_code="complete" if calls > 1 else "incomplete",
        )

    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    request = _request()
    negative = pipeline.analyze(request)
    positive = pipeline.analyze(request)

    assert not negative.safe_for_completion_reasoning
    assert negative.evidence_claim_references == ()
    assert positive.safe_for_completion_reasoning
    assert positive.cache_status is PipelineCacheStatus.PRODUCED
    assert calls == 2
    assert pipeline.metrics.negative_rejections >= 1


def test_declared_digests_cannot_hide_changed_analysis_inputs(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    pipeline.analyze(_request(query="first query", query_digest="claimed-same"))

    changed_query = pipeline.analyze(
        _request(query="second query", query_digest="claimed-same")
    )
    changed_analyzer = pipeline.analyze(
        _request(
            query="second query",
            query_digest="claimed-same",
            analyzer_id="different.analyzer",
        )
    )
    changed_retrieval = pipeline.analyze(
        _request(
            query="second query",
            query_digest="claimed-same",
            analyzer_id="different.analyzer",
            retrieval_inputs={"records": [{"record_id": "different"}]},
        )
    )

    assert AnalysisCacheReason.QUERY_DIGEST_CHANGED.value in (
        changed_query.cache_reason_codes
    )
    assert AnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED.value in (
        changed_analyzer.cache_reason_codes
    )
    assert AnalysisCacheReason.CONFIGURATION_DIGEST_CHANGED.value in (
        changed_retrieval.cache_reason_codes
    )
    assert analyzer.calls == 4


def test_pipeline_execution_policy_is_part_of_cache_authority(
    tmp_path: Path,
) -> None:
    cache = AnalysisCache(tmp_path)
    analyzer = _Analyzer()
    first_pipeline = AnalysisPipeline(cache, analyzer)
    first_pipeline.analyze(_request())
    changed_pipeline = AnalysisPipeline(
        cache,
        analyzer,
        policy=AnalysisPipelinePolicy(negative_ttl_seconds=17),
    )

    result = changed_pipeline.analyze(_request())

    assert result.cache_status is PipelineCacheStatus.PRODUCED
    assert AnalysisCacheReason.POLICY_DIGEST_CHANGED.value in (
        result.cache_reason_codes
    )
    assert analyzer.calls == 2


def test_negative_cache_policy_is_applied_inside_single_flight(
    tmp_path: Path,
) -> None:
    calls = 0

    def incomplete(context):
        nonlocal calls
        calls += 1
        return make_analysis_stage_receipt(
            context.request,
            successful=False,
            coverage_complete=False,
            reason_code="incomplete",
        )

    cache = AnalysisCache(tmp_path)
    pipeline = AnalysisPipeline(
        cache,
        incomplete,
        policy=AnalysisPipelinePolicy(cache_negative_results=False),
    )
    first = pipeline.analyze(_request())
    second = pipeline.analyze(_request())

    assert calls == 2
    assert first.cache_status is PipelineCacheStatus.INCONCLUSIVE
    assert second.cache_status is PipelineCacheStatus.INCONCLUSIVE
    assert cache.lookup(
        first.request.cache_key, require_completion_evidence=False
    ).miss


def test_negative_cache_uses_configured_bounded_ttl(tmp_path: Path) -> None:
    now = [25.0]
    cache = AnalysisCache(tmp_path, clock=lambda: now[0])

    def incomplete(context):
        return make_analysis_stage_receipt(
            context.request,
            successful=False,
            coverage_complete=False,
            reason_code="incomplete",
        )

    pipeline = AnalysisPipeline(
        cache,
        incomplete,
        policy=AnalysisPipelinePolicy(negative_ttl_seconds=7),
    )
    result = pipeline.analyze(_request())
    stored = cache.lookup(
        result.request.cache_key, require_completion_evidence=False
    )

    assert stored.hit
    assert stored.entry is not None
    assert stored.entry.expires_at_ms == 32_000
    now[0] = 33.0
    assert cache.lookup(
        result.request.cache_key, require_completion_evidence=False
    ).invalidated


def test_exact_reuse_witness_cannot_be_forged_or_detached(
    tmp_path: Path,
) -> None:
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), _Analyzer())
    pipeline.analyze(_request())
    warm = pipeline.analyze(_request())
    assert warm.exact_tree_reuse_evidence is not None
    assert warm.cache_lookup is not None

    with pytest.raises(AnalysisBindingError, match="authoritative HIT lookup"):
        AnalysisPipelineResult(
            request=warm.request,
            packet=warm.packet,
            cache_status=PipelineCacheStatus.EXACT_HIT,
            cache_lookup_status=AnalysisCacheLookupStatus.MISS,
            cache_reason_codes=("cache_miss",),
            exact_tree_reuse_evidence=warm.exact_tree_reuse_evidence,
        )
    with pytest.raises(AnalysisBindingError, match="typed lookup"):
        AnalysisPipelineResult(
            request=warm.request,
            packet=warm.packet,
            cache_status=PipelineCacheStatus.EXACT_HIT,
            cache_lookup_status=AnalysisCacheLookupStatus.HIT,
            cache_reason_codes=("exact_key_hit",),
            exact_tree_reuse_evidence=replace(
                warm.exact_tree_reuse_evidence,
                tree_id="tree:sha256:forged",
            ),
            cache_lookup=warm.cache_lookup,
        )
    assert warm.cache_lookup.entry is not None
    forged_lookup = replace(
        warm.cache_lookup,
        entry=replace(
            warm.cache_lookup.entry,
            entry_digest="sha256:" + ("f" * 64),
        ),
    )
    assert not warm.exact_tree_reuse_evidence.proves_for(
        warm.request, warm.packet, forged_lookup
    )
    with pytest.raises(AnalysisBindingError, match="typed lookup"):
        AnalysisPipelineResult(
            request=warm.request,
            packet=warm.packet,
            cache_status=PipelineCacheStatus.EXACT_HIT,
            cache_lookup_status=AnalysisCacheLookupStatus.HIT,
            cache_reason_codes=("exact_key_hit",),
            exact_tree_reuse_evidence=warm.exact_tree_reuse_evidence,
            cache_lookup=forged_lookup,
        )
    with pytest.raises(AnalysisBindingError, match="typed lookup"):
        AnalysisPipelineResult(
            request=warm.request,
            packet=warm.packet,
            cache_status=PipelineCacheStatus.EXACT_HIT,
            cache_lookup_status=AnalysisCacheLookupStatus.HIT,
            cache_reason_codes=("exact_key_hit",),
            exact_tree_reuse_evidence=replace(
                warm.exact_tree_reuse_evidence,
                cache_entry_digest="sha256:" + ("0" * 64),
            ),
            cache_lookup=warm.cache_lookup,
        )


def test_repeated_fixture_exceeds_reuse_target_without_stale_authority(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    results = [pipeline.analyze(_request()) for _ in range(10)]

    assert analyzer.calls == 1
    assert sum(result.reused for result in results) == 9
    assert pipeline.metrics.reuse_ratio == pytest.approx(0.9)
    assert pipeline.metrics.reuse_ratio >= 0.7
    assert pipeline.metrics.stale_authoritative_hits == 0


def test_identical_concurrent_misses_collapse_before_expensive_analysis(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    call_lock = threading.Lock()

    class Backend:
        __version__ = "counting-backend@1"

        def __init__(self):
            self.capability_calls = 0
            self.requests = []

        def capabilities(self):
            self.capability_calls += 1
            return {
                "protocol_versions": [1],
                "available": True,
                "operations": ["graph_retrieval"],
                "provider_version": self.__version__,
            }

        def retrieve(self, request):
            self.requests.append(request)
            return {
                "status": "completed",
                "results": [
                    {
                        "evidence_id": "provider-evidence-1",
                        "path": "src/cache.py",
                        "symbol": "AnalysisCache.lookup",
                    }
                ],
            }

    def analyzer(context):
        nonlocal calls
        with call_lock:
            calls += 1
        entered.set()
        assert release.wait(5)
        return make_analysis_stage_receipt(
            context.request,
            successful=True,
            reason_code="complete",
        )

    backend = Backend()
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path),
        analyzer,
        provider=IpfsDatasetsAnalysisProvider(backend=backend),
    )
    request = _request()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(pipeline.analyze, request) for _ in range(8)]
        assert entered.wait(5)
        deadline = time.monotonic() + 5
        while (
            pipeline.coordinator.metrics().followers < 7
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert pipeline.coordinator.metrics().followers == 7
        release.set()
        results = [future.result(timeout=10) for future in futures]

    assert calls == 1
    assert backend.capability_calls == 1
    assert len(backend.requests) == 1
    assert len({result.packet.packet_id for result in results}) == 1
    assert sum(result.joined_existing_flight for result in results) == 7
    assert all(result.safe_for_completion_reasoning for result in results)
    evidence_ids = {
        result.single_flight_collapse_evidence.evidence_id
        for result in results
        if result.single_flight_collapse_evidence is not None
    }
    assert len(evidence_ids) == 1
    assert all(
        result.operational_evidence_claim_references
        == (SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID,)
        for result in results
    )
    assert all(
        result.authoritative_evidence_claim_references == ()
        for result in results
    )
    assert all(
        SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID
        in result.all_evidence_claim_references
        for result in results
    )
    assert all(
        result.to_dict()["operational_evidence_claim_references"]
        == [SINGLE_FLIGHT_COLLAPSE_REQUIREMENT_ID]
        for result in results
    )

    produced = next(
        result
        for result in results
        if result.cache_status is PipelineCacheStatus.PRODUCED
    )
    assert produced.single_flight_collapse_evidence is not None
    with pytest.raises(AnalysisBindingError, match="dropped"):
        replace(produced, single_flight_collapse_evidence=None)
    detached = replace(
        produced.single_flight_collapse_evidence,
        receipt_id="analysis-packet:sha256:" + ("0" * 64),
    )
    with pytest.raises(AnalysisBindingError, match="detached"):
        replace(produced, single_flight_collapse_evidence=detached)


def test_retrieval_is_integrated_and_bounded_before_local_analysis(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)

    result = pipeline.analyze(_request())

    assert result.retrieval_response_id
    context = analyzer.contexts[0]
    assert context.retrieval.results
    assert context.retrieval.results[0].evidence_id
    assert context.retrieval.truncation.returned_count == 1


def test_incremental_ast_index_is_projected_into_live_retrieval(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    ast_record = build_python_ast_blob_record(
        """class AnalysisCache:
    def lookup(self, key):
        return key
""",
        blob_identity="blob:analysis-cache-v1",
    )
    request = _request(
        query={
            "text": "AnalysisCache lookup",
            "objective_terms": ["AnalysisCache", "lookup"],
        },
        ast_records=(("src/analysis_cache.py", ast_record),),
        retrieval_inputs={},
    )

    result = pipeline.analyze(request)
    context = analyzer.contexts[0]

    assert context.ast_index is not None
    assert result.ast_index_id == context.ast_index.index_id
    assert context.retrieval.results
    assert any(
        item.path == "src/analysis_cache.py"
        for item in context.retrieval.results
    )


def test_optional_provider_degrades_explicitly_without_poisoning_local_result(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    imports = []

    def unavailable(name):
        imports.append(name)
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsAnalysisProvider(importer=unavailable)
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path), analyzer, provider=provider
    )
    assert imports == []

    result = pipeline.analyze(_request())

    assert imports == ["ipfs_datasets_py"]
    assert result.safe_for_completion_reasoning
    assert result.provider_result.status is AnalysisProviderStatus.UNAVAILABLE
    assert result.provider_request is not None
    assert result.provider_policy is not None
    assert result.provider_result.proved_requirement_ids_for(
        result.provider_request, result.provider_policy
    ) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.advisory_evidence_claim_references == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.authoritative_evidence_claim_references == ()
    assert result.evidence_claim_references == ()
    assert result.all_evidence_claim_references == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.to_dict()["advisory_evidence_claim_references"] == [
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID
    ]
    assert not result.provider_result.safe_for_completion_reasoning
    assert analyzer.contexts[0].provider_result is result.provider_result


def test_native_provider_operation_alias_is_validated_canonically(
    tmp_path: Path,
) -> None:
    provider = IpfsDatasetsAnalysisProvider(
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))
    )
    result = AnalysisPipeline(
        AnalysisCache(tmp_path),
        _Analyzer(),
        provider=provider,
    ).analyze(_request(provider_operation="retrieve"))

    assert result.safe_for_completion_reasoning
    assert result.provider_result.operation.value == "graph_retrieval"
    assert result.provider_request is not None
    assert result.provider_request.operation.value == "graph_retrieval"
    assert result.advisory_evidence_claim_references == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.evidence_claim_references == ()


def test_raising_optional_adapter_is_projected_and_local_analysis_continues(
    tmp_path: Path,
) -> None:
    class BrokenProvider:
        def analyze(self, *args, **kwargs):
            raise RuntimeError("optional backend is broken")

    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path), analyzer, provider=BrokenProvider()
    )

    result = pipeline.analyze(_request())

    assert result.safe_for_completion_reasoning
    assert result.provider_result.status == "failed"
    assert (
        result.provider_result.reason_code
        == "optional_provider_invocation_failed"
    )
    assert result.advisory_evidence_claim_references == ()
    assert result.evidence_claim_references == ()
    assert result.all_evidence_claim_references == ()
    assert not result.provider_result.safe_for_completion_reasoning


@pytest.mark.parametrize(
    "provider_result",
    [
        {
            "repository_id": "repo:other",
            "tree_id": "tree:sha256:111",
            "objective_revision": "objective@1",
            "operation": "graph_retrieval",
            "safe_for_completion_reasoning": False,
        },
        {
            "repository_id": "repo:fixture",
            "tree_id": "tree:sha256:111",
            "objective_revision": "objective@1",
            "operation": "graph_retrieval",
            "completion_authority": True,
        },
    ],
)
def test_optional_provider_rebinding_or_authority_claim_is_rejected_advisory(
    tmp_path: Path,
    provider_result,
) -> None:
    class UntrustedProvider:
        def analyze(self, *args, **kwargs):
            return provider_result

    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path),
        analyzer,
        provider=UntrustedProvider(),
    )

    result = pipeline.analyze(_request())

    assert result.safe_for_completion_reasoning
    assert result.provider_result.status == "failed"
    assert result.provider_result.reason_code in {
        "optional_provider_identity_mismatch",
        "optional_provider_authority_claim_rejected",
    }
    assert result.advisory_evidence_claim_references == ()
    assert result.evidence_claim_references == ()
    assert result.all_evidence_claim_references == ()
    assert not result.provider_result.safe_for_completion_reasoning
    assert analyzer.contexts[0].provider_result is result.provider_result


def test_only_exact_request_and_policy_bound_adapter_evidence_is_projected(
    tmp_path: Path,
) -> None:
    """A typed result alone does not prove the integrated offload boundary."""

    imports = []

    def unavailable(name):
        imports.append(name)
        raise ModuleNotFoundError(name)

    real_provider = IpfsDatasetsAnalysisProvider(importer=unavailable)
    request = _request()
    detached_request = real_provider.build_request(
        request.query,
        operation=request.provider_operation,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_revision=request.objective_revision,
        payload=request.provider_payload,
        limits=AnalysisPipelinePolicy().retrieval_limits,
    )
    detached = real_provider.analyze(detached_request)
    assert detached.proved_requirement_ids_for(
        detached_request, real_provider.policy
    ) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )

    class DetachedProvider:
        def analyze(self, *args, **kwargs):
            return detached

    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path),
        analyzer,
        provider=DetachedProvider(),
    )
    result = pipeline.analyze(request)

    assert result.safe_for_completion_reasoning
    assert result.provider_result is detached
    assert result.advisory_evidence_claim_references == ()
    assert result.evidence_claim_references == ()
    assert imports == ["ipfs_datasets_py"]


def test_advisory_provider_claim_cannot_be_detached_from_typed_result(
    tmp_path: Path,
) -> None:
    provider = IpfsDatasetsAnalysisProvider(
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))
    )
    result = AnalysisPipeline(
        AnalysisCache(tmp_path),
        _Analyzer(),
        provider=provider,
    ).analyze(_request())

    with pytest.raises(AnalysisBindingError, match="active-request bound"):
        replace(
            result,
            advisory_evidence_claim_references=("unproved-requirement",),
        )


def test_advisory_provider_claim_cannot_be_replayed_from_another_query(
    tmp_path: Path,
) -> None:
    provider = IpfsDatasetsAnalysisProvider(
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))
    )
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path),
        _Analyzer(),
        provider=provider,
    )
    result = pipeline.analyze(_request())
    detached_request = provider.build_request(
        {"text": "different provider query"},
        operation=result.request.provider_operation,
        repository_id=result.request.repository_id,
        tree_id=result.request.tree_id,
        objective_revision=result.request.objective_revision,
        payload=result.request.provider_payload,
        limits=pipeline.policy.retrieval_limits,
    )
    detached_result = provider.analyze(detached_request)
    assert detached_result.proved_requirement_ids_for(
        detached_request, provider.policy
    ) == (IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,)

    with pytest.raises(AnalysisBindingError, match="detached"):
        replace(
            result,
            provider_result=detached_result,
            provider_request=detached_request,
        )


@pytest.mark.parametrize(
    ("provider_factory", "expected_reason"),
    [
        (
            lambda: type(
                "MissingBindingsProvider",
                (),
                {"analyze": lambda self, *args, **kwargs: {}},
            )(),
            "optional_provider_identity_missing",
        ),
        (
            lambda: type(
                "MismatchedOperationProvider",
                (),
                {
                    "analyze": lambda self, *args, **kwargs: {
                        "repository_id": "repo:fixture",
                        "tree_id": "tree:sha256:111",
                        "objective_revision": "objective@1",
                        "operation": "dataset_query",
                    }
                },
            )(),
            "optional_provider_identity_mismatch",
        ),
    ],
)
def test_malformed_optional_provider_identity_degrades_locally(
    tmp_path: Path,
    provider_factory,
    expected_reason: str,
) -> None:
    result = AnalysisPipeline(
        AnalysisCache(tmp_path),
        _Analyzer(),
        provider=provider_factory(),
    ).analyze(_request())

    assert result.safe_for_completion_reasoning
    assert result.provider_result.reason_code == expected_reason
    assert result.advisory_evidence_claim_references == ()


def test_hostile_optional_provider_inspection_never_aborts_local_analysis(
    tmp_path: Path,
) -> None:
    class HostileResult:
        def to_dict(self):
            return {
                "repository_id": "repo:fixture",
                "tree_id": "tree:sha256:111",
                "objective_revision": "objective@1",
                "operation": "graph_retrieval",
            }

        @property
        def safe_for_completion_reasoning(self):
            raise RuntimeError("hostile property")

    class HostileProvider:
        def analyze(self, *args, **kwargs):
            return HostileResult()

    result = AnalysisPipeline(
        AnalysisCache(tmp_path / "property"),
        _Analyzer(),
        provider=HostileProvider(),
    ).analyze(_request())

    class RaisingSerializationResult:
        def to_dict(self):
            raise RuntimeError("hostile serialization")

    class RaisingSerializationProvider:
        def analyze(self, *args, **kwargs):
            return RaisingSerializationResult()

    serialization_result = AnalysisPipeline(
        AnalysisCache(tmp_path / "serialization"),
        _Analyzer(),
        provider=RaisingSerializationProvider(),
    ).analyze(_request())

    assert result.safe_for_completion_reasoning
    assert (
        result.provider_result.reason_code
        == "optional_provider_inspection_failed"
    )
    assert serialization_result.safe_for_completion_reasoning
    assert (
        serialization_result.provider_result.reason_code
        == "optional_provider_inspection_failed"
    )
    assert result.advisory_evidence_claim_references == ()
    assert serialization_result.advisory_evidence_claim_references == ()


def test_callable_authority_claim_and_awaitable_result_degrade_locally(
    tmp_path: Path,
) -> None:
    class CallableAuthorityResult:
        def to_dict(self):
            return {
                "repository_id": "repo:fixture",
                "tree_id": "tree:sha256:111",
                "objective_revision": "objective@1",
                "operation": "graph_retrieval",
            }

        def safe_for_completion_reasoning(self):
            return True

    class CallableAuthorityProvider:
        def analyze(self, *args, **kwargs):
            return CallableAuthorityResult()

    authority = AnalysisPipeline(
        AnalysisCache(tmp_path / "authority"),
        _Analyzer(),
        provider=CallableAuthorityProvider(),
    ).analyze(_request())

    class AsyncProvider:
        async def analyze(self, *args, **kwargs):
            return None

    awaitable = AnalysisPipeline(
        AnalysisCache(tmp_path / "awaitable"),
        _Analyzer(),
        provider=AsyncProvider(),
    ).analyze(_request())

    assert (
        authority.provider_result.reason_code
        == "optional_provider_authority_claim_rejected"
    )
    assert (
        awaitable.provider_result.reason_code
        == "optional_provider_async_result_unsupported"
    )
    assert authority.safe_for_completion_reasoning
    assert awaitable.safe_for_completion_reasoning
    assert authority.advisory_evidence_claim_references == ()
    assert awaitable.advisory_evidence_claim_references == ()


def test_restart_preserves_ranked_retrieval_health_and_truncation_semantics(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "restart-cache"
    request = _request(
        ast_records=(
            (
                "src/cache.py",
                build_python_ast_blob_record(
                    "class RestartCache:\n"
                    "    def lookup(self):\n"
                    "        return 'warm'\n"
                ),
            ),
        )
    )
    first_analyzer = _Analyzer()
    cold = AnalysisPipeline(
        AnalysisCache(cache_path), first_analyzer
    ).analyze(request)

    restarted_analyzer = _Analyzer()
    warm = AnalysisPipeline(
        AnalysisCache(cache_path), restarted_analyzer
    ).analyze(request)

    assert cold.cache_status is PipelineCacheStatus.PRODUCED
    assert warm.cache_status is PipelineCacheStatus.EXACT_HIT
    assert warm.packet.packet_id == cold.packet.packet_id
    assert warm.ast_index_id == cold.ast_index_id
    assert warm.ast_index_id
    assert warm.retrieval_response_id == cold.retrieval_response_id
    assert warm.evidence_references == cold.evidence_references
    assert warm.backend_health == cold.backend_health
    assert warm.truncation == cold.truncation
    assert warm.truncation["returned_count"] == len(
        {item["evidence_id"] for item in warm.evidence_references}
    )
    assert first_analyzer.calls == 1
    assert restarted_analyzer.calls == 0


def test_g094_completion_bridge_requires_every_fresh_current_tree_proof(
    tmp_path: Path,
) -> None:
    """G094 cannot narrow or self-supply its mandatory completion evidence."""

    request = _request(
        ast_records=(
            (
                "src/integrated.py",
                build_python_ast_blob_record(
                    "class IntegratedAnalysis:\n"
                    "    def retrieve(self):\n"
                    "        return 'cache ast vector graph'\n"
                ),
            ),
        ),
        retrieval_inputs={
            "records": [
                {
                    "record_id": "record-cache",
                    "title": "bounded cache analysis",
                    "task_id": "ASI-065",
                }
            ],
            "todo_records": [
                {
                    "task_id": "ASI-065",
                    "title": "integrated AST retrieval",
                    "status": "active",
                }
            ],
        },
    )
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path / "cache"),
        analyzer,
        provider=IpfsDatasetsAnalysisProvider(
            importer=lambda name: (_ for _ in ()).throw(
                ModuleNotFoundError(name)
            )
        ),
    )
    cold = pipeline.analyze(request)
    result = pipeline.analyze(request)

    assert cold.cache_status is PipelineCacheStatus.PRODUCED
    assert cold.ast_index_id
    assert cold.retrieval_response_id
    assert cold.provider_result is not None
    assert result.cache_status is PipelineCacheStatus.EXACT_HIT
    assert result.evidence_claim_references == (
        EXACT_TREE_REUSE_REQUIREMENT_ID,
    )
    assert analyzer.calls == 1

    now = datetime(2026, 7, 24, 14, 0, tzinfo=timezone.utc)
    validation_command = (
        "python -m pytest "
        "test/api/test_agent_supervisor_analysis_pipeline.py "
        "test/api/test_agent_supervisor_ipfs_datasets_analysis_provider.py "
        "test/api/test_agent_supervisor_cache_coordinator.py -q"
    )
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-065",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": result.request.tree_id,
                "command": validation_command,
            },
            validation_passed=True,
            repository_id=result.request.repository_id,
            repository_tree=result.request.tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-065:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            EXACT_TREE_REUSE_ACCEPTANCE_CRITERIA, start=1
        )
    )
    coverage = {
        "repository_tree": result.request.tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "analysis_pipeline.py"
                    if index != 5
                    else (
                        "ipfs_accelerate_py/agent_supervisor/"
                        "cache_coordinator.py"
                    )
                ),
                "validation": (
                    "test/api/test_agent_supervisor_"
                    "analysis_pipeline.py"
                    if index != 5
                    else (
                        "test/api/test_agent_supervisor_"
                        "cache_coordinator.py"
                    )
                ),
            }
            for index, criterion in enumerate(
                EXACT_TREE_REUSE_ACCEPTANCE_CRITERIA, start=1
            )
        ],
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "analyzer_version": result.request.analyzer_version,
    }
    binding = {
        "repository_id": result.request.repository_id,
        "tree_id": result.request.tree_id,
        "analyzer_version": result.request.analyzer_version,
        "configuration_revision": result.request.configuration_digest,
        "objective_revision": result.request.objective_revision,
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-065-exhaustive-a",
                "evidence_channel": "ast-and-runtime",
                "receipt_cid": "scan:asi-065:exhaustive-a",
                "binding": binding,
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-065-exhaustive-b",
                "evidence_channel": "cache-and-validation",
                "receipt_cid": "scan:asi-065:exhaustive-b",
                "binding": binding,
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    provisional = result.evaluate_exact_tree_reuse_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert not provisional.verified
    assert provisional.acceptance_criteria == (
        EXACT_TREE_REUSE_ACCEPTANCE_CRITERIA
    )
    assert provisional.gate is not None and provisional.gate.passed
    assert provisional.gate.evaluated_evidence["analysis_result"] == {}
    assert "provisional_transition_required" in provisional.reason_codes

    verified = result.evaluate_exact_tree_reuse_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-065:failed",
        validation_passed=False,
        validation_receipt={
            "status": "failed",
            "tree_id": result.request.tree_id,
            "command": validation_command,
        },
    )
    failed_validation = result.evaluate_exact_tree_reuse_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (*evidence, failed)},
    )
    assert not failed_validation.verified
    assert "failed_validation" in failed_validation.reason_codes

    missing = result.evaluate_exact_tree_reuse_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": evidence[:-1]},
    )
    assert not missing.verified
    assert EXACT_TREE_REUSE_ACCEPTANCE_CRITERIA[-1] in missing.missing_criteria

    missing_binding = {
        **coverage,
        "criteria": [
            *coverage["criteria"][:-1],
            {**coverage["criteria"][-1], "validation": ""},
        ],
    }
    unmapped = result.evaluate_exact_tree_reuse_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "coverage": missing_binding},
    )
    assert not unmapped.verified
    assert "coverage_unverified" in unmapped.reason_codes

    for invalid_health in (
        {"status": "healthy"},
        {**health, "safe_for_completion_reasoning": False},
        {**health, "healthy": False},
        {**health, "analyzer_version": "fixture-analyzer@foreign"},
    ):
        unhealthy = result.evaluate_exact_tree_reuse_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "analyzer_health": invalid_health},
        )
        assert not unhealthy.verified
        assert any(
            code in unhealthy.reason_codes
            for code in ("analyzer_unhealthy", "analyzer_completion_unsafe")
        )

    invalid_quorums = (
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "evidence_channel": "ast-and-runtime",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "receipt_cid": "scan:asi-065:exhaustive-a",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "scan_mode": "audit"},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {**quorum["members"][1], "healthy": False},
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "finished_at": "2026-07-24T12:00:00+00:00",
                },
            ],
        },
        {
            **quorum,
            "members": [
                quorum["members"][0],
                {
                    **quorum["members"][1],
                    "binding": {**binding, "tree_id": "tree:sha256:foreign"},
                },
            ],
        },
        {
            **quorum,
            "binding": {
                **binding,
                "configuration_revision": "sha256:foreign",
            },
        },
    )
    for invalid_quorum in invalid_quorums:
        no_quorum = result.evaluate_exact_tree_reuse_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, "exhaustion_quorum": invalid_quorum},
        )
        assert not no_quorum.verified
        assert any(
            code.startswith("exhaustion_quorum")
            for code in no_quorum.reason_codes
        )

    foreign_tree = replace(
        evidence[0],
        repository_tree="tree:sha256:foreign",
        tree_id="tree:sha256:foreign",
        provenance_cid="validation:asi-065:foreign",
    )
    detached = result.evaluate_exact_tree_reuse_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (foreign_tree, *evidence[1:])},
    )
    assert not detached.verified
    assert "repository_tree_mismatch" in detached.reason_codes


def test_g095_completion_bridge_requires_fresh_complete_current_tree_proof(
    tmp_path: Path,
) -> None:
    """The analysis/provider result cannot bypass the objective proof gate."""

    result = AnalysisPipeline(
        AnalysisCache(tmp_path / "cache"),
        _Analyzer(),
        provider=IpfsDatasetsAnalysisProvider(
            importer=lambda name: (_ for _ in ()).throw(
                ModuleNotFoundError(name)
            )
        ),
    ).analyze(_request())
    now = datetime(2026, 7, 24, 14, 0, tzinfo=timezone.utc)
    evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-066",
            producer_kind="task",
            validation_receipt={
                "status": "passed",
                "tree_id": result.request.tree_id,
                "command": (
                    "python -m pytest "
                    "test/api/test_agent_supervisor_analysis_pipeline.py "
                    "test/api/test_agent_supervisor_"
                    "ipfs_datasets_analysis_provider.py "
                    "test/api/test_agent_supervisor_cache_coordinator.py -q"
                ),
            },
            validation_passed=True,
            repository_id=result.request.repository_id,
            repository_tree=result.request.tree_id,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-066:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(G095_ACCEPTANCE_CRITERIA, start=1)
    )
    coverage = {
        "repository_tree": result.request.tree_id,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "ipfs_datasets_analysis_provider.py"
                ),
                "validation": (
                    "test/api/test_agent_supervisor_"
                    "ipfs_datasets_analysis_provider.py"
                ),
            }
            for criterion in G095_ACCEPTANCE_CRITERIA
        ],
    }
    health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "analyzer_version": result.request.analyzer_version,
    }
    binding = {
        "repository_id": result.request.repository_id,
        "tree_id": result.request.tree_id,
        "analyzer_version": result.request.analyzer_version,
        "configuration_revision": result.request.configuration_digest,
        "objective_revision": result.request.objective_revision,
    }
    quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": binding,
        "members": [
            {
                "member_id": "asi-066-exhaustive",
                "evidence_channel": "exhaustive",
                "receipt_cid": "scan:asi-066:exhaustive",
                "binding": binding,
                "scan_mode": "exhaustive",
                "finished_at": now.isoformat(),
            },
            {
                "member_id": "asi-066-audit",
                "evidence_channel": "audit",
                "receipt_cid": "scan:asi-066:audit",
                "binding": binding,
                "scan_mode": "audit",
                "finished_at": now.isoformat(),
            },
        ],
    }
    values = {
        "acceptance_criteria": G095_ACCEPTANCE_CRITERIA,
        "evidence": evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    # Even a fully passing gate cannot jump directly from active to verified.
    provisional = result.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.verified is False
    assert provisional.gate is not None and provisional.gate.passed
    assert "provisional_transition_required" in provisional.reason_codes
    assert provisional.gate.evaluated_evidence["analysis_result"] == {}

    verified = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified is True
    assert verified.gate is not None and verified.gate.passed

    # Every submitted validation is authoritative: a failed extra receipt
    # cannot be hidden by the seven valid records.
    failed = replace(
        evidence[0],
        provenance_cid="validation:asi-066:failed",
        validation_passed=False,
        validation_receipt={
            "status": "failed",
            "tree_id": result.request.tree_id,
        },
    )
    invalid_validation = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (*evidence, failed)},
    )
    assert invalid_validation.state is GoalState.PROVISIONALLY_COMPLETE
    assert invalid_validation.verified is False
    assert invalid_validation.gate is not None
    assert "failed_validation" in invalid_validation.reason_codes
    assert (
        "validation_evidence_incomplete"
        in invalid_validation.gate.fail_reason_codes
    )

    missing_criterion = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": evidence[:-1]},
    )
    assert missing_criterion.state is GoalState.PROVISIONALLY_COMPLETE
    assert G095_ACCEPTANCE_CRITERIA[-1] in missing_criterion.missing_criteria
    assert "validation_evidence_incomplete" in (
        missing_criterion.gate.fail_reason_codes
    )

    unsafe_health = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "analyzer_health": {
                **health,
                "safe_for_completion_reasoning": False,
            },
        },
    )
    assert unsafe_health.state is GoalState.PROVISIONALLY_COMPLETE
    assert "analyzer_completion_unsafe" in unsafe_health.reason_codes

    duplicate_quorum = {
        **quorum,
        "members": [
            quorum["members"][0],
            {
                **quorum["members"][1],
                "evidence_channel": "exhaustive",
            },
        ],
    }
    insufficient_independence = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "exhaustion_quorum": duplicate_quorum},
    )
    assert insufficient_independence.state is GoalState.PROVISIONALLY_COMPLETE
    assert "exhaustion_quorum_inconsistent" in (
        insufficient_independence.reason_codes
    )

    foreign_tree = replace(
        evidence[0],
        repository_tree="tree:sha256:foreign",
        tree_id="tree:sha256:foreign",
        provenance_cid="validation:asi-066:foreign",
    )
    wrong_tree = result.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "evidence": (foreign_tree, *evidence[1:])},
    )
    assert wrong_tree.state is GoalState.PROVISIONALLY_COMPLETE
    assert "repository_tree_mismatch" in wrong_tree.reason_codes


def test_low_backlog_analysis_uses_pipeline_only_as_bounded_nomination_context(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path / "cache"), analyzer)
    policy = AnalysisEscalationPolicy(
        backlog_target=1,
        max_router_calls=1,
        max_router_retries=0,
    )

    def run_once():
        return run_low_backlog_analysis(
            tmp_path,
            healthy_backlog_count=0,
            objective_terms=["cache ast retrieval"],
            policy=policy,
            incremental_scanner=lambda: [],
            ast_scanner=lambda: {
                "healthy": True,
                "complete": True,
                "candidates": [{"candidate": "ast-backed proposal"}],
            },
            analysis_pipeline=pipeline,
        )

    cold = run_once()
    warm = run_once()

    assert cold.pipeline_result["cache_status"] == "produced"
    assert warm.pipeline_result["cache_status"] == "exact_hit"
    assert warm.pipeline_result["cache_lookup_status"] == "hit"
    assert warm.pipeline_result["retrieval_response_id"] == cold.pipeline_result[
        "retrieval_response_id"
    ]
    assert warm.pipeline_result["retrieval_backend_health"] == cold.pipeline_result[
        "retrieval_backend_health"
    ]
    assert warm.pipeline_result["retrieval_truncation"] == cold.pipeline_result[
        "retrieval_truncation"
    ]
    assert warm.pipeline_result["nomination_only"] is True
    assert warm.pipeline_result["safe_for_completion_reasoning"] is False
    assert warm.safe_for_completion_reasoning is False
    assert analyzer.calls == 1
