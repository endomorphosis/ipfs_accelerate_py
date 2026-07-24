from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.ipfs_datasets_analysis_provider import (
    IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    IPFS_DATASETS_OFFLOAD_COORDINATION_BOUNDARY,
    MAX_CONCURRENT_PROVIDER_DISPATCHES,
    PROVIDER_REQUEST_SCHEMA,
    PROVIDER_RESULT_SCHEMA,
    AnalysisProviderBounds,
    AnalysisProviderCapability,
    AnalysisProviderHealth,
    AnalysisProviderOperation,
    AnalysisProviderPolicy,
    AnalysisProviderRequest,
    AnalysisProviderResult,
    AnalysisProviderStatus,
    IpfsDatasetsAnalysisProviderError,
    IpfsDatasetsProviderDegradationEvidence,
    IpfsDatasetsAnalysisProvider,
    inspect_analysis_provider_capability,
    normalize_analysis_provider_operation,
)
from ipfs_accelerate_py.agent_supervisor.analysis_retrieval import RetrievalLimits


def _request(**changes):
    values = {
        "operation": "graph_retrieval",
        "repository_id": "repo:fixture",
        "tree_id": "tree:sha256:111",
        "objective_revision": "objective@1",
        "query": {"text": "cache authority"},
    }
    values.update(changes)
    return values


def test_construction_and_capability_declaration_do_not_import() -> None:
    calls = []
    backend_calls = []

    def importer(name):
        calls.append(name)
        raise AssertionError("capability declaration imported optional code")

    class ExplosiveBackend:
        def capabilities(self):
            backend_calls.append("capabilities")
            raise AssertionError("local inspection probed optional backend")

    provider = IpfsDatasetsAnalysisProvider(
        importer=importer,
        backend=ExplosiveBackend(),
    )
    state_before = {name: id(value) for name, value in vars(provider).items()}
    dispatch_threads_before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "ipfs-datasets-analysis-provider"
    }
    with ThreadPoolExecutor(max_workers=8) as executor:
        capabilities = tuple(
            executor.map(lambda _: provider.capabilities(), range(32))
        )
    first = capabilities[0]
    second = provider.capability()

    assert all(item == first for item in capabilities)
    assert all(item.to_dict() == first.to_dict() for item in capabilities)
    assert all(item.capability_id == first.capability_id for item in capabilities)
    assert calls == []
    assert backend_calls == []
    assert {name: id(value) for name, value in vars(provider).items()} == state_before
    assert {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "ipfs-datasets-analysis-provider"
    } == dispatch_threads_before
    assert first == second
    assert first.health is AnalysisProviderHealth.LAZY
    assert first.imported is False
    assert first.non_authoritative
    assert AnalysisProviderCapability.from_dict(first.to_dict()) == first

    # The standalone inspection surface depends only on canonical policy
    # metadata. Operation order and duplicate inputs cannot change its ID.
    pure = inspect_analysis_provider_capability(
        {
            "operations": [
                "dataset_query",
                "graph_retrieval",
                "dataset_query",
            ]
        }
    )
    reordered = inspect_analysis_provider_capability(
        {
            "operations": [
                "graph_retrieval",
                "dataset_query",
            ]
        }
    )
    assert pure.to_dict() == reordered.to_dict()
    assert pure.capability_id == reordered.capability_id


def test_missing_optional_module_degrades_explicitly_with_typed_evidence() -> None:
    calls = []

    def importer(name):
        calls.append(name)
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsAnalysisProvider(importer=importer)
    request = provider.build_request(_request())
    result = provider.analyze(request)

    assert calls == ["ipfs_datasets_py"]
    assert result.status is AnalysisProviderStatus.UNAVAILABLE
    assert result.reason_code == "optional_module_unavailable"
    assert result.backend_health is AnalysisProviderHealth.UNAVAILABLE
    assert result.degraded
    assert not result.safe_for_completion_reasoning
    assert result.proved_requirement_ids == ()
    assert result.diagnostic_requirement_ids == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.proved_requirement_ids_for(request, provider.policy) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.degradation_evidence is not None
    assert result.degradation_evidence.import_attempted
    assert result.degradation_evidence.fallback == "local_deterministic_analysis"
    assert result.degradation_evidence.request_id == result.request_id
    assert result.degradation_evidence.repository_id == result.repository_id
    assert result.degradation_evidence.tree_id == result.tree_id
    assert (
        result.degradation_evidence.objective_revision
        == result.objective_revision
    )
    assert result.degradation_evidence.request_bound
    assert result.degradation_evidence.proof_bound
    assert result.degradation_evidence.policy_id == provider.policy.policy_id
    assert result.degradation_evidence.proves_for(
        request, provider.policy
    )
    assert not result.degradation_evidence.proves_for(
        request,
        AnalysisProviderPolicy(
            bounds=AnalysisProviderBounds(max_results=1)
        ),
    )
    assert result.degradation_evidence.evidence_id


def test_unsupported_operation_never_loads_backend() -> None:
    calls = []
    policy = AnalysisProviderPolicy(
        operations=(AnalysisProviderOperation.GRAPH_RETRIEVAL,)
    )
    provider = IpfsDatasetsAnalysisProvider(
        policy,
        importer=lambda name: calls.append(name),
    )

    request = provider.build_request(_request(operation="dataset_query"))
    result = provider.analyze(request)

    assert calls == []
    assert result.status is AnalysisProviderStatus.UNSUPPORTED
    assert result.reason_code == "operation_not_allowlisted"
    assert result.proved_requirement_ids == ()
    assert result.proved_requirement_ids_for(request, policy) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert result.degradation_evidence is not None
    assert not result.degradation_evidence.import_attempted


class _Backend:
    __version__ = "fixture@1"

    def __init__(self):
        self.requests = []

    def capabilities(self):
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
            "authoritative": True,
            "results": [
                {
                    "evidence_id": "evidence-1",
                    "path": "src/cache.py",
                    "symbol": "AnalysisCache.lookup",
                    "score": 0.75,
                }
            ],
            "provenance": [{"artifact_id": "artifact-1", "kind": "ast"}],
        }


def test_supported_backend_receives_bounds_and_remains_advisory() -> None:
    backend = _Backend()
    provider = IpfsDatasetsAnalysisProvider(backend=backend)

    result = provider.analyze(
        {"text": "cache authority"},
        operation="graph_retrieval",
        repository_id="repo:fixture",
        tree_id="tree:sha256:111",
        objective_revision="objective@1",
        payload={},
        limits=RetrievalLimits(max_results=2, max_bytes=8_192),
    )

    assert result.status is AnalysisProviderStatus.COMPLETED
    # A backend cannot smuggle an authority claim through the response.
    assert result.reason_code == "bounded_provider_result"
    assert not result.safe_for_completion_reasoning
    assert not result.is_completion_evidence
    request = backend.requests[0]
    assert request["bounds"]["max_results"] == 2
    assert request["bounds"]["max_response_bytes"] == 8_192


def test_compact_success_is_deterministic_bounded_and_non_authoritative() -> None:
    backend = _Backend()

    def clean(request):
        backend.requests.append(request)
        return {
            "status": "completed",
            "results": [
                {
                    "evidence_id": "evidence-1",
                    "path": "src/cache.py",
                    "symbol": "AnalysisCache.lookup",
                    "score": 0.75,
                }
            ],
            "provenance": [{"artifact_id": "artifact-1", "kind": "ast"}],
        }

    backend.retrieve = clean
    provider = IpfsDatasetsAnalysisProvider(backend=backend)
    first = provider.analyze(_request())
    second = provider.analyze(_request())

    assert first.status is AnalysisProviderStatus.COMPLETED
    assert first.backend_health is AnalysisProviderHealth.HEALTHY
    assert first.evidence_references[0]["score_millionths"] == 750_000
    assert first.result_id == second.result_id
    assert not first.safe_for_completion_reasoning
    assert first.proved_requirement_ids == ()


def test_direct_provider_dispatch_does_not_create_a_competing_cache_boundary() -> None:
    backend = _Backend()
    provider = IpfsDatasetsAnalysisProvider(backend=backend)

    first = provider.analyze(_request())
    second = provider.analyze(_request())

    assert IPFS_DATASETS_OFFLOAD_COORDINATION_BOUNDARY == (
        "analysis_pipeline.single_flight"
    )
    assert len(backend.requests) == 2
    assert first.result_id == second.result_id
    assert first.proved_requirement_ids == ()
    assert second.proved_requirement_ids == ()


def test_heavy_or_malformed_backend_payload_fails_closed() -> None:
    class Backend(_Backend):
        def retrieve(self, request):
            return {
                "status": "completed",
                "results": [{"evidence_id": "evidence-1", "source": "secret"}],
            }

    result = IpfsDatasetsAnalysisProvider(backend=Backend()).analyze(_request())

    assert result.status is AnalysisProviderStatus.MALFORMED
    assert result.reason_code == "malformed_backend_response"
    assert result.evidence_references == ()
    assert not result.safe_for_completion_reasoning


def test_backend_exception_is_typed_and_async_facade_leaks_no_coroutine() -> None:
    class Backend(_Backend):
        async def retrieve(self, request):
            await asyncio.sleep(0)
            raise RuntimeError("fixture backend failure")

    async def scenario():
        provider = IpfsDatasetsAnalysisProvider(backend=Backend())
        return await provider.analyze_async(_request())

    result = asyncio.run(scenario())

    assert result.status is AnalysisProviderStatus.FAILED
    assert result.reason_code == "backend_execution_failed"
    assert result.degradation_evidence is not None
    assert result.proved_requirement_ids == ()
    assert not result.safe_for_completion_reasoning


def test_disabled_provider_is_explicit_without_import() -> None:
    calls = []
    provider = IpfsDatasetsAnalysisProvider(
        AnalysisProviderPolicy(enabled=False),
        importer=lambda name: calls.append(name),
    )

    request = provider.build_request(_request())
    result = provider.analyze(request)

    assert calls == []
    assert result.status is AnalysisProviderStatus.DISABLED
    assert result.reason_code == "provider_disabled"
    assert result.proved_requirement_ids == ()
    assert result.proved_requirement_ids_for(request, provider.policy) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )


def test_typed_backend_result_reenters_bounded_projection() -> None:
    class TypedBackend(_Backend):
        def retrieve(self, request):
            return AnalysisProviderResult(
                request_id=request["request_id"],
                operation=request["operation"],
                repository_id=request["repository_id"],
                tree_id=request["tree_id"],
                objective_revision=request["objective_revision"],
                status=AnalysisProviderStatus.COMPLETED,
                reason_code="backend_claim",
                evidence_references=(
                    {"evidence_id": "one"},
                    {"evidence_id": "two"},
                ),
                backend_health=AnalysisProviderHealth.HEALTHY,
            )

    result = IpfsDatasetsAnalysisProvider(
        backend=TypedBackend()
    ).analyze(_request(bounds=AnalysisProviderBounds(max_results=1)))

    assert result.status is AnalysisProviderStatus.COMPLETED
    assert len(result.evidence_references) == 1
    assert result.truncated
    assert result.reason_code == "bounded_provider_result"


def test_serialized_invariants_and_content_ids_are_fail_closed() -> None:
    capability = AnalysisProviderCapability(
        health=AnalysisProviderHealth.LAZY,
        operations=(AnalysisProviderOperation.GRAPH_RETRIEVAL,),
    )
    capability_record = capability.to_dict()
    assert AnalysisProviderCapability.from_dict(capability_record) == capability
    forged_capability = dict(capability_record)
    forged_capability["completion_authority"] = True
    with pytest.raises(IpfsDatasetsAnalysisProviderError):
        AnalysisProviderCapability.from_dict(forged_capability)

    degradation = IpfsDatasetsProviderDegradationEvidence(
        status=AnalysisProviderStatus.UNAVAILABLE,
        operation=AnalysisProviderOperation.GRAPH_RETRIEVAL,
        reason_code="missing",
        import_attempted=True,
    )
    forged_degradation = degradation.to_dict()
    forged_degradation["completion_authority"] = True
    with pytest.raises(IpfsDatasetsAnalysisProviderError):
        IpfsDatasetsProviderDegradationEvidence.from_dict(
            forged_degradation
        )

    with pytest.raises(
        IpfsDatasetsAnalysisProviderError, match="identity does not match"
    ):
        AnalysisProviderRequest(
            operation=AnalysisProviderOperation.GRAPH_RETRIEVAL,
            repository_id="repo",
            tree_id="tree",
            objective_revision="objective",
            query="query",
            request_id="forged-request-id",
        )


def test_degradation_evidence_cannot_be_detached_or_replayed() -> None:
    def unavailable(name):
        raise ModuleNotFoundError(name)

    provider = IpfsDatasetsAnalysisProvider(importer=unavailable)
    request = provider.build_request(_request())
    original = provider.analyze(request)
    other = provider.analyze(
        _request(
            tree_id="tree:sha256:222",
            objective_revision="objective@2",
        )
    )

    assert original.proved_requirement_ids == ()
    assert original.proved_requirement_ids_for(request, provider.policy) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert AnalysisProviderResult.from_dict(original.to_dict()) == original

    replayed = original.to_dict()
    replayed["degradation_evidence"] = (
        other.degradation_evidence.to_dict()
    )
    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="is not result-bound",
    ):
        AnalysisProviderResult.from_dict(replayed)

    mismatched = IpfsDatasetsProviderDegradationEvidence(
        status=AnalysisProviderStatus.FAILED,
        operation=AnalysisProviderOperation.DATASET_QUERY,
        reason_code="backend_execution_failed",
        import_attempted=True,
    )
    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="is not request/policy-bound",
    ):
        AnalysisProviderResult(
            request_id=original.request_id,
            operation=original.operation,
            repository_id=original.repository_id,
            tree_id=original.tree_id,
            objective_revision=original.objective_revision,
            status=original.status,
            reason_code=original.reason_code,
            backend_health=original.backend_health,
            degradation_evidence=mismatched,
        )


def test_self_consistent_fabricated_degradation_state_cannot_claim_requirement(
) -> None:
    request = AnalysisProviderRequest.from_value(_request())
    policy = AnalysisProviderPolicy()
    fabricated = IpfsDatasetsProviderDegradationEvidence(
        status=AnalysisProviderStatus.UNAVAILABLE,
        operation=request.operation,
        reason_code="fabricated_optional_failure",
        import_attempted=False,
        request_id=request.request_id,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_revision=request.objective_revision,
        policy_id=policy.policy_id,
        backend_health=AnalysisProviderHealth.HEALTHY,
        fallback="remote_completion",
    )
    result = AnalysisProviderResult(
        request_id=request.request_id,
        operation=request.operation,
        repository_id=request.repository_id,
        tree_id=request.tree_id,
        objective_revision=request.objective_revision,
        status=fabricated.status,
        reason_code=fabricated.reason_code,
        backend_health=fabricated.backend_health,
        degradation_evidence=fabricated,
    )

    assert fabricated.proof_bound
    assert not fabricated.proves_requirement
    assert not fabricated.proves_for(request, policy)
    assert fabricated.proved_requirement_ids == ()
    assert result.proved_requirement_ids == ()
    assert not result.safe_for_completion_reasoning


def test_requirement_witness_requires_every_semantic_and_binding_dimension(
) -> None:
    def unavailable(name):
        raise ModuleNotFoundError(name)

    request = AnalysisProviderRequest.from_value(_request())
    policy = AnalysisProviderPolicy()
    evidence = IpfsDatasetsAnalysisProvider(
        policy, importer=unavailable
    ).analyze(request).degradation_evidence

    assert evidence is not None
    assert evidence.proves_for(request, policy)
    assert evidence.proved_requirement_ids == ()
    assert evidence.diagnostic_requirement_ids == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert evidence.proved_requirement_ids_for(request, policy) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )

    semantic_mismatches = (
        replace(evidence, status=AnalysisProviderStatus.FAILED),
        replace(evidence, reason_code="fabricated_optional_failure"),
        replace(evidence, import_attempted=False),
        replace(evidence, backend_health=AnalysisProviderHealth.HEALTHY),
        replace(evidence, fallback="remote_completion"),
        replace(evidence, request_id=""),
        replace(evidence, policy_id=""),
    )
    for mismatched in semantic_mismatches:
        assert not mismatched.proves_requirement
        assert mismatched.proved_requirement_ids == ()
        assert not mismatched.proves_for(request, policy)

    binding_mismatches = (
        replace(evidence, operation=AnalysisProviderOperation.DATASET_QUERY),
        replace(evidence, repository_id="repo:other"),
        replace(evidence, tree_id="tree:sha256:222"),
        replace(evidence, objective_revision="objective@2"),
        replace(evidence, policy_id=AnalysisProviderPolicy(enabled=False).policy_id),
    )
    for mismatched in binding_mismatches:
        assert not mismatched.proves_for(request, policy)


def test_active_policy_semantics_cannot_be_forged_with_matching_policy_id(
) -> None:
    request = AnalysisProviderRequest.from_value(_request())
    enabled_policy = AnalysisProviderPolicy()
    disabled_policy = AnalysisProviderPolicy(enabled=False)
    disabled = IpfsDatasetsAnalysisProvider(disabled_policy).analyze(request)

    assert disabled.proves_requirement_for(request, disabled_policy)
    assert disabled.proved_requirement_ids_for(request, disabled_policy) == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )

    disabled_evidence = disabled.degradation_evidence
    assert disabled_evidence is not None
    forged_disabled = replace(
        disabled,
        degradation_evidence=replace(
            disabled_evidence, policy_id=enabled_policy.policy_id
        ),
    )
    assert forged_disabled.proved_requirement_ids == ()
    # The explicitly diagnostic property records a semantically shaped claim,
    # while the active-context API rejects the impossible policy relationship.
    assert forged_disabled.diagnostic_requirement_ids == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert not forged_disabled.proves_requirement_for(request, enabled_policy)
    assert forged_disabled.proved_requirement_ids_for(
        request, enabled_policy
    ) == ()

    dataset_request = AnalysisProviderRequest.from_value(
        _request(operation="dataset_query")
    )
    restricted_policy = AnalysisProviderPolicy(
        operations=(AnalysisProviderOperation.GRAPH_RETRIEVAL,)
    )
    rejected = IpfsDatasetsAnalysisProvider(restricted_policy).analyze(
        dataset_request
    )
    assert rejected.proves_requirement_for(dataset_request, restricted_policy)

    rejected_evidence = rejected.degradation_evidence
    assert rejected_evidence is not None
    forged_allowlist = replace(
        rejected,
        degradation_evidence=replace(
            rejected_evidence, policy_id=enabled_policy.policy_id
        ),
    )
    assert forged_allowlist.proved_requirement_ids == ()
    assert forged_allowlist.diagnostic_requirement_ids == (
        IPFS_DATASETS_LAZY_DEGRADATION_REQUIREMENT_ID,
    )
    assert not forged_allowlist.proves_requirement_for(
        dataset_request, enabled_policy
    )


def test_capability_dependency_failure_records_actual_adapter_import_history(
) -> None:
    class MissingCapabilityDependency:
        def capabilities(self):
            raise ModuleNotFoundError("optional capability dependency")

    request = AnalysisProviderRequest.from_value(_request())
    policy = AnalysisProviderPolicy()
    injected = IpfsDatasetsAnalysisProvider(
        policy, backend=MissingCapabilityDependency()
    ).analyze(request)
    imported = IpfsDatasetsAnalysisProvider(
        policy, importer=lambda name: MissingCapabilityDependency()
    ).analyze(request)

    for result, import_attempted in ((injected, False), (imported, True)):
        assert result.status is AnalysisProviderStatus.UNAVAILABLE
        assert result.reason_code == "optional_capability_unavailable"
        assert result.degradation_evidence is not None
        assert result.degradation_evidence.import_attempted is import_attempted
    assert injected.proved_requirement_ids == ()
    assert not injected.proves_requirement_for(request, policy)
    assert imported.proves_requirement_for(request, policy)


def test_public_request_builder_applies_limits_without_loading_backend() -> None:
    imports = []
    provider = IpfsDatasetsAnalysisProvider(
        importer=lambda name: imports.append(name)
    )

    request = provider.build_request(
        {"text": "cache authority"},
        operation="retrieve",
        repository_id="repo:fixture",
        tree_id="tree:sha256:111",
        objective_revision="objective@1",
        payload={},
        limits=RetrievalLimits(max_results=2, max_bytes=8_192),
    )

    assert imports == []
    assert normalize_analysis_provider_operation(
        "retrieve"
    ) is AnalysisProviderOperation.GRAPH_RETRIEVAL
    assert request.operation is AnalysisProviderOperation.GRAPH_RETRIEVAL
    assert request.bounds.max_results == 2
    assert request.bounds.max_response_bytes == 8_192

    bounded = IpfsDatasetsAnalysisProvider(
        bounds=AnalysisProviderBounds(max_results=2)
    )
    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="cannot expand provider policy",
    ):
        bounded.build_request(
            {
                **_request(),
                "bounds": AnalysisProviderBounds(max_results=3).to_dict(),
            }
        )


@pytest.mark.parametrize(
    ("backend", "status", "reason_code", "health"),
    [
        (
            type(
                "IncompatibleBackend",
                (_Backend,),
                {
                    "capabilities": lambda self: {
                        "protocol_versions": [999],
                        "available": True,
                        "operations": ["graph_retrieval"],
                    }
                },
            )(),
            AnalysisProviderStatus.UNSUPPORTED,
            "protocol_incompatible",
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
        (
            type(
                "UnhealthyBackend",
                (_Backend,),
                {
                    "capabilities": lambda self: {
                        "protocol_versions": [1],
                        "available": False,
                        "operations": ["graph_retrieval"],
                    }
                },
            )(),
            AnalysisProviderStatus.UNSUPPORTED,
            "backend_unhealthy",
            AnalysisProviderHealth.DEGRADED,
        ),
        (
            type(
                "MalformedCapabilityBackend",
                (_Backend,),
                {"capabilities": lambda self: {"available": "false"}},
            )(),
            AnalysisProviderStatus.MALFORMED,
            "malformed_capability",
            AnalysisProviderHealth.INCOMPATIBLE,
        ),
    ],
)
def test_capability_degradation_is_typed_bound_and_non_authoritative(
    backend, status, reason_code, health
) -> None:
    result = IpfsDatasetsAnalysisProvider(backend=backend).analyze(_request())

    assert result.status is status
    assert result.reason_code == reason_code
    assert result.backend_health is health
    assert result.degradation_evidence is not None
    assert result.degradation_evidence.backend_health is health
    assert result.degradation_evidence.request_bound
    assert result.degradation_evidence.proof_bound
    assert not result.is_completion_evidence
    assert not result.safe_for_completion_reasoning


def test_cancelled_and_failed_paths_are_explicit_without_false_authority() -> None:
    imports = []
    provider = IpfsDatasetsAnalysisProvider(
        importer=lambda name: imports.append(name)
    )
    cancelled = provider.analyze(
        _request(), cancellation_token=type("Token", (), {"cancelled": True})()
    )

    def broken_import(name):
        raise RuntimeError("broken optional installation")

    failed = IpfsDatasetsAnalysisProvider(
        importer=broken_import
    ).analyze(_request())

    assert imports == []
    assert cancelled.status is AnalysisProviderStatus.CANCELLED
    assert cancelled.reason_code == "cancelled_before_import"
    assert not cancelled.degradation_evidence.import_attempted
    assert failed.status is AnalysisProviderStatus.FAILED
    assert failed.reason_code == "optional_import_failed"
    assert failed.degradation_evidence.import_attempted
    for result in (cancelled, failed):
        assert result.degradation_evidence.request_bound
        assert result.proved_requirement_ids == ()
        assert not result.safe_for_completion_reasoning


def test_explicit_empty_operation_policy_is_not_expanded() -> None:
    with pytest.raises(
        IpfsDatasetsAnalysisProviderError, match="must not be empty"
    ):
        AnalysisProviderPolicy.from_value({"operations": []})

    class StringBooleanBackend(_Backend):
        def capabilities(self):
            return {
                "protocol_versions": [1],
                "available": "false",
                "operations": ["graph_retrieval"],
            }

    result = IpfsDatasetsAnalysisProvider(
        backend=StringBooleanBackend()
    ).analyze(_request())
    assert result.status is AnalysisProviderStatus.MALFORMED
    assert not result.safe_for_completion_reasoning


def test_repeated_timeouts_have_bounded_backend_concurrency() -> None:
    release = threading.Event()
    entered = 0
    lock = threading.Lock()

    class BlockingBackend(_Backend):
        def retrieve(self, request):
            nonlocal entered
            with lock:
                entered += 1
            release.wait(5)
            return {"status": "completed", "results": []}

    bounds = AnalysisProviderBounds(timeout_ms=1)
    provider = IpfsDatasetsAnalysisProvider(
        AnalysisProviderPolicy(bounds=bounds),
        backend=BlockingBackend(),
    )
    try:
        results = [
            provider.analyze(_request(bounds=bounds))
            for _ in range(MAX_CONCURRENT_PROVIDER_DISPATCHES + 1)
        ]
    finally:
        release.set()

    assert entered == MAX_CONCURRENT_PROVIDER_DISPATCHES
    assert all(
        result.status is AnalysisProviderStatus.TIMED_OUT for result in results
    )
    assert results[-1].reason_code == "provider_capacity_exhausted"
    assert all(result.degradation_evidence is not None for result in results)
    assert all(
        result.degradation_evidence.request_bound for result in results
    )
    assert all(
        not result.degradation_evidence.import_attempted for result in results
    )
    assert all(result.proved_requirement_ids == () for result in results)


def test_schema_and_bound_negotiation_precede_dispatch() -> None:
    class NegotiatingBackend(_Backend):
        def __init__(self, capability):
            super().__init__()
            self._capability = capability

        def capabilities(self):
            return self._capability

    incompatible = NegotiatingBackend(
        {
            "protocol_versions": [1],
            "request_schemas": ["vendor/request@9"],
            "result_schemas": [PROVIDER_RESULT_SCHEMA],
            "available": True,
            "operations": ["graph_retrieval"],
        }
    )
    schema_result = IpfsDatasetsAnalysisProvider(
        backend=incompatible
    ).analyze(_request())

    assert schema_result.status is AnalysisProviderStatus.UNSUPPORTED
    assert schema_result.reason_code == "schema_incompatible"
    assert incompatible.requests == []

    bounded = NegotiatingBackend(
        {
            "protocol_versions": [1],
            "request_schema": PROVIDER_REQUEST_SCHEMA,
            "result_schema": PROVIDER_RESULT_SCHEMA,
            "available": True,
            "operations": ["graph_retrieval"],
            "bounds": {"max_results": 1},
        }
    )
    bounds_result = IpfsDatasetsAnalysisProvider(backend=bounded).analyze(
        _request()
    )

    assert bounds_result.status is AnalysisProviderStatus.UNSUPPORTED
    assert bounds_result.reason_code == "request_bounds_unsupported"
    assert bounded.requests == []
    assert bounds_result.degradation_evidence is not None
    assert not bounds_result.safe_for_completion_reasoning


def test_explicit_cancellation_capability_is_enforced_before_dispatch() -> None:
    backend = _Backend()

    def capabilities():
        return {
            "protocol_versions": [1],
            "request_schema": PROVIDER_REQUEST_SCHEMA,
            "result_schema": PROVIDER_RESULT_SCHEMA,
            "available": True,
            "operations": ["graph_retrieval"],
            "supports_cancellation": False,
        }

    backend.capabilities = capabilities
    result = IpfsDatasetsAnalysisProvider(backend=backend).analyze(
        _request(),
        cancellation_token=type("Token", (), {"cancelled": False})(),
    )

    assert result.status is AnalysisProviderStatus.UNSUPPORTED
    assert result.reason_code == "cancellation_not_supported"
    assert backend.requests == []


def test_separate_health_probe_runs_before_operation_dispatch() -> None:
    events = []

    class HealthBackend(_Backend):
        def capabilities(self):
            events.append("capabilities")
            return {
                "protocol_versions": [1],
                "request_schema": PROVIDER_REQUEST_SCHEMA,
                "response_schema": PROVIDER_RESULT_SCHEMA,
                "operations": ["graph_retrieval"],
            }

        def health(self):
            events.append("health")
            return {"health": "unavailable", "available": False}

        def retrieve(self, request):
            events.append("dispatch")
            return super().retrieve(request)

    result = IpfsDatasetsAnalysisProvider(
        backend=HealthBackend()
    ).analyze(_request())

    assert events == ["capabilities", "health"]
    assert result.status is AnalysisProviderStatus.UNAVAILABLE
    assert result.reason_code == "backend_unavailable"
    assert result.backend_health is AnalysisProviderHealth.UNAVAILABLE


def test_queries_and_artifacts_cannot_smuggle_heavy_payloads() -> None:
    provider = IpfsDatasetsAnalysisProvider(backend=_Backend())

    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="query contains forbidden heavy fields",
    ):
        provider.build_request(_request(query={"graph": {"nodes": []}}))

    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="artifact reference contains unsupported fields",
    ):
        provider.build_request(
            _request(
                artifact_references=(
                    {"artifact_id": "artifact:1", "content": "source body"},
                )
            )
        )


def test_related_requests_are_dispatched_as_one_compact_bounded_batch() -> None:
    class BatchBackend:
        __version__ = "batch@1"

        def __init__(self):
            self.requests = []

        def capabilities(self):
            return {
                "protocol_versions": [1],
                "request_schema": PROVIDER_REQUEST_SCHEMA,
                "result_schema": PROVIDER_RESULT_SCHEMA,
                "available": True,
                "operations": ["batch_analysis"],
                "supports_cancellation": True,
            }

        def analyze_batch(self, request):
            self.requests.append(request)
            children = request["query"]["requests"]
            return {
                "status": "completed",
                "results": [
                    {
                        "evidence_id": f"evidence:{index}",
                        "record_id": child["request_id"],
                    }
                    for index, child in enumerate(children)
                ],
                "provenance": [
                    {"artifact_id": "dataset:fixture", "kind": "dataset"}
                ],
                "resource_use": {
                    "batch_requests": len(children),
                    "input_bytes": 512,
                },
            }

    backend = BatchBackend()
    provider = IpfsDatasetsAnalysisProvider(backend=backend)
    requests = (
        provider.build_request(_request(query={"text": "cache authority"})),
        provider.build_request(
            _request(
                operation="premise_selection",
                query={"requirement_id": "requirement:1"},
            )
        ),
    )

    result = asyncio.run(provider.analyze_batch_async(requests))

    assert result.status is AnalysisProviderStatus.COMPLETED
    assert result.operation is AnalysisProviderOperation.BATCH_ANALYSIS
    assert result.resource_use["batch_requests"] == 2
    assert len(result.evidence_references) == 2
    assert len(backend.requests) == 1
    dispatched = backend.requests[0]
    assert dispatched["payload"] == {"batch_size": 2}
    assert dispatched["bounds"]["max_batch_requests"] == 2
    assert {
        child["operation"] for child in dispatched["query"]["requests"]
    } == {"graph_retrieval", "premise_selection"}
    assert all(
        "repository_id" not in child
        for child in dispatched["query"]["requests"]
    )
    assert not result.safe_for_completion_reasoning


def test_batch_rejects_unrelated_or_nested_requests_without_import() -> None:
    imports = []
    provider = IpfsDatasetsAnalysisProvider(
        importer=lambda name: imports.append(name)
    )
    related = provider.build_request(_request())
    unrelated = provider.build_request(
        _request(tree_id="tree:sha256:different")
    )
    nested = provider.build_request(
        _request(operation="batch_analysis")
    )

    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="must share repository, tree, and objective",
    ):
        provider.analyze_batch((related, unrelated))
    with pytest.raises(
        IpfsDatasetsAnalysisProviderError,
        match="nested batch",
    ):
        provider.analyze_batch((nested,))

    assert imports == []
