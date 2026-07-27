from __future__ import annotations

import asyncio
import importlib
import sys
import types
from datetime import datetime, timedelta, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_transport import (
    ANALYSIS_TRANSPORT_REQUEST_SCHEMA,
    ANALYSIS_TRANSPORT_RESULT_SCHEMA,
    AnalysisCancellationToken,
    AnalysisCapability,
    AnalysisProviderHealth,
    AnalysisProviderKind,
    AnalysisRequest,
    AnalysisResult,
    AnalysisTransport,
    AnalysisTransportBounds,
    AnalysisTransportError,
    AnalysisTransportPolicy,
    AnalysisTransportStatus,
)


def _capability(
    provider_id: str,
    *,
    kind: AnalysisProviderKind = AnalysisProviderKind.LOCAL,
    revision: str = "capability:1",
    batching: bool = False,
    progress: bool = True,
    protocols: tuple[int, ...] = (1,),
    request_schemas: tuple[str, ...] = (ANALYSIS_TRANSPORT_REQUEST_SCHEMA,),
    result_schemas: tuple[str, ...] = (ANALYSIS_TRANSPORT_RESULT_SCHEMA,),
) -> AnalysisCapability:
    return AnalysisCapability(
        provider_id=provider_id,
        provider_kind=kind,
        provider_version="fixture:1",
        capability_revision=revision,
        operations=("symbol_impact", "premise_selection"),
        protocol_versions=protocols,
        request_schemas=request_schemas,
        result_schemas=result_schemas,
        health=AnalysisProviderHealth.LAZY,
        max_batch_size=8 if batching else 1,
        max_concurrency=2,
        supports_cancellation=True,
        supports_progress=progress,
        supports_batching=batching,
    )


def _request(name: str = "one", *, provider_id: str = "") -> AnalysisRequest:
    return AnalysisRequest(
        request_id=name,
        operation="symbol_impact",
        question="Which symbols depend on the changed interface?",
        artifact_references=(
            {
                "artifact_id": "source-tree",
                "digest": "sha256:" + "a" * 64,
                "path": "ipfs_accelerate_py/agent_supervisor",
            },
        ),
        preferred_provider_id=provider_id,
        metadata={"tree_id": "tree:fixture"},
        timeout_ms=2_000,
    )


def _response(request: AnalysisRequest, negotiated: object, **updates: object) -> dict:
    payload = {
        "schema": negotiated.result_schema,
        "protocol_version": negotiated.protocol_version,
        "request_id": request.request_id,
        "operation": request.operation,
        "capability_id": negotiated.capability_id,
        "capability_revision": negotiated.capability_revision,
        "evidence_references": (
            {
                "evidence_id": f"evidence:{request.request_id}",
                "digest": "sha256:" + "b" * 64,
                "summary": "The interface is used by two bounded call sites.",
            },
        ),
        "provenance_references": (
            {
                "record_id": "ast-index:1",
                "digest": "sha256:" + "c" * 64,
                "producer_id": "fixture",
            },
        ),
        "cost": {"input_units": 3, "output_units": 2},
        "verdict": "candidate_support",
        "truncated": False,
        "non_authoritative": True,
        "completion_authority": False,
        "safe_for_completion_reasoning": False,
    }
    payload.update(updates)
    return payload


class _Provider:
    def __init__(self, capability: AnalysisCapability) -> None:
        self.capability = capability
        self.calls = 0

    async def analyze(
        self,
        request: AnalysisRequest,
        *,
        negotiated_capability: object,
        progress: object = None,
        cancellation_token: object = None,
    ) -> dict:
        self.calls += 1
        if progress:
            progress({"message": "indexed", "completed_units": 1, "total_units": 2})
            progress({"message": "ranked", "completed_units": 2, "total_units": 2})
        return _response(request, negotiated_capability)


def test_discovery_and_negotiation_are_lazy_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    real_import = importlib.import_module

    def guarded_import(name: str, package: str | None = None) -> object:
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    capability = _capability(
        "datasets",
        kind=AnalysisProviderKind.IPFS_DATASETS,
        protocols=(3, 2, 1),
        request_schemas=("request@future", ANALYSIS_TRANSPORT_REQUEST_SCHEMA),
        result_schemas=("result@future", ANALYSIS_TRANSPORT_RESULT_SCHEMA),
    )
    transport = AnalysisTransport()
    transport.register_optional_module(
        capability,
        module_name="module_that_must_not_be_imported_during_discovery",
        attribute="provider",
    )

    discovered = transport.discover_capabilities("symbol_impact")
    negotiated = transport.negotiate("datasets", "symbol_impact")

    assert discovered == (capability,)
    assert imported == []
    assert negotiated is not None
    assert negotiated.protocol_version == 1
    assert negotiated.request_schema == ANALYSIS_TRANSPORT_REQUEST_SCHEMA
    assert negotiated.result_schema == ANALYSIS_TRANSPORT_RESULT_SCHEMA
    assert discovered[0].non_authoritative


def test_dispatch_carries_compact_refs_progress_cost_and_no_authority() -> None:
    async def scenario() -> AnalysisResult:
        capability = _capability("local")
        provider = _Provider(capability)
        progress = []
        transport = AnalysisTransport(
            local_provider=provider,
            local_capability=capability,
        )
        result = await transport.dispatch(
            _request(),
            progress_callback=progress.append,
        )
        assert provider.calls == 1
        assert [item.message for item in progress] == ["indexed", "ranked"]
        return result

    result = asyncio.run(scenario())

    assert result.status is AnalysisTransportStatus.COMPLETED
    assert result.verdict == "candidate_support"
    assert result.evidence_references[0]["evidence_id"] == "evidence:one"
    assert result.provenance_references[0]["record_id"] == "ast-index:1"
    assert result.cost["input_units"] == 3
    assert result.cost["provider_calls"] == 1
    assert [item.sequence for item in result.progress] == [0, 1]
    assert result.non_authoritative is True
    assert result.completion_authority is False
    assert result.safe_for_completion_reasoning is False
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored == result
    assert restored.result_id == result.result_id
    with pytest.raises(TypeError):
        bool(result)


def test_reference_and_progress_bounds_truncate_without_embedding_payloads() -> None:
    class VerboseProvider(_Provider):
        async def analyze(
            self,
            request: AnalysisRequest,
            *,
            negotiated_capability: object,
            progress: object,
        ) -> dict:
            for index in range(5):
                progress(f"step {index}")
            references = tuple(
                {
                    "evidence_id": f"evidence:{index}",
                    "digest": "sha256:" + str(index) * 64,
                }
                for index in range(5)
            )
            return _response(
                request,
                negotiated_capability,
                evidence_references=references,
            )

    bounds = AnalysisTransportBounds(
        max_evidence_references=2,
        max_progress_events=2,
    )
    capability = _capability("local")
    result = asyncio.run(
        AnalysisTransport(
            policy=AnalysisTransportPolicy(bounds=bounds),
            local_provider=VerboseProvider(capability),
            local_capability=capability,
        ).dispatch(_request())
    )

    assert result.status is AnalysisTransportStatus.COMPLETED
    assert len(result.evidence_references) == 2
    assert result.truncated is True
    assert len(result.progress) == 2
    assert result.progress_truncated is True

    with pytest.raises(AnalysisTransportError, match="forbidden"):
        AnalysisRequest(
            operation="symbol_impact",
            question="inspect",
            artifact_references=(
                {"artifact_id": "bad", "source_body": "not a reference"},
            ),
        )


def test_provider_loss_uses_deterministic_local_fallback() -> None:
    class LostProvider:
        async def analyze(self, request: AnalysisRequest, **_: object) -> object:
            raise ConnectionError("gone")

    optional = _capability(
        "datasets", kind=AnalysisProviderKind.IPFS_DATASETS
    )
    local = _capability("local")
    transport = AnalysisTransport(
        policy=AnalysisTransportPolicy(fallback_provider_id="local"),
    )
    transport.register_provider(optional, provider=LostProvider())
    transport.register_provider(local, provider=_Provider(local))

    result = asyncio.run(
        transport.dispatch(_request(provider_id="datasets"))
    )

    assert result.status is AnalysisTransportStatus.FALLBACK
    assert result.provider_id == "local"
    assert result.fallback_from_provider_id == "datasets"
    assert result.fallback_reason_code == AnalysisTransportStatus.PROVIDER_LOST.value
    assert result.reason_code == "provider_connection_lost"
    assert result.non_authoritative
    health = transport.health_snapshot()
    by_id = {item.provider_id: item.health for item in health.providers}
    assert by_id["datasets"] is AnalysisProviderHealth.UNAVAILABLE
    assert by_id["local"] is AnalysisProviderHealth.HEALTHY


def test_timeout_and_cancellation_are_typed_terminal_outcomes() -> None:
    class SlowProvider:
        async def analyze(self, request: AnalysisRequest, **_: object) -> object:
            await asyncio.sleep(1)
            raise AssertionError("deadline should cancel this coroutine")

    async def scenario() -> tuple[AnalysisResult, AnalysisResult]:
        capability = _capability("local")
        transport = AnalysisTransport(
            local_provider=SlowProvider(),
            local_capability=capability,
        )
        timed_out = await transport.dispatch(_request(), timeout_ms=20)

        token = AnalysisCancellationToken()
        task = asyncio.create_task(
            transport.dispatch(_request("cancelled"), cancellation_token=token)
        )
        await asyncio.sleep(0.02)
        token.cancel()
        cancelled = await task
        return timed_out, cancelled

    timed_out, cancelled = asyncio.run(scenario())
    assert timed_out.status is AnalysisTransportStatus.TIMED_OUT
    assert timed_out.reason_code == "deadline_exceeded"
    assert cancelled.status is AnalysisTransportStatus.CANCELLED
    assert cancelled.reason_code == "cancelled_during_dispatch"


def test_malformed_output_and_capability_drift_fail_closed() -> None:
    class Malformed:
        async def analyze(
            self,
            request: AnalysisRequest,
            *,
            negotiated_capability: object,
        ) -> object:
            return _response(
                request,
                negotiated_capability,
                completion_authority=True,
            )

    capability = _capability("provider")
    malformed = AnalysisTransport(
        local_provider=Malformed(),
        local_capability=capability,
    )
    malformed_result = asyncio.run(malformed.dispatch(_request()))
    assert malformed_result.status is AnalysisTransportStatus.MALFORMED_OUTPUT
    assert malformed_result.evidence_references == ()

    provider = _Provider(capability)
    drift = AnalysisTransport(
        local_provider=provider,
        local_capability=capability,
    )
    provider.capability = _capability("provider", revision="capability:2")
    drift_result = asyncio.run(drift.dispatch(_request()))
    assert drift_result.status is AnalysisTransportStatus.CAPABILITY_DRIFT
    assert drift_result.reason_code == "capability_drift"
    assert provider.calls == 0


def test_optional_import_occurs_only_at_first_dispatch_and_loss_is_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "_analysis_transport_optional_fixture"
    imports = 0
    real_import = importlib.import_module
    capability = _capability(
        "datasets", kind=AnalysisProviderKind.IPFS_DATASETS
    )
    provider = _Provider(capability)
    module = types.ModuleType(module_name)
    module.provider = provider
    sys.modules[module_name] = module

    def counted_import(name: str, package: str | None = None) -> object:
        nonlocal imports
        if name == module_name:
            imports += 1
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", counted_import)
    try:
        transport = AnalysisTransport()
        transport.register_optional_module(
            capability, module_name=module_name, attribute="provider"
        )
        assert transport.discover_capabilities() == (capability,)
        assert imports == 0

        first = asyncio.run(
            transport.dispatch(_request(provider_id="datasets"))
        )
        second = asyncio.run(
            transport.dispatch(_request("two", provider_id="datasets"))
        )
        assert first.status is AnalysisTransportStatus.COMPLETED
        assert second.status is AnalysisTransportStatus.COMPLETED
        assert imports == 1
    finally:
        sys.modules.pop(module_name, None)

    unavailable_capability = _capability(
        "missing", kind=AnalysisProviderKind.IPFS_DATASETS
    )
    unavailable = AnalysisTransport()
    unavailable.register_optional_module(
        unavailable_capability,
        module_name="_analysis_transport_definitely_missing",
    )
    result = asyncio.run(
        unavailable.dispatch(_request(provider_id="missing"))
    )
    assert result.status is AnalysisTransportStatus.UNAVAILABLE
    assert result.reason_code == "provider_activation_unavailable"


def test_native_batching_uses_one_provider_call_and_preserves_member_bounds() -> None:
    class BatchProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def analyze_batch(
            self,
            requests: tuple[AnalysisRequest, ...],
            *,
            negotiated_capability: object,
        ) -> list[dict]:
            self.calls += 1
            return [
                _response(request, negotiated_capability)
                for request in requests
            ]

    capability = _capability("batch", batching=True, progress=False)
    provider = BatchProvider()
    transport = AnalysisTransport(
        local_provider=provider,
        local_capability=capability,
    )
    results = asyncio.run(
        transport.dispatch_batch(
            (_request("one", provider_id="batch"), _request("two", provider_id="batch")),
            provider_id="batch",
        )
    )

    assert provider.calls == 1
    assert [item.request_id for item in results] == ["one", "two"]
    assert all(item.status is AnalysisTransportStatus.COMPLETED for item in results)
    assert all(item.cost["provider_calls"] == 1 for item in results)


def test_backpressure_is_bounded_and_deadline_can_expire_in_queue() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider:
        async def analyze(
            self,
            request: AnalysisRequest,
            *,
            negotiated_capability: object,
        ) -> dict:
            entered.set()
            await release.wait()
            return _response(request, negotiated_capability)

    async def scenario() -> tuple[AnalysisResult, AnalysisResult, AnalysisResult]:
        bounds = AnalysisTransportBounds(
            max_concurrency=1,
            max_queue_size=1,
            timeout_ms=2_000,
        )
        capability = _capability("local", progress=False)
        transport = AnalysisTransport(
            policy=AnalysisTransportPolicy(bounds=bounds),
            local_provider=BlockingProvider(),
            local_capability=capability,
        )
        first_task = asyncio.create_task(transport.dispatch(_request("first")))
        await entered.wait()
        queued_task = asyncio.create_task(
            transport.dispatch(_request("queued"), timeout_ms=30)
        )
        await asyncio.sleep(0.01)
        rejected = await transport.dispatch(_request("rejected"))
        queued = await queued_task
        release.set()
        first = await first_task
        health = transport.health_snapshot()
        assert health.rejected_requests == 2
        assert health.active_requests == 0
        assert health.queued_requests == 0
        return first, queued, rejected

    first, queued, rejected = asyncio.run(scenario())
    assert first.status is AnalysisTransportStatus.COMPLETED
    assert queued.status is AnalysisTransportStatus.TIMED_OUT
    assert queued.reason_code == "deadline_expired_in_queue"
    assert rejected.status is AnalysisTransportStatus.BACKPRESSURE
    assert rejected.reason_code == "transport_queue_full"


def test_request_deadline_and_request_bounds_fail_before_provider_execution() -> None:
    capability = _capability("local")
    provider = _Provider(capability)
    transport = AnalysisTransport(
        policy=AnalysisTransportPolicy(
            bounds=AnalysisTransportBounds(max_question_bytes=16)
        ),
        local_provider=provider,
        local_capability=capability,
    )
    with pytest.raises(AnalysisTransportError, match="question exceeds"):
        asyncio.run(
            transport.dispatch(
                AnalysisRequest(
                    operation="symbol_impact",
                    question="x" * 17,
                )
            )
        )
    assert provider.calls == 0

    expired = AnalysisRequest(
        operation="symbol_impact",
        question="bounded",
        deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    result = asyncio.run(transport.dispatch(expired))
    assert result.status is AnalysisTransportStatus.TIMED_OUT
    assert provider.calls == 0
