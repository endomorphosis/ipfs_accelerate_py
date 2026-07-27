"""Contract tests for pure deployment source adapters."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.model_catalog import LifecycleState, Operation, canonical_json
from ipfs_accelerate_py.model_catalog.sources.deployments import (
    BackendDeploymentSource,
    DeploymentCatalogSource,
    HealthSample,
    ServedEndpointDeploymentSource,
    adapt_backend_deployments,
    adapt_served_endpoints,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def fixed_clock():
    return NOW


def served(endpoint="http://127.0.0.1:8080/v1", **overrides):
    value = {
        "id": "leanstral_local",
        "model_id": "leanstral_local",
        "logical_model_id": "leanstral_local",
        "transport_model_id": "labs/Leanstral-1.5",
        "name": "leanstral_local",
        "provider": "llamacpp",
        "transport": "llamacpp",
        "endpoint": endpoint,
        "status": "available",
        "served": True,
        "capabilities": ["text-generation", "chat", "streaming"],
        "metadata": {},
    }
    value.update(overrides)
    return value


def test_local_and_remote_endpoint_identity_is_complete_and_secret_free():
    result = adapt_served_endpoints(
        [
            served(),
            served(
                "HTTPS://API.EXAMPLE.COM:443/v1",
                transport_model_id="remote/model",
                logical_model_id="remote/model",
                model_id="remote/model",
                id="remote/model",
                provider="remote-provider",
            ),
        ],
        observed_at=NOW,
        clock=fixed_clock,
    )

    assert len(result.deployments) == 2
    by_locality = {dict(item.labels)["locality"]: item for item in result.deployments}
    assert by_locality["local"].endpoint_uri == "http://127.0.0.1:8080/v1"
    assert by_locality["remote"].endpoint_uri == "https://api.example.com/v1"
    for deployment in result.deployments:
        labels = dict(deployment.labels)
        assert labels["service"] == "openai-compatible"
        assert labels["provider"]
        assert labels["model"]
        assert labels["protocol"] == "openai.http"
        assert labels["endpoint-id"].startswith("endpoint_")
        assert labels["locality"] in {"local", "remote"}
        assert "Bearer" not in canonical_json(deployment)


def test_logical_and_transport_model_aliases_are_preserved():
    result = adapt_served_endpoints([served()], clock=fixed_clock)

    assert len(result.models) == 1
    model = result.models[0]
    assert model.name == "leanstral_local"
    assert "labs/leanstral-1.5" in model.aliases
    assert model.provider_id == result.providers[0].provider_id
    assert result.deployments[0].model_id == model.model_id
    assert {
        operation
        for capability in result.deployments[0].capabilities
        for operation in capability.operations
    } == {Operation.TEXT_GENERATE, Operation.TEXT_CHAT, Operation.STREAM}


def test_normal_reads_never_probe_and_refresh_uses_only_injected_probe():
    calls = []

    def probe(target):
        calls.append(target)
        assert target.inference_allowed is False
        assert target.purpose == "liveness-readiness"
        return {
            "reachable": True,
            "live": True,
            "ready": False,
            "healthy": True,
            "routable": False,
            "ttl_seconds": 30,
            "diagnostics": ["warmup in progress"],
        }

    source = ServedEndpointDeploymentSource(
        [served()], probe=probe, clock=fixed_clock
    )
    initial = source.load()
    assert calls == []
    assert initial.deployments[0].state.reachable is None
    assert initial.health_samples == ()

    refreshed = source.refresh()
    assert len(calls) == 1
    sample = refreshed.health_samples[0]
    assert sample.observed_at == "2026-07-27T12:00:00.000000Z"
    assert sample.expires_at == "2026-07-27T12:00:30.000000Z"
    assert sample.provenance == "model-manager.served.probe"
    assert sample.reachable is True
    assert sample.live is True
    assert sample.ready is False
    assert sample.healthy is True
    assert sample.routable is False
    assert sample.diagnostics == ("warmup in progress",)
    assert refreshed.deployments[0].state.reachable is True
    assert refreshed.deployments[0].state.healthy is True
    assert refreshed.deployments[0].state.routable is False
    # Readiness is not silently promoted from liveness or health.
    assert refreshed.deployments[0].lifecycle == LifecycleState.READY


def test_configured_reachable_live_ready_healthy_and_routable_are_distinct():
    source = DeploymentCatalogSource(
        [
            {
                "backend_id": "orthogonal",
                "backend_type": "api",
                "endpoint": "https://example.test/v1",
                "provider": "example",
                "model": "chat",
                "configured": True,
                "authorized": False,
                "routable": True,
            }
        ],
        clock=fixed_clock,
    )

    result = source.refresh(
        probe=lambda target: {
            "configured": False,
            "reachable": True,
            "live": False,
            "ready": True,
            "healthy": False,
            "routable": False,
        }
    )
    sample = result.health_samples[0]
    assert (
        sample.configured,
        sample.reachable,
        sample.live,
        sample.ready,
        sample.healthy,
        sample.routable,
    ) == (False, True, False, True, False, False)
    assert result.deployments[0].state.authorized is False


def test_stopped_backend_is_configured_without_inventing_reachability():
    result = adapt_backend_deployments(
        [
            {
                "backend_id": "local-vllm",
                "backend_type": "api",
                "name": "vLLM",
                "endpoint": "http://localhost:8000/v1",
                "status": "stopped",
                "provider": "vllm",
                "capabilities": {
                    "supported_models": ["org/model"],
                    "supported_tasks": ["text-generation"],
                    "protocols": ["http"],
                },
            }
        ],
        clock=fixed_clock,
    )

    deployment = result.deployments[0]
    assert deployment.lifecycle == LifecycleState.STOPPED
    assert deployment.state.configured is True
    assert deployment.state.reachable is None
    assert deployment.state.healthy is None
    assert deployment.state.routable is None


def test_stale_health_sample_is_retained_but_not_projected():
    source = BackendDeploymentSource(
        [
            {
                "backend_id": "old",
                "backend_type": "api",
                "endpoint": "https://old.example/v1",
                "provider": "old",
                "model": "chat",
                "health": {
                    "observed_at": "2026-07-27T11:58:00Z",
                    "ttl_seconds": 30,
                    "provenance": "backend.health",
                    "reachable": True,
                    "healthy": True,
                },
            }
        ],
        clock=fixed_clock,
    )
    result = source.load()

    assert len(result.health_samples) == 1
    assert result.health_samples[0].is_stale(NOW)
    assert result.deployments[0].state.reachable is None
    assert result.deployments[0].state.healthy is None
    assert len(result.deployments[0].provenance) == 1


@pytest.mark.parametrize(
    "endpoint",
    [
        "not-a-url",
        "ftp://example.test/model",
        "http://",
        "http://[not-an-ipv6]/v1",
        "unix://relative",
    ],
)
def test_malformed_urls_are_bounded_diagnostics(endpoint):
    result = adapt_served_endpoints(
        [served(endpoint), served()],
        clock=fixed_clock,
    )

    assert len(result.deployments) == 1
    assert result.error_count == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "malformed_endpoint"
    assert len(diagnostic.message.encode("utf-8")) <= 512


def test_credentials_and_raw_bearer_urls_are_redacted():
    raw = (
        "https://Bearer%20abcdefghijklmnopqrstuvwxyz:secret@example.test:443/v1"
        "?api_key=sk-abcdefghijklmnopqrstuvwxyz#token"
    )
    result = adapt_served_endpoints([served(raw)], clock=fixed_clock)
    rendered = canonical_json(result.to_dict(), reject_secrets=False)

    assert result.redacted_fields == 1
    assert result.deployments[0].endpoint_uri == "https://example.test/v1"
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "secret" not in rendered
    assert "api_key" not in rendered
    assert result.diagnostics[0].code == "redacted"


def test_duplicate_endpoints_coalesce_and_snapshots_are_deterministic():
    rows = [
        served(status="available"),
        served(status="available", metadata={"irrelevant": "one"}),
        served(endpoint="http://127.0.0.1:8080/v1/"),
    ]
    forward = adapt_served_endpoints(rows, observed_at=NOW, clock=fixed_clock)
    reverse = adapt_served_endpoints(
        list(reversed(rows)), observed_at=NOW, clock=fixed_clock
    )

    # Trailing slash is endpoint identity, while exact duplicate advertisements
    # coalesce independent of source ordering.
    assert len(forward.deployments) == 2
    assert forward.snapshot.revision == reverse.snapshot.revision
    assert canonical_json(forward.to_dict()) == canonical_json(reverse.to_dict())


def test_backend_dataclass_shape_and_multiple_models_are_projected():
    @dataclasses.dataclass
    class Capabilities:
        supported_tasks: set
        supported_models: set
        protocols: set
        supports_streaming: bool = False

    @dataclasses.dataclass
    class Backend:
        backend_id: str
        backend_type: str
        name: str
        endpoint: str
        status: str
        capabilities: Capabilities
        instance: object
        metadata: dict

    record = Backend(
        backend_id="remote-api",
        backend_type="api",
        name="Remote API",
        endpoint="https://models.example/v1",
        status="healthy",
        capabilities=Capabilities(
            supported_tasks={"text-generation"},
            supported_models={"model-b", "model-a"},
            protocols={"http"},
        ),
        instance=object(),
        metadata={"provider": "example"},
    )
    result = adapt_backend_deployments([record], clock=fixed_clock)

    assert sorted(model.name for model in result.models) == ["model-a", "model-b"]
    assert len(result.deployments) == 2


def test_probe_errors_and_diagnostics_are_redacted_and_bounded():
    def failing_probe(target):
        raise RuntimeError(
            "Authorization Bearer abcdefghijklmnopqrstuvwxyz " + "x" * 2_000
        )

    result = ServedEndpointDeploymentSource(
        [served()], probe=failing_probe, clock=fixed_clock
    ).refresh()
    rendered = canonical_json(result.to_dict(), reject_secrets=False)

    assert len(result.health_samples) == 1
    assert result.health_samples[0].reachable is None
    assert result.health_samples[0].diagnostics == (
        "credential-shaped diagnostic was redacted",
    )
    assert result.diagnostics[-1].code == "probe_failed"
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered


def test_live_manager_is_rejected_instead_of_implicitly_calling_it():
    class Manager:
        calls = 0

        def list_backends(self):
            self.calls += 1
            return []

    manager = Manager()
    source = BackendDeploymentSource(manager, clock=fixed_clock)
    with pytest.raises(ValueError, match="inject list_backends"):
        source.load()
    assert manager.calls == 0


def test_refresh_requires_probe_and_health_sample_validates_ttl():
    source = ServedEndpointDeploymentSource([served()], clock=fixed_clock)
    with pytest.raises(ValueError, match="requires an injected probe"):
        source.refresh()
    with pytest.raises(ValueError, match="ttl_seconds"):
        HealthSample(
            deployment_id=source.load().deployments[0].deployment_id,
            observed_at=NOW,
            ttl_seconds=0,
            provenance="test",
        )
