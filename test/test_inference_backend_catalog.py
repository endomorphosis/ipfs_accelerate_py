"""Typed catalog contracts for :mod:`inference_backend_manager`."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from ipfs_accelerate_py.inference_backend_manager import (
    BackendCapabilities,
    BackendRegistration,
    BackendStatus,
    BackendType,
    InferenceBackendManager,
    ProviderRegistration,
)
from ipfs_accelerate_py.model_catalog import (
    CapabilityDescriptor,
    DeploymentDescriptor,
    LifecycleState,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog


def manager(**config):
    return InferenceBackendManager(
        {
            "persist_registry": False,
            "enable_health_checks": False,
            **config,
        }
    )


def capabilities(*models: str) -> BackendCapabilities:
    return BackendCapabilities(
        supported_tasks={"text-generation"},
        supported_models=set(models),
        supports_streaming=True,
        protocols={"http"},
    )


def register(
    value: InferenceBackendManager,
    backend_id: str,
    *,
    instance=None,
    endpoint: str | None = None,
    models=(),
    aliases=(),
):
    return value.register_backend(
        backend_id=backend_id,
        backend_type=BackendType.API,
        name=backend_id,
        instance=instance or object(),
        capabilities=capabilities(*models),
        endpoint=endpoint or f"https://{backend_id}.example/v1",
        metadata={"provider": backend_id},
        aliases=aliases,
    )


def typed_registration(instance=None) -> BackendRegistration:
    capability = CapabilityDescriptor(
        operations=(Operation.TEXT_GENERATE, Operation.STREAM),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
    )
    provenance = (Provenance(source="test.backend.catalog"),)
    provider = ProviderDescriptor(
        name="acme",
        aliases=("acme-ai",),
        capabilities=(capability,),
        lifecycle=LifecycleState.CONFIGURED,
        state=OperationalState(known=True, configured=True),
        provenance=provenance,
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="chat",
        aliases=("chat-latest",),
        capabilities=(capability,),
        lifecycle=LifecycleState.CONFIGURED,
        state=OperationalState(known=True, configured=True),
        provenance=provenance,
    )
    deployment = DeploymentDescriptor(
        provider_id=provider.provider_id,
        model_id=model.model_id,
        name="acme/chat/primary",
        endpoint_uri="https://typed.example/v1",
        capabilities=(capability,),
        lifecycle=LifecycleState.CONFIGURED,
        state=OperationalState(known=True, configured=True),
        provenance=provenance,
    )
    binding = RouterBinding(
        router="inference_backend_manager",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        deployment_id=deployment.deployment_id,
        operations=(Operation.TEXT_GENERATE,),
        state=OperationalState(known=True, configured=True),
        provenance=provenance,
    )
    return BackendRegistration(
        backend_id="typed",
        backend_type=BackendType.API,
        name="Typed backend",
        instance=instance or object(),
        capabilities=capabilities("chat"),
        endpoint=deployment.endpoint_uri,
        metadata={"provider": "acme"},
        aliases=("typed-alias",),
        provider=provider,
        models=(model,),
        deployments=(deployment,),
        bindings=(binding,),
    )


def test_builtin_provider_registry_uses_named_records_and_legacy_projection():
    spec = InferenceBackendManager._PROVIDER_REGISTRY["xai"]

    assert isinstance(spec, ProviderRegistration)
    assert spec.backend_module_path == "ipfs_accelerate_py.api_backends.xai"
    assert spec.module_path == spec.backend_module_path
    with pytest.warns(DeprecationWarning):
        assert spec[0] == spec.backend_module_path


def test_legacy_provider_tuple_uses_one_adapter_with_equivalent_behavior():
    old = (
        "package.backend",
        "Backend",
        "PRIMARY_KEY",
        "SECONDARY_KEY",
        "https://legacy.example/v1",
        "Legacy",
        {"text-generation"},
    )
    typed = ProviderRegistration(
        name="legacy",
        backend_module_path=old[0],
        backend_class_name=old[1],
        env_key_primary=old[2],
        env_key_secondary=old[3],
        default_base_url=old[4],
        display_name=old[5],
        supported_tasks=frozenset(old[6]),
    )

    with pytest.warns(DeprecationWarning, match="Tuple-shaped"):
        old_manager = InferenceBackendManager(
            {
                "persist_registry": False,
                "enable_health_checks": False,
            },
            provider_registry={"legacy": old},
        )
    new_manager = InferenceBackendManager(
        {"persist_registry": False, "enable_health_checks": False},
        provider_registry={"legacy": typed},
    )
    backend_class = MagicMock(return_value=MagicMock())
    module = MagicMock(Backend=backend_class)
    with patch("importlib.import_module", return_value=module):
        old_result = old_manager.configure_provider("legacy", api_key="key")
    old_call = backend_class.call_args
    backend_class.reset_mock()
    with patch("importlib.import_module", return_value=module):
        new_result = new_manager.configure_provider("legacy", api_key="key")

    assert old_result == new_result
    assert old_call == backend_class.call_args
    assert old_manager.get_backend("api_legacy").capabilities == (
        new_manager.get_backend("api_legacy").capabilities
    )


@pytest.mark.parametrize(
    "registration",
    [
        {
            "backend_id": "",
            "backend_type": "api",
            "name": "bad",
        },
        {
            "backend_id": "bad-type",
            "backend_type": "not-a-type",
            "name": "bad",
        },
        {
            "backend_id": "bad-endpoint",
            "backend_type": "api",
            "name": "bad",
            "endpoint": "not a URI",
        },
        {
            "backend_id": "unknown-field",
            "backend_type": "api",
            "name": "bad",
            "surprise": True,
        },
    ],
)
def test_malformed_registration_is_atomic(registration):
    value = manager()
    revision = value.catalog_revision

    assert value.register_backend(registration) is False
    assert value.list_backends() == []
    assert value.catalog_revision == revision


def test_typed_registration_drives_lookup_alias_selection_and_snapshot():
    value = manager()
    registration = typed_registration()

    assert value.register_backend(registration) is True
    backend = value.get_backend("typed-alias")
    assert backend.provider == registration.provider
    assert backend.models == registration.models
    assert backend.deployments == registration.deployments
    assert backend.bindings == registration.bindings
    assert value.get_backend_by_deployment(
        registration.deployments[0].deployment_id
    ) is backend
    assert value.get_provider_descriptor("acme-ai") == registration.provider
    assert value.select_backend_for_task(
        "text-generation",
        model="chat-latest",
        provider="acme-ai",
        deployment_id=registration.deployments[0].deployment_id,
    ) is backend

    snapshot = value.get_catalog_snapshot()
    assert snapshot.providers == (registration.provider,)
    assert snapshot.models == registration.models
    assert snapshot.deployments == registration.deployments
    assert snapshot.bindings == registration.bindings


def test_reregistration_replaces_all_secondary_indexes():
    value = manager()
    assert register(value, "replace", aliases=("old",))
    replacement = BackendRegistration(
        backend_id="replace",
        backend_type=BackendType.GPU,
        name="replacement",
        capabilities=BackendCapabilities(
            supported_tasks={"embedding"},
            protocols={"http"},
        ),
        endpoint="http://localhost:9910",
        metadata={"provider": "replacement"},
        aliases=("new",),
    )
    assert value.register_backend(replacement)

    assert value.get_backend("old") is None
    assert value.get_backend("new").name == "replacement"
    assert value.task_routing.get("text-generation") is None
    assert value.task_routing["embedding"] == ["replace"]
    assert value.backends_by_type[BackendType.GPU] == ["replace"]
    assert BackendType.API not in value.backends_by_type


def test_register_unregister_publishes_to_injected_catalog():
    catalog = AIServiceCatalog()
    value = manager(catalog=catalog)
    empty_revision = catalog.snapshot().revision

    assert register(value, "dynamic", models=("one",))
    published = catalog.snapshot()
    assert published.revision != empty_revision
    assert len(published.deployments) == 1
    assert published.providers[0].name == "dynamic"
    assert published.bindings[0].router == "inference_backend_manager"

    assert value.unregister_backend("dynamic")
    assert catalog.snapshot().revision == empty_revision
    assert value.catalog_source.load().snapshot.deployments == ()


def test_concurrent_updates_produce_deterministic_complete_generations():
    first = manager()
    ids = [f"backend-{index:02d}" for index in range(24)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(lambda item: register(first, item), ids))

    assert sorted(item.name for item in first.catalog_snapshot().providers) == ids
    stable_revision = first.catalog_revision
    assert stable_revision == first.catalog_revision

    second = manager()
    for backend_id in reversed(ids):
        assert register(second, backend_id)
    assert second.catalog_revision == stable_revision

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert all(executor.map(first.unregister_backend, ids[::2]))
    assert sorted(item.name for item in first.catalog_snapshot().providers) == ids[1::2]


@pytest.mark.asyncio
async def test_liveness_readiness_are_separate_and_never_invoke_inference():
    class Backend:
        inference_calls = 0
        health_calls = 0

        def health_check(self):
            self.health_calls += 1
            return {
                "reachable": True,
                "live": True,
                "ready": False,
                "healthy": True,
                "routable": False,
            }

        def run_inference(self, **kwargs):
            self.inference_calls += 1
            return {"text": "not called"}

    instance = Backend()
    value = manager()
    assert register(value, "live", instance=instance)
    before = value.catalog_snapshot().deployments[0]
    assert before.state.reachable is None
    assert before.state.healthy is None
    assert before.state.routable is None

    assert await value.check_backend_health("live")
    backend = value.get_backend("live")
    after = value.catalog_snapshot().deployments[0]
    assert (backend.live, backend.ready) == (True, False)
    assert after.state.reachable is True
    assert after.state.healthy is True
    assert after.state.routable is False
    assert after.lifecycle == LifecycleState.CONFIGURED
    assert instance.health_calls == 1
    assert instance.inference_calls == 0


def test_status_only_does_not_invent_liveness():
    value = manager()
    assert register(value, "status")
    assert value.update_backend_status("status", BackendStatus.UNHEALTHY)

    backend = value.get_backend("status")
    deployment = value.catalog_snapshot().deployments[0]
    assert backend.status == BackendStatus.UNHEALTHY
    assert (
        backend.reachable,
        backend.live,
        backend.ready,
        backend.healthy,
        backend.routable,
    ) == (None, None, None, None, None)
    assert deployment.lifecycle == LifecycleState.CONFIGURED
    assert deployment.state.healthy is None


def test_endpoint_lifecycle_replaces_identity_and_clears_observations():
    value = manager()
    assert register(value, "endpoint", endpoint="https://one.example/v1")
    assert value.update_backend_liveness(
        "endpoint",
        reachable=True,
        live=True,
        ready=True,
        healthy=True,
        routable=True,
    )
    first = value.catalog_snapshot().deployments[0]

    assert value.update_backend_endpoint("endpoint", "https://two.example/v1")
    second = value.catalog_snapshot().deployments[0]
    backend = value.get_backend("endpoint")
    assert second.endpoint_uri == "https://two.example/v1"
    assert second.deployment_id != first.deployment_id
    assert second.state.healthy is None
    assert backend.status == BackendStatus.INITIALIZING
    assert backend.live is None


@pytest.mark.asyncio
async def test_invocation_delegates_to_backend_not_catalog_or_model_manager():
    calls = []

    class Backend:
        async def generate(self, **kwargs):
            calls.append(kwargs)
            return {"text": "delegated"}

    catalog = AIServiceCatalog()
    value = manager(catalog=catalog)
    assert register(value, "delegate", instance=Backend(), models=("chat",))

    result = await value.execute_task(
        task="text-generation",
        model="chat",
        inputs=["hello"],
        parameters={"temperature": 0},
    )

    assert result["text"] == "delegated"
    assert result["backend_id"] == "delegate"
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["temperature"] == 0
    assert not hasattr(catalog, "execute_task")


def test_provider_alias_configuration_and_round_robin_are_unchanged():
    value = manager(load_balancing="round_robin")
    spec = value._provider_registry["xai"]
    backend_class = MagicMock(return_value=MagicMock())
    module = MagicMock(xai=backend_class)
    with patch("importlib.import_module", return_value=module):
        result = value.configure_provider("grok", api_key="key")
    assert result == {
        "provider": "xai",
        "configured": True,
        "backend_id": "api_xai",
    }
    assert value.get_backend("api_xai").endpoint == spec.default_base_url

    other = manager(load_balancing="round_robin")
    assert register(other, "one")
    assert register(other, "two")
    assert other.select_backend_for_task("text-generation").backend_id == "one"
    assert other.select_backend_for_task("text-generation").backend_id == "two"
