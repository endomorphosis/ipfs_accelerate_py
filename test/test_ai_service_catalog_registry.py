"""Focused contracts for catalog merging, snapshots, and resolution."""

from __future__ import annotations

import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ipfs_accelerate_py.model_catalog.registry import (
    AmbiguousAliasError,
    CatalogRegistry,
    RegistryDiagnostic,
    RegistryView,
)
from ipfs_accelerate_py.model_catalog.resolver import (
    CatalogResolver,
    ResolutionRequest,
    ResolutionResult,
)
from ipfs_accelerate_py.model_catalog.schema import (
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
from ipfs_accelerate_py.model_catalog.snapshot import (
    CatalogPage,
    InvalidCursorError,
    StaleCursorError,
    deserialize_snapshot,
    paginate_snapshot,
    serialize_snapshot,
)


def capability(context=8192):
    return CapabilityDescriptor(
        operations=(Operation.TEXT_CHAT, Operation.STREAM),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
        max_context_tokens=context,
    )


def service(
    name,
    *,
    alias=(),
    priority=0,
    context=8192,
    locality="remote",
    device="cpu",
    health=True,
    policy="standard",
):
    cap = capability(context)
    provider = ProviderDescriptor(
        name=name,
        aliases=alias,
        capabilities=(cap,),
        lifecycle=LifecycleState.READY,
        state=OperationalState(known=True, configured=True),
        labels={"locality": locality, "policy.tier": policy},
    )
    model = ModelDescriptor(
        provider_id=provider.provider_id,
        name="%s-chat" % name,
        aliases=("%s-model" % name,),
        capabilities=(cap,),
        lifecycle=LifecycleState.READY,
        labels={"device": device},
    )
    deployment = DeploymentDescriptor(
        provider_id=provider.provider_id,
        model_id=model.model_id,
        name="%s-production" % name,
        endpoint_uri="https://%s.example.test/v1" % name,
        capabilities=(cap,),
        lifecycle=LifecycleState.READY,
        state=OperationalState(
            authorized=True,
            reachable=True,
            healthy=health,
        ),
    )
    binding = RouterBinding(
        router="llm_router",
        provider_id=provider.provider_id,
        model_id=model.model_id,
        deployment_id=deployment.deployment_id,
        operations=(Operation.TEXT_CHAT, Operation.STREAM),
        priority=priority,
        state=OperationalState(routable=True),
    )
    return provider, model, deployment, binding


def register_service(registry, records, source="router", precedence=10):
    registry.register_many(records, source=source, precedence=precedence)


def test_duplicate_claims_coalesce_and_complementary_facts_merge():
    provider = ProviderDescriptor(
        name="merged",
        aliases=("merge-alias",),
        state=OperationalState(known=True),
        provenance=(Provenance(source="static"),),
    )
    configured = dataclasses.replace(
        provider,
        aliases=("second-alias",),
        state=OperationalState(configured=True),
        provenance=(Provenance(source="runtime"),),
    )
    registry = CatalogRegistry()
    registry.register(provider, source="static")
    registry.register(provider, source="static")  # idempotent per source
    registry.register(configured, source="runtime")

    snapshot = registry.snapshot(at="2030-01-01T00:00:00Z")
    assert len(snapshot.providers) == 1
    merged = snapshot.providers[0]
    assert merged.aliases == ("merge-alias", "second-alias")
    assert merged.state.known is True
    assert merged.state.configured is True
    assert {item.source for item in merged.provenance} == {"static", "runtime"}
    assert len(registry.claims(provider.provider_id)) == 2


def test_equal_authority_conflict_remains_visible_and_fails_closed():
    first = ProviderDescriptor(name="conflict", description="first")
    second = dataclasses.replace(first, description="second")
    registry = CatalogRegistry()
    registry.register(first, source="source-a", precedence=5)
    registry.register(second, source="source-b", precedence=5)

    view = registry.view(at="2030-01-01T00:00:00Z")
    assert not view.snapshot.providers
    assert len(view.claims) == 2
    diagnostic = next(item for item in view.diagnostics if item.code == "ambiguous_claim")
    assert diagnostic.ambiguous
    assert diagnostic.field == "description"


def test_source_precedence_resolves_conflict_without_hiding_losing_claim():
    lower = ProviderDescriptor(name="preferred", description="static")
    higher = dataclasses.replace(lower, description="router")
    registry = CatalogRegistry(source_precedence={"static": 1, "router": 20})
    registry.register(lower, source="static")
    registry.register(higher, source="router")

    view = registry.view(at="2030-01-01T00:00:00Z")
    assert view.snapshot.providers[0].description == "router"
    assert {claim.record.description for claim in view.claims} == {"static", "router"}
    diagnostic = next(item for item in view.diagnostics if item.code == "precedence_conflict")
    assert diagnostic.winner_source == "router"
    assert not diagnostic.ambiguous


def test_aliases_resolve_and_collisions_fail_closed():
    one = ProviderDescriptor(name="one", aliases=("shared", "one-alias"))
    two = ProviderDescriptor(name="two", aliases=("shared",))
    registry = CatalogRegistry((one, two))
    assert registry.resolve_alias("provider", "one-alias") == one.provider_id
    with pytest.raises(AmbiguousAliasError):
        registry.resolve_alias("provider", "shared")
    assert any(item.code == "alias_collision" for item in registry.diagnostics())


def test_expired_claim_is_excluded_at_snapshot_time_but_evidence_remains():
    stale = ProviderDescriptor(
        name="stale",
        provenance=(
            Provenance(
                source="remote",
                observed_at="2029-01-01T00:00:00Z",
                expires_at="2029-01-02T00:00:00Z",
            ),
        ),
    )
    registry = CatalogRegistry()
    registry.register(stale, source="remote")
    before = registry.view(at="2029-01-01T12:00:00Z")
    after = registry.view(at="2029-01-03T00:00:00Z")
    assert before.snapshot.providers == (stale,)
    assert not after.snapshot.providers
    assert after.claims[0].record == stale
    assert after.diagnostics[0].code == "stale_claim"


def test_snapshot_cid_depends_only_on_canonical_content_and_order_is_stable():
    providers = [ProviderDescriptor(name="p-%d" % index) for index in range(4)]
    left = CatalogRegistry(reversed(providers)).snapshot(
        at="2030-01-01T00:00:00Z",
        created_at="2030-01-01T00:00:00Z",
    )
    right = CatalogRegistry(providers).snapshot(
        at="2031-01-01T00:00:00Z",
        created_at="2031-01-01T00:00:00Z",
    )
    assert left.cid == right.cid
    assert left.created_at != right.created_at
    assert left.providers == right.providers
    assert [item.provider_id for item in left.providers] == sorted(
        item.provider_id for item in providers
    )
    changed = CatalogRegistry(providers + [ProviderDescriptor(name="extra")]).snapshot()
    assert changed.cid != left.cid


def test_pagination_cursor_is_snapshot_and_query_bound():
    registry = CatalogRegistry(ProviderDescriptor(name="page-%d" % index) for index in range(5))
    original = registry.snapshot(at="2030-01-01T00:00:00Z")
    first = paginate_snapshot(original, "providers", limit=2)
    second = paginate_snapshot(original, "providers", limit=2, cursor=first.next_cursor)
    third = paginate_snapshot(original, "providers", limit=2, cursor=second.next_cursor)
    assert [len(first.items), len(second.items), len(third.items)] == [2, 2, 1]
    assert first.total == second.total == third.total == 5
    assert third.next_cursor is None

    registry.register(ProviderDescriptor(name="page-new"))
    changed = registry.snapshot(at="2030-01-01T00:00:00Z")
    with pytest.raises(StaleCursorError):
        paginate_snapshot(changed, "providers", cursor=first.next_cursor)
    with pytest.raises(InvalidCursorError):
        paginate_snapshot(original, "models", cursor=first.next_cursor)
    with pytest.raises(InvalidCursorError):
        paginate_snapshot(original, "providers", cursor=first.next_cursor[:-2] + "xx")

    filtered = paginate_snapshot(
        original,
        "providers",
        limit=1,
        predicate=lambda item: item.name.endswith(("0", "2")),
        query={"suffixes": ["0", "2"]},
    )
    with pytest.raises(InvalidCursorError):
        paginate_snapshot(
            original,
            "providers",
            limit=1,
            cursor=filtered.next_cursor,
            predicate=lambda item: item.name.endswith(("0", "2")),
            query={"suffixes": ["0"]},
        )


def test_resolution_filters_complete_intersection_and_explains_ranking():
    local = service(
        "local",
        alias=("preferred",),
        priority=5,
        context=16_384,
        locality="local",
        device="cuda",
        policy="premium",
    )
    remote = service("remote", priority=1, context=4_096, policy="standard")
    registry = CatalogRegistry()
    register_service(registry, remote)
    register_service(registry, local)
    snapshot = registry.snapshot(at="2030-01-01T00:00:00Z")

    result = CatalogResolver().resolve(
        snapshot,
        operation=Operation.TEXT_CHAT,
        modality=Modality.TEXT,
        provider="preferred",
        model="local-model",
        deployment="local-production",
        policy={"tier": "premium"},
        device="cuda",
        context_tokens=12_000,
        healthy=True,
        locality="local",
        configured=True,
        authorized=True,
        reachable=True,
        routable=True,
    )
    assert result.found
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.provider.name == "local"
    assert candidate.model.name == "local-chat"
    assert candidate.deployment.name == "local-production"
    assert any("binding priority" in reason for reason in candidate.reasons)
    assert any("positive state" in reason for reason in candidate.reasons)


def test_ranking_is_deterministic_and_uses_priority_then_stable_identity():
    high = service("high", priority=10)
    low = service("low", priority=1)
    tied_a = service("tie-a", priority=3)
    tied_b = service("tie-b", priority=3)
    all_records = high + low + tied_b + tied_a
    expected = None
    for records in (all_records, tuple(reversed(all_records))):
        registry = CatalogRegistry()
        register_service(registry, records)
        result = CatalogResolver().resolve(
            registry.snapshot(at="2030-01-01T00:00:00Z"),
            operation="text.chat",
        )
        order = tuple(item.provider.name for item in result.candidates)
        assert order[0] == "high"
        assert order[-1] == "low"
        expected = order if expected is None else expected
        assert order == expected


@pytest.mark.parametrize(
    "constraints,fragment",
    [
        ({"context": 100_000}, "context"),
        ({"device": "tpu"}, "device"),
        ({"policy": {"tier": "forbidden"}}, "policy"),
        ({"health": False}, "healthy"),
        ({"locality": "local"}, "locality"),
        ({"modality": "audio"}, "modality"),
        ({"provider": "absent"}, "provider"),
    ],
)
def test_no_candidate_reasons_cover_constraint_failures(constraints, fragment):
    records = service("only", context=2_048)
    registry = CatalogRegistry()
    register_service(registry, records)
    result = CatalogResolver().resolve(
        registry.snapshot(at="2030-01-01T00:00:00Z"), operation="text.chat", **constraints
    )
    assert not result.found
    assert result.total_candidates == 0
    assert "no candidates" in result.reasons[0]
    assert any(fragment in reason for reason in result.reasons[1:])


def test_model_alias_collision_is_scoped_by_provider_but_global_use_fails():
    first = service("first")
    second = service("second")
    second_model = dataclasses.replace(second[1], aliases=("first-model",))
    second = (second[0], second_model, second[2], second[3])
    registry = CatalogRegistry()
    register_service(registry, first)
    register_service(registry, second)
    snapshot = registry.snapshot(at="2030-01-01T00:00:00Z")
    resolver = CatalogResolver()

    ambiguous = resolver.resolve(snapshot, operation="text.chat", model="first-model")
    assert not ambiguous.found
    assert any("ambiguous" in reason for reason in ambiguous.reasons)
    scoped = resolver.resolve(
        snapshot,
        operation="text.chat",
        provider="first",
        model="first-model",
    )
    assert scoped.found
    assert scoped.candidates[0].provider.name == "first"


def test_registry_iteration_and_concurrent_readers_are_deterministic_and_safe():
    records = service("concurrent")
    registry = CatalogRegistry()
    register_service(registry, records, source="initial")
    errors = []
    start = threading.Barrier(5)

    def reader():
        start.wait()
        observed = []
        try:
            for _ in range(100):
                snapshot = registry.snapshot(at="2030-01-01T00:00:00Z")
                observed.append(
                    tuple(
                        getattr(
                            item,
                            "provider_id",
                            getattr(item, "binding_id", ""),
                        )
                        for item in snapshot.providers
                        + snapshot.models
                        + snapshot.deployments
                        + snapshot.bindings
                    )
                )
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)
        return observed

    def writer():
        start.wait()
        provider = records[0]
        for index in range(100):
            registry.register(
                dataclasses.replace(provider, description="revision-%d" % index),
                source="writer",
                precedence=20,
            )

    with ThreadPoolExecutor(max_workers=5) as pool:
        readers = [pool.submit(reader) for _ in range(4)]
        write = pool.submit(writer)
        observations = [future.result() for future in readers]
        write.result()
    assert not errors
    assert all(len(items) == 100 for items in observations)
    assert tuple(registry) == tuple(
        registry.snapshot().providers
        + registry.snapshot().models
        + registry.snapshot().deployments
        + registry.snapshot().bindings
    )


def test_serialization_round_trips_for_snapshot_page_view_and_resolution():
    records = service("roundtrip", alias=("rt",))
    registry = CatalogRegistry()
    register_service(registry, records)
    view = registry.view(at="2030-01-01T00:00:00Z")
    snapshot = deserialize_snapshot(serialize_snapshot(view.snapshot))
    assert snapshot == view.snapshot
    assert RegistryView.from_dict(json.loads(json.dumps(view.to_dict(), sort_keys=True))) == view
    assert (
        RegistryDiagnostic.from_dict(view.diagnostics[0].to_dict()) == view.diagnostics[0]
        if view.diagnostics
        else True
    )

    page = paginate_snapshot(snapshot, "providers", limit=1)
    assert CatalogPage.from_dict(page.to_dict()) == page
    request = ResolutionRequest(
        operation="text.chat",
        provider="rt",
        modality="text",
        context=1_024,
        health=True,
    )
    assert ResolutionRequest.from_dict(request.to_dict()) == request
    result = CatalogResolver().resolve(snapshot, request)
    assert (
        ResolutionResult.from_dict(json.loads(json.dumps(result.to_dict(), sort_keys=True)))
        == result
    )
