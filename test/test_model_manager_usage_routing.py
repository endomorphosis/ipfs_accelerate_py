"""Usage snapshots and usage-aware resolution via ModelManager (AICAT-028)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import ipfs_accelerate_py.model_manager as model_manager_module
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.resolution import (
    USAGE_AWARE_RESOLUTION_REQUIREMENT_ID,
    USAGE_REVISION_OFF,
    RevisionMismatch,
    StaleSnapshotPolicy,
    UnknownLimitPolicy,
    UsageRoutingRequest,
    UsageServiceUnavailable,
    composite_usage_revision,
    hard_filter_candidate,
    ranking_sort_key,
    resolve_usage_aware,
    saturation_micros,
    tightest_dimensions,
    StaticCandidate,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    AvailabilityState,
    DimensionHeadroom,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    Provenance,
    Quantity,
    QuantityKind,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageLimit,
    UsageSnapshot,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore
from ipfs_accelerate_py.endpoint_usage.schema import EndpointUsageScope, ProtocolKind
from ipfs_accelerate_py.model_catalog import (
    CapabilityDescriptor,
    CatalogSnapshot,
    DeploymentDescriptor,
    Modality,
    ModelDescriptor,
    Operation,
    OperationalState,
    ProviderDescriptor,
    Provenance as CatalogProvenance,
    RouterBinding,
)
from ipfs_accelerate_py.model_catalog.catalog import AIServiceCatalog
from ipfs_accelerate_py.model_catalog.sources.static import (
    CatalogSourceResult,
    SourceMetadata,
)
from ipfs_accelerate_py.model_manager import ModelManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager_factory(monkeypatch, tmp_path):
    for name in (
        "HAVE_STORAGE_WRAPPER",
        "HAVE_IPFS_KIT_STORAGE",
        "HAVE_DATASETS_INTEGRATION",
        "HAVE_GRAPHRAG",
    ):
        monkeypatch.setattr(model_manager_module, name, False)
    created = []

    def factory(*, catalog=None, project_legacy_models=False, usage_service=None, name=None):
        path = tmp_path / (name or "mgr-%d.json" % len(created))
        manager = ModelManager(
            storage_path=str(path),
            use_database=False,
            enable_ipfs=False,
            catalog=catalog,
            project_legacy_models=project_legacy_models,
            usage_service=usage_service,
        )
        created.append(manager)
        return manager

    yield factory
    for manager in created:
        manager.close()


def _scope(provider_id: str, operation: str = "text.chat", **overrides) -> EndpointUsageScope:
    defaults = {
        "provider_id": provider_id,
        "protocol": ProtocolKind.HTTPS,
        "operation": operation,
        "deployment_id": stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        "credential_pseudonym": credential_configuration_pseudonym(
            "env:USAGE_TEST_KEY", key_id="usage-default"
        ),
    }
    defaults.update(overrides)
    return EndpointUsageScope(**defaults)


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    used: int = 0,
    remaining: int | None = None,
    enforcement: LimitEnforcement = LimitEnforcement.HARD,
    currency: str | None = None,
    reset_at: str | None = None,
) -> UsageLimit:
    rem = ceiling - used if remaining is None else remaining
    window = LimitWindow(
        kind=WindowKind.FIXED,
        length_ms=60_000,
        reset_at=reset_at,
    )
    kwargs = {
        "scope_id": scope_id,
        "dimension": dimension,
        "ceiling": Quantity.finite(ceiling),
        "window": window,
        "remaining": Quantity.finite(rem),
        "used": Quantity.finite(used),
        "enforcement": enforcement,
        "provenance": Provenance(source=LimitSource.CONFIGURED),
    }
    if currency is not None:
        kwargs["currency"] = currency
    return UsageLimit(**kwargs)


def _snapshot(
    scope_id: str,
    *,
    state: AvailabilityState = AvailabilityState.AVAILABLE,
    limits=(),
    headroom=(),
    next_eligible_at: str | None = None,
    observed_at: str = "2026-07-28T12:00:00Z",
    fresh_until: str = "2026-07-28T12:05:00Z",
    reason_codes=(),
) -> UsageSnapshot:
    if not headroom and limits:
        headroom = tuple(
            DimensionHeadroom(
                dimension=lim.dimension,
                available=lim.remaining
                if lim.remaining.kind is QuantityKind.FINITE
                else Quantity.finite(0),
                ceiling=lim.ceiling,
                reserved=Quantity.finite(0),
                currency=lim.currency,
                state=state,
                next_eligible_at=next_eligible_at,
            )
            for lim in limits
        )
    return UsageSnapshot(
        scope_id=scope_id,
        observed_at=observed_at,
        fresh_until=fresh_until,
        state=state,
        limits=tuple(limits),
        headroom=tuple(headroom),
        next_eligible_at=next_eligible_at,
        reason_codes=tuple(reason_codes),
    )


def _catalog_pair():
    """Two providers/models/bindings for ranking and rejection tests."""

    capability = CapabilityDescriptor(
        operations=(Operation.TEXT_CHAT,),
        input_modalities=(Modality.TEXT,),
        output_modalities=(Modality.TEXT,),
    )
    prov = (CatalogProvenance(source="usage.fixture"),)
    p_a = ProviderDescriptor(
        name="provider-a",
        capabilities=(capability,),
        state=OperationalState(configured=True, authorized=True, routable=True, healthy=True),
        provenance=prov,
        labels=(("locality", "remote"),),
    )
    p_b = ProviderDescriptor(
        name="provider-b",
        capabilities=(capability,),
        state=OperationalState(configured=True, authorized=True, routable=True, healthy=True),
        provenance=prov,
        labels=(("locality", "local"),),
    )
    m_a = ModelDescriptor(
        provider_id=p_a.provider_id,
        name="model-a",
        capabilities=(capability,),
        provenance=prov,
    )
    m_b = ModelDescriptor(
        provider_id=p_b.provider_id,
        name="model-b",
        capabilities=(capability,),
        provenance=prov,
    )
    d_a = DeploymentDescriptor(
        provider_id=p_a.provider_id,
        model_id=m_a.model_id,
        name="deploy-a",
        endpoint_uri="https://api.a.example.test/v1",
        state=OperationalState(configured=True, authorized=True, routable=True, healthy=True),
        provenance=prov,
        labels=(("locality", "remote"),),
    )
    d_b = DeploymentDescriptor(
        provider_id=p_b.provider_id,
        model_id=m_b.model_id,
        name="deploy-b",
        endpoint_uri="https://api.b.example.test/v1",
        state=OperationalState(configured=True, authorized=True, routable=True, healthy=True),
        provenance=prov,
        labels=(("locality", "local"),),
    )
    b_a = RouterBinding(
        router="fixture_router",
        provider_id=p_a.provider_id,
        model_id=m_a.model_id,
        deployment_id=d_a.deployment_id,
        operations=(Operation.TEXT_CHAT,),
        priority=10,
        state=OperationalState(configured=True, authorized=True, routable=True, healthy=True),
        provenance=prov,
        labels=(("locality", "remote"),),
    )
    b_b = RouterBinding(
        router="fixture_router",
        provider_id=p_b.provider_id,
        model_id=m_b.model_id,
        deployment_id=d_b.deployment_id,
        operations=(Operation.TEXT_CHAT,),
        priority=5,
        state=OperationalState(configured=True, authorized=True, routable=True, healthy=True),
        provenance=prov,
        labels=(("locality", "local"),),
    )
    snap = CatalogSnapshot(
        providers=(p_a, p_b),
        models=(m_a, m_b),
        deployments=(d_a, d_b),
        bindings=(b_a, b_b),
    )
    return snap, b_a, b_b, p_a, p_b


class _MemorySource:
    """Minimal non-side-effecting catalog source for tests."""

    def __init__(self, source: str, snapshot: CatalogSnapshot, *, precedence: int = 10) -> None:
        self.source = source
        self.precedence = precedence
        self.side_effecting = False
        self.current = snapshot
        self.load_calls = 0

    def _result(self) -> CatalogSourceResult:
        return CatalogSourceResult(
            snapshot=self.current,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=self.current.revision,
            ),
        )

    def load(self) -> CatalogSourceResult:
        self.load_calls += 1
        return self._result()

    def refresh(self) -> CatalogSourceResult:
        return self._result()


def _loaded_catalog(snapshot: CatalogSnapshot) -> AIServiceCatalog:
    source = _MemorySource("usage.fixture", snapshot, precedence=10)
    return AIServiceCatalog({source.source: source})


# ---------------------------------------------------------------------------
# Requirement / pure resolution
# ---------------------------------------------------------------------------


def test_requirement_id_stable():
    assert USAGE_AWARE_RESOLUTION_REQUIREMENT_ID == "requirement:usage-aware-resolution.v1"


def test_unlike_dimensions_never_summed_into_one_score():
    headroom = (
        DimensionHeadroom(
            dimension=UsageDimension.INPUT_TOKENS,
            available=Quantity.finite(500),
            ceiling=Quantity.finite(1000),
            state=AvailabilityState.AVAILABLE,
        ),
        DimensionHeadroom(
            dimension=UsageDimension.REQUESTS,
            available=Quantity.finite(1),
            ceiling=Quantity.finite(10),
            state=AvailabilityState.NEAR_LIMIT,
        ),
    )
    required = UsageVector.of(input_tokens=50, requests=1)
    tight = tightest_dimensions(headroom, required)
    assert tight[0] == "requests"
    sat_req = saturation_micros(headroom[1].available, headroom[1].ceiling)
    sat_tok = saturation_micros(headroom[0].available, headroom[0].ceiling)
    assert sat_req == 900_000
    assert sat_tok == 500_000
    # Sort key uses per-dimension saturations, not a sum.
    key_a = ranking_sort_key(
        {
            "tightest_dimension": "requests",
            "second_tightest_dimension": "input_tokens",
            "sat_requests": sat_req,
            "sat_input_tokens": sat_tok,
            "catalog_score": 10,
        }
    )
    key_b = ranking_sort_key(
        {
            "tightest_dimension": "requests",
            "second_tightest_dimension": "input_tokens",
            "sat_requests": 100_000,
            "sat_input_tokens": 999_000,  # worse secondary, better primary
            "catalog_score": 10,
        }
    )
    assert key_b < key_a
    # Primary dimension alone decides; unlike units are never summed.
    assert key_a[6] == sat_req
    assert key_b[6] == 100_000
    # Secondary is a later key — a worse secondary cannot offset a better primary.
    assert key_b[7] == 999_000
    assert key_a[7] == sat_tok


def test_hard_filter_rejects_exhausted_before_score():
    scope = _scope(stable_id("provider", "hard-filter"))
    snap = _snapshot(
        scope.scope_id,
        state=AvailabilityState.EXHAUSTED,
        limits=(
            _limit(scope.scope_id, UsageDimension.REQUESTS, 10, used=10, remaining=0),
        ),
        next_eligible_at="2026-07-28T13:00:00Z",
    )
    cand = StaticCandidate(
        binding_id=stable_id("binding", "hard"),
        provider_id=scope.provider_id,
        scope_id=scope.scope_id,
        catalog_score=999_999,
    )
    policy = RoutingPolicy(mode=RoutingMode.ENFORCE)
    ureq = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now="2026-07-28T12:00:00Z",
    )
    ok, reasons, _headroom = hard_filter_candidate(cand, snap, ureq, policy)
    assert ok is False
    assert "limit_exhausted" in reasons
    # High catalog score is irrelevant — candidate is rejected.


def test_unknown_and_stale_follow_explicit_policy():
    scope = _scope(stable_id("provider", "stale-unknown"))
    stale = _snapshot(
        scope.scope_id,
        state=AvailabilityState.AVAILABLE,
        limits=(_limit(scope.scope_id, UsageDimension.REQUESTS, 10),),
        observed_at="2026-07-28T11:00:00Z",
        fresh_until="2026-07-28T11:01:00Z",
    )
    cand = StaticCandidate(
        binding_id=stable_id("binding", "stale"),
        provider_id=scope.provider_id,
        scope_id=scope.scope_id,
    )
    policy = RoutingPolicy(mode=RoutingMode.ENFORCE)
    deny = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now="2026-07-28T12:00:00Z",
        stale_snapshot_policy=StaleSnapshotPolicy.DENY,
    )
    ok, reasons, _ = hard_filter_candidate(cand, stale, deny, policy)
    assert ok is False
    assert "stale_snapshot" in reasons

    allow = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now="2026-07-28T12:00:00Z",
        stale_snapshot_policy=StaleSnapshotPolicy.ALLOW,
    )
    ok2, reasons2, _ = hard_filter_candidate(cand, stale, allow, policy)
    assert ok2 is True
    assert "stale_snapshot" not in reasons2

    unknown = _snapshot(scope.scope_id, state=AvailabilityState.UNKNOWN, limits=())
    ureq_unknown = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now="2026-07-28T12:00:00Z",
        unknown_limit_policy=UnknownLimitPolicy.DENY,
    )
    ok3, reasons3, _ = hard_filter_candidate(cand, unknown, ureq_unknown, policy)
    assert ok3 is False
    assert "unknown_state" in reasons3


def test_composite_usage_revision_detects_snapshot_change():
    scope = _scope(stable_id("provider", "rev"))
    a = _snapshot(
        scope.scope_id,
        limits=(_limit(scope.scope_id, UsageDimension.REQUESTS, 10, used=0),),
    )
    b = _snapshot(
        scope.scope_id,
        limits=(_limit(scope.scope_id, UsageDimension.REQUESTS, 10, used=5, remaining=5),),
        observed_at="2026-07-28T12:00:01Z",
    )
    assert a.usage_revision != b.usage_revision
    assert composite_usage_revision([a]) != composite_usage_revision([b])


# ---------------------------------------------------------------------------
# ModelManager facades
# ---------------------------------------------------------------------------


def test_missing_usage_service_preserves_resolve_and_fails_usage_facades(manager_factory):
    snap, _ba, _bb, _pa, _pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    manager = manager_factory(catalog=catalog)
    assert manager.usage_service is None

    # Existing APIs still work.
    result = manager.resolve(operation=Operation.TEXT_CHAT)
    assert result.found
    assert result.snapshot_revision == manager.catalog_revision
    page = manager.list_services(limit=10)
    assert page is not None

    with pytest.raises(UsageServiceUnavailable):
        manager.usage_snapshot("scope_x")
    with pytest.raises(UsageServiceUnavailable):
        manager.list_usage_limits("scope_x")
    with pytest.raises(UsageServiceUnavailable):
        manager.get_endpoint_headroom("scope_x")

    # OFF mode resolve_for_routing works without a service.
    resolution = manager.resolve_for_routing(
        operation=Operation.TEXT_CHAT,
        routing_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert resolution.usage_revision == USAGE_REVISION_OFF
    assert resolution.candidates
    assert "usage_routing_off" in resolution.reason_codes

    # Enforce without service fails closed.
    with pytest.raises(UsageServiceUnavailable):
        manager.resolve_for_routing(
            operation=Operation.TEXT_CHAT,
            routing_policy=RoutingPolicy(mode=RoutingMode.ENFORCE),
            scope_by_binding={result.candidates[0].binding_id: "scope_x"},
        )


def test_usage_snapshot_limits_headroom_side_effect_free(manager_factory):
    clock = FakeClock(datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc))
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w1", fence=1)
    coord = UsageCoordinator(store, writer_id="w1", fence=1)
    scope = _scope(stable_id("provider", "mm-snap"))
    coord.configure_limits(
        scope.scope_id,
        [
            _limit(scope.scope_id, UsageDimension.REQUESTS, 100),
            _limit(scope.scope_id, UsageDimension.INPUT_TOKENS, 10_000),
        ],
    )
    snap, ba, bb, _pa, _pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    before_rev = catalog.snapshot().revision
    manager = manager_factory(catalog=catalog, usage_service=coord)

    usage = manager.usage_snapshot(scope.scope_id)
    assert usage.scope_id == scope.scope_id
    assert usage.usage_revision
    assert usage.state in (
        AvailabilityState.AVAILABLE,
        AvailabilityState.UNKNOWN,
        AvailabilityState.NEAR_LIMIT,
    )

    page = manager.list_usage_limits(scope.scope_id, limit=1)
    assert page.total == 2
    assert len(page.items) == 1
    assert page.next_cursor is not None
    page2 = manager.list_usage_limits(
        scope.scope_id, limit=1, cursor=page.next_cursor
    )
    assert len(page2.items) == 1
    assert page.items[0].limit_id != page2.items[0].limit_id

    filtered = manager.list_usage_limits(
        scope.scope_id, dimension=UsageDimension.REQUESTS
    )
    assert filtered.total == 1
    assert filtered.items[0].dimension is UsageDimension.REQUESTS

    headroom = manager.get_endpoint_headroom(scope.scope_id)
    assert headroom
    dims = {item.dimension for item in headroom}
    assert UsageDimension.REQUESTS in dims
    assert UsageDimension.INPUT_TOKENS in dims

    only_req = manager.get_endpoint_headroom(
        scope.scope_id, dimension=UsageDimension.REQUESTS
    )
    assert len(only_req) == 1

    # Static catalog CID/revision unchanged by usage reads.
    assert manager.catalog_revision == before_rev
    assert catalog.snapshot().revision == before_rev
    # No reservation created by read paths.
    assert usage.reservations == ()


def test_resolve_for_routing_hard_filters_then_ranks(manager_factory):
    clock = FakeClock(datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc))
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w1", fence=1)
    coord = UsageCoordinator(store, writer_id="w1", fence=1)

    snap, ba, bb, pa, pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    catalog_rev = catalog.snapshot().revision

    scope_a = _scope(pa.provider_id, deployment_id=ba.deployment_id)
    scope_b = _scope(
        pb.provider_id,
        deployment_id=bb.deployment_id,
        credential_pseudonym=credential_configuration_pseudonym(
            "env:USAGE_TEST_KEY_B", key_id="usage-default"
        ),
    )
    # Inject immutable snapshots: A exhausted, B has headroom.
    # Catalog priority prefers A (10 > 5); usage hard-filter must select B.
    snap_a = _snapshot(
        scope_a.scope_id,
        state=AvailabilityState.EXHAUSTED,
        limits=(_limit(scope_a.scope_id, UsageDimension.REQUESTS, 1, used=1, remaining=0),),
        headroom=(
            DimensionHeadroom(
                dimension=UsageDimension.REQUESTS,
                available=Quantity.finite(0),
                ceiling=Quantity.finite(1),
                state=AvailabilityState.EXHAUSTED,
                next_eligible_at="2026-07-28T13:00:00Z",
            ),
            DimensionHeadroom(
                dimension=UsageDimension.INPUT_TOKENS,
                available=Quantity.finite(0),
                ceiling=Quantity.finite(100),
                state=AvailabilityState.EXHAUSTED,
            ),
        ),
        next_eligible_at="2026-07-28T13:00:00Z",
        reason_codes=("limit_exhausted",),
    )
    snap_b = _snapshot(
        scope_b.scope_id,
        state=AvailabilityState.AVAILABLE,
        limits=(
            _limit(scope_b.scope_id, UsageDimension.REQUESTS, 100),
            _limit(scope_b.scope_id, UsageDimension.INPUT_TOKENS, 50_000),
        ),
        headroom=(
            DimensionHeadroom(
                dimension=UsageDimension.REQUESTS,
                available=Quantity.finite(100),
                ceiling=Quantity.finite(100),
                state=AvailabilityState.AVAILABLE,
            ),
            DimensionHeadroom(
                dimension=UsageDimension.INPUT_TOKENS,
                available=Quantity.finite(50_000),
                ceiling=Quantity.finite(50_000),
                state=AvailabilityState.AVAILABLE,
            ),
        ),
    )

    manager = manager_factory(catalog=catalog, usage_service=coord)
    static = manager.resolve(operation=Operation.TEXT_CHAT)
    assert static.candidates[0].binding_id == ba.binding_id  # higher priority

    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.CROSS_PROVIDER,
        prefer_local=True,
    )
    ureq = UsageRoutingRequest(
        required=UsageVector.of(requests=1, input_tokens=100),
        now=clock.to_rfc3339(),
        latency_ms_by_binding={ba.binding_id: 50, bb.binding_id: 20},
        health_by_binding={ba.binding_id: True, bb.binding_id: True},
    )
    resolution = manager.resolve_for_routing(
        operation=Operation.TEXT_CHAT,
        usage_request=ureq,
        routing_policy=policy,
        scope_by_binding={
            ba.binding_id: scope_a.scope_id,
            bb.binding_id: scope_b.scope_id,
        },
        snapshots_by_scope={
            scope_a.scope_id: snap_a,
            scope_b.scope_id: snap_b,
        },
        expected_catalog_revision=catalog_rev,
    )

    assert resolution.catalog_revision == catalog_rev
    assert resolution.usage_revision.startswith("urev_")
    assert resolution.selected_binding_id == bb.binding_id
    assert [c.binding_id for c in resolution.candidates] == [bb.binding_id]
    rejected_ids = {c.binding_id for c in resolution.rejected}
    assert ba.binding_id in rejected_ids
    rejected_a = next(c for c in resolution.rejected if c.binding_id == ba.binding_id)
    assert rejected_a.rejection_reasons
    assert any(
        code in rejected_a.rejection_reasons
        for code in ("limit_exhausted", "insufficient_headroom_requests")
    )
    # Explanations expose ranking inputs / headroom / next eligible.
    winner = resolution.candidates[0]
    ranking_names = {name for name, _ in winner.ranking_inputs}
    assert "tightest_dimension" in ranking_names or "catalog_score" in ranking_names
    assert winner.headroom
    # Static catalog revision still identical.
    assert manager.catalog_revision == catalog_rev


def test_resolve_for_routing_exposes_explanation_fields(manager_factory):
    scope = _scope(stable_id("provider", "explain"))
    reset = "2026-07-28T13:00:00Z"
    snap = _snapshot(
        scope.scope_id,
        state=AvailabilityState.NEAR_LIMIT,
        limits=(
            _limit(
                scope.scope_id,
                UsageDimension.INPUT_TOKENS,
                1000,
                used=900,
                remaining=100,
                reset_at=reset,
            ),
            _limit(scope.scope_id, UsageDimension.REQUESTS, 10, used=1, remaining=9),
        ),
        next_eligible_at=reset,
    )
    binding_id = stable_id("binding", "explain")
    cand = StaticCandidate(
        binding_id=binding_id,
        provider_id=scope.provider_id,
        scope_id=scope.scope_id,
        catalog_score=42,
        locality="local",
        healthy=True,
    )
    policy = RoutingPolicy(mode=RoutingMode.ENFORCE, prefer_local=True)
    ureq = UsageRoutingRequest(
        required=UsageVector.of(input_tokens=50, requests=1),
        now="2026-07-28T12:00:00Z",
        latency_ms_by_binding={binding_id: 12},
        circuit_open_by_binding={binding_id: False},
        affinity_binding_id=binding_id,
    )
    resolution = resolve_usage_aware(
        catalog_revision="catalog-rev-test",
        candidates=[cand],
        snapshots_by_scope={scope.scope_id: snap},
        policy=policy,
        request=ureq,
    )
    assert resolution.selected_binding_id == binding_id
    winner = resolution.candidates[0]
    inputs = dict(winner.ranking_inputs)
    assert inputs.get("tightest_dimension") == "input_tokens"
    assert "sat_input_tokens" in inputs
    assert "sat_requests" in inputs
    assert inputs.get("latency_ms") == 12
    assert inputs.get("health") is True
    assert inputs.get("circuit_open") is False
    assert inputs.get("affinity") is True
    assert str(inputs.get("next_eligible_at") or "").startswith("2026-07-28T13:00:00")
    assert str(inputs.get("reset_horizon") or inputs.get("next_eligible_at") or "").startswith(
        "2026-07-28T13:00:00"
    )
    assert winner.headroom
    assert str(resolution.next_eligible_at or "").startswith("2026-07-28T13:00:00")


def test_revision_mismatch_visible_not_silently_mixed(manager_factory):
    clock = FakeClock(datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc))
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w1", fence=1)
    coord = UsageCoordinator(store, writer_id="w1", fence=1)
    scope = _scope(stable_id("provider", "cas"))
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, 50)],
    )
    snap, ba, _bb, _pa, _pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    manager = manager_factory(catalog=catalog, usage_service=coord)

    first = manager.usage_snapshot(scope.scope_id)
    # Mutate usage state.
    coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="req-1",
        attempt_id="1",
        idempotency_key="idem-1",
        owner_id="owner-1",
    )
    second = manager.usage_snapshot(scope.scope_id)
    assert first.usage_revision != second.usage_revision

    with pytest.raises(RevisionMismatch):
        manager.usage_snapshot(
            scope.scope_id, expected_usage_revision=first.usage_revision
        )

    with pytest.raises(RevisionMismatch):
        manager.resolve_for_routing(
            operation=Operation.TEXT_CHAT,
            routing_policy=RoutingPolicy(mode=RoutingMode.ENFORCE),
            usage_request=UsageRoutingRequest(
                required=UsageVector.of(requests=1),
                now=clock.to_rfc3339(),
            ),
            scope_by_binding={ba.binding_id: scope.scope_id},
            snapshots_by_scope={scope.scope_id: first},
            expected_usage_revision=second.usage_revision,
        )


def test_catalog_revision_mismatch_and_static_cids_stable(manager_factory):
    snap, ba, bb, _pa, _pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    manager = manager_factory(catalog=catalog)
    rev = manager.catalog_revision
    provider_cid = snap.providers[0].cid

    with pytest.raises(RevisionMismatch) as exc:
        manager.resolve_for_routing(
            operation=Operation.TEXT_CHAT,
            routing_policy=RoutingPolicy(mode=RoutingMode.OFF),
            expected_catalog_revision="not-the-real-revision",
        )
    assert exc.value.kind == "catalog"

    # Usage-off planning does not mutate catalog records or CIDs.
    resolution = manager.resolve_for_routing(
        operation=Operation.TEXT_CHAT,
        routing_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert manager.catalog_revision == rev
    assert catalog.snapshot().providers[0].cid == provider_cid
    assert resolution.catalog_revision == rev


def test_media_cost_deadline_and_circuit_hard_gates():
    scope = _scope(stable_id("provider", "gates"))
    snap = _snapshot(
        scope.scope_id,
        limits=(
            _limit(scope.scope_id, UsageDimension.MEDIA_BYTES, 1_000),
            _limit(
                scope.scope_id,
                UsageDimension.COST_MICROS,
                5_000,
                currency="USD",
            ),
            _limit(scope.scope_id, UsageDimension.REQUESTS, 10),
        ),
        next_eligible_at="2026-07-28T12:30:00Z",
    )
    # Rebuild headroom with currency for cost.
    snap = UsageSnapshot(
        scope_id=scope.scope_id,
        observed_at="2026-07-28T12:00:00Z",
        fresh_until="2026-07-28T12:05:00Z",
        state=AvailabilityState.AVAILABLE,
        limits=snap.limits,
        headroom=(
            DimensionHeadroom(
                dimension=UsageDimension.MEDIA_BYTES,
                available=Quantity.finite(500),
                ceiling=Quantity.finite(1_000),
                state=AvailabilityState.AVAILABLE,
            ),
            DimensionHeadroom(
                dimension=UsageDimension.COST_MICROS,
                available=Quantity.finite(5_000),
                ceiling=Quantity.finite(5_000),
                currency="USD",
                state=AvailabilityState.AVAILABLE,
            ),
            DimensionHeadroom(
                dimension=UsageDimension.REQUESTS,
                available=Quantity.finite(10),
                ceiling=Quantity.finite(10),
                state=AvailabilityState.AVAILABLE,
            ),
        ),
        next_eligible_at="2026-07-28T12:30:00Z",
    )
    binding_id = stable_id("binding", "gates")
    cand = StaticCandidate(
        binding_id=binding_id,
        provider_id=scope.provider_id,
        scope_id=scope.scope_id,
    )
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        cost_ceiling_micros=1000,
        cost_currency="USD",
        allow_wait=False,
    )
    # Media too large.
    ureq = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        media_bytes=800,
        now="2026-07-28T12:00:00Z",
    )
    ok, reasons, _ = hard_filter_candidate(cand, snap, ureq, policy)
    assert ok is False
    assert "insufficient_headroom_media_bytes" in reasons

    # Cost ceiling exceeded by required vector.
    ureq_cost = UsageRoutingRequest(
        required=UsageVector.of(requests=1, cost_micros=2000, currency="USD"),
        now="2026-07-28T12:00:00Z",
    )
    ok2, reasons2, _ = hard_filter_candidate(cand, snap, ureq_cost, policy)
    assert ok2 is False
    assert "cost_ceiling_exceeded" in reasons2

    # Circuit open.
    ureq_circuit = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now="2026-07-28T12:00:00Z",
        circuit_open_by_binding={binding_id: True},
    )
    # Need a snapshot that covers requests so only circuit rejects.
    ok3, reasons3, _ = hard_filter_candidate(cand, snap, ureq_circuit, policy)
    assert ok3 is False
    assert "circuit_open" in reasons3


def test_pin_constraints_applied_by_static_resolve(manager_factory):
    snap, ba, bb, pa, _pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    manager = manager_factory(catalog=catalog)
    # Explicit provider pin — only A.
    pinned = manager.resolve(
        operation=Operation.TEXT_CHAT,
        provider=pa.provider_id,
    )
    assert pinned.found
    assert all(c.provider_id == pa.provider_id for c in pinned.candidates)

    resolution = manager.resolve_for_routing(
        operation=Operation.TEXT_CHAT,
        provider=pa.provider_id,
        routing_policy=RoutingPolicy(mode=RoutingMode.OFF),
    )
    assert resolution.candidates
    assert all(
        c.binding_id == ba.binding_id or c.binding_id in {ba.binding_id}
        for c in resolution.candidates
    )
    assert bb.binding_id not in {c.binding_id for c in resolution.candidates}


def test_read_paths_do_not_reserve(manager_factory):
    clock = FakeClock(datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc))
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w1", fence=1)
    coord = UsageCoordinator(store, writer_id="w1", fence=1)
    scope = _scope(stable_id("provider", "no-reserve"))
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, 5)],
    )
    snap, ba, bb, pa, pb = _catalog_pair()
    catalog = _loaded_catalog(snap)
    manager = manager_factory(catalog=catalog, usage_service=coord)

    before = manager.usage_snapshot(scope.scope_id)
    manager.list_usage_limits(scope.scope_id)
    manager.get_endpoint_headroom(scope.scope_id)
    manager.resolve_for_routing(
        operation=Operation.TEXT_CHAT,
        routing_policy=RoutingPolicy(mode=RoutingMode.ENFORCE),
        usage_request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=clock.to_rfc3339(),
        ),
        scope_by_binding={
            ba.binding_id: scope.scope_id,
            bb.binding_id: scope.scope_id,
        },
    )
    after = manager.usage_snapshot(scope.scope_id)
    assert before.reservations == ()
    assert after.reservations == ()
    # Headroom unchanged (no reserve side effect).
    assert before.usage_revision == after.usage_revision


def test_pagination_cursor_and_bounds(manager_factory):
    clock = FakeClock(datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc))
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="w1", fence=1)
    coord = UsageCoordinator(store, writer_id="w1", fence=1)
    scope = _scope(stable_id("provider", "page"))
    dims = [
        UsageDimension.REQUESTS,
        UsageDimension.INPUT_TOKENS,
        UsageDimension.OUTPUT_TOKENS,
        UsageDimension.TOTAL_TOKENS,
    ]
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, dim, 100 + i) for i, dim in enumerate(dims)],
    )
    manager = manager_factory(
        catalog=_loaded_catalog(_catalog_pair()[0]),
        usage_service=coord,
    )
    p1 = manager.list_usage_limits(scope.scope_id, limit=2)
    assert p1.total == 4
    assert len(p1.items) == 2
    assert p1.next_cursor
    p2 = manager.list_usage_limits(scope.scope_id, limit=2, cursor=p1.next_cursor)
    assert len(p2.items) == 2
    ids = {item.limit_id for item in p1.items + p2.items}
    assert len(ids) == 4

    with pytest.raises(Exception):
        manager.list_usage_limits(scope.scope_id, limit=0)
    with pytest.raises(Exception):
        manager.list_usage_limits(scope.scope_id, cursor="not-a-real-limit-id")
