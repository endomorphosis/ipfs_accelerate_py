"""Direct deployment bindings use the real owner and refuse configuration drift."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.direct_provider_usage import DirectProviderUsageBinding
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import (
    ConsumerId,
    ProviderUsageMigrationError,
    build_consumer_call_context,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    LimitEnforcement,
    LimitWindow,
    ProtocolKind,
    Quantity,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore


DIMENSIONS = (UsageDimension.REQUESTS, UsageDimension.INPUT_TOKENS, UsageDimension.OUTPUT_TOKENS)


def _limits(scope, *, dimensions=DIMENSIONS, ceiling=100, enforcement=LimitEnforcement.HARD):
    return [UsageLimit(
        scope_id=scope.scope_id, dimension=dimension, ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.LIFETIME), enforcement=enforcement,
    ) for dimension in dimensions]


@pytest.fixture
def owner():
    clock = FakeClock()
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="direct-writer", fence=7)
    coordinator = UsageCoordinator(store, writer_id="direct-writer", fence=7)
    scope = EndpointUsageScope(
        provider_id=stable_id("provider", "direct-fixture"), protocol=ProtocolKind.HTTPS,
        operation="text.generate", deployment_id=stable_id("deployment", "direct-fixture"),
        credential_pseudonym=stable_id("credential", "direct-fixture"),
    )
    coordinator.configure_limits(scope.scope_id, _limits(scope))
    arguments = dict(
        coordinator=coordinator, scope=scope, provider_id="authored-provider",
        model="authored-model", reservation_ttl_ms=20_000, fence=7,
    )
    binding = DirectProviderUsageBinding(**arguments)
    yield SimpleNamespace(
        clock=clock, store=store, coordinator=coordinator, scope=scope,
        binding=binding, arguments=arguments,
    )
    store.close()


def _context(owner, **overrides):
    view = owner.coordinator.admission_snapshot(owner.scope.scope_id)
    fields = dict(
        consumer_id=ConsumerId.PROMPT_GOAL_PLANNER,
        provider_id=owner.binding.provider_id, endpoint_scope_id=owner.scope.scope_id,
        fence_id="7", catalog_revision=owner.binding.catalog_revision,
        usage_revision=view.snapshot.usage_revision, admission_revision=view.admission_revision,
        side_effect="text.generate", estimated_requests=1, estimated_input_tokens=5,
        estimated_output_tokens=2, request_id="request:direct-fixture",
        idempotency_key="idem:direct-fixture",
    )
    fields.update(overrides)
    return build_consumer_call_context(**fields)


def _dispatch(owner, calls, **overrides):
    arguments = dict(
        consumer_id=ConsumerId.PROMPT_GOAL_PLANNER, provider_id=owner.binding.provider_id,
        model=owner.binding.model, timeout_ms=10_000, estimated_input_tokens=5,
        estimated_output_tokens=2, invoke=lambda: calls.append("invoked") or "authored output",
    )
    arguments.update(overrides)
    return owner.binding.dispatch(**arguments)


def test_explicit_binding_enforces_even_when_environment_is_off(owner, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "off")
    calls = []
    result = _dispatch(owner, calls)
    assert calls == ["invoked"] and result.phase == "settled" and result.metered
    assert result.mode.value == "enforce"
    assert result.usage_receipt.scope.endpoint_scope_id == owner.scope.scope_id
    assert result.usage_receipt.scope.fence_id == "7"
    assert result.usage_receipt.scope.admission_revision.startswith("uadm_")
    assert result.execution_result.observation["settlement_basis"] == "estimated_ceiling"
    reservation = next(iter(owner.store.read()["reservations"].values()))
    assert reservation["charged_amounts"] == {"requests": 1, "input_tokens": 5, "output_tokens": 2}


def test_exact_explicit_context_replay_preserves_pin_and_never_reinvokes(owner):
    context = _context(owner)
    calls = []
    first = _dispatch(owner, calls, context=context)
    owner.clock.advance(milliseconds=10)
    second = _dispatch(owner, calls, context=context)
    assert first.phase == "settled" and second.phase == "replayed"
    assert calls == ["invoked"] and len(owner.store.read()["reservations"]) == 1
    assert second.usage_receipt.scope.admission_revision == context.admission_revision
    assert second.text == "" and "generated_text_unavailable" in second.reason_codes


def test_reconstructed_gateway_never_dispatches_prior_reservation(owner):
    context = _context(owner)
    calls = []
    assert _dispatch(owner, calls, context=context).phase == "settled"
    owner.binding = DirectProviderUsageBinding(**owner.arguments)
    result = _dispatch(owner, calls, context=context)
    assert result.phase == "denied"
    assert "reservation_replay_requires_reconciliation" in result.reason_codes
    assert calls == ["invoked"] and len(owner.store.read()["reservations"]) == 1


def test_supplied_pin_is_not_refreshed_after_budget_mutation(owner):
    context = _context(owner)
    owner.coordinator.configure_caller_budget(owner.scope.scope_id, UsageVector.of(requests=0))
    calls = []
    with pytest.raises(ProviderUsageMigrationError) as error:
        _dispatch(owner, calls, context=context)
    assert "stale_coordination" in error.value.reason_codes
    assert calls == [] and owner.store.read()["reservations"] == {}


@pytest.mark.parametrize("field,value", [
    ("provider_id", "another-provider"), ("model", "another-model"),
    ("timeout_ms", 0), ("timeout_ms", True), ("timeout_ms", 15_000),
    ("estimated_input_tokens", 0), ("estimated_output_tokens", -1),
    ("estimated_input_tokens", float("inf")), ("estimated_output_tokens", True),
])
def test_invalid_dispatch_boundaries_refuse_before_callback_or_ledger_read(owner, monkeypatch, field, value):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid dispatch read the ledger")

    monkeypatch.setattr(owner.coordinator, "admission_snapshot", forbidden)
    calls = []
    with pytest.raises(ProviderUsageMigrationError):
        _dispatch(owner, calls, **{field: value})
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("provider_id", "another-provider"), ("endpoint_scope_id", stable_id("scope", "foreign")),
    ("fence_id", "8"), ("catalog_revision", "catalog:foreign"),
    ("estimated_requests", 2), ("estimated_input_tokens", 6),
    ("estimated_output_tokens", 3), ("side_effect", "text.chat"),
    ("consumer_id", ConsumerId.RESCUE_PLANNER), ("admission_revision", ""),
    ("usage_revision", "usage:placeholder"),
])
def test_explicit_context_cannot_change_deployment_or_drop_pins(owner, field, value):
    context = _context(owner, **{field: value})
    calls = []
    before = owner.store.read()
    with pytest.raises(ProviderUsageMigrationError):
        _dispatch(owner, calls, context=context)
    assert calls == [] and owner.store.read() == before


def test_context_constructor_fields_cannot_override_binding(owner):
    calls = []
    with pytest.raises(ProviderUsageMigrationError) as error:
        _dispatch(owner, calls, endpoint_scope_id=stable_id("scope", "foreign"))
    assert "direct_usage_binding_mismatch" in error.value.reason_codes and calls == []
    with pytest.raises(ProviderUsageMigrationError) as error:
        _dispatch(owner, calls, context=_context(owner), task_id="task:override")
    assert "direct_usage_context_conflict" in error.value.reason_codes


@pytest.mark.parametrize("kind", ["absent", "input_missing", "soft", "unlimited", "unknown"])
def test_caller_budget_cannot_replace_configured_finite_hard_limits(owner, kind):
    limits = _limits(owner.scope)
    if kind == "absent":
        limits = []
    elif kind == "input_missing":
        limits = [item for item in limits if item.dimension is not UsageDimension.INPUT_TOKENS]
    elif kind == "soft":
        limits = _limits(owner.scope, enforcement=LimitEnforcement.SOFT)
    else:
        quantity = Quantity.unlimited() if kind == "unlimited" else Quantity.unknown()
        limits = [replace(item, ceiling=quantity) for item in limits]
    owner.coordinator.configure_limits(owner.scope.scope_id, limits)
    owner.coordinator.configure_caller_budget(
        owner.scope.scope_id, UsageVector.of(requests=100, input_tokens=100, output_tokens=100),
    )
    calls = []
    before = owner.store.read()
    with pytest.raises(ProviderUsageMigrationError) as error:
        _dispatch(owner, calls)
    assert error.value.reason_codes == ("direct_usage_limits_required",)
    assert calls == [] and owner.store.read() == before


def test_finite_zero_quota_is_a_real_typed_denial(owner):
    owner.coordinator.configure_limits(owner.scope.scope_id, _limits(owner.scope, ceiling=0))
    calls = []
    result = _dispatch(owner, calls)
    assert result.phase == "denied" and not result.metered and calls == []
    assert "limit_exhausted" in result.reason_codes
    assert owner.store.read()["reservations"] == {}


def test_fence_takeover_refuses_without_callback(owner):
    owner.store.bump_fence(writer_id="next-writer")
    calls = []
    before = owner.store.read()
    with pytest.raises(ProviderUsageMigrationError) as error:
        _dispatch(owner, calls)
    assert error.value.reason_codes == ("direct_usage_fence_mismatch",)
    assert calls == [] and owner.store.read() == before


def test_limits_changed_after_paired_view_fail_at_authoritative_cas(owner, monkeypatch):
    original = owner.coordinator.admission_snapshot

    def remove_limits(scope_id):
        view = original(scope_id)
        owner.coordinator.configure_limits(scope_id, [])
        return view

    monkeypatch.setattr(owner.coordinator, "admission_snapshot", remove_limits)
    calls = []
    with pytest.raises(ProviderUsageMigrationError) as error:
        _dispatch(owner, calls)
    assert "stale_coordination" in error.value.reason_codes
    assert calls == [] and owner.store.read()["reservations"] == {}


def test_one_paired_view_and_immutable_binding(owner, monkeypatch):
    original = owner.coordinator.admission_snapshot
    reads = []

    def counted(scope_id):
        reads.append(scope_id)
        return original(scope_id)

    monkeypatch.setattr(owner.coordinator, "admission_snapshot", counted)
    assert _dispatch(owner, []).phase == "settled"
    assert reads == [owner.scope.scope_id]
    with pytest.raises(FrozenInstanceError):
        owner.binding.model = "another-model"


@pytest.mark.parametrize("field,value", [
    ("provider_id", "llm_router:auto"), ("model", "auto"), ("model", "efficient"),
    ("provider_id", "router"), ("provider_id", "llm_router"), ("provider_id", "grok"),
    ("provider_id", "*"),
    ("model", ""), ("reservation_ttl_ms", 0), ("settlement_margin_ms", -1),
    ("fence", True), ("fence", 8), ("coordinator", object()),
])
def test_invalid_deployment_binding_is_rejected(owner, field, value):
    arguments = {**owner.arguments, field: value}
    with pytest.raises(ProviderUsageMigrationError):
        DirectProviderUsageBinding(**arguments)


def test_unknown_scope_and_non_text_operation_are_not_direct_authority(owner):
    unknown = EndpointUsageScope(
        provider_id=owner.scope.provider_id, protocol=ProtocolKind.HTTPS,
        operation="text.generate", unknown_scope=True,
    )
    other_operation = EndpointUsageScope(
        provider_id=owner.scope.provider_id, protocol=ProtocolKind.HTTPS,
        operation="text.chat", deployment_id=owner.scope.deployment_id,
        credential_pseudonym=owner.scope.credential_pseudonym,
    )
    for scope in (unknown, other_operation):
        with pytest.raises(ProviderUsageMigrationError):
            DirectProviderUsageBinding(**{**owner.arguments, "scope": scope})


def test_unfenced_coordinator_is_not_accepted(owner):
    coordinator = UsageCoordinator(owner.store)
    with pytest.raises(ProviderUsageMigrationError) as error:
        DirectProviderUsageBinding(**{**owner.arguments, "coordinator": coordinator})
    assert error.value.reason_codes == ("direct_usage_fence_mismatch",)
