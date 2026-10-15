"""Stable admission pins retain supervisor lineage and observational receipts."""

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import (
    CoordinationState,
    ProviderExecutionError,
    ProviderExecutionGateway,
    ProviderExecutionMode,
    ProviderExecutionPhase,
    ProviderExecutionRequest,
    SideEffectBoundary,
    build_execution_request,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage import (
    ProviderUsageValidationError,
    SupervisorToEndpointRequest,
    SupervisorUsageEnvelope,
    SupervisorUsageScope,
    build_child_envelope,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import (
    ConsumerId,
    ProviderUsageMigrationError,
    build_bridge_request,
    build_consumer_call_context,
    build_request_lineage_envelope,
    dispatch_migrated_provider_call,
    request_leaf_envelope,
)
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_usage_rollout import (
    SupervisorStage,
    _execution_request,
    build_harness,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitWindow,
    Quantity,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    FakeClock,
    InMemoryUsageLedgerStore,
    SystemClock,
)


SCOPE = stable_id("scope", "supervisor-admission-revision")
OTHER_SCOPE = stable_id("scope", "supervisor-admission-revision-other")
OTHER_PIN = "uadm_" + "f" * 64


class AdvancingClock(FakeClock):
    def now(self):
        return self.advance(milliseconds=1)


class RecordingCoordinator(UsageCoordinator):
    def __init__(self, store):
        super().__init__(store)
        self.arguments = []

    def reserve(self, *args, **kwargs):
        self.arguments.append(dict(kwargs))
        return super().reserve(*args, **kwargs)


def _limits(scope=SCOPE, ceiling=1_000):
    return [UsageLimit(
        scope_id=scope, dimension=dimension, ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.LIFETIME),
    ) for dimension in (UsageDimension.REQUESTS, UsageDimension.INPUT_TOKENS, UsageDimension.OUTPUT_TOKENS)]


def _coordinator(clock=None):
    store = InMemoryUsageLedgerStore(clock=clock or FakeClock())
    coordinator = RecordingCoordinator(store)
    coordinator.configure_limits(SCOPE, _limits())
    return coordinator, store


def _context(coordinator, *, stable=True, **overrides):
    admission = coordinator.admission_snapshot(SCOPE)
    values = dict(
        consumer_id=ConsumerId.TASK_PROPOSAL_ROUTER,
        provider_id="provider:admission-revision",
        endpoint_scope_id=SCOPE,
        usage_revision=admission.snapshot.usage_revision,
        admission_revision=admission.admission_revision if stable else "",
        request_id="request:admission-revision",
        idempotency_key="idem:admission-revision",
        fence_id="1",
        estimated_input_tokens=5,
        estimated_output_tokens=2,
    )
    values.update(overrides)
    return build_consumer_call_context(**values)


def _request(context, *, mode=ProviderExecutionMode.ENFORCE):
    lineage = build_request_lineage_envelope(context)
    leaf = request_leaf_envelope(lineage)
    bridge = build_bridge_request(context, leaf)
    return build_execution_request(
        bridge=bridge, envelope=lineage, provider_id=context.provider_id,
        modality="text", side_effect_boundary=SideEffectBoundary.SIDE_EFFECTING,
        operation="text.generate", mode=mode,
    )


def _invoke(calls):
    def invoke(request):
        calls.append(request.request_content_id)
        return {"status": "authored", "units": UsageVector.of(requests=1, input_tokens=3, output_tokens=1).to_dict()}
    return invoke


@pytest.mark.parametrize("clock", [AdvancingClock, SystemClock])
def test_real_owner_accepts_stable_pin_with_advancing_clock_and_keeps_observation_identity(clock):
    coordinator, store = _coordinator(clock())
    context = _context(coordinator)
    request = _request(context)
    later = coordinator.admission_snapshot(SCOPE)
    assert later.snapshot.usage_revision != context.usage_revision
    assert later.admission_revision == context.admission_revision
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(request)
    assert result.success and result.phase is ProviderExecutionPhase.SETTLED
    assert len(calls) == 1 and len(store.read()["reservations"]) == 1
    assert coordinator.arguments[0]["expected_admission_revision"] == context.admission_revision
    assert "expected_usage_revision" not in coordinator.arguments[0]
    assert result.receipt.usage_revision == context.usage_revision
    assert result.receipt.scope.admission_revision == context.admission_revision
    assert result.usage_revision != context.usage_revision


def test_stable_pin_is_inherited_by_every_budget_node_and_round_trips():
    coordinator, _store = _coordinator()
    context = _context(coordinator)
    request = _request(context)
    for node in request.envelope.walk():
        assert node.scope.admission_revision == context.admission_revision
        for child in node.children:
            assert child.scope.shares_binding_with(node.scope)
            assert child.budget.is_lower_or_equal(node.budget)
    roundtrip = ProviderExecutionRequest.from_dict(request.to_record())
    assert roundtrip == request
    assert roundtrip.request_content_id == request.request_content_id
    assert SupervisorToEndpointRequest.from_dict(request.bridge.to_record()) == request.bridge


def test_legacy_empty_pin_is_absent_from_all_serialized_contract_payloads():
    coordinator, _store = _coordinator()
    request = _request(_context(coordinator, stable=False))
    assert "admission_revision" not in request.to_record()
    assert "admission_revision" not in request.bridge.to_record()
    for node in request.envelope.walk():
        record = node.scope.to_record()
        assert "admission_revision" not in record
        assert SupervisorUsageScope.from_dict(record).content_id == node.scope.content_id
        explicit_empty = dict(record, admission_revision="")
        assert SupervisorUsageScope.from_dict(explicit_empty).content_id == node.scope.content_id
    assert ProviderExecutionRequest.from_dict(request.to_record()).request_content_id == request.request_content_id


def test_legacy_gateway_keeps_exact_observational_pin_argument():
    coordinator, _store = _coordinator()
    context = _context(coordinator, stable=False)
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(_request(context))
    assert result.success and len(calls) == 1
    assert coordinator.arguments[0]["expected_usage_revision"] == context.usage_revision
    assert "expected_admission_revision" not in coordinator.arguments[0]


def test_legacy_stale_observational_pin_is_never_silently_dropped():
    clock = AdvancingClock()
    coordinator, store = _coordinator(clock)
    request = _request(_context(coordinator, stable=False))
    calls = []
    with pytest.raises(ProviderExecutionError, match="stale"):
        ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(request)
    assert calls == [] and store.read()["reservations"] == {}
    assert "expected_usage_revision" in coordinator.arguments[0]


def test_mutated_ledger_invalidates_stable_pin_before_provider_invocation():
    coordinator, store = _coordinator(AdvancingClock())
    request = _request(_context(coordinator))
    coordinator.configure_limits(SCOPE, _limits(ceiling=900))
    calls = []
    with pytest.raises(ProviderExecutionError, match="stale"):
        ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(request)
    assert calls == [] and store.read()["reservations"] == {}


def test_foreign_endpoint_scope_cannot_reuse_stable_pin():
    coordinator, store = _coordinator()
    coordinator.configure_limits(OTHER_SCOPE, _limits(OTHER_SCOPE))
    context = _context(coordinator, endpoint_scope_id=OTHER_SCOPE)
    calls = []
    with pytest.raises(ProviderExecutionError, match="stale"):
        ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(_request(context))
    assert calls == [] and store.read()["reservations"] == {}


def test_scope_bridge_and_parent_cannot_drop_or_replace_inherited_pin():
    coordinator, _store = _coordinator()
    request = _request(_context(coordinator))
    for pin in ("", OTHER_PIN):
        with pytest.raises(ProviderUsageValidationError, match="admission_revision"):
            replace(request.bridge, admission_revision=pin)
        parent = request.envelope
        child = parent.children[0]
        with pytest.raises(ProviderUsageValidationError, match="stale|foreign"):
            build_child_envelope(
                parent, level=child.scope.level, budget=child.budget,
                admission_revision=pin, supervisor_run_id=child.scope.supervisor_run_id,
            )


@pytest.mark.parametrize("pin", [True, 10, "usage:old", "uadm_bad", "uadm_" + "G" * 64])
def test_malformed_stable_pin_cannot_enter_consumer_context(pin):
    coordinator, _store = _coordinator()
    with pytest.raises(ProviderUsageMigrationError, match="admission_revision"):
        _context(coordinator, admission_revision=pin)


def test_inherited_budget_still_blocks_widened_request_under_valid_stable_pin():
    coordinator, store = _coordinator()
    request = _request(_context(coordinator))
    widened = replace(request, bridge=replace(
        request.bridge, estimated=UsageVector.of(requests=1, input_tokens=6, output_tokens=2),
    ))
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(widened)
    assert result.phase is ProviderExecutionPhase.DENIED
    assert "caller_budget_exceeded" in result.reason_codes
    assert calls == [] and coordinator.arguments == [] and store.read()["reservations"] == {}


def test_pinned_replay_uses_exact_contract_and_does_not_reserve_or_invoke_twice():
    coordinator, store = _coordinator(AdvancingClock())
    request = _request(_context(coordinator))
    calls = []
    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls))
    first = gateway.execute(request)
    second = gateway.execute(request)
    assert first.success and second.success and second.replayed
    assert len(calls) == len(coordinator.arguments) == len(store.read()["reservations"]) == 1


def test_changed_pin_separates_execution_identity_and_gateway_cache():
    coordinator, _store = _coordinator()
    context = _context(coordinator)
    first = _request(context, mode=ProviderExecutionMode.OFF)
    second = _request(replace(context, admission_revision=OTHER_PIN), mode=ProviderExecutionMode.OFF)
    assert first.request_key == second.request_key and first.request_content_id != second.request_content_id
    calls = []
    gateway = ProviderExecutionGateway(invoker=_invoke(calls))
    gateway.execute(first)
    result = gateway.execute(second)
    assert len(calls) == 2 and not result.replayed


def test_refreshed_pin_with_same_attempt_cannot_reinvoke_or_recharge_real_owner():
    coordinator, store = _coordinator(AdvancingClock())
    first = _request(_context(coordinator))
    calls = []
    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls))
    assert gateway.execute(first).success
    refreshed = _request(_context(coordinator))
    assert refreshed.request_key == first.request_key
    assert refreshed.attempt_key == first.attempt_key
    assert refreshed.bridge.admission_revision != first.bridge.admission_revision
    assert refreshed.request_content_id != first.request_content_id
    before = store.read()
    denied = gateway.execute(refreshed)
    assert denied.phase is ProviderExecutionPhase.DENIED and not denied.replayed
    assert len(coordinator.arguments) == 2 and len(calls) == 1
    assert store.read() == before
    record, = store.read()["reservations"].values()
    assert record["charged_amounts"]["requests"] == 1


def test_unsupported_owner_cannot_retry_new_pin_as_legacy_or_unpinned():
    coordinator, store = _coordinator()
    request = _request(_context(coordinator))

    class LegacyOwner:
        body_calls = 0

        def reserve(self, scope_id, requested, *, request_id, attempt_id, idempotency_key,
                    owner_id, lease_id, estimate, expected_usage_revision=None,
                    ttl_ms=30_000, caller_budget=None, fence=None):
            self.body_calls += 1
            raise AssertionError("stable pin was silently discarded")

    owner = LegacyOwner()
    calls = []
    result = ProviderExecutionGateway(coordinator=owner, invoker=_invoke(calls)).execute(request)
    assert result.phase is ProviderExecutionPhase.DENIED
    assert "typeerror" in result.reason_codes and owner.body_calls == 0
    assert calls == [] and store.read()["reservations"] == {}


@pytest.mark.parametrize("coordination", [
    CoordinationState.STALE, CoordinationState.UNKNOWN, CoordinationState.UNAVAILABLE,
])
@pytest.mark.parametrize("stale_pin", [False, True])
def test_enforced_stable_pin_cannot_downgrade_to_degraded_local_budget(coordination, stale_pin):
    coordinator, store = _coordinator()
    request = replace(
        _request(_context(coordinator)), coordination_state=coordination,
        degraded_budget_id="reviewed-budget:local",
    )
    if stale_pin:
        coordinator.configure_limits(SCOPE, _limits(ceiling=900))
    before = store.read()
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(request)
    assert result.phase is ProviderExecutionPhase.DENIED
    assert "coordination_fail_closed" in result.reason_codes
    assert "admission_revision_requires_coordination" in result.reason_codes
    assert calls == [] and coordinator.arguments == []
    assert store.read() == before and store.read()["reservations"] == {}


@pytest.mark.parametrize("coordination", [
    CoordinationState.STALE, CoordinationState.UNKNOWN, CoordinationState.UNAVAILABLE,
])
def test_legacy_degraded_budget_behavior_remains_explicit_without_stable_pin(coordination):
    coordinator, store = _coordinator()
    request = replace(
        _request(_context(coordinator, stable=False)), coordination_state=coordination,
        degraded_budget_id="reviewed-budget:local",
    )
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoke(calls)).execute(request)
    assert result.phase is ProviderExecutionPhase.DEGRADED and calls
    assert coordinator.arguments == [] and store.read()["reservations"] == {}


def test_migrated_consumer_threads_stable_pin_without_losing_estimate_or_receipt_binding():
    coordinator, _store = _coordinator(AdvancingClock())
    context = _context(coordinator)
    calls = []
    result = dispatch_migrated_provider_call(
        context=context, invoke=lambda: calls.append("provider") or "authored result",
        mode="enforce", gateway=ProviderExecutionGateway(coordinator=coordinator),
    )
    assert result.execution_result.success and result.text == "authored result"
    assert calls == ["provider"]
    assert result.execution_result.receipt.scope.admission_revision == context.admission_revision
    assert result.execution_result.receipt.usage_revision == context.usage_revision
    assert result.execution_result.settled.get(UsageDimension.INPUT_TOKENS).amount.value == 5
    assert result.execution_result.settled.get(UsageDimension.OUTPUT_TOKENS).amount.value == 2


def test_rollout_harness_builds_both_bindings_from_one_admission_read(monkeypatch):
    harness = build_harness()
    original = harness.coordinator.admission_snapshot
    reads = []

    def observed(scope):
        result = original(scope)
        reads.append(result)
        return result

    monkeypatch.setattr(harness.coordinator, "admission_snapshot", observed)
    request = _execution_request(harness, stage=SupervisorStage.PLANNING, request_id="request:pinned-rollout")
    assert len(reads) == 1
    assert request.bridge.usage_revision == reads[0].snapshot.usage_revision
    assert request.bridge.admission_revision == reads[0].admission_revision
    assert request.bridge.scope.admission_revision == reads[0].admission_revision


def test_serialized_execution_cannot_override_nested_bridge_pin():
    coordinator, _store = _coordinator()
    request = _request(_context(coordinator))
    record = request.to_record()
    record.pop("content_id", None)
    record.pop("request_content_id", None)
    record["admission_revision"] = OTHER_PIN
    with pytest.raises(ProviderExecutionError, match="admission_revision"):
        ProviderExecutionRequest.from_dict(record)
