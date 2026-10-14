"""Gateway lifecycle checks against an isolated authoritative usage ledger."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import (
    ProviderExecutionError,
    ProviderExecutionGateway,
    ProviderExecutionMode,
    ProviderExecutionPhase,
    SideEffectBoundary,
    build_execution_request,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageScope,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    LimitSource, LimitWindow, Provenance, Quantity, UsageDimension,
    UsageLimit, UsageVector, UsageVectorEntry, WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import InMemoryUsageLedgerStore


SCOPE = stable_id("scope", "gateway-safety")


def _coordinator(kind=UsageCoordinator):
    store = InMemoryUsageLedgerStore()
    coordinator = kind(store)
    coordinator.configure_limits(SCOPE, [
        UsageLimit(
            scope_id=SCOPE, dimension=UsageDimension.REQUESTS,
            ceiling=Quantity.finite(20),
            window=LimitWindow(kind=WindowKind.LIFETIME),
            provenance=Provenance(source=LimitSource.CONFIGURED),
        ),
    ])
    return coordinator, store


def _request(coordinator=None, *, ceiling=4, requested=1, mode=ProviderExecutionMode.ENFORCE):
    scope = SupervisorUsageScope(
        level=SupervisorUsageLevel.REQUEST,
        repository_id="repository:safety", state_id="state:safety", tree_id="tree:safety",
        policy_id="policy:safety", policy_revision="policy:1", supervisor_run_id="run:safety",
        goal_id="goal:safety", objective_id="objective:safety", objective_revision="objective:1",
        task_id="task:safety", attempt=1, stage="inference", lane="lane:safety",
        request_id="request:safety", catalog_revision="catalog:1",
        usage_revision=coordinator.snapshot(SCOPE).usage_revision if coordinator else "usage:1",
        endpoint_scope_id=SCOPE, caller_id="caller:safety", deadline_at="2099-01-01T00:00:00Z",
        idempotency_key="idem:safety", lease_id="lease:safety", fence_id="1",
        parent_scope_id="parent:safety",
    )
    envelope = SupervisorUsageEnvelope(
        scope=scope,
        budget=SupervisorUsageBudget.of(window=LimitWindow(kind=WindowKind.LIFETIME), requests=ceiling),
    )
    bridge = SupervisorToEndpointRequest(
        scope=scope, envelope_id=envelope.envelope_id, endpoint_scope_id=SCOPE,
        catalog_revision=scope.catalog_revision, usage_revision=scope.usage_revision,
        estimated=UsageVector.of(requests=requested), request_id=scope.request_id,
        attempt=scope.attempt, idempotency_key=scope.idempotency_key, caller_id=scope.caller_id,
        deadline_at=scope.deadline_at, lease_id=scope.lease_id, fence_id=scope.fence_id,
    )
    return build_execution_request(
        bridge=bridge, envelope=envelope, provider_id="provider:safety", modality="text",
        side_effect_boundary=SideEffectBoundary.SIDE_EFFECTING, operation="text.generate", mode=mode,
    )


def _invoker(calls):
    def invoke(request):
        calls.append(request.request_id if hasattr(request, "request_id") else request.bridge.request_id)
        return {"units": UsageVector.of(requests=1).to_dict(), "status": "ok"}
    return invoke


def _records(store):
    return tuple(store.read()["reservations"].values())


@pytest.mark.parametrize("degraded", ["", "degraded-budget:label"])
def test_enforce_never_substitutes_simulation_for_missing_coordinator(degraded):
    calls = []
    request = replace(_request(), degraded_budget_id=degraded)
    result = ProviderExecutionGateway(invoker=_invoker(calls)).execute(request)
    assert result.phase is ProviderExecutionPhase.DENIED
    assert "coordinator_required" in result.reason_codes
    assert not result.success and result.receipt is None
    assert calls == []


def test_enforce_requires_invoker_before_reserving_or_charging_real_owner():
    coordinator, store = _coordinator()
    request = _request(coordinator)
    before = coordinator.snapshot(SCOPE)
    gateway = ProviderExecutionGateway(coordinator=coordinator)
    result = gateway.execute(request)
    assert result.phase is ProviderExecutionPhase.DENIED
    assert "invoker_required" in result.reason_codes
    assert not result.success and result.receipt is None and not result.settled.entries
    assert not _records(store) and gateway.invoke_count(request.attempt_key) == 0
    assert coordinator.snapshot(SCOPE).headroom == before.headroom


def test_dispatch_mark_failure_prevents_callback_and_releases_unstarted_hold():
    class RefuseDispatch(UsageCoordinator):
        def mark_dispatched(self, reservation_id):
            raise RuntimeError("dispatch mark unavailable")

    coordinator, store = _coordinator(RefuseDispatch)
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls)).execute(_request(coordinator))
    assert result.phase is ProviderExecutionPhase.FAILED
    assert "dispatch_mark_failed" in result.reason_codes
    assert calls == [] and result.receipt is None
    assert _records(store)[0]["state"] == "released"


def test_lost_dispatch_acknowledgement_conservatively_settles_without_callback():
    class LostDispatchAck(UsageCoordinator):
        def mark_dispatched(self, reservation_id):
            super().mark_dispatched(reservation_id)
            raise RuntimeError("ack lost")

    coordinator, store = _coordinator(LostDispatchAck)
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls)).execute(_request(coordinator))
    assert not result.success and calls == []
    assert _records(store)[0]["state"] == "committed"
    assert result.settled.get(UsageDimension.REQUESTS).amount.value == 1


@pytest.mark.parametrize("acknowledgement", ["missing", "foreign", "wrong_state"])
def test_invalid_dispatch_acknowledgement_cannot_authorize_callback(acknowledgement):
    class InvalidDispatch(UsageCoordinator):
        def mark_dispatched(self, reservation_id):
            result = super().mark_dispatched(reservation_id)
            if acknowledgement == "missing":
                return None
            return SimpleNamespace(
                reservation_id="reservation:foreign" if acknowledgement == "foreign" else result.reservation_id,
                state="committed" if acknowledgement == "wrong_state" else "held",
            )

    coordinator, store = _coordinator(InvalidDispatch)
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls)).execute(_request(coordinator))
    assert not result.success and result.receipt is None and calls == []
    assert "dispatch_mark_failed" in result.reason_codes
    assert _records(store)[0]["state"] == "committed"


def test_unacknowledged_commit_retains_hold_and_caches_uncertain_outcome():
    class RefuseCommit(UsageCoordinator):
        def commit(self, *args, **kwargs):
            raise RuntimeError("settlement unavailable")

    coordinator, store = _coordinator(RefuseCommit)
    calls = []
    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls))
    request = _request(coordinator)
    first = gateway.execute(request)
    replay = gateway.execute(request)
    assert first.final_status is SupervisorUsageFinalStatus.UNKNOWN
    assert not first.success and first.receipt is None and not first.settled.entries
    assert "reconciliation_required" in first.reason_codes
    assert replay.replayed and not replay.success
    assert len(calls) == 1
    record = _records(store)[0]
    assert record["state"] == "held" and record["dispatched"] is True


@pytest.mark.parametrize("acknowledgement", ["missing", "foreign", "wrong_state", "missing_event", "nonfinite"])
def test_invalid_commit_acknowledgement_never_becomes_settlement_evidence(acknowledgement):
    class InvalidCommit(UsageCoordinator):
        def commit(self, *args, **kwargs):
            result = super().commit(*args, **kwargs)
            if acknowledgement == "missing":
                return None
            changes = {}
            if acknowledgement == "foreign":
                changes["reservation_id"] = "reservation:foreign"
            elif acknowledgement == "wrong_state":
                changes["state"] = "held"
            elif acknowledgement == "missing_event":
                changes["event_id"] = ""
            else:
                changes["charged"] = UsageVector(entries=(UsageVectorEntry(
                    dimension=UsageDimension.REQUESTS, amount=Quantity.unknown(),
                ),))
            return replace(result, **changes)

    coordinator, store = _coordinator(InvalidCommit)
    calls = []
    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls))
    request = _request(coordinator)
    result = gateway.execute(request)
    assert result.final_status is SupervisorUsageFinalStatus.UNKNOWN
    assert not result.success and result.receipt is None and not result.settled.entries
    assert "reconciliation_required" in result.reason_codes
    assert gateway.execute(request).replayed and len(calls) == 1
    assert _records(store)[0]["state"] == "committed"


def test_missing_conservative_cancellation_acknowledgement_is_pending():
    class MissingCancelAck(UsageCoordinator):
        def cancel(self, *args, **kwargs):
            super().cancel(*args, **kwargs)
            return None

    coordinator, store = _coordinator(MissingCancelAck)
    request = replace(_request(coordinator), cancelled=True, post_dispatch=True)
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls)).execute(request)
    assert result.final_status is SupervisorUsageFinalStatus.UNKNOWN
    assert not result.success and result.receipt is None and not result.settled.entries
    assert "reconciliation_required" in result.reason_codes
    assert calls == [] and _records(store)[0]["state"] == "committed"


def test_malformed_usage_after_callback_conservatively_charges_without_success():
    coordinator, store = _coordinator()
    gateway = ProviderExecutionGateway(
        coordinator=coordinator, invoker=lambda _request: {"units": {"requests": "bad"}},
    )
    result = gateway.execute(_request(coordinator))
    assert not result.success and result.receipt is None
    assert "invoke_failed" in result.reason_codes
    assert _records(store)[0]["state"] == "committed"
    assert result.settled.get(UsageDimension.REQUESTS).amount.value == 1


def test_actual_envelope_ceiling_is_passed_to_reservation_owner():
    class RecordBudget(UsageCoordinator):
        def reserve(self, *args, **kwargs):
            self.received_budget = kwargs["caller_budget"]
            return super().reserve(*args, **kwargs)

    coordinator, _store = _coordinator(RecordBudget)
    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker([]))
    result = gateway.execute(_request(coordinator, ceiling=4, requested=1))
    assert result.success
    assert coordinator.received_budget.get(UsageDimension.REQUESTS).amount.value == 4


def test_request_above_envelope_is_refused_before_creating_hold():
    coordinator, store = _coordinator()
    calls = []
    result = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls)).execute(
        _request(coordinator, ceiling=1, requested=2)
    )
    assert result.phase is ProviderExecutionPhase.DENIED
    assert "caller_budget_exceeded" in result.reason_codes
    assert calls == [] and not _records(store)


def test_transient_gateway_denial_does_not_cache_forever():
    class DenyFirst(UsageCoordinator):
        denied = False

        def reserve(self, *args, **kwargs):
            if not self.denied:
                self.denied = True
                return SimpleNamespace(granted=False, reason_codes=("capacity_denied",))
            return super().reserve(*args, **kwargs)

    coordinator, _store = _coordinator(DenyFirst)
    calls = []
    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls))
    request = _request(coordinator)
    assert gateway.execute(request).phase is ProviderExecutionPhase.DENIED
    assert gateway.execute(request).success
    assert gateway.execute(request).replayed
    assert len(calls) == 1


def test_per_call_invoker_preserves_gateway_replay_and_has_single_owner():
    coordinator, _store = _coordinator()
    calls = []
    gateway = ProviderExecutionGateway(coordinator=coordinator)
    request = _request(coordinator)
    assert gateway.has_invoker is False
    assert gateway.execute(request, invoker=_invoker(calls)).success
    assert gateway.execute(request, invoker=_invoker(calls)).replayed
    assert len(calls) == 1
    configured = ProviderExecutionGateway(coordinator=coordinator, invoker=_invoker(calls))
    with pytest.raises(ProviderExecutionError, match="one invoker"):
        configured.execute(request, invoker=_invoker(calls))


def test_off_mode_remains_an_explicit_unmetered_invocation():
    calls = []
    result = ProviderExecutionGateway().execute(
        _request(mode=ProviderExecutionMode.OFF), invoker=_invoker(calls),
    )
    assert len(calls) == 1 and result.phase is ProviderExecutionPhase.SETTLED
    assert result.receipt is None and result.final_status is SupervisorUsageFinalStatus.UNKNOWN


def test_caller_idempotency_string_cannot_replay_foreign_provider_outcome():
    calls = []
    gateway = ProviderExecutionGateway(invoker=_invoker(calls))
    first_request = _request(mode=ProviderExecutionMode.OFF)
    first = gateway.execute(first_request)
    second = gateway.execute(replace(first_request, provider_id="provider:other"))
    assert first.provider_id != second.provider_id
    assert len(calls) == 2 and not second.replayed


def test_migrated_text_callback_keeps_all_reserved_token_estimates():
    from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import (
        ConsumerId, build_consumer_call_context, dispatch_migrated_provider_call,
    )

    coordinator, store = _coordinator()
    coordinator.configure_limits(SCOPE, [
        UsageLimit(
            scope_id=SCOPE, dimension=dimension, ceiling=Quantity.finite(1_000),
            window=LimitWindow(kind=WindowKind.LIFETIME),
            provenance=Provenance(source=LimitSource.CONFIGURED),
        )
        for dimension in (UsageDimension.REQUESTS, UsageDimension.INPUT_TOKENS, UsageDimension.OUTPUT_TOKENS)
    ])
    context = build_consumer_call_context(
        consumer_id=ConsumerId.TASK_PROPOSAL_ROUTER,
        provider_id="provider:migration-safety", endpoint_scope_id=SCOPE,
        usage_revision=coordinator.snapshot(SCOPE).usage_revision,
        request_id="request:migration-safety", idempotency_key="idem:migration-safety",
        fence_id="1", estimated_input_tokens=50, estimated_output_tokens=20,
    )
    result = dispatch_migrated_provider_call(
        context=context, invoke=lambda: "authored response", mode="enforce",
        gateway=ProviderExecutionGateway(coordinator=coordinator),
    )
    assert result.execution_result.success
    assert result.execution_result.observation["settlement_basis"] == "estimated_ceiling"
    assert result.execution_result.settled.get(UsageDimension.INPUT_TOKENS).amount.value == 50
    assert result.execution_result.settled.get(UsageDimension.OUTPUT_TOKENS).amount.value == 20
    record, = _records(store)
    assert record["charged_amounts"] == {"requests": 1, "input_tokens": 50, "output_tokens": 20}
