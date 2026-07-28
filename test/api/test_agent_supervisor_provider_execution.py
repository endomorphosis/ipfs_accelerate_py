"""Tests for ASI-166 reservation-aware supervisor provider execution gateway."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_execution import (
    RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID,
    CoordinationState,
    ProviderExecutionError,
    ProviderExecutionGateway,
    ProviderExecutionMode,
    ProviderExecutionPhase,
    ProviderExecutionRequest,
    SideEffectBoundary,
    accounting_bounds,
    build_execution_request,
    conservative_estimate,
    discover_schemas,
    new_attempt_idempotency_key,
)
from ipfs_accelerate_py.agent_supervisor.provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageScope,
    build_child_envelope,
)
from ipfs_accelerate_py.endpoint_usage import (
    LimitWindow,
    UsageDimension,
    UsageVector,
    WindowKind,
)


def _window() -> LimitWindow:
    return LimitWindow(kind=WindowKind.LIFETIME)


def _budget(**dimensions: int) -> SupervisorUsageBudget:
    return SupervisorUsageBudget.of(
        window=_window(),
        currency="USD",
        **dimensions,
    )


def _base_scope_kwargs() -> dict[str, Any]:
    return {
        "repository_id": "repository:supervisor",
        "state_id": "state:lane-4",
        "tree_id": "tree:abc123",
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "policy:implementation-daemon@1",
        "catalog_revision": "catalog:rev-1",
        "usage_revision": "usage:rev-1",
        "supervisor_run_id": "",
        "goal_id": "",
        "objective_id": "",
        "objective_revision": "",
        "task_id": "",
        "attempt": 0,
        "stage": "",
        "lane": "",
        "request_id": "",
        "endpoint_scope_id": "",
        "caller_id": "",
        "deadline_at": "",
        "idempotency_key": "",
        "lease_id": "",
        "fence_id": "",
        "parent_scope_id": "",
    }


def _lineage() -> SupervisorUsageEnvelope:
    root = SupervisorUsageEnvelope(
        scope=SupervisorUsageScope(
            level=SupervisorUsageLevel.DEPLOYMENT,
            **_base_scope_kwargs(),
        ),
        budget=_budget(
            requests=1_000,
            input_tokens=100_000,
            output_tokens=50_000,
            cost_micros=10_000_000,
        ),
    )
    run = build_child_envelope(
        root,
        level=SupervisorUsageLevel.RUN,
        budget=_budget(
            requests=100,
            input_tokens=10_000,
            output_tokens=5_000,
            cost_micros=1_000_000,
        ),
        supervisor_run_id="run:1",
    )
    goal = build_child_envelope(
        run,
        level=SupervisorUsageLevel.GOAL,
        budget=_budget(
            requests=50,
            input_tokens=5_000,
            output_tokens=2_500,
            cost_micros=500_000,
        ),
        goal_id="ASI-G510",
        objective_id="objective:usage",
        objective_revision="objective:usage@1",
    )
    task = build_child_envelope(
        goal,
        level=SupervisorUsageLevel.TASK,
        budget=_budget(
            requests=10,
            input_tokens=1_000,
            output_tokens=500,
            cost_micros=100_000,
        ),
        task_id="task:asi-166",
    )
    attempt = build_child_envelope(
        task,
        level=SupervisorUsageLevel.ATTEMPT,
        budget=_budget(
            requests=5,
            input_tokens=500,
            output_tokens=250,
            cost_micros=50_000,
        ),
        attempt=1,
    )
    stage = build_child_envelope(
        attempt,
        level=SupervisorUsageLevel.STAGE,
        budget=_budget(
            requests=3,
            input_tokens=300,
            output_tokens=150,
            cost_micros=30_000,
        ),
        stage="implementation",
    )
    lane = build_child_envelope(
        stage,
        level=SupervisorUsageLevel.LANE,
        budget=_budget(
            requests=2,
            input_tokens=200,
            output_tokens=100,
            cost_micros=20_000,
        ),
        lane="provider-execution",
    )
    request = build_child_envelope(
        lane,
        level=SupervisorUsageLevel.REQUEST,
        budget=_budget(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            cost_micros=10_000,
        ),
        request_id="request:asi-166-1",
        endpoint_scope_id="endpoint-scope:chat-1",
        caller_id="caller:supervisor",
        deadline_at="2026-07-28T12:00:00Z",
        idempotency_key="idem:asi-166-1",
        lease_id="lease:1",
        fence_id="1",
    )
    # Rebuild root with full child tree.
    return SupervisorUsageEnvelope(
        scope=root.scope,
        budget=root.budget,
        children=(
            SupervisorUsageEnvelope(
                scope=run.scope,
                budget=run.budget,
                children=(
                    SupervisorUsageEnvelope(
                        scope=goal.scope,
                        budget=goal.budget,
                        children=(
                            SupervisorUsageEnvelope(
                                scope=task.scope,
                                budget=task.budget,
                                children=(
                                    SupervisorUsageEnvelope(
                                        scope=attempt.scope,
                                        budget=attempt.budget,
                                        children=(
                                            SupervisorUsageEnvelope(
                                                scope=stage.scope,
                                                budget=stage.budget,
                                                children=(
                                                    SupervisorUsageEnvelope(
                                                        scope=lane.scope,
                                                        budget=lane.budget,
                                                        children=(request,),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def _request_envelope(tree: SupervisorUsageEnvelope | None = None) -> SupervisorUsageEnvelope:
    lineage = tree or _lineage()
    for node in lineage.walk():
        if node.scope.level is SupervisorUsageLevel.REQUEST:
            return node
    raise AssertionError("missing request envelope")


def _bridge(
    request_env: SupervisorUsageEnvelope,
    *,
    estimated: UsageVector | None = None,
) -> SupervisorToEndpointRequest:
    scope = request_env.scope
    return SupervisorToEndpointRequest(
        scope=scope,
        envelope_id=request_env.envelope_id,
        endpoint_scope_id=scope.endpoint_scope_id,
        catalog_revision=scope.catalog_revision,
        usage_revision=scope.usage_revision,
        estimated=estimated
        or UsageVector.of(requests=1, input_tokens=80, output_tokens=40),
        request_id=scope.request_id,
        attempt=scope.attempt,
        idempotency_key=scope.idempotency_key,
        caller_id=scope.caller_id,
        deadline_at=scope.deadline_at,
        lease_id=scope.lease_id,
        fence_id=scope.fence_id,
    )


def _execution_request(
    *,
    cancelled: bool = False,
    post_dispatch: bool = False,
    timeout_expired: bool = False,
    mode: ProviderExecutionMode = ProviderExecutionMode.ENFORCE,
    coordination_state: CoordinationState = CoordinationState.AVAILABLE,
    degraded_budget_id: str = "",
    attempt: int | None = None,
    idempotency_key: str | None = None,
    request_id: str | None = None,
) -> ProviderExecutionRequest:
    request_env = _request_envelope()
    scope = request_env.scope
    if (
        attempt is not None
        or idempotency_key is not None
        or request_id is not None
    ):
        # Rebuild request-level scope with overrides via child envelope helpers.
        parent = None
        lineage = _lineage()
        for node in lineage.walk():
            if (
                node.scope.level is SupervisorUsageLevel.LANE
                and node.scope.lane == scope.lane
            ):
                parent = node
                break
        assert parent is not None
        request_env = build_child_envelope(
            parent,
            level=SupervisorUsageLevel.REQUEST,
            budget=request_env.budget,
            request_id=request_id or scope.request_id,
            endpoint_scope_id=scope.endpoint_scope_id,
            caller_id=scope.caller_id,
            deadline_at=scope.deadline_at,
            idempotency_key=idempotency_key or scope.idempotency_key,
            lease_id=scope.lease_id,
            fence_id=scope.fence_id,
            attempt=attempt or scope.attempt,
        )
    bridge = _bridge(request_env)
    return build_execution_request(
        bridge=bridge,
        envelope=request_env,
        provider_id="provider:example",
        modality="text",
        side_effect_boundary=SideEffectBoundary.IDEMPOTENT,
        operation="text.generate",
        mode=mode,
        cancelled=cancelled,
        post_dispatch=post_dispatch,
        timeout_expired=timeout_expired,
        degraded_budget_id=degraded_budget_id,
        coordination_state=coordination_state,
        metadata={"stage": "implementation"},
    )


class _FakeDecision:
    def __init__(
        self,
        *,
        granted: bool = True,
        reservation_id: str = "reservation:1",
        usage_revision: str = "usage:rev-1",
        reason_codes: tuple[str, ...] = (),
    ) -> None:
        self.granted = granted
        self.reservation_id = reservation_id
        self.usage_revision = usage_revision
        self.reason_codes = reason_codes


class _FakeSettlement:
    def __init__(
        self,
        *,
        charged: UsageVector | None = None,
        event_id: str = "event:1",
        usage_revision: str = "usage:rev-2",
        state: str = "committed",
    ) -> None:
        self.charged = charged or UsageVector.of(requests=1)
        self.event_id = event_id
        self.usage_revision = usage_revision
        self.state = state


class FakeCoordinator:
    """Offline coordinator double for hermetic gateway tests."""

    def __init__(self) -> None:
        self.reservations: dict[str, dict[str, Any]] = {}
        self.reserve_calls = 0
        self.commit_calls = 0
        self.cancel_calls = 0
        self.release_calls = 0
        self.dispatch_calls = 0
        self.force_stale = False
        self.deny = False

    def reserve(self, scope_id: str, requested: Any, **kwargs: Any) -> _FakeDecision:
        self.reserve_calls += 1
        if self.force_stale:
            raise RuntimeError("stale snapshot revision")
        if self.deny:
            return _FakeDecision(
                granted=False,
                reservation_id="",
                reason_codes=("capacity_denied",),
            )
        reservation_id = f"reservation:{self.reserve_calls}"
        self.reservations[reservation_id] = {
            "scope_id": scope_id,
            "requested": requested,
            "kwargs": kwargs,
            "dispatched": False,
            "terminal": False,
        }
        return _FakeDecision(reservation_id=reservation_id)

    def mark_dispatched(self, reservation_id: str) -> None:
        self.dispatch_calls += 1
        if reservation_id in self.reservations:
            self.reservations[reservation_id]["dispatched"] = True

    def cancel(self, reservation_id: str, *, reason: str = "cancelled") -> _FakeSettlement:
        self.cancel_calls += 1
        record = self.reservations.get(reservation_id, {})
        record["terminal"] = True
        charged = UsageVector.of(requests=1) if record.get("dispatched") else UsageVector()
        return _FakeSettlement(
            charged=charged,
            event_id=f"cancel-event:{reservation_id}",
            state="committed" if record.get("dispatched") else "released",
        )

    def release(self, reservation_id: str, *, reason: str = "released") -> _FakeSettlement:
        self.release_calls += 1
        if reservation_id in self.reservations:
            self.reservations[reservation_id]["terminal"] = True
        return _FakeSettlement(
            charged=UsageVector(),
            event_id=f"release-event:{reservation_id}",
            state="released",
        )

    def commit(
        self,
        reservation_id: str,
        actual: Any = None,
        *,
        observation_id: str | None = None,
        release_unused: bool = True,
    ) -> _FakeSettlement:
        self.commit_calls += 1
        if reservation_id in self.reservations:
            self.reservations[reservation_id]["terminal"] = True
        charged = actual if isinstance(actual, UsageVector) else UsageVector.of(requests=1)
        return _FakeSettlement(
            charged=charged,
            event_id=f"commit-event:{reservation_id}",
            state="committed",
        )


def test_request_binds_scope_envelope_attempt_revisions_deadline_lease_fence() -> None:
    req = _execution_request()
    assert req.bridge.scope.level is SupervisorUsageLevel.REQUEST
    assert req.bridge.attempt == 1
    assert req.bridge.idempotency_key
    assert req.bridge.catalog_revision == "catalog:rev-1"
    assert req.bridge.usage_revision == "usage:rev-1"
    assert req.bridge.endpoint_scope_id
    assert req.bridge.deadline_at
    assert req.bridge.lease_id
    assert req.bridge.fence_id
    assert req.side_effect_boundary is SideEffectBoundary.IDEMPOTENT
    assert req.request_key == req.bridge.idempotency_key
    assert req.attempt_key.endswith("#1")
    record = req.to_record()
    assert record["schema"].endswith("provider-execution-request@1")
    assert "prompt" not in record
    assert req.content_id


def test_gateway_estimate_reserve_invoke_settle_with_redacted_receipt() -> None:
    coordinator = FakeCoordinator()
    calls = {"n": 0}

    def invoker(request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {
            "provider_id": request.provider_id,
            "endpoint_scope_id": request.bridge.endpoint_scope_id,
            "units": {"requests": 1, "input_tokens": 40, "output_tokens": 10},
            "endpoint_receipt_id": "endpoint-receipt:1",
            "status": "ok",
            # Forbidden leakage must be stripped.
            "prompt": "should-not-appear",
            "output": "secret-model-text",
        }

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    result = gateway.execute(_execution_request())
    assert result.phase is ProviderExecutionPhase.SETTLED
    assert result.final_status is SupervisorUsageFinalStatus.COMMITTED
    assert result.granted is True
    assert result.reservation_id.startswith("reservation:")
    assert result.receipt is not None
    assert result.supervisor_receipt_id == result.receipt.receipt_id
    assert result.endpoint_receipt_id == "endpoint-receipt:1"
    assert result.redacted_endpoint.startswith("endpoint:")
    assert "http" not in result.redacted_endpoint
    assert "prompt" not in result.observation
    assert "output" not in result.observation
    assert not result.authorizes_usage
    assert not result.is_completion_evidence
    assert calls["n"] == 1
    assert coordinator.reserve_calls == 1
    assert coordinator.dispatch_calls == 1
    assert coordinator.commit_calls == 1


def test_exact_replay_cannot_reinvoke_or_recharge() -> None:
    coordinator = FakeCoordinator()
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"units": {"requests": 1}, "status": "ok"}

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    request = _execution_request()
    first = gateway.execute(request)
    second = gateway.execute(request)
    assert first.reservation_id == second.reservation_id
    assert second.replayed is True
    assert "exact_replay" in second.reason_codes or "single_flight" in second.reason_codes
    assert calls["n"] == 1
    assert coordinator.reserve_calls == 1
    assert coordinator.commit_calls == 1
    assert gateway.invoke_count(request.attempt_key) == 1


def test_pre_dispatch_cancel_releases_without_invoke() -> None:
    coordinator = FakeCoordinator()
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"units": {"requests": 1}}

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    result = gateway.execute(_execution_request(cancelled=True, post_dispatch=False))
    assert result.phase is ProviderExecutionPhase.CANCELLED
    assert result.final_status is SupervisorUsageFinalStatus.CANCELLED
    assert calls["n"] == 0
    assert coordinator.reserve_calls == 0
    assert "pre_dispatch_cancelled" in result.reason_codes


def test_post_dispatch_timeout_conservatively_settles() -> None:
    coordinator = FakeCoordinator()
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"units": {"requests": 1}}

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    result = gateway.execute(
        _execution_request(cancelled=False, post_dispatch=True, timeout_expired=True)
    )
    assert result.phase is ProviderExecutionPhase.SETTLED
    assert result.final_status is SupervisorUsageFinalStatus.COMMITTED
    assert "conservative_settle" in result.reason_codes
    assert "post_dispatch_timeout" in result.reason_codes
    assert calls["n"] == 0
    assert coordinator.reserve_calls == 1
    assert coordinator.dispatch_calls == 1
    assert coordinator.cancel_calls == 1


def test_enforce_mode_fails_closed_on_unknown_coordination_without_degraded_budget() -> None:
    gateway = ProviderExecutionGateway(coordinator=None, invoker=lambda r: {"status": "ok"})
    # Explicit unknown coordination with enforce and no degraded budget.
    result = gateway.execute(
        _execution_request(
            mode=ProviderExecutionMode.ENFORCE,
            coordination_state=CoordinationState.UNKNOWN,
            degraded_budget_id="",
        )
    )
    assert result.phase is ProviderExecutionPhase.DENIED
    assert result.final_status is SupervisorUsageFinalStatus.CAPACITY_UNAVAILABLE
    assert "coordination_fail_closed" in result.reason_codes


def test_enforce_mode_allows_reviewed_degraded_budget_local_fallback() -> None:
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"status": "local_ok"}

    gateway = ProviderExecutionGateway(coordinator=None, invoker=invoker)
    result = gateway.execute(
        _execution_request(
            mode=ProviderExecutionMode.ENFORCE,
            coordination_state=CoordinationState.STALE,
            degraded_budget_id="degraded-budget:local-deterministic",
        )
    )
    assert result.phase is ProviderExecutionPhase.DEGRADED
    assert "degraded_local_fallback" in result.reason_codes
    assert "reviewed_degraded_budget" in result.reason_codes
    assert calls["n"] == 1


def test_off_mode_invokes_without_reservation() -> None:
    coordinator = FakeCoordinator()
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"status": "ok"}

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    result = gateway.execute(
        _execution_request(mode=ProviderExecutionMode.OFF)
    )
    assert result.phase is ProviderExecutionPhase.SETTLED
    assert "off_mode" in result.reason_codes
    assert calls["n"] == 1
    assert coordinator.reserve_calls == 0


def test_capacity_denial_is_terminal_without_invoke() -> None:
    coordinator = FakeCoordinator()
    coordinator.deny = True
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"status": "ok"}

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    result = gateway.execute(_execution_request())
    assert result.phase is ProviderExecutionPhase.DENIED
    assert calls["n"] == 0
    assert coordinator.reserve_calls == 1


def test_new_attempt_gets_distinct_idempotency_and_can_invoke() -> None:
    coordinator = FakeCoordinator()
    calls = {"n": 0}

    def invoker(_request: ProviderExecutionRequest) -> Mapping[str, Any]:
        calls["n"] += 1
        return {"units": {"requests": 1}, "status": "ok"}

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=invoker)
    first = gateway.execute(_execution_request())
    # Distinct request identity (new attempt/idempotency) must not hit replay.
    request_env = _request_envelope()
    parent = None
    for node in _lineage().walk():
        if node.scope.level is SupervisorUsageLevel.LANE:
            parent = node
            break
    assert parent is not None
    second_env = build_child_envelope(
        parent,
        level=SupervisorUsageLevel.REQUEST,
        budget=request_env.budget,
        request_id="request:asi-166-2",
        endpoint_scope_id=request_env.scope.endpoint_scope_id,
        caller_id=request_env.scope.caller_id,
        deadline_at=request_env.scope.deadline_at,
        idempotency_key=new_attempt_idempotency_key("idem:asi-166-1", 2),
        lease_id=request_env.scope.lease_id,
        fence_id=request_env.scope.fence_id,
        # REQUEST level inherits attempt from parent ATTEMPT scope.
    )
    second_req = build_execution_request(
        bridge=_bridge(second_env),
        envelope=second_env,
        provider_id="provider:example",
        operation="text.generate",
    )
    second = gateway.execute(second_req)
    assert first.request_key != second.request_key
    assert first.reservation_id != second.reservation_id
    assert calls["n"] == 2
    assert coordinator.reserve_calls == 2


def test_authority_flags_are_hard_false() -> None:
    bounds = accounting_bounds()
    assert bounds == {
        "authorizes_usage": False,
        "rewrites_provider_settlement": False,
        "is_completion_evidence": False,
        "is_correctness_evidence": False,
    }
    result = ProviderExecutionGateway().execute(_execution_request())
    with pytest.raises(ProviderExecutionError, match="cannot be true"):
        type(result)(
            phase=result.phase,
            final_status=result.final_status,
            granted=result.granted,
            reservation_id=result.reservation_id,
            usage_revision=result.usage_revision,
            catalog_revision=result.catalog_revision,
            provider_id=result.provider_id,
            redacted_endpoint=result.redacted_endpoint,
            attempt_key=result.attempt_key,
            request_key=result.request_key,
            request_id=result.request_id,
            authorizes_usage=True,
        )


def test_conservative_estimate_never_empty() -> None:
    estimate = conservative_estimate(
        scope_id="endpoint-scope:1",
        operation="text.generate",
        requested=UsageVector(),
    )
    assert estimate.requested.entries
    assert estimate.scope_id.startswith("scope_")
    assert estimate.operation == "text.generate"


def test_result_rejects_prompt_and_secret_fields() -> None:
    request_env = _request_envelope()
    bridge = _bridge(request_env)
    req = build_execution_request(
        bridge=bridge,
        envelope=request_env,
        provider_id="provider:x",
        metadata={"note": "ok"},
    )
    with pytest.raises(ProviderExecutionError):
        build_execution_request(
            bridge=bridge,
            envelope=request_env,
            provider_id="provider:x",
            metadata={"prompt": "leak"},
        )
    assert "prompt" not in req.to_record()


def test_discover_schemas_and_requirement_id() -> None:
    schemas = discover_schemas()
    assert schemas["requirement_id"] == RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID
    assert schemas["authorizes_usage"] == "false"
    assert schemas["goal_id"] == "ASI-G510"


def test_cold_import_is_side_effect_free() -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import ipfs_accelerate_py.agent_supervisor.provider_execution as pe
assert pe.RESERVATION_AWARE_PROVIDER_EXECUTION_REQUIREMENT_ID.startswith("requirement:")
schemas = pe.discover_schemas()
assert schemas["authorizes_usage"] == "false"
import importlib
importlib.reload(pe)
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.provider_execution"
    )
    for name in (
        "ProviderExecutionRequest",
        "ProviderExecutionResult",
        "ProviderExecutionGateway",
        "SideEffectBoundary",
        "ProviderExecutionMode",
    ):
        assert hasattr(module, name)


def test_simulated_path_without_coordinator_is_offline() -> None:
    gateway = ProviderExecutionGateway()
    result = gateway.execute(
        _execution_request(coordination_state=CoordinationState.AVAILABLE)
    )
    # No coordinator → simulated reserve path.
    assert result.phase is ProviderExecutionPhase.SETTLED
    assert result.reservation_id.startswith("sim:")
    assert result.receipt is not None
