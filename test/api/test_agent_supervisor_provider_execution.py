"""Tests for ASI-166 reservation-aware provider execution gateway."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_execution import (
    ProviderExecutionError,
    ProviderExecutionGateway,
    ProviderExecutionPhase,
    ProviderExecutionResult,
    build_execution_request,
    new_attempt_idempotency_key,
)
from ipfs_accelerate_py.agent_supervisor.provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageLevel,
    SupervisorUsageScope,
    build_child_envelope,
)
from ipfs_accelerate_py.endpoint_usage import (
    EndpointUsageScope,
    LimitWindow,
    ProtocolKind,
    UsageVector,
    WindowKind,
    credential_configuration_pseudonym,
    stable_id,
)


def _window() -> LimitWindow:
    return LimitWindow(kind=WindowKind.LIFETIME)


def _budget(**dimensions: int) -> SupervisorUsageBudget:
    return SupervisorUsageBudget.of(window=_window(), currency="USD", **dimensions)


def _base_scope_kwargs() -> dict:
    return {
        "repository_id": "repository:supervisor",
        "state_id": "state:lane-0",
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


def _endpoint_scope() -> EndpointUsageScope:
    provider_id = stable_id("provider", "example-ai")
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id("deployment", provider_id, "chat"),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:EXAMPLE_API_KEY", key_id="ledger-default"
        ),
    )


def _request_lineage() -> SupervisorUsageEnvelope:
    root = SupervisorUsageEnvelope(
        scope=SupervisorUsageScope(
            level=SupervisorUsageLevel.DEPLOYMENT,
            **_base_scope_kwargs(),
        ),
        budget=_budget(requests=1000),
    )
    run = build_child_envelope(
        root, level=SupervisorUsageLevel.RUN, budget=_budget(requests=500),
        supervisor_run_id="run:asi-166",
    )
    goal = build_child_envelope(
        run, level=SupervisorUsageLevel.GOAL, budget=_budget(requests=200),
        goal_id="goal:ASI-G510", objective_id="ASI-166",
        objective_revision="objective:asi-166@1",
    )
    task = build_child_envelope(
        goal, level=SupervisorUsageLevel.TASK, budget=_budget(requests=50),
        task_id="ASI-166",
    )
    attempt = build_child_envelope(
        task, level=SupervisorUsageLevel.ATTEMPT, budget=_budget(requests=10),
        attempt=1,
    )
    stage = build_child_envelope(
        attempt, level=SupervisorUsageLevel.STAGE, budget=_budget(requests=5),
        stage="implementation",
    )
    lane = build_child_envelope(
        stage, level=SupervisorUsageLevel.LANE, budget=_budget(requests=3),
        lane="lane-0",
    )
    endpoint = _endpoint_scope()
    request = build_child_envelope(
        lane,
        level=SupervisorUsageLevel.REQUEST,
        budget=_budget(requests=1),
        request_id="request:1",
        endpoint_scope_id=endpoint.scope_id or "",
        caller_id="caller:implementation-daemon",
        deadline_at="2099-07-28T01:00:00Z",
        idempotency_key="idem:request:1",
        lease_id="lease:1",
        fence_id="fence:1",
    )
    # Rebuild nested tree bottom-up so root validates the full lineage.
    lane = SupervisorUsageEnvelope(scope=lane.scope, budget=lane.budget, children=(request,))
    stage = SupervisorUsageEnvelope(scope=stage.scope, budget=stage.budget, children=(lane,))
    attempt = SupervisorUsageEnvelope(
        scope=attempt.scope, budget=attempt.budget, children=(stage,)
    )
    task = SupervisorUsageEnvelope(scope=task.scope, budget=task.budget, children=(attempt,))
    goal = SupervisorUsageEnvelope(scope=goal.scope, budget=goal.budget, children=(task,))
    run = SupervisorUsageEnvelope(scope=run.scope, budget=run.budget, children=(goal,))
    return SupervisorUsageEnvelope(scope=root.scope, budget=root.budget, children=(run,))


def _request_envelope() -> SupervisorUsageEnvelope:
    root = _request_lineage()
    # Walk to the request leaf.
    node = root
    while node.children:
        node = node.children[0]
    return node


def _bridge(envelope: SupervisorUsageEnvelope) -> SupervisorToEndpointRequest:
    scope = envelope.scope
    return SupervisorToEndpointRequest(
        scope=scope,
        envelope_id=envelope.envelope_id,
        endpoint_scope_id=scope.endpoint_scope_id,
        catalog_revision=scope.catalog_revision,
        usage_revision=scope.usage_revision,
        estimated=UsageVector.of(requests=1),
        request_id=scope.request_id,
        attempt=scope.attempt,
        idempotency_key=scope.idempotency_key,
        caller_id=scope.caller_id,
        deadline_at=scope.deadline_at,
        lease_id=scope.lease_id,
        fence_id=scope.fence_id,
    )


def test_gateway_reserve_invoke_settle_and_exact_replay() -> None:
    envelope = _request_envelope()
    bridge = _bridge(envelope)
    request = build_execution_request(
        bridge=bridge,
        envelope=envelope,
        provider_id="provider:simulated",
        modality="text",
    )
    calls = {"n": 0}

    def invoker(_req):
        calls["n"] += 1
        return {
            "endpoint": "https://secret.example/v1/chat",
            "status": "ok",
            "units": UsageVector.of(requests=1).to_dict(),
        }

    gateway = ProviderExecutionGateway(invoker=invoker)
    first = gateway.execute(request)
    second = gateway.execute(request)

    assert first.phase is ProviderExecutionPhase.SETTLED
    assert first.granted is True
    assert first.receipt is not None
    assert first.receipt.is_completion_evidence is False
    assert first.redacted_endpoint.startswith("endpoint:")
    assert "secret.example" not in first.redacted_endpoint
    assert second.replayed is True
    assert "exact_replay" in second.reason_codes
    assert calls["n"] == 1


def test_pre_dispatch_cancel_does_not_invoke() -> None:
    envelope = _request_envelope()
    bridge = _bridge(envelope)
    request = build_execution_request(
        bridge=bridge,
        envelope=envelope,
        provider_id="provider:simulated",
        cancelled=True,
    )
    gateway = ProviderExecutionGateway(
        invoker=lambda _r: (_ for _ in ()).throw(AssertionError("should not invoke"))
    )
    result = gateway.execute(request)
    assert result.phase is ProviderExecutionPhase.CANCELLED
    assert result.granted is False
    assert "pre_dispatch_cancelled" in result.reason_codes


def test_retry_uses_new_attempt_key() -> None:
    key1 = new_attempt_idempotency_key("idem-base", 1)
    key2 = new_attempt_idempotency_key("idem-base", 2)
    assert key1 != key2
    assert key2.endswith("#attempt-2")


def test_result_cannot_claim_completion_authority() -> None:
    with pytest.raises(ProviderExecutionError, match="completion"):
        ProviderExecutionResult(is_completion_evidence=True)
