"""Tests for todo_daemon LLM helpers and ASI-166 gateway adapter."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.provider_execution import (
    ProviderExecutionPhase,
    build_execution_request,
)
from ipfs_accelerate_py.agent_supervisor.provider_usage import (
    SupervisorToEndpointRequest,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageLevel,
    SupervisorUsageScope,
    build_child_envelope,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import llm as todo_llm
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


def _request_envelope() -> SupervisorUsageEnvelope:
    root = SupervisorUsageEnvelope(
        scope=SupervisorUsageScope(
            level=SupervisorUsageLevel.DEPLOYMENT, **_base_scope_kwargs()
        ),
        budget=_budget(requests=1000),
    )
    run = build_child_envelope(
        root, level=SupervisorUsageLevel.RUN, budget=_budget(requests=500),
        supervisor_run_id="run:todo-llm",
    )
    goal = build_child_envelope(
        run, level=SupervisorUsageLevel.GOAL, budget=_budget(requests=200),
        goal_id="goal:todo", objective_id="TODO", objective_revision="objective:todo@1",
    )
    task = build_child_envelope(
        goal, level=SupervisorUsageLevel.TASK, budget=_budget(requests=50), task_id="TODO-1"
    )
    attempt = build_child_envelope(
        task, level=SupervisorUsageLevel.ATTEMPT, budget=_budget(requests=10), attempt=1
    )
    stage = build_child_envelope(
        attempt, level=SupervisorUsageLevel.STAGE, budget=_budget(requests=5),
        stage="implementation",
    )
    lane = build_child_envelope(
        stage, level=SupervisorUsageLevel.LANE, budget=_budget(requests=3), lane="lane-0"
    )
    provider_id = stable_id("provider", "example-ai")
    endpoint = EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id("deployment", provider_id, "chat"),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:EXAMPLE_API_KEY", key_id="ledger-default"
        ),
    )
    request = build_child_envelope(
        lane,
        level=SupervisorUsageLevel.REQUEST,
        budget=_budget(requests=1),
        request_id="request:todo-1",
        endpoint_scope_id=endpoint.scope_id or "",
        caller_id="caller:todo-daemon",
        deadline_at="2099-07-28T01:00:00Z",
        idempotency_key="idem:todo-1",
        lease_id="lease:todo",
        fence_id="fence:todo",
    )
    lane = SupervisorUsageEnvelope(scope=lane.scope, budget=lane.budget, children=(request,))
    stage = SupervisorUsageEnvelope(scope=stage.scope, budget=stage.budget, children=(lane,))
    attempt = SupervisorUsageEnvelope(
        scope=attempt.scope, budget=attempt.budget, children=(stage,)
    )
    task = SupervisorUsageEnvelope(scope=task.scope, budget=task.budget, children=(attempt,))
    goal = SupervisorUsageEnvelope(scope=goal.scope, budget=goal.budget, children=(task,))
    run = SupervisorUsageEnvelope(scope=run.scope, budget=run.budget, children=(goal,))
    root = SupervisorUsageEnvelope(scope=root.scope, budget=root.budget, children=(run,))
    node = root
    while node.children:
        node = node.children[0]
    return node


def test_todo_llm_module_exports_process_helpers() -> None:
    assert callable(todo_llm.collect_descendant_pids)
    assert callable(todo_llm.terminate_active_llm_process)
    assert callable(todo_llm.execute_via_provider_gateway)


def test_todo_llm_gateway_adapter_settles() -> None:
    envelope = _request_envelope()
    scope = envelope.scope
    bridge = SupervisorToEndpointRequest(
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
    request = build_execution_request(
        bridge=bridge,
        envelope=envelope,
        provider_id="provider:todo",
    )
    result = todo_llm.execute_via_provider_gateway(request)
    assert result.phase is ProviderExecutionPhase.SETTLED
    assert result.granted is True
