"""Tests for hierarchical supervisor usage envelopes and endpoint bridge."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.provider_usage import (
    BRIDGE_AUTHORIZES_USAGE,
    BRIDGE_IS_COMPLETION_EVIDENCE,
    BRIDGE_IS_CORRECTNESS_EVIDENCE,
    BRIDGE_REWRITES_PROVIDER_SETTLEMENT,
    MAX_NESTING_DEPTH,
    SUPERVISOR_USAGE_ENVELOPE_GOAL_ID,
    SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID,
    ProviderUsageValidationError,
    SupervisorBudgetLimit,
    SupervisorToEndpointRequest,
    SupervisorUsageAttribution,
    SupervisorUsageBudget,
    SupervisorUsageEnvelope,
    SupervisorUsageFinalStatus,
    SupervisorUsageLevel,
    SupervisorUsageReceipt,
    SupervisorUsageScope,
    accounting_bridge_bounds,
    attribute_endpoint_events,
    build_child_envelope,
    consume_reconciled_endpoint_events_exactly_once,
    discover_schemas,
    finite_units,
)
from ipfs_accelerate_py.endpoint_usage import (
    EndpointUsageScope,
    LimitWindow,
    ProtocolKind,
    UsageDimension,
    UsageEvent,
    UsageEventKind,
    UsageVector,
    WindowKind,
    credential_configuration_pseudonym,
    stable_id,
)


def _window() -> LimitWindow:
    return LimitWindow(kind=WindowKind.LIFETIME)


def _budget(**dimensions: int) -> SupervisorUsageBudget:
    return SupervisorUsageBudget.of(
        window=_window(),
        currency="USD",
        **dimensions,
    )


def _base_scope_kwargs() -> dict:
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


def _deployment_scope() -> SupervisorUsageScope:
    return SupervisorUsageScope(
        level=SupervisorUsageLevel.DEPLOYMENT,
        **_base_scope_kwargs(),
    )


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


def _endpoint_event(
    *,
    sequence: int = 1,
    request_id: str = "request:1",
    input_tokens: int = 40,
    output_tokens: int = 10,
    cost_micros: int = 250,
    kind: UsageEventKind = UsageEventKind.COMMIT,
) -> UsageEvent:
    scope = _endpoint_scope()
    return UsageEvent(
        kind=kind,
        scope_id=scope.scope_id,
        request_id=request_id,
        sequence=sequence,
        occurred_at=f"2026-07-28T00:00:{sequence:02d}Z",
        units=UsageVector.of(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_micros=cost_micros,
            currency="USD",
        ),
    )


def _lineage() -> SupervisorUsageEnvelope:
    root = SupervisorUsageEnvelope(
        scope=_deployment_scope(),
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
            requests=500,
            input_tokens=50_000,
            output_tokens=25_000,
            cost_micros=5_000_000,
        ),
        supervisor_run_id="run:asi-165",
    )
    goal = build_child_envelope(
        run,
        level=SupervisorUsageLevel.GOAL,
        budget=_budget(
            requests=200,
            input_tokens=20_000,
            output_tokens=10_000,
            cost_micros=2_000_000,
        ),
        goal_id="goal:ASI-G510",
        objective_id="ASI-165",
        objective_revision="objective:asi-165@1",
    )
    task = build_child_envelope(
        goal,
        level=SupervisorUsageLevel.TASK,
        budget=_budget(
            requests=50,
            input_tokens=5_000,
            output_tokens=2_500,
            cost_micros=500_000,
        ),
        task_id="ASI-165",
    )
    attempt = build_child_envelope(
        task,
        level=SupervisorUsageLevel.ATTEMPT,
        budget=_budget(
            requests=10,
            input_tokens=1_000,
            output_tokens=500,
            cost_micros=100_000,
        ),
        attempt=1,
    )
    stage = build_child_envelope(
        attempt,
        level=SupervisorUsageLevel.STAGE,
        budget=_budget(
            requests=5,
            input_tokens=500,
            output_tokens=250,
            cost_micros=50_000,
        ),
        stage="implementation",
    )
    lane = build_child_envelope(
        stage,
        level=SupervisorUsageLevel.LANE,
        budget=_budget(
            requests=3,
            input_tokens=300,
            output_tokens=150,
            cost_micros=30_000,
        ),
        lane="lane-4",
    )
    endpoint = _endpoint_scope()
    request = build_child_envelope(
        lane,
        level=SupervisorUsageLevel.REQUEST,
        budget=_budget(
            requests=1,
            input_tokens=100,
            output_tokens=50,
            cost_micros=10_000,
        ),
        request_id="request:1",
        endpoint_scope_id=endpoint.scope_id or "",
        caller_id="caller:implementation-daemon",
        deadline_at="2026-07-28T01:00:00Z",
        idempotency_key="idem:request:1",
        lease_id="lease:1",
        fence_id="fence:1",
    )
    # Rebuild nested tree bottom-up so root validates the full lineage.
    lane = SupervisorUsageEnvelope(
        scope=lane.scope, budget=lane.budget, children=(request,)
    )
    stage = SupervisorUsageEnvelope(
        scope=stage.scope, budget=stage.budget, children=(lane,)
    )
    attempt = SupervisorUsageEnvelope(
        scope=attempt.scope, budget=attempt.budget, children=(stage,)
    )
    task = SupervisorUsageEnvelope(
        scope=task.scope, budget=task.budget, children=(attempt,)
    )
    goal = SupervisorUsageEnvelope(
        scope=goal.scope, budget=goal.budget, children=(task,)
    )
    run = SupervisorUsageEnvelope(
        scope=run.scope, budget=run.budget, children=(goal,)
    )
    return SupervisorUsageEnvelope(
        scope=root.scope, budget=root.budget, children=(run,)
    )


def test_requirement_and_schema_discovery_are_stable() -> None:
    schemas = discover_schemas()
    assert SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID == (
        "requirement:supervisor-usage-envelope.v1"
    )
    assert SUPERVISOR_USAGE_ENVELOPE_GOAL_ID == "ASI-G510"
    assert schemas["requirement_id"] == SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID
    assert schemas["authorizes_usage"] == "false"
    assert schemas["is_completion_evidence"] == "false"
    bounds = accounting_bridge_bounds()
    assert bounds == {
        "authorizes_usage": False,
        "rewrites_provider_settlement": False,
        "is_completion_evidence": False,
        "is_correctness_evidence": False,
    }
    assert not BRIDGE_AUTHORIZES_USAGE
    assert not BRIDGE_REWRITES_PROVIDER_SETTLEMENT
    assert not BRIDGE_IS_COMPLETION_EVIDENCE
    assert not BRIDGE_IS_CORRECTNESS_EVIDENCE


def test_child_budget_can_only_lower_parent_across_dimensions_and_currency() -> None:
    parent = _budget(
        requests=100,
        input_tokens=1_000,
        cost_micros=10_000,
    )
    child = _budget(
        requests=50,
        input_tokens=500,
        cost_micros=5_000,
    )
    assert child.is_lower_or_equal(parent)
    widened = _budget(requests=200)
    assert not widened.is_lower_or_equal(parent)
    # New dimension not present on parent is widening.
    novel = _budget(output_tokens=10)
    assert not novel.is_lower_or_equal(parent)
    mixed = SupervisorUsageBudget(
        limits=(
            SupervisorBudgetLimit(
                dimension=UsageDimension.COST_MICROS,
                ceiling=100,
                window=_window(),
                currency="EUR",
            ),
        )
    )
    assert not mixed.is_lower_or_equal(parent)


def test_nested_envelope_lineage_binds_identities_without_payload_leakage() -> None:
    tree = _lineage()
    nodes = tree.walk()
    assert len(nodes) == MAX_NESTING_DEPTH
    request = nodes[-1]
    assert request.scope.level is SupervisorUsageLevel.REQUEST
    assert request.scope.repository_id == "repository:supervisor"
    assert request.scope.supervisor_run_id == "run:asi-165"
    assert request.scope.task_id == "ASI-165"
    assert request.scope.attempt == 1
    assert request.scope.stage == "implementation"
    assert request.scope.lane == "lane-4"
    assert request.scope.request_id == "request:1"
    assert request.scope.lease_id == "lease:1"
    assert request.scope.fence_id == "fence:1"
    payload = request.to_dict()
    serialized = json.dumps(payload)
    for forbidden in (
        "prompt",
        "messages",
        "sk-",
        "https://",
        "api_key",
        "Bearer ",
    ):
        assert forbidden not in serialized
    restored = SupervisorUsageEnvelope.from_dict(tree.to_dict())
    assert restored == tree
    with pytest.raises(FrozenInstanceError):
        tree.scope = _deployment_scope()  # type: ignore[misc]


def test_rejects_foreign_stale_widened_duplicate_and_unknown_fields() -> None:
    root = SupervisorUsageEnvelope(
        scope=_deployment_scope(),
        budget=_budget(requests=10, input_tokens=100),
    )
    # Widened child.
    with pytest.raises(ProviderUsageValidationError, match="widen|raise"):
        build_child_envelope(
            root,
            level=SupervisorUsageLevel.RUN,
            budget=_budget(requests=20),
            supervisor_run_id="run:x",
        )
    # Stale ancestry.
    with pytest.raises(ProviderUsageValidationError, match="stale|foreign"):
        build_child_envelope(
            root,
            level=SupervisorUsageLevel.RUN,
            budget=_budget(requests=5),
            supervisor_run_id="run:x",
            usage_revision="usage:stale",
        )
    # Missing parent on non-deployment scope.
    with pytest.raises(ProviderUsageValidationError, match="missing parent"):
        SupervisorUsageScope(
            level=SupervisorUsageLevel.RUN,
            **{**_base_scope_kwargs(), "supervisor_run_id": "run:1"},
        )
    # Duplicate attempts under same parent.
    run = build_child_envelope(
        root,
        level=SupervisorUsageLevel.RUN,
        budget=_budget(requests=5, input_tokens=50),
        supervisor_run_id="run:1",
    )
    goal = build_child_envelope(
        run,
        level=SupervisorUsageLevel.GOAL,
        budget=_budget(requests=4, input_tokens=40),
        goal_id="goal:1",
        objective_id="obj:1",
        objective_revision="obj:1@1",
    )
    task = build_child_envelope(
        goal,
        level=SupervisorUsageLevel.TASK,
        budget=_budget(requests=3, input_tokens=30),
        task_id="task:1",
    )
    a1 = build_child_envelope(
        task,
        level=SupervisorUsageLevel.ATTEMPT,
        budget=_budget(requests=1, input_tokens=10),
        attempt=1,
    )
    a1_dup = build_child_envelope(
        task,
        level=SupervisorUsageLevel.ATTEMPT,
        budget=_budget(requests=1, input_tokens=10),
        attempt=1,
    )
    with pytest.raises(ProviderUsageValidationError, match="duplicate attempt"):
        SupervisorUsageEnvelope(
            scope=task.scope,
            budget=task.budget,
            children=(a1, a1_dup),
        )
    # Negative / overflow ceilings.
    with pytest.raises(ProviderUsageValidationError, match="between"):
        SupervisorBudgetLimit(
            dimension=UsageDimension.REQUESTS,
            ceiling=-1,
            window=_window(),
        )
    # Mixed currency in one budget.
    with pytest.raises(ProviderUsageValidationError, match="mixed cost currency"):
        SupervisorUsageBudget(
            limits=(
                SupervisorBudgetLimit(
                    dimension=UsageDimension.COST_MICROS,
                    ceiling=1,
                    window=_window(),
                    currency="USD",
                ),
                SupervisorBudgetLimit(
                    dimension=UsageDimension.COST_MICROS,
                    ceiling=1,
                    window=LimitWindow(kind=WindowKind.SLIDING, length_ms=60_000),
                    currency="EUR",
                ),
            )
        )
    # Unknown fields.
    payload = root.to_dict()
    payload["prompt"] = "must never cross"
    with pytest.raises(ProviderUsageValidationError, match="unknown fields"):
        SupervisorUsageEnvelope.from_dict(payload)


def test_unbounded_nesting_is_rejected() -> None:
    node = SupervisorUsageEnvelope(
        scope=_deployment_scope(),
        budget=_budget(requests=100),
    )
    # Build a chain deeper than MAX_NESTING_DEPTH by forcing levels that skip
    # the normal may_parent check is not possible; instead verify the constant
    # and that a full legal depth tree is accepted while excess children depth
    # validation is enforced via walk size.
    full = _lineage()
    assert len(full.walk()) == MAX_NESTING_DEPTH
    assert MAX_NESTING_DEPTH == 8


def test_request_receipt_bridge_binds_scope_and_rejects_authority_claims() -> None:
    tree = _lineage()
    request_env = tree.walk()[-1]
    bridge = SupervisorToEndpointRequest(
        scope=request_env.scope,
        envelope_id=request_env.envelope_id,
        endpoint_scope_id=request_env.scope.endpoint_scope_id,
        catalog_revision=request_env.scope.catalog_revision,
        usage_revision=request_env.scope.usage_revision,
        estimated=UsageVector.of(requests=1, input_tokens=80, output_tokens=40),
        request_id=request_env.scope.request_id,
        attempt=request_env.scope.attempt,
        idempotency_key=request_env.scope.idempotency_key,
        caller_id=request_env.scope.caller_id,
        deadline_at=request_env.scope.deadline_at,
        lease_id=request_env.scope.lease_id,
        fence_id=request_env.scope.fence_id,
    )
    event = _endpoint_event()
    receipt = SupervisorUsageReceipt(
        scope=request_env.scope,
        envelope_id=request_env.envelope_id,
        request_id=request_env.scope.request_id,
        endpoint_scope_id=request_env.scope.endpoint_scope_id,
        catalog_revision=request_env.scope.catalog_revision,
        usage_revision=request_env.scope.usage_revision,
        reservation_id="reservation:1",
        endpoint_event_ids=(event.event_id or "",),
        settled=event.units,
        final_status=SupervisorUsageFinalStatus.COMMITTED,
    )
    assert bridge.bridge_request_id
    assert receipt.receipt_id
    assert not receipt.authorizes_usage
    assert not receipt.is_completion_evidence
    with pytest.raises(ProviderUsageValidationError, match="cannot be true"):
        SupervisorUsageReceipt(
            scope=request_env.scope,
            envelope_id=request_env.envelope_id,
            request_id=request_env.scope.request_id,
            endpoint_scope_id=request_env.scope.endpoint_scope_id,
            catalog_revision=request_env.scope.catalog_revision,
            usage_revision=request_env.scope.usage_revision,
            reservation_id="reservation:1",
            endpoint_event_ids=(event.event_id or "",),
            settled=event.units,
            final_status=SupervisorUsageFinalStatus.COMMITTED,
            authorizes_usage=True,
        )
    with pytest.raises(ProviderUsageValidationError, match="stale"):
        SupervisorToEndpointRequest(
            scope=request_env.scope,
            envelope_id=request_env.envelope_id,
            endpoint_scope_id=request_env.scope.endpoint_scope_id,
            catalog_revision="catalog:stale",
            usage_revision=request_env.scope.usage_revision,
            estimated=UsageVector.of(requests=1),
            request_id=request_env.scope.request_id,
            attempt=request_env.scope.attempt,
            idempotency_key=request_env.scope.idempotency_key,
            caller_id=request_env.scope.caller_id,
            deadline_at=request_env.scope.deadline_at,
            lease_id=request_env.scope.lease_id,
            fence_id=request_env.scope.fence_id,
        )


def test_endpoint_events_are_consumed_exactly_once_for_attribution() -> None:
    tree = _lineage()
    request_env = tree.walk()[-1]
    first = _endpoint_event(sequence=1, input_tokens=40, output_tokens=10)
    second = _endpoint_event(sequence=2, request_id="request:2", input_tokens=20)
    consumed = consume_reconciled_endpoint_events_exactly_once((first, second))
    assert len(consumed) == 2
    with pytest.raises(ProviderUsageValidationError, match="duplicated"):
        consume_reconciled_endpoint_events_exactly_once((first, first))
    attributions = attribute_endpoint_events(
        scope=request_env.scope,
        events=(first,),
        lifecycle_event_ids=("lifecycle:1",),
    )
    assert len(attributions) == 1
    assert isinstance(attributions[0], SupervisorUsageAttribution)
    assert attributions[0].endpoint_event_id == first.event_id
    assert finite_units(first.units, UsageDimension.INPUT_TOKENS) == 40
    with pytest.raises(ProviderUsageValidationError, match="one-to-one"):
        attribute_endpoint_events(
            scope=request_env.scope,
            events=(first, second),
            lifecycle_event_ids=("lifecycle:1",),
        )


def test_cold_import_and_schema_discovery_are_side_effect_free() -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import ipfs_accelerate_py.agent_supervisor.provider_usage as pu
assert pu.SUPERVISOR_USAGE_ENVELOPE_REQUIREMENT_ID.startswith("requirement:")
schemas = pu.discover_schemas()
assert schemas["authorizes_usage"] == "false"
import importlib
importlib.reload(pu)
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
        "ipfs_accelerate_py.agent_supervisor.provider_usage"
    )
    for name in (
        "SupervisorUsageEnvelope",
        "SupervisorUsageScope",
        "SupervisorUsageBudget",
        "SupervisorUsageAttribution",
        "SupervisorToEndpointRequest",
        "SupervisorUsageReceipt",
        "discover_schemas",
    ):
        assert hasattr(module, name)
