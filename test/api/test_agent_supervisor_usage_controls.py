"""ASI-169: authorized supervisor usage-governance controls and metrics."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    SUPERVISOR_USAGE_ADMIN_AUTHORITY,
    SUPERVISOR_USAGE_BUDGET_AUTHORITY,
    SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID,
    SUPERVISOR_USAGE_CORRECTION_AUTHORITY,
    SUPERVISOR_USAGE_POLICY_AUTHORITY,
    SUPERVISOR_USAGE_READ_AUTHORITY,
    SUPERVISOR_USAGE_READ_DETAIL_AUTHORITY,
    SUPERVISOR_USAGE_REASON_CODES,
    SUPERVISOR_USAGE_RESET_AUTHORITY,
    USAGE_CONTROL_MUTATION_OPERATIONS,
    USAGE_CONTROL_READ_OPERATIONS,
    USAGE_HEADROOM_BANDS,
    SupervisorUsageControlOperation,
    discover_usage_control_catalog,
    usage_control_authorities,
    usage_control_operations,
    usage_control_reason_codes,
    usage_headroom_band,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    ProviderUsageControl,
    SupervisorControlService,
    provider_usage_controls,
)
from ipfs_accelerate_py.agent_supervisor.runtime.scheduler_metrics import (
    SUPERVISOR_USAGE_METRICS_REQUIREMENT_ID,
    forbidden_usage_metric_label_keys,
    project_usage_governance_metrics,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)


def _read_auth() -> list[str]:
    return [SUPERVISOR_USAGE_READ_AUTHORITY]


def _admin_auth() -> list[str]:
    return [
        SUPERVISOR_USAGE_READ_AUTHORITY,
        SUPERVISOR_USAGE_ADMIN_AUTHORITY,
        SUPERVISOR_USAGE_BUDGET_AUTHORITY,
        SUPERVISOR_USAGE_POLICY_AUTHORITY,
        SUPERVISOR_USAGE_CORRECTION_AUTHORITY,
        SUPERVISOR_USAGE_RESET_AUTHORITY,
    ]


def _service() -> tuple[ProviderUsageControl, SupervisorControlService]:
    usage = ProviderUsageControl(
        catalog_revision_provider=lambda: "catalog-rev-test",
        supervisor_revision_provider=lambda: "supervisor-rev-test",
        policy_revision_provider=lambda: "policy-rev-test",
    )
    control = SupervisorControlService(
        repository_allowlist=("/tmp/repo",),
        state_allowlist=("/tmp/state",),
        usage_control=usage,
        lease_validator=lambda _request: True,
    )
    return usage, control


def test_requirement_ids_and_shared_vocabularies() -> None:
    assert SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID.startswith("requirement:")
    assert SUPERVISOR_USAGE_METRICS_REQUIREMENT_ID.startswith("requirement:")
    assert provider_usage_controls.SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID == (
        SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
    )
    assert set(usage_control_operations()) == {
        op.value for op in SupervisorUsageControlOperation
    }
    assert {op.value for op in USAGE_CONTROL_READ_OPERATIONS}.isdisjoint(
        {op.value for op in USAGE_CONTROL_MUTATION_OPERATIONS}
    )
    assert {op.value for op in USAGE_CONTROL_READ_OPERATIONS} | {
        op.value for op in USAGE_CONTROL_MUTATION_OPERATIONS
    } == set(usage_control_operations())
    authorities = usage_control_authorities()
    assert authorities["read"] == SUPERVISOR_USAGE_READ_AUTHORITY
    assert authorities["budget"] == SUPERVISOR_USAGE_BUDGET_AUTHORITY
    assert authorities["admin"] == SUPERVISOR_USAGE_ADMIN_AUTHORITY
    assert "read_denied" in SUPERVISOR_USAGE_REASON_CODES
    assert "parent_budget_raise_denied" in SUPERVISOR_USAGE_REASON_CODES
    assert set(usage_control_reason_codes()) == SUPERVISOR_USAGE_REASON_CODES
    assert "exhausted" in USAGE_HEADROOM_BANDS


def test_discovery_is_lazy_and_side_effect_free() -> None:
    usage, service = _service()
    before = usage.usage_revision()
    catalog = usage.discover()
    via_service = service.usage_discover()
    assert before == usage.usage_revision()
    assert catalog["requirement_id"] == SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
    assert catalog["catalog_revision"] == "catalog-rev-test"
    assert catalog["usage_revision"] == before
    assert catalog["policy_revision"] == "policy-rev-test"
    assert catalog["supervisor_revision"] == "supervisor-rev-test"
    assert catalog["completion_authoritative"] is False
    assert catalog["operational_evidence_only"] is True
    assert via_service["operations"] == catalog["operations"]
    static = discover_usage_control_catalog()
    assert static["requirement_id"] == SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
    assert len(static["operations"]) == len(SupervisorUsageControlOperation)


def test_reads_are_side_effect_free_and_bind_revisions() -> None:
    usage, _ = _service()
    before = usage.usage_revision()
    auths = _read_auth()
    status = usage.status(authorities=auths)
    health = usage.health(authorities=auths)
    budgets = usage.budgets(authorities=auths)
    headroom = usage.headroom("scope:a", authorities=auths)
    reservations = usage.reservations("scope:a", authorities=auths)
    receipts = usage.receipts(authorities=auths)
    preview = usage.route_preview(
        authorities=auths,
        candidates=[{"binding_id": "b1", "provider_id": "provider:a", "scope_id": "scope:a"}],
    )
    blocked = usage.blocked_work(authorities=auths)
    next_eligible = usage.next_eligible(authorities=auths)
    adapters = usage.adapter_capabilities(authorities=auths)
    after = usage.usage_revision()

    assert before == after
    for payload in (
        status,
        health,
        budgets,
        headroom,
        reservations,
        receipts,
        preview,
        blocked,
        next_eligible,
        adapters,
    ):
        assert payload["success"] is True, payload
        assert payload["catalog_revision"] == "catalog-rev-test"
        assert payload["usage_revision"]
        assert payload["policy_revision"]
        assert payload["supervisor_revision"]
        assert payload["completion_authoritative"] is False
        assert payload["operational_evidence_only"] is True
    assert preview["reserved"] is False
    assert preview["invoked"] is False
    assert preview["probed"] is False
    assert health["healthy"] is True
    assert adapters["count"] >= 1


def test_read_requires_authority() -> None:
    usage, _ = _service()
    denied = usage.status(authorities=[])
    assert denied["success"] is False
    assert denied["error_code"] == "read_denied"


def test_default_status_redacts_credential_account_tenant_detail() -> None:
    usage, _ = _service()
    usage.record_receipt(
        {
            "receipt_id": "r1",
            "target_id": "scope:a",
            "scope_id": "scope:a",
            "credential_pseudonym": "cred:secret",
            "account_pseudonym": "acct:secret",
            "tenant_id": "tenant:secret",
        }
    )
    # Inject a synthetic budget row so status has a target; attach detail fields
    # via a fake coordinator snapshot path by recording a receipt only.
    aggregate = usage.receipts(authorities=_read_auth())
    blob = str(aggregate)
    assert "cred:secret" not in blob
    assert "acct:secret" not in blob
    assert "tenant:secret" not in blob

    detailed = usage.receipts(
        authorities=[SUPERVISOR_USAGE_READ_AUTHORITY, SUPERVISOR_USAGE_READ_DETAIL_AUTHORITY]
    )
    detailed_blob = str(detailed)
    assert "cred:secret" in detailed_blob
    assert "acct:secret" in detailed_blob


def test_pagination_is_bounded_and_cursor_safe() -> None:
    usage, _ = _service()
    for index in range(3):
        usage.record_blocked_work(
            {
                "work_id": f"work:{index}",
                "scope_id": "scope:a",
                "reason": "capacity_unavailable",
                "next_eligible_at": f"2026-01-01T00:00:0{index}Z",
            }
        )
    page = usage.blocked_work(authorities=_read_auth(), limit=1)
    assert page["success"] is True
    assert page["count"] == 1
    assert page["next_cursor"] == "work:0"
    page2 = usage.blocked_work(
        authorities=_read_auth(), limit=1, cursor=page["next_cursor"]
    )
    assert page2["count"] == 1
    assert page2["items"][0]["work_id"] == "work:1"
    oversized = usage.blocked_work(authorities=_read_auth(), limit=10_000)
    assert oversized["success"] is False
    assert oversized["error_code"] == "unbounded_page"
    bad_cursor = usage.blocked_work(
        authorities=_read_auth(), limit=1, cursor="missing"
    )
    assert bad_cursor["error_code"] == "invalid_cursor"


def test_route_preview_never_reserves_or_mutates() -> None:
    usage, _ = _service()
    before = usage.usage_revision()
    result = usage.route_preview(
        authorities=_read_auth(),
        candidates=[
            {
                "binding_id": "binding:a",
                "provider_id": "provider:openai",
                "deployment_id": "deployment:chat",
                "scope_id": "scope:a",
            }
        ],
    )
    assert result["success"] is True
    assert result["reserved"] is False
    assert result["invoked"] is False
    assert result["probed"] is False
    assert result["refreshed"] is False
    assert usage.usage_revision() == before


def test_budget_mutation_requires_guardrails_and_cannot_raise_parent() -> None:
    usage, _ = _service()
    denied = usage.set_budget(
        "child:1",
        authorities=_read_auth(),
        expected_usage_revision=usage.usage_revision(),
        idempotency_key="k1",
        lease_id="lease",
        fence=1,
        expected_effects=["set_budget"],
        budget={"limits": [{"dimension": "requests", "ceiling": 10}]},
    )
    assert denied["error_code"] in {
        "budget_authority_denied",
        "admin_denied",
    }

    parent = usage.set_budget(
        "parent:1",
        authorities=_admin_auth(),
        expected_usage_revision=usage.usage_revision(),
        idempotency_key="parent-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["set_budget"],
        budget={"limits": [{"dimension": "requests", "ceiling": 100}], "level": "goal"},
        actor="operator",
    )
    assert parent["success"] is True, parent
    assert parent["audit"]["operation"] == "usage_set_budget"

    child = usage.set_budget(
        "child:1",
        authorities=_admin_auth(),
        expected_usage_revision=usage.usage_revision(),
        idempotency_key="child-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["set_budget"],
        parent_target_id="parent:1",
        budget={"limits": [{"dimension": "requests", "ceiling": 50}], "level": "task"},
    )
    assert child["success"] is True, child

    raise_parent = usage.set_budget(
        "child:2",
        authorities=_admin_auth(),
        expected_usage_revision=usage.usage_revision(),
        idempotency_key="child-raise",
        lease_id="lease-1",
        fence=1,
        expected_effects=["set_budget"],
        parent_target_id="parent:1",
        budget={"limits": [{"dimension": "requests", "ceiling": 200}]},
    )
    assert raise_parent["success"] is False
    assert raise_parent["error_code"] == "parent_budget_raise_denied"


def test_mutations_require_revision_lease_fence_idempotency_and_reject_model_peer() -> None:
    usage, _ = _service()
    auths = _admin_auth()
    rev = usage.usage_revision()

    no_fence = usage.reset(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="k-fence",
        lease_id="lease",
        expected_effects=["reset"],
    )
    assert no_fence["error_code"] == "fence_required"

    no_lease = usage.reset(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="k-lease",
        fence=1,
        expected_effects=["reset"],
    )
    assert no_lease["error_code"] == "lease_required"

    model_denied = usage.correct(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="k-model",
        lease_id="lease",
        fence=1,
        expected_effects=["correction"],
        source="model_output",
        supersedes_event_id="evt-1",
        units={"entries": []},
    )
    assert model_denied["error_code"] == "mutation_denied_model_output"

    peer_denied = usage.set_policy(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="k-peer",
        lease_id="lease",
        fence=1,
        expected_effects=["set_policy"],
        source="remote_peer",
        policy={"mode": "observe"},
    )
    assert peer_denied["error_code"] == "mutation_denied_remote_peer"

    ok = usage.reset(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="reset-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["reset"],
        actor="operator-a",
    )
    assert ok["success"] is True, ok
    assert ok["audit"]["idempotency_key"] == "reset-1"

    replay = usage.reset(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="reset-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["reset"],
        actor="operator-a",
    )
    assert replay["success"] is True
    assert "idempotency_replay" in replay.get("reason_codes", [])

    conflict = usage.reset(
        "scope:b",
        authorities=auths,
        expected_usage_revision=usage.usage_revision(),
        idempotency_key="reset-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["reset"],
    )
    assert conflict["error_code"] == "idempotency_conflict"

    stale = usage.reset(
        "scope:a",
        authorities=auths,
        expected_usage_revision=rev,
        idempotency_key="reset-stale",
        lease_id="lease-1",
        fence=1,
        expected_effects=["reset"],
    )
    assert stale["error_code"] in {"stale_snapshot", "revision_mismatch"}


def test_headroom_bands() -> None:
    assert (
        usage_headroom_band(
            {"kind": "finite", "value": 0},
            {"kind": "finite", "value": 100},
        )
        == "exhausted"
    )
    assert (
        usage_headroom_band(
            {"kind": "finite", "value": 5},
            {"kind": "finite", "value": 100},
        )
        == "critical"
    )
    assert (
        usage_headroom_band(
            {"kind": "finite", "value": 20},
            {"kind": "finite", "value": 100},
        )
        == "low"
    )
    assert (
        usage_headroom_band(
            {"kind": "finite", "value": 40},
            {"kind": "finite", "value": 100},
        )
        == "medium"
    )
    assert (
        usage_headroom_band(
            {"kind": "finite", "value": 80},
            {"kind": "finite", "value": 100},
        )
        == "high"
    )
    assert (
        usage_headroom_band(
            {"kind": "unlimited"},
            {"kind": "unlimited"},
        )
        == "unlimited"
    )


def test_metrics_are_event_derived_and_low_cardinality() -> None:
    metrics = project_usage_governance_metrics(
        [
            {
                "type": "usage_reservation_denied",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "planning",
                "reason": "limit_exhausted",
            },
            {
                "type": "usage_estimate",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "implementation",
                "estimated": 100,
                "actual": 80,
                "dimension": "total_tokens",
            },
            {
                "type": "usage_wait",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "analysis",
                "reason": "wait",
            },
            {
                "type": "usage_reroute",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "analysis",
                "reason": "reroute",
            },
            {
                "type": "usage_fallback",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "rescue",
                "reason": "fallback",
            },
            {
                "type": "usage_reset",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "planning",
            },
            {
                "type": "usage_herd",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "planning",
            },
            {
                "type": "usage_starvation",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "validation",
            },
            {
                "type": "usage_fairness",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "planning",
                "fairness_state": "ready",
            },
            {
                "type": "usage_settlement",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "implementation",
            },
            {
                "type": "usage_correction",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "implementation",
            },
            {
                "type": "usage_headroom",
                "provider": "provider:openai",
                "deployment": "deployment:chat",
                "stage": "planning",
                "headroom_band": "low",
                "dimension": "requests",
            },
            {
                "type": "usage_ledger_health",
                "ledger_health": True,
            },
        ]
    )
    assert metrics["requirement_id"] == SUPERVISOR_USAGE_METRICS_REQUIREMENT_ID
    assert metrics["completion_authoritative"] is False
    assert metrics["operational_evidence_only"] is True
    names = {sample["name"] for sample in metrics["samples"]}
    for required in (
        "usage_denials_total",
        "usage_estimate_error_ratio_sum",
        "usage_waits_total",
        "usage_reroutes_total",
        "usage_fallbacks_total",
        "usage_resets_total",
        "usage_herd_total",
        "usage_starvation_total",
        "usage_fairness_total",
        "usage_settlements_total",
        "usage_corrections_total",
        "usage_headroom_band",
        "usage_ledger_health",
    ):
        assert required in names
    for sample in metrics["samples"]:
        labels = sample["labels"]
        assert not set(labels).intersection(forbidden_usage_metric_label_keys())
        for forbidden in (
            "request_id",
            "credential",
            "tenant",
            "prompt",
            "endpoint_url",
            "model_alias",
        ):
            assert forbidden not in labels
    from ipfs_accelerate_py.agent_supervisor.runtime import scheduler_metrics as sm

    with pytest.raises(ValueError, match="forbidden metric label"):
        sm._usage_series_key(  # noqa: SLF001
            "usage_waits_total",
            {
                "provider": "p",
                "deployment": "d",
                "stage": "planning",
                "reason": "wait",
                "request_id": "r1",
            },
        )


def test_service_execute_and_mcp_adapter_parity() -> None:
    usage, service = _service()
    auths = _admin_auth()
    via_service = service.usage_execute(
        "usage_set_budget",
        authorities=auths,
        target_id="scope:mcp",
        expected_usage_revision=usage.usage_revision(),
        idempotency_key="mcp-budget",
        lease_id="lease",
        fence=1,
        expected_effects=["set_budget"],
        budget={"limits": [{"dimension": "requests", "ceiling": 5}]},
    )
    assert via_service["success"] is True

    native_tools.set_provider_usage_control_service(usage)
    try:
        record = asyncio.run(
            native_tools.agent_supervisor_usage(
                "usage_status",
                authorities=_read_auth(),
            )
        )
        assert record["success"] is True
        assert record["requirement_id"] == SUPERVISOR_USAGE_CONTROL_REQUIREMENT_ID
        discovered = asyncio.run(native_tools.agent_supervisor_usage("discover"))
        assert discovered["success"] is True
        assert "operations" in discovered
    finally:
        native_tools.set_provider_usage_control_service(None)


def test_usage_controls_are_not_completion_evidence() -> None:
    usage, service = _service()
    status = usage.status(authorities=_read_auth())
    assert status["completion_authoritative"] is False
    assert status["operational_evidence_only"] is True
    catalog = service.usage_discover()
    assert catalog["completion_authoritative"] is False
    for operation in catalog["operations"]:
        assert operation["completion_authoritative"] is False
