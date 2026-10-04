"""AICAT-034: authorized usage controls, receipts, and observability."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from ipfs_accelerate_py.endpoint_usage.controls import (
    USAGE_ADMIN_AUTHORITY,
    USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID,
    USAGE_READ_AUTHORITY,
    USAGE_READ_DETAIL_AUTHORITY,
    USAGE_REASON_CODES,
    ControlOperation,
    UsageControlService,
    headroom_band,
    usage_control_authorities,
    usage_control_operations,
    usage_control_reason_codes,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.observability import (
    USAGE_OBSERVABILITY_REQUIREMENT_ID,
    UsageObservability,
    forbidden_metric_label_keys,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    UsageDimension,
    UsageEventKind,
    UsageLimit,
    UsageVector,
    UsageVectorEntry,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore
from ipfs_accelerate_py.mcp_server.mcplusplus import idl_registry
from ipfs_accelerate_py.mcp_server.tools.ai_router_tools import text_embedding
from ipfs_accelerate_py.mcp_server.tools.model_tools import native_model_tools


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _scope(key: str = "prov-a") -> EndpointUsageScope:
    provider_id = stable_id("provider", key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation="text.chat",
        deployment_id=stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:USAGE_CTRL_KEY", key_id="usage-ctrl"
        ),
        account_pseudonym=stable_id("account", key, "acct"),
    )


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    currency: str | None = None,
) -> UsageLimit:
    return UsageLimit(
        scope_id=scope_id,
        dimension=dimension,
        ceiling=Quantity.finite(ceiling),
        window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
        currency=currency,
    )


def _harness():
    clock = FakeClock(_now())
    store = InMemoryUsageLedgerStore(clock=clock, writer_id="writer-1", fence=1)
    coord = UsageCoordinator(store, writer_id="writer-1", fence=1)
    scope = _scope()
    snap = coord.configure_limits(
        scope.scope_id,
        [
            _limit(scope.scope_id, UsageDimension.REQUESTS, 100),
            _limit(
                scope.scope_id,
                UsageDimension.COST_MICROS,
                1_000_000,
                currency="usd",
            ),
        ],
    )
    obs = UsageObservability()
    service = UsageControlService(
        coord,
        observability=obs,
        catalog_revision_provider=lambda: "catalog-rev-test",
    )
    return scope, coord, store, service, obs, snap


def _run(coro):
    return asyncio.run(coro)


class _ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[definition["name"]] = definition


# ---------------------------------------------------------------------------
# Python control surface
# ---------------------------------------------------------------------------


def test_requirement_ids_and_shared_vocabularies() -> None:
    assert USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID.startswith("requirement:")
    assert USAGE_OBSERVABILITY_REQUIREMENT_ID.startswith("requirement:")
    assert "read_denied" in USAGE_REASON_CODES
    assert set(usage_control_operations()) == {op.value for op in ControlOperation}
    authorities = usage_control_authorities()
    assert authorities["read"] == USAGE_READ_AUTHORITY
    assert authorities["admin"] == USAGE_ADMIN_AUTHORITY


def test_reads_are_side_effect_free_and_bind_revisions() -> None:
    scope, coord, store, service, obs, snap = _harness()
    before = store.read()["revision"]
    auths = [USAGE_READ_AUTHORITY]
    status = service.status(authorities=auths)
    health = service.health(authorities=auths)
    limits = service.limits(scope.scope_id, authorities=auths)
    headroom = service.headroom(scope.scope_id, authorities=auths)
    reservations = service.reservations(scope.scope_id, authorities=auths)
    receipts = service.receipts(authorities=auths)
    adapters = service.adapter_capabilities(authorities=auths)
    after = store.read()["revision"]

    assert before == after
    for payload in (status, health, limits, headroom, reservations, receipts, adapters):
        assert payload["success"] is True
        assert payload["catalog_revision"] == "catalog-rev-test"
        assert "usage_revision" in payload
        assert_no_prompt_media_or_output(
            {k: v for k, v in payload.items() if k not in {"error"}}
            if False
            else {
                k: v
                for k, v in payload.items()
                if str(k).casefold() not in {"error"}
            }
        )
    assert status["count"] >= 1
    assert health["healthy"] is True
    assert limits["count"] >= 1
    assert headroom["items"]
    assert adapters["count"] >= 1


def test_read_requires_authority() -> None:
    _, _, _, service, _, _ = _harness()
    denied = service.status(authorities=[])
    assert denied["success"] is False
    assert denied["error_code"] == "read_denied"


def test_detail_redaction_hides_account_and_cost_without_detail_authority() -> None:
    scope, _, _, service, _, _ = _harness()
    aggregate = service.limits(scope.scope_id, authorities=[USAGE_READ_AUTHORITY])
    # Account / credential pseudonyms must not appear without detail authority.
    blob = str(aggregate)
    assert "account_pseudonym" not in blob
    assert "credential_pseudonym" not in blob
    # Cost amounts collapsed to band when present
    for item in aggregate.get("items") or []:
        if item.get("dimension") == "cost_micros":
            assert "band" in item or item.get("ceiling") is None


def test_pagination_is_bounded() -> None:
    scope, _, _, service, _, _ = _harness()
    page = service.limits(
        scope.scope_id, authorities=[USAGE_READ_AUTHORITY], limit=1
    )
    assert page["success"] is True
    assert page["count"] == 1
    oversized = service.limits(
        scope.scope_id, authorities=[USAGE_READ_AUTHORITY], limit=10_000
    )
    assert oversized["success"] is False
    assert oversized["error_code"] == "unbounded_page"


def test_route_preview_never_reserves() -> None:
    scope, coord, store, service, _, snap = _harness()
    before = store.read()["revision"]
    before_res = len(store.read().get("reservations") or {})
    binding_id = stable_id("binding", "preview-a")
    result = service.route_preview(
        authorities=[USAGE_READ_AUTHORITY],
        candidates=[
            {
                "binding_id": binding_id,
                "provider_id": scope.provider_id,
                "scope_id": scope.scope_id,
            }
        ],
        scope_by_binding={binding_id: scope.scope_id},
        catalog_revision="catalog-rev-test",
    )
    after = store.read()
    assert result["success"] is True, result
    assert result["reserved"] is False
    assert result["invoked"] is False
    assert result["probed"] is False
    assert after["revision"] == before
    assert len(after.get("reservations") or {}) == before_res


def test_admin_mutations_require_guardrails_and_audit() -> None:
    scope, coord, store, service, obs, snap = _harness()
    # Missing admin
    denied = service.reset(
        scope.scope_id,
        authorities=[USAGE_READ_AUTHORITY],
        expected_usage_revision=snap.usage_revision,
        idempotency_key="k1",
        lease_id="lease",
        fence=1,
    )
    assert denied["error_code"] == "admin_denied"

    admin = [USAGE_READ_AUTHORITY, USAGE_ADMIN_AUTHORITY]
    # Missing fence
    no_fence = service.reset(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="k2",
        lease_id="lease",
    )
    assert no_fence["error_code"] == "fence_required"

    # Model output cannot mutate
    model_denied = service.import_observation(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="k3",
        lease_id="lease",
        fence=1,
        source="model_output",
        units={
            "entries": [
                {"dimension": "requests", "amount": {"kind": "finite", "value": 1}}
            ]
        },
    )
    assert model_denied["error_code"] == "mutation_denied_model_output"

    peer_denied = service.correct(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="k4",
        lease_id="lease",
        fence=1,
        source="remote_peer",
        supersedes_event_id="evt-missing",
        units={
            "entries": [
                {"dimension": "requests", "amount": {"kind": "finite", "value": 1}}
            ]
        },
    )
    assert peer_denied["error_code"] == "mutation_denied_remote_peer"

    # Successful reset + idempotent replay
    ok = service.reset(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="reset-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["reset"],
        actor="operator-a",
    )
    assert ok["success"] is True
    assert ok["audit"]["operation"] == "reset"
    assert ok["audit"]["idempotency_key"] == "reset-1"

    replay = service.reset(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="reset-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["reset"],
        actor="operator-a",
    )
    assert replay["success"] is True
    assert "idempotency_replay" in replay.get("reason_codes", [])


def test_override_and_import_with_expected_revision() -> None:
    scope, coord, store, service, obs, snap = _harness()
    admin = [USAGE_READ_AUTHORITY, USAGE_ADMIN_AUTHORITY]
    new_limit = _limit(scope.scope_id, UsageDimension.REQUESTS, 50)
    overridden = service.override_limits(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="ovr-1",
        lease_id="lease-1",
        fence=1,
        expected_effects=["override:requests"],
        limits=[new_limit.to_dict()],
    )
    assert overridden["success"] is True
    new_rev = overridden["usage_revision"]
    assert new_rev != snap.usage_revision

    # Stale expected revision fails closed
    stale = service.override_limits(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="ovr-2",
        lease_id="lease-1",
        fence=1,
        limits=[new_limit.to_dict()],
    )
    assert stale["error_code"] in {"stale_snapshot", "revision_mismatch"}

    units = UsageVector(
        entries=(
            UsageVectorEntry(
                dimension=UsageDimension.REQUESTS,
                amount=Quantity.finite(1),
            ),
        )
    )
    imported = service.import_observation(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=new_rev,
        idempotency_key="imp-1",
        lease_id="lease-1",
        fence=1,
        kind=UsageEventKind.OBSERVATION_SUCCESS.value,
        units=units.to_dict(),
        expected_effects=["import_observation"],
    )
    assert imported["success"] is True


def test_headroom_bands() -> None:
    assert (
        headroom_band(Quantity.finite(0), Quantity.finite(100))
        == "exhausted"
    )
    assert headroom_band(Quantity.finite(5), Quantity.finite(100)) == "critical"
    assert headroom_band(Quantity.finite(20), Quantity.finite(100)) == "low"
    assert headroom_band(Quantity.finite(40), Quantity.finite(100)) == "medium"
    assert headroom_band(Quantity.finite(80), Quantity.finite(100)) == "high"
    assert headroom_band(Quantity.unlimited(), Quantity.unlimited()) == "unlimited"


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_metrics_are_event_derived_and_low_cardinality() -> None:
    scope, coord, store, service, obs, snap = _harness()
    admin = [USAGE_READ_AUTHORITY, USAGE_ADMIN_AUTHORITY]
    service.reset(
        scope.scope_id,
        authorities=admin,
        expected_usage_revision=snap.usage_revision,
        idempotency_key="m-reset",
        lease_id="lease",
        fence=1,
        expected_effects=["reset"],
    )
    obs.ingest_document(store.read())
    obs.record_denial(provider="provider:openai", deployment="deployment:chat", reason="limit_exhausted")
    obs.record_wait(provider="provider:openai", deployment="deployment:chat")
    obs.record_reroute(provider="provider:openai", deployment="deployment:chat")
    obs.record_fallback(provider="provider:openai", deployment="deployment:chat")
    obs.record_estimate_error(100, 80, provider="provider:openai", deployment="deployment:chat")
    obs.record_store_health(healthy=True)
    snap_metrics = obs.snapshot()
    assert snap_metrics["series_count"] >= 1
    names = {sample["name"] for sample in snap_metrics["samples"]}
    assert "usage_resets_total" in names or "usage_control_mutations_total" in names
    assert "usage_reservation_denials_total" in names
    assert "usage_waits_total" in names
    assert "usage_reroutes_total" in names
    assert "usage_fallbacks_total" in names
    assert "usage_estimate_error_ratio_sum" in names
    assert "usage_store_health" in names

    # Forbidden labels rejected
    with pytest.raises(ValueError):
        obs._update(  # noqa: SLF001 - contract enforcement
            "usage_reservations_total",
            1,
            {
                "provider": "p",
                "deployment": "d",
                "outcome": "accepted",
                "request_id": "r1",
            },
        )
    forbidden = forbidden_metric_label_keys()
    assert "request_id" in forbidden
    assert "credential" in forbidden
    assert "tenant" in forbidden
    assert "alias" in forbidden
    assert "model" in forbidden
    assert "endpoint_url" in forbidden or "endpoint_uri" in forbidden


# ---------------------------------------------------------------------------
# MCP tool wiring
# ---------------------------------------------------------------------------


def test_mcp_usage_tools_register_and_delegate() -> None:
    scope, coord, store, service, obs, snap = _harness()
    native_model_tools.set_usage_control_service(service)
    registry = _ToolRegistry()
    native_model_tools.register_native_model_tools(registry)
    text_embedding.register_native_ai_router_tools(registry)

    assert "model_catalog_usage" in registry.tools
    assert "model_catalog_usage_metrics" in registry.tools
    assert "route_preview" in registry.tools
    # Existing catalog tools preserved
    assert "model_catalog_health" in registry.tools
    assert "llm_generate" in registry.tools

    status = _run(
        native_model_tools.model_catalog_usage(
            "status",
            authorities=[USAGE_READ_AUTHORITY],
        )
    )
    assert status["success"] is True
    assert status["catalog_revision"] == "catalog-rev-test"

    binding_id = stable_id("binding", "mcp-preview")
    preview = _run(
        text_embedding.route_preview(
            candidates=[
                {
                    "binding_id": binding_id,
                    "provider_id": scope.provider_id,
                    "scope_id": scope.scope_id,
                }
            ],
            scope_by_binding={binding_id: scope.scope_id},
            authorities=[USAGE_READ_AUTHORITY],
        )
    )
    assert preview["success"] is True, preview
    assert preview["reserved"] is False

    metrics = _run(
        native_model_tools.model_catalog_usage_metrics(
            authorities=[USAGE_READ_AUTHORITY]
        )
    )
    assert metrics["success"] is True

    native_model_tools.set_usage_control_service(None)


def test_mcp_admin_mutation_through_tool() -> None:
    scope, coord, store, service, obs, snap = _harness()
    native_model_tools.set_usage_control_service(service)
    result = _run(
        native_model_tools.model_catalog_usage(
            "reset",
            scope_id=scope.scope_id,
            admin=True,
            expected_usage_revision=snap.usage_revision,
            idempotency_key="mcp-reset-1",
            lease_id="lease-mcp",
            fence=1,
            expected_effects=["reset"],
        )
    )
    assert result["success"] is True
    assert result["audit"]["operation"] == "reset"
    native_model_tools.set_usage_control_service(None)


# ---------------------------------------------------------------------------
# Python / MCP / MCP++ schema & reason-code agreement
# ---------------------------------------------------------------------------


def test_python_mcp_mcplusplus_schemas_and_reason_codes_agree() -> None:
    registry = _ToolRegistry()
    native_model_tools.register_native_model_tools(registry)
    text_embedding.register_native_ai_router_tools(registry)

    mcp_usage_schema = registry.tools["model_catalog_usage"]["input_schema"]
    mcp_preview_schema = registry.tools["route_preview"]["input_schema"]
    idl_schemas = idl_registry.ai_usage_v1_input_schemas()

    assert idl_schemas["model_catalog_usage"]["properties"]["operation"]["enum"] == (
        mcp_usage_schema["properties"]["operation"]["enum"]
    )
    assert set(idl_schemas["route_preview"]["properties"]) == set(
        mcp_preview_schema["properties"]
    )
    assert idl_schemas["model_catalog_usage"]["required"] == mcp_usage_schema["required"]

    # Authorities agree
    py_auth = usage_control_authorities()
    idl_auth = idl_registry.usage_control_authorities()
    assert py_auth == idl_auth
    assert idl_registry.AI_USAGE_READ_AUTHORITY == USAGE_READ_AUTHORITY
    assert idl_registry.AI_USAGE_ADMIN_AUTHORITY == USAGE_ADMIN_AUTHORITY

    # Reason codes agree
    py_codes = set(usage_control_reason_codes())
    idl_codes = set(idl_registry.usage_control_reason_codes())
    assert py_codes == idl_codes

    # Descriptor is well-formed and separate from frozen catalog CID
    usage_desc = idl_registry.build_ai_usage_v1_descriptor()
    assert usage_desc["name"] == idl_registry.AI_USAGE_INTERFACE_NAME
    assert set(m["operation"] for m in usage_desc["methods"]) == {
        "model_catalog_usage",
        "model_catalog_usage_metrics",
        "route_preview",
    }
    catalog_cid = idl_registry.compute_interface_cid(
        idl_registry.build_ai_catalog_v1_descriptor()
    )
    assert catalog_cid.startswith("bafk")
    # Frozen catalog surface unchanged (matches test_mcplusplus_ai_catalog_idl)
    assert catalog_cid == "bafkreiat4dykpooyzlu3lugkbvouyhao5i4s4irfwxs6h4c2ujzlx5zrlu"


def test_receipt_recording_is_redacted() -> None:
    scope, _, _, service, _, snap = _harness()
    service.record_receipt(
        {
            "schema_version": "1.0",
            "receipt_id": "receipt-1",
            "catalog_revision": "catalog-rev-test",
            "usage_revision": snap.usage_revision,
            "scope_id": scope.scope_id,
            "final_status": "committed",
            "reason_codes": ["ok"],
        }
    )
    listed = service.receipts(authorities=[USAGE_READ_AUTHORITY])
    assert listed["success"] is True
    assert listed["count"] == 1
    assert listed["items"][0]["receipt_id"] == "receipt-1"
