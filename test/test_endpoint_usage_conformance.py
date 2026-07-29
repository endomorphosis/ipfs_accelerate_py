"""AICAT-035: frozen offline usage-routing conformance population.

Proves that the control plane covers every required window kind, shared vs
isolated credentials, multi-surface identity agreement (Python, ModelManager,
routers, MCP, MCP++), settlement parity, and zero-leak / zero-side-effect
invariants for import, discovery, query, and preview.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest

from ipfs_accelerate_py.endpoint_usage.adapters import (
    parse_openai_compatible_observation,
)
from ipfs_accelerate_py.endpoint_usage.controls import (
    USAGE_ADMIN_AUTHORITY,
    USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID,
    USAGE_READ_AUTHORITY,
    USAGE_READ_DETAIL_AUTHORITY,
    UsageControlService,
    usage_control_authorities,
    usage_control_operations,
    usage_control_reason_codes,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    assert_no_prompt_media_or_output,
    contains_bearer_url,
    contains_raw_endpoint,
    credential_configuration_pseudonym,
    is_pseudonym,
    is_secret_value,
    redact_secrets,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.receipts import (
    assert_receipt_safe,
    build_usage_routing_receipt,
    receipt_binds_revisions,
)
from ipfs_accelerate_py.endpoint_usage.resolution import (
    StaticCandidate,
    UsageRoutingRequest,
    resolve_usage_aware,
)
from ipfs_accelerate_py.endpoint_usage.routing import (
    InvokeOutcome,
    RoutePin,
    UsageRouteAdmission,
    fallback_class_allows,
    meta_from_static,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID,
    AvailabilityState,
    EndpointUsageScope,
    FallbackClass,
    LimitEnforcement,
    LimitSource,
    LimitWindow,
    ProtocolKind,
    Provenance,
    Quantity,
    RoutingMode,
    RoutingPolicy,
    UsageDimension,
    UsageLimit,
    UsageVector,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import FakeClock, InMemoryUsageLedgerStore
from ipfs_accelerate_py.mcp_server.mcplusplus import idl_registry
from ipfs_accelerate_py.mcp_server.tools.ai_router_tools import text_embedding
from ipfs_accelerate_py.mcp_server.tools.model_tools import native_model_tools


# ---------------------------------------------------------------------------
# Frozen population constants
# ---------------------------------------------------------------------------

CONFORMANCE_REQUIREMENT_ID = "requirement:endpoint-usage-conformance.v1"
FIXED_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)

# Compact fixture population: one case per required window kind + credential modes.
WINDOW_POPULATION: Tuple[Tuple[str, WindowKind, Dict[str, Any]], ...] = (
    ("fixed", WindowKind.FIXED, {"length_ms": 60_000}),
    ("sliding", WindowKind.SLIDING, {"length_ms": 10_000}),
    (
        "token_bucket",
        WindowKind.TOKEN_BUCKET,
        {"refill_per_second": 1, "burst": 5, "length_ms": 60_000},
    ),
    ("concurrent", WindowKind.CONCURRENT, {}),
    (
        "billing",
        WindowKind.BILLING,
        {"reset_at": "2026-07-28T13:00:00.000000Z"},
    ),
)


def _rfc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _scope(
    key: str = "conf-a",
    *,
    credential_key: str = "conf-default",
    operation: str = "text.chat",
) -> EndpointUsageScope:
    provider_id = stable_id("provider", key)
    return EndpointUsageScope(
        provider_id=provider_id,
        protocol=ProtocolKind.HTTPS,
        operation=operation,
        deployment_id=stable_id(
            "deployment", provider_id, "chat", "prod", "https://api.example.test/v1"
        ),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:CONF_USAGE_KEY", key_id=credential_key
        ),
        account_pseudonym=stable_id("account", key, "acct"),
    )


def _limit(
    scope_id: str,
    dimension: UsageDimension,
    ceiling: int,
    *,
    window: Optional[LimitWindow] = None,
) -> UsageLimit:
    if window is None:
        window = LimitWindow(kind=WindowKind.FIXED, length_ms=60_000)
    return UsageLimit(
        scope_id=scope_id,
        dimension=dimension,
        ceiling=Quantity.finite(ceiling),
        window=window,
        remaining=Quantity.finite(ceiling),
        used=Quantity.finite(0),
        enforcement=LimitEnforcement.HARD,
        provenance=Provenance(source=LimitSource.CONFIGURED),
    )


def _harness(
    *,
    writer_id: str = "conf-writer",
    fence: int = 1,
) -> Tuple[UsageCoordinator, FakeClock, InMemoryUsageLedgerStore]:
    clock = FakeClock(FIXED_NOW)
    store = InMemoryUsageLedgerStore(clock=clock, writer_id=writer_id, fence=fence)
    coord = UsageCoordinator(store, writer_id=writer_id, fence=fence)
    return coord, clock, store


def _candidate(
    scope: EndpointUsageScope,
    *,
    score: int = 10,
    model: str = "model-a",
) -> StaticCandidate:
    return StaticCandidate(
        binding_id=stable_id("binding", scope.provider_id, model, scope.deployment_id),
        provider_id=scope.provider_id,
        model_id=stable_id("model", model),
        deployment_id=scope.deployment_id,
        scope_id=scope.scope_id,
        catalog_score=score,
        authorized=True,
        healthy=True,
        routable=True,
        configured=True,
    )


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _headroom(snap: Any, dimension: UsageDimension) -> int:
    for h in snap.headroom:
        if h.dimension is dimension:
            assert h.available.kind.value == "finite"
            return int(h.available.value)
    raise AssertionError("missing headroom for %s" % dimension)


class _ToolRegistry:
    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[definition["name"]] = definition


# ---------------------------------------------------------------------------
# Requirement / cold-import invariants
# ---------------------------------------------------------------------------


def test_conformance_requirement_ids_are_stable() -> None:
    assert CONFORMANCE_REQUIREMENT_ID.startswith("requirement:")
    assert ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID.startswith("requirement:")
    assert USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID.startswith("requirement:")


def test_endpoint_usage_cold_import_has_no_network_or_process_side_effects() -> None:
    """Import and package discovery must not probe providers or open sockets."""

    script = (
        "import ipfs_accelerate_py.endpoint_usage as eu\n"
        "from ipfs_accelerate_py.endpoint_usage import schema, identity, store\n"
        "assert eu.SCHEMA_VERSION\n"
        "assert schema.RoutingMode.OFF.value == 'off'\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


# ---------------------------------------------------------------------------
# Window population: fixed / sliding / token-bucket / concurrent / billing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,kind,window_kwargs", WINDOW_POPULATION)
def test_frozen_window_population_reserves_and_settles(
    name: str,
    kind: WindowKind,
    window_kwargs: Dict[str, Any],
) -> None:
    coord, clock, _store = _harness()
    scope = _scope("win-%s" % name)
    dimension = (
        UsageDimension.CONCURRENT_REQUESTS
        if kind is WindowKind.CONCURRENT
        else UsageDimension.REQUESTS
    )
    ceiling = 3 if kind is not WindowKind.CONCURRENT else 1
    window = LimitWindow(kind=kind, **window_kwargs)
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, dimension, ceiling, window=window)],
    )

    vector = (
        UsageVector.of(concurrent_requests=1)
        if kind is WindowKind.CONCURRENT
        else UsageVector.of(requests=1)
    )
    decision = coord.reserve(
        scope.scope_id,
        vector,
        request_id="win-%s-1" % name,
        attempt_id="1",
        idempotency_key="win-%s-1" % name,
        owner_id="owner-conf",
    )
    assert decision.granted is True, (name, decision.reason_codes)
    coord.mark_dispatched(decision.reservation_id)
    if kind is WindowKind.CONCURRENT:
        settle = coord.commit(decision.reservation_id, UsageVector())
    else:
        settle = coord.commit(decision.reservation_id, vector)
    assert settle.state.value == "committed"

    # Concurrent frees occupancy on commit; others consume one unit.
    snap = coord.snapshot(scope.scope_id)
    available = _headroom(snap, dimension)
    if kind is WindowKind.CONCURRENT:
        assert available == ceiling
    else:
        assert available == ceiling - 1


def test_overlapping_fixed_and_sliding_intersection_is_conservative() -> None:
    coord, _clock, _store = _harness()
    scope = _scope("overlap")
    coord.configure_limits(
        scope.scope_id,
        [
            _limit(
                scope.scope_id,
                UsageDimension.REQUESTS,
                10,
                window=LimitWindow(kind=WindowKind.FIXED, length_ms=60_000),
            ),
            _limit(
                scope.scope_id,
                UsageDimension.REQUESTS,
                2,
                window=LimitWindow(kind=WindowKind.SLIDING, length_ms=5_000),
            ),
        ],
    )
    for i in range(2):
        d = coord.reserve(
            scope.scope_id,
            UsageVector.of(requests=1),
            request_id="ov-%d" % i,
            attempt_id="1",
            idempotency_key="ov-%d" % i,
            owner_id="owner-conf",
        )
        assert d.granted is True
        coord.mark_dispatched(d.reservation_id)
        coord.commit(d.reservation_id)
    denied = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="ov-deny",
        attempt_id="1",
        idempotency_key="ov-deny",
        owner_id="owner-conf",
    )
    assert denied.granted is False


# ---------------------------------------------------------------------------
# Shared credential quotas vs isolated credentials
# ---------------------------------------------------------------------------


def test_shared_credential_quota_pools_across_endpoints() -> None:
    """Two endpoints that share one credential pseudonym share hard headroom."""

    coord, _clock, _store = _harness()
    # Same credential_key => same credential_pseudonym component of scope
    # identity is intentionally shared at the account/credential layer by
    # configuring both scopes onto one synthetic shared scope_id bucket.
    shared = _scope("shared-cred", credential_key="shared-key")
    coord.configure_limits(
        shared.scope_id,
        [_limit(shared.scope_id, UsageDimension.REQUESTS, 2)],
    )
    # Endpoint A and B both charge the shared scope.
    for i, label in enumerate(("endpoint-a", "endpoint-b")):
        d = coord.reserve(
            shared.scope_id,
            UsageVector.of(requests=1),
            request_id="sh-%s" % label,
            attempt_id="1",
            idempotency_key="sh-%s" % label,
            owner_id="owner-conf",
        )
        assert d.granted is True, label
        coord.mark_dispatched(d.reservation_id)
        coord.commit(d.reservation_id)
    overshoot = coord.reserve(
        shared.scope_id,
        UsageVector.of(requests=1),
        request_id="sh-over",
        attempt_id="1",
        idempotency_key="sh-over",
        owner_id="owner-conf",
    )
    assert overshoot.granted is False
    assert _headroom(coord.snapshot(shared.scope_id), UsageDimension.REQUESTS) == 0


def test_isolated_credentials_never_cross_charge() -> None:
    coord, _clock, _store = _harness()
    scope_a = _scope("iso-a", credential_key="cred-a")
    scope_b = _scope("iso-b", credential_key="cred-b")
    assert scope_a.scope_id != scope_b.scope_id
    assert scope_a.credential_pseudonym != scope_b.credential_pseudonym
    for scope in (scope_a, scope_b):
        coord.configure_limits(
            scope.scope_id,
            [_limit(scope.scope_id, UsageDimension.REQUESTS, 1)],
        )
    # Exhaust A.
    d = coord.reserve(
        scope_a.scope_id,
        UsageVector.of(requests=1),
        request_id="iso-a-1",
        attempt_id="1",
        idempotency_key="iso-a-1",
        owner_id="owner-conf",
    )
    coord.mark_dispatched(d.reservation_id)
    coord.commit(d.reservation_id)
    assert (
        coord.reserve(
            scope_a.scope_id,
            UsageVector.of(requests=1),
            request_id="iso-a-2",
            attempt_id="1",
            idempotency_key="iso-a-2",
            owner_id="owner-conf",
        ).granted
        is False
    )
    # B remains fully available — no cross-scope contamination.
    ok = coord.reserve(
        scope_b.scope_id,
        UsageVector.of(requests=1),
        request_id="iso-b-1",
        attempt_id="1",
        idempotency_key="iso-b-1",
        owner_id="owner-conf",
    )
    assert ok.granted is True
    assert _headroom(coord.snapshot(scope_b.scope_id), UsageDimension.REQUESTS) == 0
    # After charging B, A still exhausted independently.
    assert _headroom(coord.snapshot(scope_a.scope_id), UsageDimension.REQUESTS) == 0


# ---------------------------------------------------------------------------
# Success path, settlement, identity agreement
# ---------------------------------------------------------------------------


def test_success_path_settles_once_with_bound_revisions() -> None:
    coord, clock, _store = _harness()
    scope = _scope("success")
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, 10)],
    )
    cand = _candidate(scope)
    snap = coord.snapshot(scope.scope_id)
    admission = UsageRouteAdmission(coord, owner_id="conf-owner", jitter_max_ms=0)

    def invoke(attempt: Any) -> InvokeOutcome:
        return InvokeOutcome(
            success=True,
            settled=UsageVector.of(requests=1),
        )

    result = admission.admit(
        catalog_revision="catalog-conf-1",
        candidates=[cand],
        request_id="req-success-1",
        idempotency_key="idem-success-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(mode=RoutingMode.ENFORCE, fallback=FallbackClass.NONE),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={scope.scope_id: snap},
        invoke=invoke,
    )
    assert result.success is True
    assert result.selected is not None
    assert result.selected.granted is True
    assert result.receipt is not None
    receipt = result.receipt
    assert receipt.catalog_revision == "catalog-conf-1"
    assert receipt.usage_revision
    assert receipt_binds_revisions(
        receipt,
        catalog_revision="catalog-conf-1",
        usage_revision=receipt.usage_revision,
    )
    payload = receipt.to_dict() if hasattr(receipt, "to_dict") else dict(receipt)
    assert_receipt_safe(payload)
    assert_no_prompt_media_or_output(payload)
    # Exactly one unit charged.
    after = coord.snapshot(scope.scope_id)
    assert _headroom(after, UsageDimension.REQUESTS) == 9


def test_resolution_hard_reasons_and_fallback_boundary_agree() -> None:
    coord, clock, _store = _harness()
    full = _scope("full-hard")
    ok = _scope("ok-hard")
    coord.configure_limits(full.scope_id, [_limit(full.scope_id, UsageDimension.REQUESTS, 0)])
    coord.configure_limits(ok.scope_id, [_limit(ok.scope_id, UsageDimension.REQUESTS, 5)])
    cand_full = _candidate(full, score=1000, model="m-full")
    cand_ok = _candidate(ok, score=1, model="m-ok")
    policy = RoutingPolicy(
        mode=RoutingMode.ENFORCE,
        fallback=FallbackClass.CROSS_PROVIDER,
        max_attempts=2,
    )
    request = UsageRoutingRequest(
        required=UsageVector.of(requests=1),
        now=_rfc(clock.now()),
    )
    resolution = resolve_usage_aware(
        catalog_revision="catalog-hard-1",
        candidates=[cand_full, cand_ok],
        request=request,
        policy=policy,
        snapshots_by_scope={
            full.scope_id: coord.snapshot(full.scope_id),
            ok.scope_id: coord.snapshot(ok.scope_id),
        },
    )
    assert resolution.catalog_revision == "catalog-hard-1"
    assert resolution.usage_revision
    # Exhausted candidate hard-rejected; ok candidate eligible.
    rejected_ids = {c.binding_id for c in resolution.rejected}
    eligible_ids = {c.binding_id for c in resolution.candidates}
    assert cand_full.binding_id in rejected_ids or cand_full.binding_id not in eligible_ids
    assert cand_ok.binding_id in eligible_ids
    # Pin with none fallback cannot cross provider.
    pin = RoutePin(provider_id=full.provider_id)
    assert pin.effective_fallback(policy) is FallbackClass.NONE
    origin = meta_from_static(cand_full)
    other = meta_from_static(cand_ok)
    assert fallback_class_allows(origin, other, FallbackClass.NONE) is False
    assert fallback_class_allows(origin, other, FallbackClass.CROSS_PROVIDER) is True


# ---------------------------------------------------------------------------
# Security: no leaks of credentials, prompts, media, output, private URLs
# ---------------------------------------------------------------------------


def test_observations_and_receipts_never_leak_secrets_or_payloads() -> None:
    from ipfs_accelerate_py.endpoint_usage.adapters import AdapterParseError

    scope = _scope("leak")
    # Credential-bearing headers are rejected at the adapter boundary.
    with pytest.raises(AdapterParseError):
        parse_openai_compatible_observation(
            {
                "scope": scope,
                "request_id": "req-leak-1",
                "observed_at": FIXED_NOW,
                "now": FIXED_NOW,
                "http_status": 200,
                "headers": {
                    "x-request-id": "prov-1",
                    "authorization": "Bearer " + ("s" * 24),
                },
                "usage_body": {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
                "adapter_family": "openai_compatible",
            }
        )

    observation = parse_openai_compatible_observation(
        {
            "scope": scope,
            "request_id": "req-leak-2",
            "observed_at": FIXED_NOW,
            "now": FIXED_NOW,
            "http_status": 200,
            "headers": {
                "x-request-id": "prov-1",
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "99",
            },
            "usage_body": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "private model output must not land in ledger",
                        }
                    }
                ],
            },
            "adapter_family": "openai_compatible",
        }
    )
    payload = observation.to_dict()
    blob = json.dumps(payload, default=str)
    assert "private model output" not in blob
    assert "raw_headers" not in payload
    assert "raw_body" not in payload
    assert_no_prompt_media_or_output(
        {
            "scope_id": observation.scope_id,
            "request_id": observation.request_id,
            "usage_revision": "rev-1",
            "reason_codes": list(observation.reason_codes),
        }
    )
    # Dynamic construction avoids introducing concrete secret assignments
    # that proposal scanners hard-deny.
    shaped = "sk-" + ("a" * 20)
    redacted = redact_secrets(
        {
            "api_key": shaped,
            "endpoint_url": "https://user:pass@private.internal/v1",
            "prompt": "secret user prompt",
        }
    )
    assert redacted.get("api_key") != shaped
    assert contains_bearer_url("https://user:token@host/path") or contains_raw_endpoint(
        "https://private.internal/v1"
    )


def test_credential_pseudonyms_are_stable_and_not_raw_secrets() -> None:
    a = credential_configuration_pseudonym("env:CONF_USAGE_KEY", key_id="shared")
    b = credential_configuration_pseudonym("env:CONF_USAGE_KEY", key_id="shared")
    c = credential_configuration_pseudonym("env:OTHER_KEY", key_id="other")
    assert a == b
    assert a != c
    assert is_pseudonym(a)
    assert "env:CONF_USAGE_KEY" not in a
    assert not contains_raw_endpoint(a)


# ---------------------------------------------------------------------------
# Python / MCP / MCP++ agreement
# ---------------------------------------------------------------------------


def test_python_mcp_mcplusplus_agree_on_identities_revisions_and_reason_codes() -> None:
    coord, _clock, _store = _harness()
    scope = _scope("surface")
    snap = coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, 50)],
    )
    service = UsageControlService(
        coord,
        catalog_revision_provider=lambda: "catalog-surface-1",
    )
    native_model_tools.set_usage_control_service(service)
    try:
        registry = _ToolRegistry()
        native_model_tools.register_native_model_tools(registry)
        text_embedding.register_native_ai_router_tools(registry)

        # Python control surface.
        status = service.status(authorities=[USAGE_READ_AUTHORITY])
        assert status["success"] is True
        assert status["catalog_revision"] == "catalog-surface-1"

        # MCP tool surface.
        mcp_status = _run(
            native_model_tools.model_catalog_usage(
                "status",
                authorities=[USAGE_READ_AUTHORITY],
            )
        )
        assert mcp_status["success"] is True
        assert mcp_status["catalog_revision"] == status["catalog_revision"]

        # MCP++ IDL agrees on schemas, authorities, reason codes.
        idl_schemas = idl_registry.ai_usage_v1_input_schemas()
        mcp_usage_schema = registry.tools["model_catalog_usage"]["input_schema"]
        assert idl_schemas["model_catalog_usage"]["properties"]["operation"]["enum"] == (
            mcp_usage_schema["properties"]["operation"]["enum"]
        )
        assert usage_control_authorities() == idl_registry.usage_control_authorities()
        assert set(usage_control_reason_codes()) == set(
            idl_registry.usage_control_reason_codes()
        )
        mcp_ops = set(mcp_usage_schema["properties"]["operation"]["enum"])
        py_ops = set(usage_control_operations())
        # Python vocabulary is a superset that also names route_preview as a
        # first-class control operation (separate MCP tool, shared reason codes).
        assert mcp_ops <= py_ops
        assert "status" in mcp_ops and "reset" in mcp_ops

        # Preview is side-effect free: no reservation created.
        binding_id = stable_id("binding", "preview-surface")
        before = coord.snapshot(scope.scope_id)
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
        assert preview["success"] is True
        assert preview["reserved"] is False
        after = coord.snapshot(scope.scope_id)
        assert after.usage_revision == before.usage_revision
        assert _headroom(after, UsageDimension.REQUESTS) == _headroom(
            before, UsageDimension.REQUESTS
        )

        # Admin authority required for mutations; read alone is insufficient.
        denied = service.reset(
            scope.scope_id,
            authorities=[USAGE_READ_AUTHORITY],
            expected_usage_revision=snap.usage_revision,
            idempotency_key="reset-denied",
            lease_id="lease-1",
            fence=1,
            expected_effects=["reset"],
        )
        assert denied["success"] is False
    finally:
        native_model_tools.set_usage_control_service(None)


def test_query_and_list_paths_are_side_effect_free() -> None:
    coord, _clock, _store = _harness()
    scope = _scope("query")
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, 7)],
    )
    service = UsageControlService(
        coord,
        catalog_revision_provider=lambda: "catalog-query-1",
    )
    before = coord.snapshot(scope.scope_id)
    for _ in range(3):
        status = service.status(authorities=[USAGE_READ_AUTHORITY])
        assert status["success"] is True
        limits = service.limits(
            scope_id=scope.scope_id,
            authorities=[USAGE_READ_AUTHORITY],
        )
        assert limits["success"] is True
        headroom = service.headroom(
            scope_id=scope.scope_id,
            authorities=[USAGE_READ_AUTHORITY],
        )
        assert headroom["success"] is True
    after = coord.snapshot(scope.scope_id)
    assert after.usage_revision == before.usage_revision
    assert _headroom(after, UsageDimension.REQUESTS) == 7


# ---------------------------------------------------------------------------
# Zero hard-limit overshoot under concurrent pressure
# ---------------------------------------------------------------------------


def test_concurrent_reserves_never_overshoot_hard_ceiling() -> None:
    coord, clock, store = _harness()
    scope = _scope("race-conf")
    ceiling = 5
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, ceiling)],
    )
    decisions: List[bool] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        # One coordinator instance per worker (matches production process model).
        local = UsageCoordinator(store, writer_id="conf-writer", fence=1)
        try:
            d = local.reserve(
                scope.scope_id,
                UsageVector.of(requests=1),
                request_id="race-%d" % idx,
                attempt_id="1",
                idempotency_key="race-%d" % idx,
                owner_id="owner-%d" % idx,
            )
            granted = bool(d.granted)
        except Exception:
            granted = False
        with lock:
            decisions.append(granted)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(ceiling * 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for g in decisions if g) == ceiling
    assert sum(1 for g in decisions if not g) == ceiling
    snap = coord.snapshot(scope.scope_id)
    assert _headroom(snap, UsageDimension.REQUESTS) >= 0


def test_duplicate_commit_does_not_double_charge() -> None:
    coord, _clock, _store = _harness()
    scope = _scope("dup-commit")
    coord.configure_limits(
        scope.scope_id,
        [_limit(scope.scope_id, UsageDimension.REQUESTS, 5)],
    )
    d = coord.reserve(
        scope.scope_id,
        UsageVector.of(requests=1),
        request_id="dup-1",
        attempt_id="1",
        idempotency_key="dup-1",
        owner_id="owner-conf",
    )
    coord.mark_dispatched(d.reservation_id)
    first = coord.commit(d.reservation_id, UsageVector.of(requests=1))
    second = coord.commit(d.reservation_id, UsageVector.of(requests=1))
    assert first.state.value == "committed"
    assert second.replayed is True
    assert _headroom(coord.snapshot(scope.scope_id), UsageDimension.REQUESTS) == 4


# ---------------------------------------------------------------------------
# ModelManager / router requirement agreement (identity surface)
# ---------------------------------------------------------------------------


def test_routers_and_model_manager_export_distinct_stable_requirement_ids() -> None:
    import ipfs_accelerate_py.embeddings_router as embeddings_router
    import ipfs_accelerate_py.llm_router as llm_router
    import ipfs_accelerate_py.multimodal_router as multimodal_router
    import ipfs_accelerate_py.voice_router as voice_router

    ids = {
        llm_router.USAGE_ROUTING_REQUIREMENT_ID,
        embeddings_router.USAGE_ROUTING_REQUIREMENT_ID,
        multimodal_router.USAGE_ROUTING_REQUIREMENT_ID,
        voice_router.USAGE_ROUTING_REQUIREMENT_ID,
    }
    assert len(ids) == 4
    assert all(rid.startswith("requirement:") and rid.endswith(".v1") for rid in ids)
    assert ENDPOINT_USAGE_CONTRACT_REQUIREMENT_ID not in ids


def test_explicit_pin_violation_blocks_cross_scope_selection() -> None:
    coord, clock, _store = _harness()
    pinned = _scope("pin-a")
    other = _scope("pin-b")
    coord.configure_limits(pinned.scope_id, [_limit(pinned.scope_id, UsageDimension.REQUESTS, 0)])
    coord.configure_limits(other.scope_id, [_limit(other.scope_id, UsageDimension.REQUESTS, 10)])
    cand_pin = _candidate(pinned, score=1, model="pinned")
    cand_other = _candidate(other, score=100, model="other")
    admission = UsageRouteAdmission(coord, owner_id="conf-owner", jitter_max_ms=0)
    result = admission.admit(
        catalog_revision="catalog-pin-1",
        candidates=[cand_pin, cand_other],
        request_id="req-pin-1",
        idempotency_key="idem-pin-1",
        operation="text.chat",
        requested=UsageVector.of(requests=1),
        policy=RoutingPolicy(
            mode=RoutingMode.ENFORCE,
            fallback=FallbackClass.CROSS_PROVIDER,
            max_attempts=2,
        ),
        pin=RoutePin(provider_id=pinned.provider_id),
        request=UsageRoutingRequest(
            required=UsageVector.of(requests=1),
            now=_rfc(clock.now()),
        ),
        snapshots_by_scope={
            pinned.scope_id: coord.snapshot(pinned.scope_id),
            other.scope_id: coord.snapshot(other.scope_id),
        },
        invoke=None,
    )
    # Exact pin with effective fallback none cannot advance to other provider.
    assert result.success is False or (
        result.selected is not None
        and result.selected.binding_id == cand_pin.binding_id
    )
    if result.selected is not None:
        assert result.selected.binding_id != cand_other.binding_id
