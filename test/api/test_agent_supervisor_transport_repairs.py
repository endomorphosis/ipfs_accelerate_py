"""DCR-044: transport, lifecycle, and browser mediation repair operators.

Acceptance:
* Actual preview middleware tests prove no browser mutation bypass.
* Health and initialize alone never establish capability availability.
* Mutation must traverse one governed mediator; raw service proxies are
  read-only allowlisted or rejected.
* Operators remain proposal-only and never grant write/proof/semantic authority.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.transport_repairs import (
    GOVERNED_MCP_MEDIATOR_INTERFACE,
    GOVERNED_MUTATION_ROUTE,
    SERVICE_PROXY_MUTATION_JSONRPC_METHODS,
    SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS,
    TRANSPORT_REPAIR_EVIDENCE,
    TRANSPORT_REPAIR_OPERATORS_INTERFACE,
    AuthoritySource,
    BrowserMediationOperator,
    BrowserMediationPolicy,
    CapabilityState,
    LifecycleBindingOperator,
    LifecyclePhase,
    MethodEffectClass,
    OperatorRole,
    ProxyDecision,
    RepairDisposition,
    RouteKind,
    TransportBindingOperator,
    TransportEndpointBinding,
    TransportProfile,
    TransportRepairError,
    TransportRepairRequest,
    assert_no_browser_mutation_bypass,
    build_capability_report,
    build_lifecycle_binding,
    build_middleware_transcript,
    build_transport_repair_operators,
    capability_claims_available,
    classify_capability_state,
    classify_jsonrpc_effect,
    classify_service_proxy_access,
    default_browser_mediation_policy,
    materialize_transport_operator_vectors,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _cid(label: str) -> str:
    return content_identity({"dcr044": label})


def _transport(
    *,
    binding_id: str = "binding:transport:demo",
    owner: str = "ipfs_accelerate_py",
    method: str = "initialize",
    effect: MethodEffectClass = MethodEffectClass.READ,
    route_kind: RouteKind = RouteKind.SERVICE_PROXY,
    path: str | None = None,
) -> TransportEndpointBinding:
    if path is None:
        path = (
            GOVERNED_MUTATION_ROUTE
            if effect is MethodEffectClass.MUTATE
            else f"/mcp/services/{owner}/mcp"
        )
    return TransportEndpointBinding(
        binding_id=binding_id,
        owner=owner,
        endpoint_identity=_cid(f"endpoint:{owner}:{method}"),
        transport_profile=TransportProfile.HTTP,
        route_kind=(
            RouteKind.GOVERNED_MEDIATOR
            if effect is MethodEffectClass.MUTATE
            else route_kind
        ),
        method=method,
        effect_class=effect,
        same_origin_path=path,
    )


# ---------------------------------------------------------------------------
# Interface / registry
# ---------------------------------------------------------------------------


def test_interfaces_and_evidence_are_declared() -> None:
    assert TRANSPORT_REPAIR_OPERATORS_INTERFACE == "TransportRepairOperators@1"
    assert GOVERNED_MCP_MEDIATOR_INTERFACE == "GovernedMcpMediator@1"
    assert TRANSPORT_REPAIR_EVIDENCE == "dcr/transport-repair@1"
    ops = build_transport_repair_operators()
    assert ops.INTERFACE == TRANSPORT_REPAIR_OPERATORS_INTERFACE
    assert ops.EVIDENCE_ID == TRANSPORT_REPAIR_EVIDENCE
    assert ops.MEDIATOR_INTERFACE == GOVERNED_MCP_MEDIATOR_INTERFACE
    assert isinstance(ops.transport_binding, TransportBindingOperator)
    assert isinstance(ops.lifecycle_binding, LifecycleBindingOperator)
    assert isinstance(ops.browser_mediation, BrowserMediationOperator)


def test_registry_binds_transport_operators_to_transport_family() -> None:
    reg = build_default_operator_registry()
    transport = reg.require_known(OperatorKind.REPAIR_TRANSPORT_ADAPTER)
    capability = reg.require_known(OperatorKind.REPAIR_CAPABILITY_TRUTH)
    assert transport.family is OperatorFamily.TRANSPORT
    assert capability.family is OperatorFamily.TRANSPORT
    assert transport.proposal_only is True
    assert capability.proposal_only is True
    assert transport.grants_write_authority is False
    assert capability.grants_write_authority is False
    assert transport.semantic_authority is False
    assert capability.allows_source_generation is False
    assert "scope:closed_transport_adapter" in transport.write_scope
    assert "scope:closed_capability_report" in capability.write_scope
    assert reg.get("transport_adapter").kind is OperatorKind.REPAIR_TRANSPORT_ADAPTER
    assert reg.get("capability_truth").kind is OperatorKind.REPAIR_CAPABILITY_TRUTH
    assert reg.get("typed_unavailable").kind is OperatorKind.REPAIR_CAPABILITY_TRUTH


# ---------------------------------------------------------------------------
# Capability truth — no availability from health/initialize alone
# ---------------------------------------------------------------------------


def test_health_and_initialize_alone_never_claim_available() -> None:
    state = classify_capability_state(
        health_ok=True, initialize_ok=True, tools_ok=False, interfaces_ok=False
    )
    assert state is CapabilityState.INITIALIZED_NOT_AVAILABLE
    assert capability_claims_available(state) is False

    report = build_capability_report(
        owner="ipfs_accelerate_py",
        health_ok=True,
        initialize_ok=True,
        tools_ok=False,
        interfaces_ok=False,
    )
    assert report.available is False
    assert report.state is CapabilityState.INITIALIZED_NOT_AVAILABLE
    assert "health_and_initialize" in report.typed_unavailable_reason

    lifecycle = build_lifecycle_binding(
        binding_id="binding:lifecycle:init-only",
        owner="ipfs_accelerate_py",
        health_ok=True,
        initialize_ok=True,
    )
    assert lifecycle.claims_available is False
    assert lifecycle.phase is LifecyclePhase.INITIALIZED
    assert lifecycle.capability_state is CapabilityState.INITIALIZED_NOT_AVAILABLE


def test_tools_or_interfaces_surface_establishes_availability() -> None:
    tools_state = classify_capability_state(
        health_ok=True, initialize_ok=True, tools_ok=True
    )
    assert tools_state is CapabilityState.AVAILABLE
    assert capability_claims_available(tools_state) is True

    interfaces_state = classify_capability_state(
        health_ok=True, initialize_ok=True, interfaces_ok=True
    )
    assert capability_claims_available(interfaces_state) is True

    report = build_capability_report(
        owner="ipfs_datasets_py",
        health_ok=True,
        initialize_ok=True,
        tools_ok=True,
    )
    assert report.available is True
    assert report.typed_unavailable_reason == ""


def test_false_availability_claim_is_rejected_on_lifecycle_binding() -> None:
    with pytest.raises(TransportRepairError, match="health/initialize alone"):
        build_lifecycle_binding(
            binding_id="binding:bad",
            owner="ipfs_kit_py",
            health_ok=True,
            initialize_ok=True,
            tools_ok=False,
        )
        # Force an illegal claims_available=True via constructor.
        from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.transport_repairs import (
            LifecycleBinding,
        )

        LifecycleBinding(
            binding_id="binding:bad",
            owner="ipfs_kit_py",
            phase=LifecyclePhase.INITIALIZED,
            capability_state=CapabilityState.INITIALIZED_NOT_AVAILABLE,
            health_ok=True,
            initialize_ok=True,
            tools_ok=False,
            claims_available=True,
        )


def test_lifecycle_operator_repairs_capability_truth_preview() -> None:
    ops = build_transport_repair_operators()
    reviewed = build_lifecycle_binding(
        binding_id="binding:lifecycle:reviewed",
        owner="ipfs_accelerate_py",
        health_ok=True,
        initialize_ok=True,
        tools_ok=False,
    )
    # Drifted current state incorrectly claimed available via tools_ok mismatch
    # is reconstructed as truthful reviewed state.
    receipt = ops.lifecycle_binding.apply(
        TransportRepairRequest(
            role=OperatorRole.LIFECYCLE_BINDING,
            reviewed_lifecycle=reviewed,
            current_lifecycle=None,
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False
    assert receipt.capability_report is not None
    assert receipt.capability_report.available is False
    assert "no_health_initialize_availability_claim" in receipt.reason_codes
    assert receipt.preview_lifecycle is not None
    assert receipt.preview_lifecycle.claims_available is False


# ---------------------------------------------------------------------------
# Service proxy classification — no browser mutation bypass
# ---------------------------------------------------------------------------


def test_read_only_proxy_methods_are_allowlisted() -> None:
    health = classify_service_proxy_access(
        http_method="GET",
        service_path="/mcp/health",
    )
    assert health["allowed"] is True
    assert health["decision"] == ProxyDecision.ALLOW_READ.value
    assert health["effect_class"] == MethodEffectClass.READ.value

    initialize = classify_service_proxy_access(
        http_method="POST",
        service_path="/mcp",
        jsonrpc_method="initialize",
    )
    assert initialize["allowed"] is True
    assert initialize["jsonrpc_method"] == "initialize"

    tools_list = classify_service_proxy_access(
        http_method="POST",
        service_path="/mcp",
        jsonrpc_method="tools/list",
    )
    assert tools_list["allowed"] is True


@pytest.mark.parametrize(
    "method",
    sorted(SERVICE_PROXY_MUTATION_JSONRPC_METHODS)[:8]
    + ["tools/call", "mcp++/execute", "mcp++/goals/create", "mcp++/schedule/claim"],
)
def test_mutations_require_governed_mediator_not_raw_proxy(method: str) -> None:
    classification = classify_service_proxy_access(
        http_method="POST",
        service_path="/mcp/services/ipfs_accelerate_py/mcp",
        jsonrpc_method=method,
    )
    assert classification["allowed"] is False
    assert classification["effect_class"] == MethodEffectClass.MUTATE.value
    assert classification["decision"] == ProxyDecision.REQUIRE_GOVERNED_MEDIATOR.value
    assert classification["governed_route"] == GOVERNED_MUTATION_ROUTE
    assert classification["mediation"] == "governed_mediator"

    with pytest.raises(TransportRepairError, match="bypass|raw service|read-only"):
        assert_no_browser_mutation_bypass(
            http_method="POST",
            service_path="/mcp",
            jsonrpc_method=method,
            mediation_path="raw_service_proxy",
        )


def test_direct_proxy_mutation_path_is_rejected() -> None:
    with pytest.raises(TransportRepairError, match="direct_proxy"):
        assert_no_browser_mutation_bypass(
            http_method="POST",
            service_path="/mcp",
            jsonrpc_method="tools/call",
            mediation_path="direct_proxy",
        )


def test_preview_middleware_transcript_proves_no_mutation_bypass() -> None:
    cases = (
        {"http_method": "GET", "service_path": "/mcp/health", "jsonrpc_method": ""},
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "initialize",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "tools/list",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "tools/call",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "mcp++/execute",
        },
        {
            "http_method": "POST",
            "service_path": "/mcp",
            "jsonrpc_method": "mcp++/goals/create",
        },
    )
    transcript = build_middleware_transcript(cases)
    assert len(transcript) == len(cases)

    by_method = {row.jsonrpc_method or row.http_method: row for row in transcript}
    assert by_method["GET"].allowed is True
    assert by_method["initialize"].allowed is True
    assert by_method["tools/list"].allowed is True
    for mutation in ("tools/call", "mcp++/execute", "mcp++/goals/create"):
        row = by_method[mutation]
        assert row.allowed is False
        assert row.effect_class is MethodEffectClass.MUTATE
        assert row.decision is ProxyDecision.REQUIRE_GOVERNED_MEDIATOR

    ops = build_transport_repair_operators()
    receipt = ops.browser_mediation.apply(
        TransportRepairRequest(
            role=OperatorRole.BROWSER_MEDIATION,
            reviewed_mediation=default_browser_mediation_policy(),
            middleware_cases=cases,
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert "no_browser_mutation_bypass" in receipt.reason_codes
    assert "middleware_transcript" in receipt.reason_codes
    assert receipt.preview_mediation is not None
    assert receipt.preview_mediation.INTERFACE == GOVERNED_MCP_MEDIATOR_INTERFACE
    assert receipt.preview_mediation.allow_raw_proxy_mutations is False
    assert receipt.preview_mediation.governed_mutation_route == GOVERNED_MUTATION_ROUTE
    assert all(
        not (row.effect_class is MethodEffectClass.MUTATE and row.allowed)
        for row in receipt.middleware_transcript
    )


# ---------------------------------------------------------------------------
# Transport binding operator
# ---------------------------------------------------------------------------


def test_transport_binding_operator_preview_inverse_idempotent() -> None:
    ops = build_transport_repair_operators()
    reviewed = _transport()
    receipt = ops.transport_binding.apply(
        TransportRepairRequest(
            role=OperatorRole.TRANSPORT_BINDING,
            reviewed_transport=reviewed,
            current_transport=None,
        )
    )
    assert receipt.disposition is RepairDisposition.PREVIEW_READY
    assert receipt.proposal_only is True
    assert receipt.grants_write_authority is False
    assert receipt.semantic_authority is False
    assert "endpoint_identity" in receipt.reason_codes
    assert "route_kind" in receipt.reason_codes
    assert "method_effect_class" in receipt.reason_codes
    assert receipt.preview_transport is not None
    assert receipt.preview_transport.same_origin_path.startswith("/mcp/services/")
    assert "://" not in receipt.preview_transport.same_origin_path

    aligned = ops.transport_binding.apply(
        TransportRepairRequest(
            role=OperatorRole.TRANSPORT_BINDING,
            reviewed_transport=reviewed,
            current_transport=reviewed,
        )
    )
    assert aligned.disposition is RepairDisposition.ALREADY_ALIGNED
    assert aligned.inverse_transport is not None
    assert ops.transport_binding.inverse(aligned) == reviewed


def test_mutate_transport_binding_requires_governed_route() -> None:
    with pytest.raises(TransportRepairError, match="governed mediator"):
        TransportEndpointBinding(
            binding_id="binding:bad-mutate",
            owner="ipfs_accelerate_py",
            endpoint_identity=_cid("bad"),
            transport_profile=TransportProfile.HTTP,
            route_kind=RouteKind.SERVICE_PROXY,
            method="tools/call",
            effect_class=MethodEffectClass.MUTATE,
            same_origin_path="/mcp/services/ipfs_accelerate_py/mcp",
        )

    ok = _transport(
        method="tools/call",
        effect=MethodEffectClass.MUTATE,
        path=GOVERNED_MUTATION_ROUTE,
    )
    assert ok.route_kind is RouteKind.GOVERNED_MEDIATOR
    assert ok.same_origin_path == GOVERNED_MUTATION_ROUTE


def test_transport_binding_rejects_absolute_backend_endpoints() -> None:
    with pytest.raises(TransportRepairError, match="absolute backend"):
        TransportEndpointBinding(
            binding_id="binding:leaky",
            owner="ipfs_kit_py",
            endpoint_identity=_cid("leaky"),
            transport_profile=TransportProfile.HTTP,
            route_kind=RouteKind.SERVICE_PROXY,
            method="initialize",
            effect_class=MethodEffectClass.READ,
            same_origin_path="http://127.0.0.1:3003/mcp",
        )


def test_invented_authority_abstains() -> None:
    ops = build_transport_repair_operators()
    receipt = ops.transport_binding.apply(
        TransportRepairRequest(
            role=OperatorRole.TRANSPORT_BINDING,
            reviewed_transport=_transport(),
            authority=AuthoritySource.INVENTED,
        )
    )
    assert receipt.disposition is RepairDisposition.ABSTAIN
    assert "transport_source_not_reviewed" in receipt.reason_codes


def test_forbidden_payload_fields_fail_closed() -> None:
    with pytest.raises(TransportRepairError, match="forbidden field"):
        TransportEndpointBinding.from_dict(
            {
                "binding_id": "binding:x",
                "owner": "ipfs_accelerate_py",
                "endpoint_identity": "endpoint:x",
                "transport_profile": "http",
                "route_kind": "service_proxy",
                "method": "initialize",
                "effect_class": "read",
                "same_origin_path": "/mcp/services/ipfs_accelerate_py/mcp",
                "source_body": "def handler(): pass",
            }
        )


# ---------------------------------------------------------------------------
# Bundle / vectors
# ---------------------------------------------------------------------------


def test_operator_bundle_dispatches_by_role() -> None:
    ops = build_transport_repair_operators()
    transport_receipt = ops.apply(
        TransportRepairRequest(
            role=OperatorRole.TRANSPORT_BINDING,
            reviewed_transport=_transport(),
        )
    )
    assert transport_receipt.role is OperatorRole.TRANSPORT_BINDING

    lifecycle_receipt = ops.apply(
        TransportRepairRequest(
            role=OperatorRole.LIFECYCLE_BINDING,
            reviewed_lifecycle=build_lifecycle_binding(
                binding_id="binding:lifecycle:bundle",
                owner="ipfs_accelerate_py",
                health_ok=True,
                initialize_ok=True,
                tools_ok=True,
            ),
        )
    )
    assert lifecycle_receipt.capability_report is not None
    assert lifecycle_receipt.capability_report.available is True

    mediation_receipt = ops.apply(
        TransportRepairRequest(
            role=OperatorRole.BROWSER_MEDIATION,
            reviewed_mediation=BrowserMediationPolicy(
                policy_id="policy:bundle-mediator"
            ),
        )
    )
    assert mediation_receipt.middleware_transcript
    assert any(
        row.jsonrpc_method == "tools/call" and not row.allowed
        for row in mediation_receipt.middleware_transcript
    )


def test_materialize_vectors_are_content_addressed() -> None:
    vectors = materialize_transport_operator_vectors()
    assert vectors["interface"] == TRANSPORT_REPAIR_OPERATORS_INTERFACE
    assert vectors["mediator_interface"] == GOVERNED_MCP_MEDIATOR_INTERFACE
    assert vectors["evidence_id"] == TRANSPORT_REPAIR_EVIDENCE
    assert vectors["governed_mutation_route"] == GOVERNED_MUTATION_ROUTE
    assert vectors["capability_truth"]["health_initialize_only_available"] is False
    assert vectors["capability_truth"]["tools_ready_available"] is True
    assert "tools/call" in vectors["mutation_jsonrpc_methods"]
    assert "initialize" in vectors["read_only_jsonrpc_methods"]
    assert set(SERVICE_PROXY_READ_ONLY_JSONRPC_METHODS).issuperset({"initialize", "tools/list"})
    assert vectors["vector_digest"].startswith("sha256:")
    # Receipts prove middleware + capability truth evidence subset.
    mediation = vectors["receipts"]["mediation"]
    assert "no_browser_mutation_bypass" in mediation["reason_codes"]
    assert mediation["middleware_transcript"]
    lifecycle = vectors["receipts"]["lifecycle"]
    assert lifecycle["capability_report"]["available"] is False


def test_classify_jsonrpc_effect_covers_closed_sets() -> None:
    assert classify_jsonrpc_effect("initialize") is MethodEffectClass.READ
    assert classify_jsonrpc_effect("tools/call") is MethodEffectClass.MUTATE
    assert classify_jsonrpc_effect("mcp++/schedule/claim") is MethodEffectClass.MUTATE
    assert classify_jsonrpc_effect("") is MethodEffectClass.UNKNOWN
