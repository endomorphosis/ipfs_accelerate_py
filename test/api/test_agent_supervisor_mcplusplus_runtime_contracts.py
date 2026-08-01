"""Hermetic MCP++ runtime contract witnesses (VFS-018 / VFS-G061).

Also covers VFS-058 objective validation repair: exact-text discovery of
``objective validation repair``, separation of domain
``vfs/mcplusplus-runtime-witness@1`` from the synthetic validation gate,
production vs mock authority, shared HTTP/mcp+p2p admission, and typed
non-authoritative failure outcomes.  Network remains disabled unless an exact
fixture and egress policy permit it.

VFS-091 / VFS-G156 additionally proves that exact domain evidence term through
a portable, non-authoritative claim over a receipt that exercises the full
acceptance matrix.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.mcplusplus_contract_resolver import (
    TransportKind,
)
from ipfs_accelerate_py.agent_supervisor.mcplusplus_runtime_witness import (
    CONTRACT_VERSION,
    DEFAULT_TIMEOUT_MS,
    EVIDENCE_RUNTIME_WITNESS,
    MCPLUSPLUS_RUNTIME_EVIDENCE_CLAIM_SCHEMA,
    MCPLUSPLUS_RUNTIME_RECEIPT_SCHEMA,
    MCPLUSPLUS_RUNTIME_WITNESS_SCHEMA,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_EVIDENCE_GOAL_ID,
    OBJECTIVE_EVIDENCE_PARENT_GOAL_ID,
    OBJECTIVE_EVIDENCE_TASK_ID,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
    RUNTIME_WITNESS_INVARIANTS,
    WITNESS_VERSION,
    AdapterSpec,
    BackendAvailability,
    CallObservation,
    CallRequest,
    CancellationToken,
    CapabilityNegotiationRecord,
    CleanupStatus,
    HermeticMCPlusPlusRuntime,
    ImplementationKind,
    RuntimeWitness,
    RuntimeWitnessAuthorityError,
    RuntimeWitnessError,
    RuntimeWitnessReceipt,
    ToolDiscoveryRecord,
    ValidationVerdict,
    WitnessOutcome,
    WitnessPhase,
    admitted_shared_transport_profiles,
    all_covered_evidence_terms,
    covered_evidence_terms,
    default_mock_adapters,
    default_production_adapters,
    make_call_request,
    make_runtime,
    mcplusplus_runtime_witness_evidence,
    objective_validation_repair_evidence_terms,
    production_dispatch_distinguished_from_mocks,
    prove_mcplusplus_runtime_witness,
    prove_mcplusplus_runtime_witness_evidence,
    receipt_content_identity,
    replay_receipt,
    run_witness_subprocess,
    runtime_receipt_satisfies_mcplusplus_witness,
    runtime_witness_acceptance_report,
    runtime_witness_evidence_terms,
    typed_non_authoritative_failure_outcomes,
    validate_against_schema,
)
from ipfs_accelerate_py.agent_supervisor.program_assurance_contracts import (
    ClaimLevel,
)


FOREST = "forest:test-vfs-018"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime() -> HermeticMCPlusPlusRuntime:
    return make_runtime(forest_id=FOREST)


def _phases_include(witness: RuntimeWitness, *phases: WitnessPhase) -> None:
    completed = set(witness.observation.phases_completed)
    for phase in phases:
        assert phase.value in completed, (
            f"missing phase {phase.value} in {sorted(completed)}"
        )


# ---------------------------------------------------------------------------
# Schema / identity helpers
# ---------------------------------------------------------------------------


def test_evidence_kind_and_version_constants() -> None:
    assert EVIDENCE_RUNTIME_WITNESS == "vfs/mcplusplus-runtime-witness@1"
    assert WITNESS_VERSION == "mcplusplus-runtime-witness@1"
    assert CONTRACT_VERSION == 1
    assert DEFAULT_TIMEOUT_MS == 5_000


def test_objective_validation_repair_evidence_term_discoverable() -> None:
    """VFS-G061 objective validation repair: exact-text discovery key present.

    Anchors the synthetic phrase ``objective validation repair`` so objective
    scans re-find the validation gate.  Domain evidence stays separate
    (``vfs/mcplusplus-runtime-witness@1``).  The repair term never enters
    witness/receipt identity, production authority, formal-proof claims, or
    static-completeness claims.  Owned by VFS-G061 via repair task VFS-058.
    """

    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
    assert OBJECTIVE_GOAL_ID == "VFS-G061"
    assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-058"
    assert objective_validation_repair_evidence_terms() == (
        "objective validation repair",
    )
    # Domain envelope evidence remains runtime-witness only.
    assert runtime_witness_evidence_terms() == (
        "vfs/mcplusplus-runtime-witness@1",
    )
    assert "objective validation repair" not in runtime_witness_evidence_terms()
    assert covered_evidence_terms() == ("vfs/mcplusplus-runtime-witness@1",)
    assert "objective validation repair" not in covered_evidence_terms()
    # Full discovery set includes the validation-gate meta term last.
    assert all_covered_evidence_terms() == (
        "vfs/mcplusplus-runtime-witness@1",
        "objective validation repair",
    )
    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE in all_covered_evidence_terms()

    runtime = make_runtime(forest_id=FOREST)
    witness = runtime.call(make_call_request("echo", {"message": "repair"}))
    payload = witness.to_dict()
    # Witness identity envelope never absorbs the synthetic repair term.
    assert payload["evidence_kind"] == EVIDENCE_RUNTIME_WITNESS
    assert "objective validation repair" not in payload["evidence_kind"]
    assert payload.get("evidence_objective_validation_repair") is None
    assert witness.static_completeness_claimed is False
    assert witness.formal_proof_claimed is False
    assert witness.network_enabled is False

    receipt = runtime.run_suite(
        (
            make_call_request("echo", {"message": "a"}),
            make_call_request("mock.always_ok", {}),
        )
    )
    receipt_body = receipt.to_dict()
    assert receipt_body["evidence_kind"] == EVIDENCE_RUNTIME_WITNESS
    assert "objective validation repair" not in receipt_body["evidence_kind"]
    assert receipt.replaces_static_completeness is False
    assert receipt.replaces_formal_proof is False
    assert receipt.supplements_static_resolution is True
    assert receipt.supplements_formal_proof is True


def test_vfs_g061_acceptance_production_transports_and_non_authority(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    """VFS-G061 acceptance: production vs mock, shared transports, typed failures.

    Proves the full acceptance subset for objective validation repair on the
    hermetic MCP++ runtime surface, including the network-disabled refinement.
    """

    assert OBJECTIVE_GOAL_ID == "VFS-G061"
    assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-058"
    assert "objective validation repair" in all_covered_evidence_terms()
    assert production_dispatch_distinguished_from_mocks() is True
    assert admitted_shared_transport_profiles() == (
        TransportKind.HTTP.value,
        TransportKind.MCP_P2P.value,
    )

    # Real adapter dispatch is distinguished from mocks.
    production = runtime.call(make_call_request("echo", {"message": "prod"}))
    mock = runtime.call(make_call_request("mock.always_ok", {"x": 1}))
    assert production.observation.implementation_kind is ImplementationKind.PRODUCTION
    assert production.observation.grants_runtime_authority is True
    assert production.is_production_witness is True
    assert mock.observation.implementation_kind is ImplementationKind.MOCK
    assert mock.observation.grants_runtime_authority is False
    assert mock.is_production_witness is False
    assert production.evidence_kind == EVIDENCE_RUNTIME_WITNESS
    assert mock.evidence_kind == EVIDENCE_RUNTIME_WITNESS
    assert "objective validation repair" not in production.evidence_kind

    # HTTP and mcp+p2p use the same admitted contract where declared.
    w_http = runtime.call(
        make_call_request(
            "vfs.stat",
            {"path": "/docs/readme.md"},
            transport=TransportKind.HTTP.value,
        )
    )
    w_p2p = runtime.call(
        make_call_request(
            "vfs.stat",
            {"path": "/docs/readme.md"},
            transport=TransportKind.MCP_P2P.value,
            requested_profiles=("mcp++/basic", "mcp++/p2p-transport"),
        )
    )
    assert w_http.observation.outcome is WitnessOutcome.PASSED
    assert w_p2p.observation.outcome is WitnessOutcome.PASSED
    assert w_http.observation.adapter_id == w_p2p.observation.adapter_id
    assert w_http.transport == TransportKind.HTTP.value
    assert w_p2p.transport == TransportKind.MCP_P2P.value

    # Failures and unavailable services are typed, bounded, non-authoritative.
    failures = typed_non_authoritative_failure_outcomes()
    for required in (
        "malformed_call",
        "missing_tool",
        "schema_violation",
        "unavailable_backend",
        "cancelled",
        "profile_mismatch",
        "stale_manifest",
        "timed_out",
        "transport_rejected",
        "dispatch_error",
        "inconclusive",
    ):
        assert required in failures
    assert "passed" not in failures

    matrix = runtime.run_suite(
        [
            CallRequest(tool_name=""),
            make_call_request("no.such.tool", {}),
            make_call_request("echo", {}),
            make_call_request("unavailable.probe", {}),
            make_call_request("echo", {"message": "c"}, cancel=True),
            make_call_request(
                "echo",
                {"message": "p"},
                requested_profiles=("mcp++/risk-scheduling",),
            ),
            make_call_request(
                "echo",
                {"message": "s"},
                expected_manifest_cid="baguqeerastale",
            ),
            make_call_request("echo", {"message": "t"}, force_timeout=True),
        ]
    )
    for witness in matrix.witnesses:
        outcome = witness.observation.outcome
        assert outcome is not WitnessOutcome.PASSED
        assert outcome.value in failures
        assert witness.observation.grants_runtime_authority is False
        assert outcome.is_authoritative is False
        assert witness.static_completeness_claimed is False
        assert witness.formal_proof_claimed is False
        assert "objective validation repair" not in witness.evidence_kind

    # Network remains disabled unless exact fixture + egress policy permit it.
    assert runtime.network_enabled is False
    with pytest.raises(RuntimeWitnessError, match="network"):
        HermeticMCPlusPlusRuntime(
            forest_id=FOREST,
            network_enabled=True,
            adapters=default_production_adapters(),
        )


def _vfs_g156_acceptance_receipt() -> RuntimeWitnessReceipt:
    runtime = make_runtime(forest_id=FOREST)
    return runtime.run_suite(
        (
            make_call_request(
                "echo",
                {"message": "http"},
                transport=TransportKind.HTTP.value,
            ),
            make_call_request(
                "echo",
                {"message": "p2p"},
                transport=TransportKind.MCP_P2P.value,
            ),
            make_call_request("mock.always_ok", {}),
            make_call_request("unavailable.probe", {}),
            make_call_request(
                "echo",
                {"message": "bounded-timeout"},
                force_timeout=True,
            ),
            make_call_request("echo", {}),
        )
    )


def test_vfs_g156_evidence_surface_binds_objective_leaf() -> None:
    """The exact missing evidence term is owned by VFS-G156 / VFS-091."""

    assert mcplusplus_runtime_witness_evidence() == (
        "vfs/mcplusplus-runtime-witness@1"
    )
    assert runtime_witness_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert covered_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == (
        "vfs/mcplusplus-runtime-witness@1",
    )
    assert OBJECTIVE_EVIDENCE_GOAL_ID == "VFS-G156"
    assert OBJECTIVE_EVIDENCE_PARENT_GOAL_ID == OBJECTIVE_GOAL_ID == "VFS-G061"
    assert OBJECTIVE_EVIDENCE_TASK_ID == "VFS-091"
    assert len(RUNTIME_WITNESS_INVARIANTS) == 6


def test_prove_vfs_g156_runtime_witness_acceptance_matrix() -> None:
    """A real receipt proves every clause without completion authority."""

    receipt = _vfs_g156_acceptance_receipt()
    assert runtime_receipt_satisfies_mcplusplus_witness(receipt)
    assert runtime_receipt_satisfies_mcplusplus_witness(receipt.to_dict())

    report = runtime_witness_acceptance_report(receipt)
    assert report["satisfied"] is True
    assert report["failure_codes"] == []
    assert report["checks"] == {
        "receipt_integrity": True,
        "production_mock_separation": True,
        "shared_transport_contract": True,
        "typed_bounded_failures": True,
        "hermetic_network_policy": True,
        "supplemental_authority": True,
    }
    assert report["outcome_counts"] == {
        "passed": 3,
        "schema_violation": 1,
        "timed_out": 1,
        "unavailable_backend": 1,
    }
    assert report["shared_contract_bindings"][0]["tool_name"] == "echo"
    assert report["shared_contract_bindings"][0]["transports"] == [
        TransportKind.HTTP.value,
        TransportKind.MCP_P2P.value,
    ]

    claim = prove_mcplusplus_runtime_witness(receipt)
    assert claim["schema"] == MCPLUSPLUS_RUNTIME_EVIDENCE_CLAIM_SCHEMA
    assert claim["evidence"] == EVIDENCE_RUNTIME_WITNESS
    assert claim["evidence_terms"] == ["vfs/mcplusplus-runtime-witness@1"]
    assert claim["all_evidence_terms"] == list(OBJECTIVE_DOMAIN_EVIDENCE_TERMS)
    assert claim["requirement_id"] == EVIDENCE_RUNTIME_WITNESS
    assert claim["goal_id"] == "VFS-G156"
    assert claim["parent_goal_id"] == "VFS-G061"
    assert claim["task_id"] == "VFS-091"
    assert claim["receipt_id"] == receipt.receipt_id
    assert claim["fixture_id"] == receipt.fixture_id
    assert claim["forest_id"] == receipt.forest_id
    assert claim["manifest_cid"] == receipt.manifest_cid
    assert claim["invariants"] == list(RUNTIME_WITNESS_INVARIANTS)
    assert claim["satisfied"] is True
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False
    assert claim["promotion_authoritative"] is False
    assert claim["semantic_authority"] is False

    alias = prove_mcplusplus_runtime_witness_evidence(receipt.to_dict())
    assert alias == claim


@pytest.mark.parametrize(
    ("requests", "failed_check"),
    (
        (
            (
                make_call_request("echo", {"message": "http"}),
                make_call_request("mock.always_ok", {}),
                make_call_request("unavailable.probe", {}),
                make_call_request(
                    "echo", {"message": "timeout"}, force_timeout=True
                ),
            ),
            "shared_transport_contract",
        ),
        (
            (
                make_call_request("echo", {"message": "http"}),
                make_call_request(
                    "echo",
                    {"message": "p2p"},
                    transport=TransportKind.MCP_P2P.value,
                ),
                make_call_request("unavailable.probe", {}),
                make_call_request(
                    "echo", {"message": "timeout"}, force_timeout=True
                ),
            ),
            "production_mock_separation",
        ),
        (
            (
                make_call_request("echo", {"message": "http"}),
                make_call_request(
                    "echo",
                    {"message": "p2p"},
                    transport=TransportKind.MCP_P2P.value,
                ),
                make_call_request("mock.always_ok", {}),
            ),
            "typed_bounded_failures",
        ),
    ),
)
def test_vfs_g156_claim_fails_closed_on_incomplete_matrix(
    requests: tuple[CallRequest, ...],
    failed_check: str,
) -> None:
    receipt = make_runtime(forest_id=FOREST).run_suite(requests)
    claim = prove_mcplusplus_runtime_witness(receipt)
    assert claim["satisfied"] is False
    assert claim["checks"][failed_check] is False
    assert failed_check.replace("_", "-") in claim["failure_codes"]
    assert claim["completion_authoritative"] is False


def test_vfs_g156_claim_rejects_invalid_or_networked_receipts() -> None:
    invalid = prove_mcplusplus_runtime_witness({"not": "a receipt"})
    assert invalid["satisfied"] is False
    assert invalid["receipt_id"] == ""
    assert invalid["failure_codes"] == ["invalid-runtime-witness-receipt"]
    assert invalid["completion_authoritative"] is False

    body = _vfs_g156_acceptance_receipt().to_dict()
    body["network_enabled"] = True
    networked = prove_mcplusplus_runtime_witness(body)
    assert networked["satisfied"] is False
    assert networked["checks"]["hermetic_network_policy"] is False
    assert "hermetic-network-policy" in networked["failure_codes"]
    assert networked["authoritative"] is False


def test_default_adapters_distinguish_production_and_mock() -> None:
    production = default_production_adapters()
    mocks = default_mock_adapters()
    assert production
    assert mocks
    assert all(a.is_production for a in production)
    assert all(a.is_mock for a in mocks)
    assert all(
        not a.implementation_kind.grants_production_authority for a in mocks
    )
    names = {a.tool_name for a in production}
    assert "echo" in names
    assert "vfs.stat" in names
    assert "unavailable.probe" in names


def test_adapter_spec_round_trip_and_fingerprints() -> None:
    spec = default_production_adapters()[0]
    restored = AdapterSpec.from_dict(spec.to_dict())
    assert restored.content_id == spec.content_id
    assert restored.input_schema_fingerprint == spec.input_schema_fingerprint
    assert restored.tool_name == "echo"


def test_validate_against_schema_subset() -> None:
    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }
    ok, errors = validate_against_schema({"message": "hi"}, schema)
    assert ok is ValidationVerdict.VALID
    assert errors == ()
    bad, errors = validate_against_schema({}, schema)
    assert bad is ValidationVerdict.INVALID
    assert errors
    skipped, _ = validate_against_schema({"x": 1}, None)
    assert skipped is ValidationVerdict.SKIPPED


# ---------------------------------------------------------------------------
# Discovery, negotiation, transport
# ---------------------------------------------------------------------------


def test_tool_discovery_records_production_and_mock(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    discovery = runtime.discover_tools()
    assert isinstance(discovery, ToolDiscoveryRecord)
    assert "echo" in discovery.tool_names
    assert "echo" in discovery.production_tools
    assert "mock.always_ok" in discovery.mock_tools
    assert "mock.always_ok" not in discovery.production_tools
    assert discovery.manifest_cid
    assert discovery.server_name == "ipfs-accelerate-mcp++"
    assert len(discovery.adapter_ids) == len(discovery.tool_names)


def test_capability_negotiation_success_and_profile_mismatch(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    ok = runtime.negotiate(
        requested_profiles=("mcp++/basic", "mcp++/mcp-idl"),
        transport=TransportKind.HTTP,
    )
    assert ok.negotiated is True
    assert ok.active_profile == "mcp++/basic"
    assert ok.active_transport == TransportKind.HTTP.value

    bad = runtime.negotiate(
        requested_profiles=("mcp++/ucan",),
        transport=TransportKind.HTTP,
    )
    assert bad.negotiated is False
    assert bad.reason == "profile_mismatch"


def test_transport_http_and_mcp_p2p_admitted(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    http = runtime.negotiate(
        requested_profiles=("mcp++/basic",),
        transport=TransportKind.HTTP,
    )
    p2p = runtime.negotiate(
        requested_profiles=("mcp++/basic",),
        transport=TransportKind.MCP_P2P,
    )
    assert http.negotiated and p2p.negotiated
    assert http.active_transport == TransportKind.HTTP.value
    assert p2p.active_transport == TransportKind.MCP_P2P.value

    # vfs.stat admits both transports under the same contract surface.
    w_http = runtime.call(
        make_call_request(
            "vfs.stat",
            {"path": "/docs/readme.md"},
            transport=TransportKind.HTTP.value,
        )
    )
    w_p2p = runtime.call(
        make_call_request(
            "vfs.stat",
            {"path": "/docs/readme.md"},
            transport=TransportKind.MCP_P2P.value,
            requested_profiles=("mcp++/basic", "mcp++/p2p-transport"),
        )
    )
    assert w_http.observation.outcome is WitnessOutcome.PASSED
    assert w_p2p.observation.outcome is WitnessOutcome.PASSED
    assert w_http.transport == TransportKind.HTTP.value
    assert w_p2p.transport == TransportKind.MCP_P2P.value
    assert w_http.observation.adapter_id == w_p2p.observation.adapter_id


def test_network_disabled_by_default() -> None:
    rt = make_runtime(forest_id=FOREST)
    assert rt.network_enabled is False
    with pytest.raises(RuntimeWitnessError, match="network"):
        HermeticMCPlusPlusRuntime(
            forest_id=FOREST,
            network_enabled=True,
            adapters=default_production_adapters(),
        )


# ---------------------------------------------------------------------------
# Happy-path production witnesses
# ---------------------------------------------------------------------------


def test_production_echo_witness_records_full_phases(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    witness = runtime.call(make_call_request("echo", {"message": "hello"}))
    obs = witness.observation
    assert obs.outcome is WitnessOutcome.PASSED
    assert obs.grants_runtime_authority is True
    assert obs.implementation_kind is ImplementationKind.PRODUCTION
    assert obs.adapter_id == "adapter:echo:production"
    assert obs.implementation_target.endswith("_handler_echo")
    assert obs.input_validation is ValidationVerdict.VALID
    assert obs.output_validation is ValidationVerdict.VALID
    assert obs.result == {"echo": "hello", "length": 5}
    assert obs.claim_level == ClaimLevel.RUNTIME_WITNESSED.value
    assert witness.evidence_kind == EVIDENCE_RUNTIME_WITNESS
    assert witness.static_completeness_claimed is False
    assert witness.formal_proof_claimed is False
    assert witness.network_enabled is False
    _phases_include(
        witness,
        WitnessPhase.DISCOVERY,
        WitnessPhase.CAPABILITY_NEGOTIATION,
        WitnessPhase.INPUT_VALIDATION,
        WitnessPhase.DISPATCH,
        WitnessPhase.OUTPUT_SCHEMA,
        WitnessPhase.TRANSPORT,
        WitnessPhase.TIMEOUT,
        WitnessPhase.CLEANUP,
    )


def test_production_identity_and_vfs_stat(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    identity = runtime.call(make_call_request("identity", {"name": "swiss"}))
    assert identity.observation.outcome is WitnessOutcome.PASSED
    assert identity.observation.result["name"] == "swiss"
    assert identity.observation.result["identity"].startswith("b")

    stat = runtime.call(
        make_call_request("vfs.stat", {"path": "ipfs_kit_py/vfs.py"})
    )
    assert stat.observation.outcome is WitnessOutcome.PASSED
    assert stat.observation.result["exists"] is True
    assert stat.observation.grants_runtime_authority is True


def test_mock_success_does_not_grant_runtime_authority(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    witness = runtime.call(make_call_request("mock.always_ok", {"x": 1}))
    assert witness.observation.outcome is WitnessOutcome.PASSED
    assert witness.observation.implementation_kind is ImplementationKind.MOCK
    assert witness.observation.grants_runtime_authority is False
    assert witness.is_production_witness is False


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_malformed_call_empty_tool_name(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    request = CallRequest(tool_name="")
    witness = runtime.call(request)
    assert witness.observation.outcome is WitnessOutcome.MALFORMED_CALL
    assert witness.observation.grants_runtime_authority is False
    _phases_include(
        witness,
        WitnessPhase.DISCOVERY,
        WitnessPhase.CAPABILITY_NEGOTIATION,
    )


def test_missing_tool(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(make_call_request("definitely.missing", {}))
    assert witness.observation.outcome is WitnessOutcome.MISSING_TOOL
    assert "not registered" in witness.observation.reason


def test_wrong_input_schema(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(make_call_request("echo", {"not_message": 1}))
    assert witness.observation.outcome is WitnessOutcome.SCHEMA_VIOLATION
    assert witness.observation.input_validation is ValidationVerdict.INVALID
    assert witness.observation.input_errors
    assert witness.observation.grants_runtime_authority is False


def test_unavailable_backend(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(make_call_request("unavailable.probe", {}))
    assert witness.observation.outcome is WitnessOutcome.UNAVAILABLE_BACKEND
    assert witness.observation.error_code == "unavailable"
    assert witness.observation.error_schema_ok is True


def test_cancellation_via_request_flag(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    witness = runtime.call(
        make_call_request("echo", {"message": "x"}, cancel=True)
    )
    assert witness.observation.outcome is WitnessOutcome.CANCELLED
    assert witness.observation.cancelled is True


def test_cancellation_via_token(runtime: HermeticMCPlusPlusRuntime) -> None:
    token = CancellationToken(cancellation_id="cancel:vfs-018")
    token.cancel(reason="operator abort")
    witness = runtime.call(
        make_call_request("echo", {"message": "x"}),
        cancellation=token,
    )
    assert witness.observation.outcome is WitnessOutcome.CANCELLED
    assert "operator abort" in witness.observation.reason


def test_profile_mismatch(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(
        make_call_request(
            "echo",
            {"message": "x"},
            requested_profiles=("mcp++/ucan",),
        )
    )
    assert witness.observation.outcome is WitnessOutcome.PROFILE_MISMATCH
    assert witness.negotiation.negotiated is False


def test_adapter_profile_mismatch() -> None:
    """Requested profile is admitted by runtime but not by the adapter."""

    rt = HermeticMCPlusPlusRuntime(
        forest_id=FOREST,
        adapters=(
            AdapterSpec(
                tool_name="echo",
                adapter_id="adapter:echo:basic-only",
                implementation_kind=ImplementationKind.PRODUCTION,
                implementation_target=(
                    "ipfs_accelerate_py.agent_supervisor."
                    "mcplusplus_runtime_witness._handler_echo"
                ),
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "echo": {"type": "string"},
                        "length": {"type": "integer"},
                    },
                    "required": ["echo", "length"],
                },
                profiles=("mcp++/basic",),
                transports=(TransportKind.HTTP.value,),
            ),
        ),
        admitted_profiles=("mcp++/basic", "mcp++/mcp-idl"),
    )
    witness = rt.call(
        make_call_request(
            "echo",
            {"message": "x"},
            requested_profiles=("mcp++/mcp-idl",),
        )
    )
    assert witness.observation.outcome is WitnessOutcome.PROFILE_MISMATCH
    assert "adapter" in witness.observation.reason


def test_stale_manifest(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(
        make_call_request(
            "echo",
            {"message": "x"},
            expected_manifest_cid="baguqeerastalemanifestcid0000000000000000000000000000000000",
        )
    )
    assert witness.observation.outcome is WitnessOutcome.STALE_MANIFEST
    assert "manifest" in witness.observation.reason


def test_force_timeout(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(
        make_call_request(
            "echo",
            {"message": "x"},
            force_timeout=True,
        )
    )
    assert witness.observation.outcome is WitnessOutcome.TIMED_OUT
    assert witness.observation.timed_out is True
    _phases_include(witness, WitnessPhase.TIMEOUT)


def test_dispatch_error_typed_error_code(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    witness = runtime.call(make_call_request("identity", {"name": ""}))
    assert witness.observation.outcome is WitnessOutcome.DISPATCH_ERROR
    assert witness.observation.error_code == "invalid_name"
    assert witness.observation.error_schema_ok is True
    _phases_include(witness, WitnessPhase.ERROR_SCHEMA)


def test_transport_rejected_when_not_admitted() -> None:
    rt = HermeticMCPlusPlusRuntime(
        forest_id=FOREST,
        adapters=default_production_adapters()[:1],
        admitted_transports=(TransportKind.HTTP.value,),
    )
    witness = rt.call(
        make_call_request(
            "echo",
            {"message": "x"},
            transport=TransportKind.MCP_P2P.value,
        )
    )
    assert witness.observation.outcome is WitnessOutcome.TRANSPORT_REJECTED


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_is_clean_after_suite(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    runtime.call(make_call_request("echo", {"message": "x"}))
    status = runtime.cleanup()
    assert status is CleanupStatus.CLEAN
    assert runtime.is_cleaned_up is True
    # Post-cleanup calls are non-authoritative.
    later = runtime.call(make_call_request("echo", {"message": "y"}))
    assert later.observation.outcome is WitnessOutcome.INCONCLUSIVE


# ---------------------------------------------------------------------------
# Receipts and deterministic replay
# ---------------------------------------------------------------------------


def test_run_suite_receipt_and_deterministic_replay(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    requests = (
        make_call_request("echo", {"message": "a"}),
        make_call_request("vfs.stat", {"path": "/tmp/x"}),
        make_call_request("missing.tool", {}),
        make_call_request("mock.always_ok", {}),
    )
    receipt = runtime.run_suite(requests)
    assert isinstance(receipt, RuntimeWitnessReceipt)
    assert receipt.schema == MCPLUSPLUS_RUNTIME_RECEIPT_SCHEMA
    assert receipt.evidence_kind == EVIDENCE_RUNTIME_WITNESS
    assert len(receipt.witnesses) == 4
    assert receipt.production_witness_count == 2
    assert receipt.mock_witness_count == 1
    assert receipt.network_enabled is False
    assert receipt.supplements_static_resolution is True
    assert receipt.supplements_formal_proof is True
    assert receipt.replaces_static_completeness is False
    assert receipt.replaces_formal_proof is False

    replayed = replay_receipt(receipt)
    assert replayed.content_id == receipt.content_id
    assert replayed.to_json() == receipt.to_json()
    assert receipt_content_identity(receipt.to_dict()) == receipt.content_id

    # Second independent suite with same inputs yields identical receipt id
    # only when fixture identity is fixed.
    rt2 = HermeticMCPlusPlusRuntime(
        forest_id=FOREST,
        adapters=list(default_production_adapters())
        + list(default_mock_adapters()),
        fixture_id=runtime.fixture_id,
        manifest_cid=runtime.manifest_cid,
        timeout_ms=runtime.timeout_ms,
    )
    receipt2 = rt2.run_suite(requests)
    assert receipt2.content_id == receipt.content_id


def test_receipt_rejects_replacing_static_or_proof() -> None:
    base = RuntimeWitnessReceipt(
        fixture_id="fix:1",
        forest_id=FOREST,
        manifest_cid="baguqeeramanifest",
        witnesses=(),
    )
    with pytest.raises(RuntimeWitnessAuthorityError):
        RuntimeWitnessReceipt(
            fixture_id="fix:1",
            forest_id=FOREST,
            manifest_cid="baguqeeramanifest",
            witnesses=(),
            replaces_static_completeness=True,
        )
    with pytest.raises(RuntimeWitnessAuthorityError):
        RuntimeWitnessReceipt(
            fixture_id="fix:1",
            forest_id=FOREST,
            manifest_cid="baguqeeramanifest",
            witnesses=(),
            replaces_formal_proof=True,
        )
    with pytest.raises(RuntimeWitnessAuthorityError):
        RuntimeWitnessReceipt(
            fixture_id="fix:1",
            forest_id=FOREST,
            manifest_cid="baguqeeramanifest",
            witnesses=(),
            supplements_static_resolution=False,
        )
    assert base.receipt_id


def test_observation_rejects_mock_authority_grant() -> None:
    with pytest.raises(RuntimeWitnessAuthorityError):
        CallObservation(
            outcome=WitnessOutcome.PASSED,
            tool_name="mock.always_ok",
            implementation_kind=ImplementationKind.MOCK,
            grants_runtime_authority=True,
        )


def test_witness_rejects_static_completeness_claim(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    good = runtime.call(make_call_request("echo", {"message": "z"}))
    payload = good.to_dict()
    payload["static_completeness_claimed"] = True
    with pytest.raises(RuntimeWitnessAuthorityError):
        RuntimeWitness.from_dict(payload)


def test_witness_round_trip(runtime: HermeticMCPlusPlusRuntime) -> None:
    witness = runtime.call(make_call_request("echo", {"message": "rt"}))
    assert witness.schema == MCPLUSPLUS_RUNTIME_WITNESS_SCHEMA
    restored = RuntimeWitness.from_dict(witness.to_dict())
    assert restored.content_id == witness.content_id
    assert restored.witness_id == witness.witness_id


# ---------------------------------------------------------------------------
# Bounded subprocess fixture
# ---------------------------------------------------------------------------


def test_subprocess_runtime_fixture_records_discovery_and_outcomes() -> None:
    receipt = run_witness_subprocess(
        [
            make_call_request("echo", {"message": "subprocess"}),
            make_call_request("missing.tool", {}),
            make_call_request(
                "echo",
                {"message": "x"},
                requested_profiles=("mcp++/ucan",),
            ),
            make_call_request("unavailable.probe", {}),
            make_call_request("mock.always_ok", {}),
        ],
        forest_id="forest:subprocess-vfs-018",
        subprocess_timeout_s=60.0,
    )
    assert len(receipt.witnesses) == 5
    outcomes = [w.observation.outcome for w in receipt.witnesses]
    assert outcomes[0] is WitnessOutcome.PASSED
    assert outcomes[1] is WitnessOutcome.MISSING_TOOL
    assert outcomes[2] is WitnessOutcome.PROFILE_MISMATCH
    assert outcomes[3] is WitnessOutcome.UNAVAILABLE_BACKEND
    assert outcomes[4] is WitnessOutcome.PASSED
    assert receipt.witnesses[0].observation.grants_runtime_authority is True
    assert receipt.witnesses[4].observation.grants_runtime_authority is False
    assert receipt.network_enabled is False

    # Discovery identity is present on every witness.
    for witness in receipt.witnesses:
        assert "echo" in witness.discovery.production_tools
        assert witness.discovery.manifest_cid == receipt.manifest_cid
        assert WitnessPhase.DISCOVERY.value in (
            witness.observation.phases_completed
        )

    # Deterministic replay of the subprocess receipt.
    assert replay_receipt(receipt).content_id == receipt.content_id


def test_subprocess_rejects_empty_requests() -> None:
    with pytest.raises(RuntimeWitnessError, match="at least one"):
        run_witness_subprocess([])


# ---------------------------------------------------------------------------
# Dispatch target identity
# ---------------------------------------------------------------------------


def test_dispatch_target_identity_is_recorded(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    witness = runtime.call(make_call_request("vfs.stat", {"path": "/a"}))
    obs = witness.observation
    assert obs.adapter_id == "adapter:vfs.stat:production"
    assert "mcplusplus_runtime_witness._handler_vfs_stat" in (
        obs.implementation_target
    )
    assert obs.implementation_kind is ImplementationKind.PRODUCTION


# ---------------------------------------------------------------------------
# Records: negotiation / discovery standalone
# ---------------------------------------------------------------------------


def test_negotiation_and_discovery_round_trip() -> None:
    discovery = ToolDiscoveryRecord(
        tool_names=("echo",),
        adapter_ids=("adapter:echo:production",),
        production_tools=("echo",),
        mock_tools=(),
        fixture_tools=(),
        manifest_cid="baguqeeram1",
    )
    assert ToolDiscoveryRecord.from_dict(discovery.to_dict()).content_id == (
        discovery.content_id
    )
    negotiation = CapabilityNegotiationRecord(
        requested_profiles=("mcp++/basic",),
        admitted_profiles=("mcp++/basic",),
        active_profile="mcp++/basic",
        requested_transport=TransportKind.HTTP.value,
        admitted_transports=(TransportKind.HTTP.value,),
        active_transport=TransportKind.HTTP.value,
        negotiated=True,
        reason="negotiated",
    )
    assert CapabilityNegotiationRecord.from_dict(
        negotiation.to_dict()
    ).content_id == negotiation.content_id


def test_call_request_normalizes_tool_name() -> None:
    request = make_call_request("VFS.Stat", {"path": "/x"})
    assert request.tool_name == "vfs.stat"


def test_cancellation_token_requires_identity() -> None:
    with pytest.raises(RuntimeWitnessError, match="cancellation identity"):
        CancellationToken(cancellation_id="")


def test_unavailable_adapter_spec_flag() -> None:
    specs = {
        s.tool_name: s for s in default_production_adapters()
    }
    assert (
        specs["unavailable.probe"].backend_availability
        is BackendAvailability.UNAVAILABLE
    )


def test_runtime_fixture_identity_stable_for_same_registry() -> None:
    a = make_runtime(forest_id=FOREST)
    b = make_runtime(forest_id=FOREST)
    assert a.fixture_id == b.fixture_id
    assert a.manifest_cid == b.manifest_cid


def test_suite_covers_acceptance_failure_matrix(
    runtime: HermeticMCPlusPlusRuntime,
) -> None:
    """Single suite exercising the acceptance failure matrix compactly."""

    receipt = runtime.run_suite(
        [
            make_call_request("echo", {"message": "ok"}),  # production pass
            make_call_request("mock.always_ok", {}),  # mock pass
            CallRequest(tool_name=""),  # malformed
            make_call_request("no.such.tool", {}),  # missing
            make_call_request("echo", {}),  # wrong schema
            make_call_request("unavailable.probe", {}),  # unavailable
            make_call_request("echo", {"message": "c"}, cancel=True),
            make_call_request(
                "echo",
                {"message": "p"},
                requested_profiles=("mcp++/risk-scheduling",),
            ),
            make_call_request(
                "echo",
                {"message": "s"},
                expected_manifest_cid="baguqeerastale",
            ),
            make_call_request(
                "echo", {"message": "t"}, force_timeout=True
            ),
        ]
    )
    expected = [
        WitnessOutcome.PASSED,
        WitnessOutcome.PASSED,
        WitnessOutcome.MALFORMED_CALL,
        WitnessOutcome.MISSING_TOOL,
        WitnessOutcome.SCHEMA_VIOLATION,
        WitnessOutcome.UNAVAILABLE_BACKEND,
        WitnessOutcome.CANCELLED,
        WitnessOutcome.PROFILE_MISMATCH,
        WitnessOutcome.STALE_MANIFEST,
        WitnessOutcome.TIMED_OUT,
    ]
    assert [w.observation.outcome for w in receipt.witnesses] == expected
    # Only the first production pass grants authority.
    grants = [w.observation.grants_runtime_authority for w in receipt.witnesses]
    assert grants == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    # Receipt still only supplements static/formal layers.
    assert receipt.supplements_static_resolution
    assert not receipt.replaces_static_completeness
    assert not receipt.replaces_formal_proof
    assert replay_receipt(receipt).content_id == receipt.content_id
