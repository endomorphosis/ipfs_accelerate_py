"""SCA-051 / SCA-629 tests for MCP++ contract parity (SCAEV051PARITY).

Proves objective evidence SCAEV051PARITY for SCA-G051: schema, argument,
result, policy, transport, discovery/execution, compatibility, and the six
failure-state distinctions remain non-collapsible across routes.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_analysis import (
    DEFAULT_FAILURE_STATES,
    MCP_CONTRACT_ANALYSIS_INTERFACE,
    PARITY_CLAIM_FAMILIES,
    SCAEV051PARITY,
    SCAEV051PARITY_COVERAGE,
    SCAEV051PARITY_EVIDENCE,
    ContractParityClaim,
    McpContractAnalysis,
    McpContractAnalysisError,
    McpContractAnalyzer,
    ParityState,
    ReviewedAlias,
    analyze_mcp_contract,
    analyze_mcp_contracts,
    scaev051_parity_evidence,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
    McpClaimFamily,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_invocation_trace import (
    InvocationTerminalState,
    McpInvocationTrace,
)


OPERATION = "repo.inspect"
FAILURES = [
    "unsupported",
    "unavailable",
    "denied",
    "timed_out",
    "malformed",
    "partial",
]


def _input_schema(
    *,
    repo_name: str = "repo",
    repo_type: str = "string",
    include_default: bool = True,
) -> dict:
    repo = {"type": repo_type}
    if include_default:
        repo["default"] = "main"
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["status", "files"]},
            repo_name: repo,
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def _output_schema(*, data_type: str = "object") -> dict:
    return {
        "type": "object",
        "properties": {
            "data": {"type": data_type},
            "ok": {"type": "boolean"},
        },
        "required": ["data", "ok"],
        "additionalProperties": False,
    }


def _expected() -> dict:
    return {
        "operation_id": OPERATION,
        "input_schema": _input_schema(),
        "output_schema": _output_schema(),
        "result_envelope": [
            "content",
            "error",
            "provenance",
            "receipt",
        ],
        "failure_states": list(FAILURES),
        "required_policies": ["authorize", "fence"],
        "transports": ["http", "stdio"],
        "require_provenance": True,
        "require_receipt": True,
        "complete": True,
    }


def _route(
    route_id: str,
    transport: str,
    *,
    path_class: str = "direct",
) -> dict:
    return {
        "route_id": route_id,
        "transport": transport,
        "path_class": path_class,
        "callable": True,
        "input_schema": _input_schema(),
        "output_schema": _output_schema(),
        "argument_map": {"action": "action", "repo": "repo"},
        "result_envelope": [
            "content",
            "error",
            "provenance",
            "receipt",
        ],
        "failure_states": list(FAILURES),
        "failure_mapping": {state: state for state in FAILURES},
        "events": [
            "policy:authorize",
            "policy:fence",
            "effect:repository_read",
        ],
        "mutation_capable": True,
        "provenance": True,
        "receipt": True,
        "source_ids": [f"source:{route_id}"],
    }


def _observed(*, compatibility: bool = False) -> dict:
    return {
        "operation_id": OPERATION,
        "discovery": {"tools": [OPERATION]},
        "routes": [
            _route("route:stdio", "stdio"),
            _route(
                "route:http",
                "http",
                path_class="compatibility" if compatibility else "direct",
            ),
        ],
        "complete": True,
    }


def _reasons(report: McpContractAnalysis, family: McpClaimFamily) -> set[str]:
    return set(report.claim(family).reason_codes)


def test_seeded_conformant_fixture_satisfies_every_parity_family() -> None:
    report = analyze_mcp_contract(_expected(), _observed())

    assert MCP_CONTRACT_ANALYSIS_INTERFACE == "McpContractAnalysis@1"
    assert report.passed is True
    assert report.complete is True
    assert report.state is ParityState.SATISFIED
    assert {claim.family for claim in report.claims} == set(
        PARITY_CLAIM_FAMILIES
    )
    assert all(claim.state is ParityState.SATISFIED for claim in report.claims)
    assert report.analysis_id.startswith("b")
    assert report.expected_contract_id.startswith("b")
    assert report.observed_contract_id.startswith("b")


def test_argument_rename_requires_an_exact_reviewed_mapping() -> None:
    observed = _observed()
    for route in observed["routes"]:
        route["input_schema"] = _input_schema(repo_name="repository")
        route["argument_map"]["repo"] = "repository"

    unreviewed = analyze_mcp_contract(_expected(), observed)
    argument_claim = unreviewed.claim(McpClaimFamily.ARGUMENTS_PRESERVED)
    assert argument_claim.state is ParityState.REFUTED
    assert "argument_rename_unreviewed" in argument_claim.reason_codes

    alias = ReviewedAlias(
        source_name="repo",
        target_name="repository",
        review_id="review:alias:repo-v1",
        source_ids=("contract:mcp-idl:v1",),
    )
    reviewed = analyze_mcp_contract(_expected(), observed, aliases=(alias,))
    assert reviewed.claim(
        McpClaimFamily.ARGUMENTS_PRESERVED
    ).state is ParityState.SATISFIED
    assert reviewed.claim(
        McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES
    ).state is ParityState.SATISFIED
    assert reviewed.passed is True


@pytest.mark.parametrize(
    ("mutate", "family", "reason"),
    [
        (
            lambda route: route.update(
                input_schema=_input_schema(include_default=False)
            ),
            McpClaimFamily.ARGUMENTS_PRESERVED,
            "argument_default_changed",
        ),
        (
            lambda route: route.update(
                input_schema=_input_schema(repo_type="integer")
            ),
            McpClaimFamily.ARGUMENTS_PRESERVED,
            "argument_type_changed",
        ),
        (
            lambda route: route["input_schema"]["properties"]["repo"].pop(
                "type"
            ),
            McpClaimFamily.ARGUMENTS_PRESERVED,
            "argument_type_lost",
        ),
        (
            lambda route: route.update(
                result_envelope=["content", "error", "provenance"]
            ),
            McpClaimFamily.RESULT_ENVELOPE_PRESERVED,
            "result_envelope_field_lost",
        ),
        (
            lambda route: route.update(
                events=[
                    "policy:authorize",
                    "effect:repository_write",
                    "policy:fence",
                ]
            ),
            McpClaimFamily.POLICY_BEFORE_EFFECT,
            "policy_after_effect",
        ),
        (
            lambda route: route.update(
                failure_mapping={state: "error" for state in FAILURES}
            ),
            McpClaimFamily.FAILURE_PARITY,
            "failure_states_collapsed",
        ),
    ],
)
def test_argument_result_policy_and_failure_regressions_are_refuted(
    mutate, family: McpClaimFamily, reason: str
) -> None:
    observed = _observed()
    mutate(observed["routes"][0])

    report = analyze_mcp_contract(_expected(), observed)

    assert report.passed is False
    assert report.claim(family).state is ParityState.REFUTED
    assert reason in _reasons(report, family)


def test_input_contravariance_and_output_covariance_are_checked_recursively() -> None:
    observed = _observed()
    # The handler rejects a descriptor-valid action.
    observed["routes"][0]["input_schema"]["properties"]["action"]["enum"] = [
        "status"
    ]
    # The handler may produce a result forbidden by the descriptor.
    observed["routes"][1]["output_schema"]["properties"]["data"] = {
        "type": "string"
    }

    report = analyze_mcp_contract(_expected(), observed)
    claim = report.claim(McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES)

    assert claim.state is ParityState.REFUTED
    assert "input_schema_enum_variance" in claim.reason_codes
    assert "output_schema_type_variance" in claim.reason_codes
    assert {
        witness.boundary_id for witness in claim.counterexamples
    } == {"route:http", "route:stdio"}


def test_tools_list_and_tools_call_drift_is_detected_from_observed_routes() -> None:
    observed = _observed()
    observed["discovery"] = {"tools": [OPERATION]}
    for route in observed["routes"]:
        route["callable"] = False

    report = analyze_mcp_contract(_expected(), observed)

    claim = report.claim(McpClaimFamily.DISCOVERY_EXECUTION_PARITY)
    assert claim.state is ParityState.REFUTED
    assert claim.reason_codes == ("tools_list_call_drift",)


def test_exact_invocation_trace_is_authoritative_for_tools_call_reachability() -> None:
    trace = McpInvocationTrace(
        operation_id=OPERATION,
        graph_root="graph:fixture",
        snapshot_id="snapshot:fixture",
        source_node_id="node:descriptor",
        target_node_ids=("node:handler",),
        terminal_state=InvocationTerminalState.REFUTED,
        reason_code="no_invocation_path",
    )

    report = analyze_mcp_contract(_expected(), _observed(), trace=trace)

    claim = report.claim(McpClaimFamily.DISCOVERY_EXECUTION_PARITY)
    assert claim.state is ParityState.REFUTED
    assert trace.trace_id in claim.premise_ids
    assert report.trace_id == trace.trace_id


def test_one_transport_bypass_is_distinguished_from_common_semantics() -> None:
    observed = _observed()
    observed["routes"][1]["receipt"] = False

    report = analyze_mcp_contract(_expected(), observed)

    claim = report.claim(McpClaimFamily.TRANSPORT_PARITY)
    assert claim.state is ParityState.REFUTED
    assert "transport_only_bypass" in claim.reason_codes
    witnesses = [
        item
        for item in claim.counterexamples
        if item.reason_code == "transport_only_bypass"
    ]
    assert {item.expected["transport"] for item in witnesses} == {"http"}
    assert "route_receipt_bypass" in {
        item.expected["semantic_requirement"] for item in witnesses
    }


def test_compatibility_route_cannot_bypass_receipt_or_policy() -> None:
    observed = _observed(compatibility=True)
    compatibility = observed["routes"][1]
    compatibility["receipt"] = False
    compatibility["events"] = ["effect:repository_write"]

    report = analyze_mcp_contract(_expected(), observed)

    claim = report.claim(McpClaimFamily.NO_COMPATIBILITY_BYPASS)
    assert claim.state is ParityState.REFUTED
    assert claim.reason_codes == ("compatibility_bypass",)
    semantic_requirements = {
        item.expected["semantic_requirement"]
        for item in claim.counterexamples
    }
    assert "route_receipt_bypass" in semantic_requirements
    assert "required_policy_missing" in semantic_requirements


def test_compatibility_route_inherits_schema_and_argument_requirements() -> None:
    observed = _observed(compatibility=True)
    compatibility = observed["routes"][1]
    compatibility["input_schema"] = _input_schema(repo_type="integer")

    report = analyze_mcp_contract(_expected(), observed)

    claim = report.claim(McpClaimFamily.NO_COMPATIBILITY_BYPASS)
    assert claim.state is ParityState.REFUTED
    assert any(
        item.expected["semantic_requirement"] == "argument_type_changed"
        for item in claim.counterexamples
    )


def test_success_and_error_envelope_variants_cannot_collapse() -> None:
    expected = _expected()
    expected["result_envelopes"] = {
        "success": ["content", "receipt"],
        "error": ["code", "message"],
    }
    observed = _observed()
    for route in observed["routes"]:
        route["result_envelopes"] = deepcopy(expected["result_envelopes"])
        route["envelope_mapping"] = {
            "success": "response",
            "error": "response",
        }

    report = analyze_mcp_contract(expected, observed)

    claim = report.claim(McpClaimFamily.RESULT_ENVELOPE_PRESERVED)
    assert claim.state is ParityState.REFUTED
    assert "result_envelopes_collapsed" in claim.reason_codes


def test_missing_evidence_stays_partial_or_ambiguous_and_never_passes() -> None:
    observed = _observed()
    del observed["discovery"]
    del observed["routes"][0]["events"]
    del observed["routes"][0]["failure_states"]

    report = analyze_mcp_contract(_expected(), observed)

    assert report.complete is False
    assert report.passed is False
    assert report.claim(
        McpClaimFamily.DISCOVERY_EXECUTION_PARITY
    ).state is ParityState.AMBIGUOUS
    assert report.claim(
        McpClaimFamily.POLICY_BEFORE_EFFECT
    ).state is ParityState.PARTIAL
    assert report.claim(
        McpClaimFamily.FAILURE_PARITY
    ).state is ParityState.PARTIAL


def test_unsupported_schema_fragment_is_typed_and_not_a_false_pass() -> None:
    expected = _expected()
    expected["input_schema"]["allOf"] = [{"type": "object"}]

    report = analyze_mcp_contract(expected, _observed())
    claim = report.claim(McpClaimFamily.DESCRIPTOR_SCHEMA_MATCHES)

    assert report.passed is False
    assert claim.state is ParityState.UNSUPPORTED
    assert "unsupported_schema_keyword:$/allOf" in claim.reason_codes


def test_reports_are_order_invariant_round_trip_and_reject_tampering() -> None:
    expected = _expected()
    observed = _observed()
    reordered_expected = dict(reversed(list(expected.items())))
    reordered_observed = dict(reversed(list(observed.items())))
    reordered_observed["routes"] = list(reversed(observed["routes"]))

    first = analyze_mcp_contract(expected, observed)
    second = analyze_mcp_contract(reordered_expected, reordered_observed)

    assert second.analysis_id == first.analysis_id
    assert second.to_json() == first.to_json()
    assert McpContractAnalysis.from_json(first.to_json()) == first

    tampered = first.to_dict()
    tampered["claims"][0]["reason_codes"] = ["tampered"]
    with pytest.raises(McpContractAnalysisError, match="claim identity"):
        McpContractAnalysis.from_dict(tampered)

    tampered = first.to_dict()
    tampered["passed"] = False
    with pytest.raises(McpContractAnalysisError, match="passed claim"):
        McpContractAnalysis.from_dict(tampered)


def test_batch_analysis_is_sorted_and_duplicate_operations_fail_closed() -> None:
    expected_a = _expected()
    observed_a = _observed()
    expected_b = deepcopy(expected_a)
    observed_b = deepcopy(observed_a)
    expected_b["operation_id"] = "repo.apply"
    observed_b["operation_id"] = "repo.apply"
    observed_b["discovery"]["tools"] = ["repo.apply"]

    reports = analyze_mcp_contracts(
        [
            {"expected": expected_a, "observed": observed_a},
            {"expected": expected_b, "observed": observed_b},
        ]
    )
    assert [item.operation_id for item in reports] == [
        "repo.apply",
        "repo.inspect",
    ]

    with pytest.raises(McpContractAnalysisError, match="duplicate operation"):
        McpContractAnalyzer().analyze_many(
            [
                (expected_a, observed_a),
                (expected_a, observed_a),
            ]
        )


def test_alias_and_claim_serialization_validate_authority_and_identity() -> None:
    with pytest.raises(McpContractAnalysisError, match="source_ids"):
        ReviewedAlias("repo", "repository", "review:v1", ())

    report = analyze_mcp_contract(_expected(), _observed())
    claim = report.claim(McpClaimFamily.ARGUMENTS_PRESERVED)
    assert ContractParityClaim.from_dict(claim.to_dict()) == claim


def test_scaev051parity_evidence_markers_and_coverage() -> None:
    """Exact-text SCAEV051PARITY markers for objective evidence admission."""

    assert SCAEV051PARITY == "SCAEV051PARITY"
    assert SCAEV051PARITY_EVIDENCE == SCAEV051PARITY
    payload = scaev051_parity_evidence()
    assert payload["evidence"] == SCAEV051PARITY
    assert payload["requirement_ids"] == [SCAEV051PARITY]
    assert payload["coverage"] == list(SCAEV051PARITY_COVERAGE)
    assert payload["interface"] == MCP_CONTRACT_ANALYSIS_INTERFACE
    assert set(payload["claim_families"]) == {
        family.value for family in PARITY_CLAIM_FAMILIES
    }
    assert tuple(payload["failure_states"]) == DEFAULT_FAILURE_STATES
    assert DEFAULT_FAILURE_STATES == (
        "unsupported",
        "unavailable",
        "denied",
        "timed_out",
        "malformed",
        "partial",
    )
    assert "discovery-execution-parity-tools-list-call" in SCAEV051PARITY_COVERAGE
    assert "transport-parity" in SCAEV051PARITY_COVERAGE
    assert any("failure-state-distinctions" in item for item in SCAEV051PARITY_COVERAGE)


def test_each_default_failure_state_loss_is_independently_refuted() -> None:
    """SCAEV051PARITY: the six failure distinctions never silently drop."""

    for lost in DEFAULT_FAILURE_STATES:
        observed = _observed()
        reduced = [state for state in FAILURES if state != lost]
        for route in observed["routes"]:
            route["failure_states"] = list(reduced)
            route["failure_mapping"] = {state: state for state in reduced}

        report = analyze_mcp_contract(_expected(), observed)
        claim = report.claim(McpClaimFamily.FAILURE_PARITY)
        assert claim.state is ParityState.REFUTED, lost
        assert "failure_state_lost" in claim.reason_codes
        assert any(
            item.expected == lost and item.reason_code == "failure_state_lost"
            for item in claim.counterexamples
        ), lost


def test_transport_discovery_list_call_route_drift_is_refuted() -> None:
    """SCAEV051PARITY: per-transport tools/list must agree with tools/call."""

    observed = _observed()
    observed["discovery"] = {
        "tools": [OPERATION],
        "transports": {
            "stdio": [OPERATION],
            "http": [OPERATION],
        },
    }
    observed["routes"][1]["callable"] = False
    observed["routes"][1]["discoverable"] = True

    report = analyze_mcp_contract(_expected(), observed)
    claim = report.claim(McpClaimFamily.DISCOVERY_EXECUTION_PARITY)
    assert claim.state is ParityState.REFUTED
    assert "tools_list_call_route_drift" in claim.reason_codes
    assert any(
        item.boundary_id == "route:http"
        and item.reason_code == "tools_list_call_route_drift"
        for item in claim.counterexamples
    )


def test_missing_transport_route_and_shared_semantics_are_distinguished() -> None:
    """SCAEV051PARITY: absent transport routes are not treated as parity."""

    expected = _expected()
    expected["transports"] = ["http", "stdio", "ws"]
    observed = _observed()

    report = analyze_mcp_contract(expected, observed)
    claim = report.claim(McpClaimFamily.TRANSPORT_PARITY)
    assert claim.state is ParityState.REFUTED
    assert "transport_route_missing" in claim.reason_codes
    assert any(
        item.boundary_id == "ws" and item.reason_code == "transport_route_missing"
        for item in claim.counterexamples
    )
