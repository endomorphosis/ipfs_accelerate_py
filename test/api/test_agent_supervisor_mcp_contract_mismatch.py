"""DCR-024 consumes the typed, canonical DCR-023 observation epoch."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_mismatch import (
    analyze_mcp_contract_mismatches,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    MCP_LIVE_OBSERVATION_EPOCH_INTERFACE,
    MCP_LIVE_OBSERVATION_EPOCH_SCHEMA,
    MCP_LIVE_OBSERVER_INTERFACE,
    MCP_LIVE_TRANSCRIPT_SCHEMA,
    McpObservationTranscript,
    ObservationStatus,
    RequiredMcpObservation,
    build_mcp_observation_epoch,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

_ROOTS = {"descriptor": "bafy-descriptor", "policy": "bafy-policy"}
_SNAPSHOT = {"forest": "bafy-forest", "graph": "bafy-graph"}
_GRAPH = {
    "graph_cid": "bafy-graph-cid",
    "edges": [
        {"id": "edge-protocol", "relation": "expects_descriptor"},
        {"id": "edge-schema", "relation": "defines_method_schema"},
        {"id": "edge-mediation", "relation": "binds_mediator_route"},
        {"id": "edge-implementation", "relation": "dispatches_to_handler"},
        {"id": "edge-identity", "relation": "emits_receipt_runtime_identity"},
    ],
    "blockers": [],
}


def _semantic(operation: str) -> dict[str, str]:
    return {
        "package": "accelerate",
        "operation": operation,
        "direction": "request",
        "schema": "CatalogRequest@1",
        "profile": "mcp++",
        "transport": "mcp",
    }


def _transcript(checks: list[dict[str, object]]) -> dict[str, object]:
    required = [
        {
            "service_role": "accelerate",
            "edge_id": str(check["edge_id"]),
            "semantic_key": dict(check.get("semantic_key") or _semantic("catalog.read")),
        }
        for check in checks
        if isinstance(check.get("semantic_key") or _semantic("catalog.read"), dict)
    ] or [
        {
            "service_role": "accelerate",
            "edge_id": "edge-protocol",
            "semantic_key": _semantic("catalog.read"),
        }
    ]
    receipt_body: dict[str, object] = {
        "schema": MCP_LIVE_TRANSCRIPT_SCHEMA,
        "interface": MCP_LIVE_OBSERVER_INTERFACE,
        "authoritative": False,
        "completion_authoritative": False,
        "status": "observed",
        "failure": None,
        "service_role": "accelerate",
        "transport": "mcp",
        "operation": "catalog.read",
        "endpoint": "http://127.0.0.1:9010",
        "request_base64": "e30=",
        "response_base64": "e30=",
        "request_digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        "response_digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        "graph_cid": _GRAPH["graph_cid"],
        "runtime_receipt_id": "bafy-runtime",
        "process_witness_cid": "bafy-process",
        "template_cid": "bafy-template",
    }
    receipt = {**receipt_body, "receipt_id": content_identity(receipt_body)}
    for check in checks:
        if check.get("status") == "passed":
            check.setdefault("receipt_id", receipt["receipt_id"])
            check.setdefault("request_digest", receipt_body["request_digest"])
            check.setdefault("response_digest", receipt_body["response_digest"])
    body: dict[str, object] = {
        "schema": MCP_LIVE_OBSERVATION_EPOCH_SCHEMA,
        "interface": MCP_LIVE_OBSERVATION_EPOCH_INTERFACE,
        "authoritative": False,
        "completion_authoritative": False,
        "valid": True,
        "graph_cid": _GRAPH["graph_cid"],
        "semantic_roots": _ROOTS,
        "snapshot_roots": _SNAPSHOT,
        "required_observations": required,
        "receipts": [receipt],
        "checks": checks,
    }
    return {**body, "epoch_cid": content_identity(body)}


def test_absent_or_stale_dcr023_epoch_is_an_integration_pending_liveness_finding() -> None:
    absent = analyze_mcp_contract_mismatches(
        graph=_GRAPH, semantic_roots=_ROOTS, snapshot_roots=_SNAPSHOT
    )
    assert absent["production_readiness"] == "integration_pending"

    stale = _transcript(
        [
            {
                "mismatch_class": "liveness",
                "status": "passed",
                "edge_id": "edge-protocol",
                "semantic_key": _semantic("catalog.read"),
            }
        ]
    )
    stale["semantic_roots"] = {"descriptor": "stale"}
    stale["epoch_cid"] = content_identity(
        {key: value for key, value in stale.items() if key != "epoch_cid"}
    )
    report = analyze_mcp_contract_mismatches(
        graph=_GRAPH, semantic_roots=_ROOTS, snapshot_roots=_SNAPSHOT, transcript=stale
    )
    assert report["dcr023_current_valid"] is False
    assert report["production_readiness"] == "integration_pending"


def test_typed_current_epoch_from_dcr023_builder_is_accepted_directly() -> None:
    receipt = McpObservationTranscript(
        status=ObservationStatus.OBSERVED,
        failure=None,
        service_role="accelerate",
        transport="mcp",
        operation="catalog.read",
        endpoint="http://127.0.0.1:9010",
        request_bytes=b'{"method":"catalog.read"}',
        response_bytes=b'{"result":{}}',
        graph_cid=_GRAPH["graph_cid"],
        runtime_receipt_id="bafy-runtime-receipt",
        process_witness_cid="bafy-process-witness",
        template_cid="bafy-template",
    )
    epoch = build_mcp_observation_epoch(
        graph_cid=_GRAPH["graph_cid"],
        semantic_roots=_ROOTS,
        snapshot_roots=_SNAPSHOT,
        required_observations=(
            RequiredMcpObservation(
                service_role="accelerate",
                edge_id="edge-protocol",
                **_semantic("catalog.read"),
            ),
        ),
        receipts=(receipt,),
    )
    report = analyze_mcp_contract_mismatches(
        graph=_GRAPH, semantic_roots=_ROOTS, snapshot_roots=_SNAPSHOT, transcript=epoch
    )
    assert epoch.valid is True
    assert report["dcr023_current_valid"] is True
    assert report["production_readiness"] == "ready"
    assert report["findings"] == []

    forged = epoch.to_dict()
    forged["checks"][0]["request_digest"] = "sha256:forged"
    forged["epoch_cid"] = content_identity(
        {key: value for key, value in forged.items() if key != "epoch_cid"}
    )
    rejected = analyze_mcp_contract_mismatches(
        graph=_GRAPH, semantic_roots=_ROOTS, snapshot_roots=_SNAPSHOT, transcript=forged
    )
    assert rejected["dcr023_current_valid"] is False


def test_nonpassing_epoch_is_not_current_even_when_its_cid_is_self_consistent() -> None:
    checks = [
        {
            "mismatch_class": "schema",
            "status": "missing",
            "edge_id": "edge-schema",
            "semantic_key": _semantic("catalog.read"),
        },
        {
            "mismatch_class": "schema",
            "status": "missing",
            "edge_id": "edge-schema",
            "semantic_key": _semantic("catalog.read"),
        },
        {
            "mismatch_class": "schema",
            "status": "missing",
            "edge_id": "edge-identity",
            "semantic_key": _semantic("catalog.inspect"),
        },
        {
            "mismatch_class": "identity",
            "status": "unobserved",
            "edge_id": "edge-identity",
            "semantic_key": _semantic("catalog.read"),
        },
    ]
    report = analyze_mcp_contract_mismatches(
        graph=_GRAPH,
        semantic_roots=_ROOTS,
        snapshot_roots=_SNAPSHOT,
        transcript=_transcript(checks),
    )
    assert report["dcr023_current_valid"] is False
    assert report["production_readiness"] == "integration_pending"
    assert report["findings"][0]["edge_id"] == "dcr023:current-transcript"


def test_semantic_key_absence_cannot_make_a_nonpassing_epoch_current() -> None:
    checks = [
        {"mismatch_class": "schema", "status": "missing", "edge_id": "edge-schema"},
        {"mismatch_class": "schema", "status": "missing", "edge_id": "edge-identity"},
    ]
    report = analyze_mcp_contract_mismatches(
        graph=_GRAPH,
        semantic_roots=_ROOTS,
        snapshot_roots=_SNAPSHOT,
        transcript=_transcript(checks),
    )
    assert report["dcr023_current_valid"] is False
    assert report["findings"][0]["edge_id"] == "dcr023:current-transcript"


def test_current_epoch_rejects_wrong_schema_and_graph_blockers_remain_nonpassing() -> None:
    epoch = _transcript(
        [
            {
                "mismatch_class": "liveness",
                "status": "passed",
                "edge_id": "edge-protocol",
                "semantic_key": _semantic("catalog.read"),
            }
        ]
    )
    epoch["schema"] = "forged"
    epoch["epoch_cid"] = content_identity(
        {key: value for key, value in epoch.items() if key != "epoch_cid"}
    )
    invalid = analyze_mcp_contract_mismatches(
        graph=_GRAPH, semantic_roots=_ROOTS, snapshot_roots=_SNAPSHOT, transcript=epoch
    )
    assert invalid["dcr023_current_valid"] is False

    graph = {**_GRAPH, "blockers": [{"kind": "authority_conflict", "operation": "catalog.read"}]}
    report = analyze_mcp_contract_mismatches(
        graph=graph,
        semantic_roots=_ROOTS,
        snapshot_roots=_SNAPSHOT,
        transcript=_transcript(
            [
                {
                    "mismatch_class": "liveness",
                    "status": "passed",
                    "edge_id": "edge-protocol",
                    "semantic_key": _semantic("catalog.read"),
                }
            ]
        ),
    )
    assert report["production_readiness"] == "integration_pending"
    assert report["findings"][0]["mismatch_class"] == "authority"
