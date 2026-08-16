"""DCR-044 static transport operator tests; no target runtime is touched."""

from __future__ import annotations

import ast
import hashlib

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    MCP_CONTRACT_GRAPH_INTERFACE,
    MCP_CONTRACT_GRAPH_SCHEMA,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_live_observer import (
    McpObservationEpoch,
    McpObservationTranscript,
    ObservationStatus,
    RequiredMcpObservation,
    build_mcp_observation_epoch,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorDescriptor,
    OperatorRegistry,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.transport_repairs import (
    TRANSPORT_REPAIR_ACTIVATION,
    TransportRepairStatus,
    build_transport_repair_preview,
    transport_ast_span_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)


def _registry() -> OperatorRegistry:
    descriptor = OperatorDescriptor.from_mapping(
        {
            "operator_id": "transport.normalize",
            "kind": "replace_exact_bytes",
            "input_schema": {
                "type": "object",
                "required": ["anchor", "relative_path", "source_digest"],
                "properties": {
                    "anchor": "cid",
                    "relative_path": "path",
                    "source_digest": "sha256",
                },
                "additional_properties": False,
            },
            "owner_root": "swissknife",
            "write_scope": ["transport.py"],
            "before_predicates": ["unique_anchor"],
            "after_predicates": ["governed_mediator"],
            "applicability_proofs": ["static_ast"],
            "preview": {"kind": "metadata_only", "fields": ["source_digest"]},
            "inverse": {"kind": "restore_exact_before_bytes", "binding": "source_digest"},
            "validation_commands": [["pytest", "transport.py"]],
        }
    )
    return OperatorRegistry(
        (descriptor,), reviewed_manifest={descriptor.operator_id: descriptor.descriptor_id}
    )


def _graph() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": MCP_CONTRACT_GRAPH_SCHEMA,
        "interface": MCP_CONTRACT_GRAPH_INTERFACE,
        "authoritative": False,
        "nodes": [{"id": "source"}, {"id": "target"}],
        "edges": [
            {
                "id": "edge:mediator",
                "source": "source",
                "target": "target",
                "relation": "binds_mediator_route",
                "authority_class": "observed_provider",
            }
        ],
    }
    return {
        **body,
        "canonical_bytes": canonical_json_bytes(body).decode("utf-8"),
        "graph_cid": content_identity(body),
    }


def _epoch(graph: dict[str, object]) -> McpObservationEpoch:
    requirement = RequiredMcpObservation(
        service_role="transport-mediator",
        edge_id="edge:mediator",
        package="swissknife.desktop",
        operation="mcp.health",
        direction="request",
        schema="cid:health-schema",
        profile="mcp-http-jsonrpc",
        transport="mcp",
    )
    receipt = McpObservationTranscript(
        status=ObservationStatus.OBSERVED,
        failure=None,
        service_role="transport-mediator",
        transport="mcp",
        operation="mcp.health",
        endpoint="http://localhost:7001/",
        request_bytes=b'{"method":"mcp.health"}',
        response_bytes=b'{"ok":true}',
        graph_cid=str(graph["graph_cid"]),
        runtime_receipt_id="cid:runtime-receipt",
        process_witness_cid="cid:process-witness",
        template_cid="cid:template",
    )
    return build_mcp_observation_epoch(
        graph_cid=str(graph["graph_cid"]),
        semantic_roots={"semantic": "cid:semantic"},
        snapshot_roots={"forest": "cid:forest"},
        required_observations=(requirement,),
        receipts=(receipt,),
    )


def _request(*, source: bytes | None = None, endpoint: str = "http://localhost:7001/mcp"):
    registry = _registry()
    graph = _graph()
    epoch = _epoch(graph)
    source = source or (
        b'configure_transport(endpoint="http://localhost:7000/mcp", '
        b'profile="mcp-http-jsonrpc", mediator="governed_mediator", '
        b'lifecycle="strict_failure")\n'
    )
    anchor = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call))
    descriptor = registry.enumerate()[0]
    return {
        "action": "normalize_endpoint_profile",
        "operator_id": descriptor.operator_id,
        "descriptor_id": descriptor.descriptor_id,
        "manifest_cid": registry.report()["registry_cid"],
        "owner_root": "swissknife",
        "relative_path": "transport.py",
        "source_bytes": source,
        "source_digest": "sha256:" + hashlib.sha256(source).hexdigest(),
        "anchor": transport_ast_span_identity(source, anchor),
        "target_api": "configure_transport",
        "endpoint": endpoint,
        "profile": "mcp-http-jsonrpc",
        "graph": graph,
        "graph_edge_id": "edge:mediator",
        "semantic_roots": {"semantic": "cid:semantic"},
        "snapshot_roots": {"forest": "cid:forest"},
        "observation_epoch": epoch,
        "observation_epoch_cid": epoch.epoch_cid,
        "behavioral_postcondition": {
            "mediator": "governed_mediator",
            "requires_mediator": True,
            "lifecycle_errors_fail": True,
            "raw_proxy_exposed": False,
        },
    }, registry


def test_static_loopback_preview_is_reversible_and_runtime_pending() -> None:
    request, registry = _request()
    preview = build_transport_repair_preview(request, registry=registry)

    assert preview.status is TransportRepairStatus.PREVIEWED
    assert b"localhost:7001" in preview.after_bytes
    assert preview.forward_cid and preview.inverse_cid
    payload = preview.to_dict()
    assert payload["activation_status"] == TRANSPORT_REPAIR_ACTIVATION
    assert payload["execution_authorized"] is False
    assert payload["completion_authorized"] is False
    assert payload["model_call_count"] == payload["provider_call_count"] == 0


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://example.test:7001/mcp",
        "http://user@localhost:7001/mcp",
        "http://localhost:7001/mcp?proxy=1",
    ),
)
def test_remote_userinfo_and_query_endpoint_tricks_are_rejected(endpoint: str) -> None:
    request, registry = _request(endpoint=endpoint)
    preview = build_transport_repair_preview(request, registry=registry)
    assert preview.status is TransportRepairStatus.REJECTED
    assert preview.to_dict()["execution_authorized"] is False


def test_dynamic_multiple_anchor_raw_proxy_and_stale_epoch_are_rejected() -> None:
    request, registry = _request(
        source=(
            b'configure_transport(endpoint="http://localhost:7000/mcp", '
            b'profile="mcp-http-jsonrpc", mediator="governed_mediator", '
            b'lifecycle="strict_failure")\n'
            b'configure_transport(endpoint="http://localhost:7002/mcp", '
            b'profile="mcp-http-jsonrpc", mediator="governed_mediator", '
            b'lifecycle="strict_failure")\n'
        )
    )
    assert (
        build_transport_repair_preview(request, registry=registry).status
        is TransportRepairStatus.REJECTED
    )

    raw_proxy, _ = _request(
        source=(
            b'configure_transport(endpoint="http://localhost:7000/mcp", '
            b'profile="mcp-http-jsonrpc", mediator="raw_proxy", '
            b'lifecycle="strict_failure")\n'
        )
    )
    assert (
        build_transport_repair_preview(raw_proxy, registry=registry).status
        is TransportRepairStatus.REJECTED
    )

    stale = dict(_request()[0])
    stale["observation_epoch_cid"] = "cid:stale"
    assert (
        build_transport_repair_preview(stale, registry=registry).status
        is TransportRepairStatus.REJECTED
    )


def test_forged_self_consistent_epoch_is_rejected_by_current_epoch_validation() -> None:
    request, registry = _request()
    graph = request["graph"]
    assert isinstance(graph, dict)
    forged = McpObservationEpoch(
        graph_cid=str(graph["graph_cid"]),
        semantic_roots={"semantic": "cid:semantic"},
        snapshot_roots={"forest": "cid:forest"},
        required_observations=(),
        receipts=(),
        checks=(),
        valid=True,
    )
    request["observation_epoch"] = forged
    request["observation_epoch_cid"] = forged.epoch_cid

    assert build_transport_repair_preview(request, registry=registry).status is TransportRepairStatus.REJECTED
