"""SCA-050: exact, fail-closed MCP++ invocation reachability tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_invocation_trace import (
    MCP_INVOCATION_TRACE_INTERFACE,
    InvocationPathClass,
    InvocationTerminalState,
    InvocationTraceRequest,
    McpInvocationTrace,
    McpInvocationTraceError,
    McpInvocationTracer,
    TraceBounds,
    compute_mcp_invocation_trace,
    compute_mcp_invocation_traces,
)
from ipfs_accelerate_py.agent_supervisor.analysis.symbolic_contract_graph import (
    GRAPH_VERSION,
    ContractAuthority,
    ContractEdgeKind,
    ContractGraphEdge,
    ContractGraphNode,
    ContractNodeKind,
    ContractProvenance,
    SymbolicContractGraph,
)


SNAPSHOT = "repository-snapshot:sha256:mcp-trace-fixture"


def _span(path: str, line: int) -> dict[str, object]:
    return {
        "path": path,
        "source_sha256": "sha256:" + f"{line:064x}",
        "start_line": line,
        "start_column": 0,
        "end_line": line,
        "end_column": 20,
    }


def _node(
    key: str,
    kind: ContractNodeKind,
    *,
    payload: dict[str, object] | None = None,
) -> ContractGraphNode:
    return ContractGraphNode(
        kind=kind,
        stable_key=key,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        payload=payload or {"label": key},
        source_refs=(f"source:{key}",),
    )


def _edge(
    source: ContractGraphNode,
    target: ContractGraphNode,
    kind: ContractEdgeKind,
    *,
    line: int,
    compatibility: bool = False,
    mandatory: bool = True,
    payload: dict[str, object] | None = None,
) -> ContractGraphEdge:
    values: dict[str, object] = {
        "source_span": _span("fixture/mcp_path.py", line),
    }
    if compatibility:
        values["compatibility"] = True
        values["route_kind"] = "compatibility_route"
    if payload:
        values.update(payload)
    return ContractGraphEdge(
        kind=kind,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        mandatory=mandatory,
        payload=values,
        source_refs=(f"ast-source:{line}",),
    )


def _direct_and_compatibility_graph():
    declaration = _node("descriptor:ipfs.add", ContractNodeKind.METHOD)
    connector = _node("connector:tools/call", ContractNodeKind.SYMBOL)
    compatibility = _node(
        "compatibility:/api/v0/add", ContractNodeKind.TRANSPORT
    )
    tool = _node("tool:ipfs.add", ContractNodeKind.TOOL)
    handler = _node("handler:add", ContractNodeKind.HANDLER)
    implementation = _node("implementation:add", ContractNodeKind.SYMBOL)
    edges = (
        _edge(
            declaration,
            connector,
            ContractEdgeKind.DISPATCHES_TO,
            line=10,
        ),
        _edge(
            connector,
            tool,
            ContractEdgeKind.REGISTERS,
            line=20,
        ),
        _edge(
            declaration,
            compatibility,
            ContractEdgeKind.TRANSPORTED_BY,
            line=30,
            compatibility=True,
        ),
        _edge(
            compatibility,
            tool,
            ContractEdgeKind.DISPATCHES_TO,
            line=40,
            compatibility=True,
        ),
        _edge(tool, handler, ContractEdgeKind.HANDLED_BY, line=50),
        _edge(handler, implementation, ContractEdgeKind.IMPLEMENTS, line=60),
    )
    graph = SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(
            declaration,
            connector,
            compatibility,
            tool,
            handler,
            implementation,
        ),
        edges=edges,
    )
    return graph, declaration, implementation


def test_exact_direct_and_compatibility_paths_retain_edges_and_spans() -> None:
    graph, declaration, implementation = _direct_and_compatibility_graph()

    trace = compute_mcp_invocation_trace(
        graph,
        "ipfs_kit_py:ipfs.add",
        declaration.node_id,
        (implementation.node_id,),
    )

    assert MCP_INVOCATION_TRACE_INTERFACE == "McpInvocationTrace@1"
    assert trace.terminal_state is InvocationTerminalState.REACHABLE
    assert trace.state is InvocationTerminalState.REACHABLE
    assert trace.reason_code == "exact_authoritative_path"
    assert trace.graph_root == graph.graph_root
    assert len(trace.direct_paths) == 1
    assert len(trace.compatibility_paths) == 1
    assert trace.direct_paths[0].path_class is InvocationPathClass.DIRECT
    assert (
        trace.compatibility_paths[0].path_class
        is InvocationPathClass.COMPATIBILITY
    )
    for path in trace.all_paths:
        assert path.proof_eligible
        assert path.edge_ids
        assert path.source_spans
        assert all(segment.edge_id for segment in path.segments)
        assert all(segment.source_ids for segment in path.segments)
        assert all(segment.source_spans for segment in path.segments)
    projection = trace.to_dict()
    assert projection["interface"] == MCP_INVOCATION_TRACE_INTERFACE
    assert projection["terminal_state"] == "reachable"
    assert "reachable" not in {
        key for key in projection if key != "terminal_state"
    }
    assert McpInvocationTrace.from_json(trace.to_json()) == trace


def test_unresolved_dynamic_segment_never_counts_as_reachable() -> None:
    declaration = _node("descriptor:dynamic", ContractNodeKind.METHOD)
    dynamic = _node(
        "unresolved:runtime-tool-name",
        ContractNodeKind.UNRESOLVED,
        payload={
            "resolution_state": "unresolved",
            "unresolved_id": "dynamic-tool-name",
            "source_span": _span("fixture/connector.ts", 8),
        },
    )
    handler = _node("handler:maybe", ContractNodeKind.HANDLER)
    graph = SymbolicContractGraph(
        SNAPSHOT,
        (declaration, dynamic, handler),
        (
            _edge(
                declaration,
                dynamic,
                ContractEdgeKind.DISPATCHES_TO,
                line=8,
                payload={"unresolved_id": "dynamic-tool-name"},
            ),
            _edge(
                dynamic,
                handler,
                ContractEdgeKind.HANDLED_BY,
                line=9,
            ),
        ),
    )

    trace = McpInvocationTracer(graph).trace(
        "dynamic.operation",
        declaration.node_id,
        (handler.node_id,),
    )

    assert trace.terminal_state is InvocationTerminalState.AMBIGUOUS
    assert trace.reason_code == "unresolved_dynamic_segment"
    assert trace.proved_paths == ()
    assert trace.unresolved_paths
    assert all(path.dynamic for path in trace.unresolved_paths)
    assert all(not path.proof_eligible for path in trace.unresolved_paths)


def test_closed_terminal_state_set_is_exercised_exactly() -> None:
    declaration = _node("descriptor:operation", ContractNodeKind.METHOD)
    left = _node("handler:left", ContractNodeKind.HANDLER)
    right = _node("handler:right", ContractNodeKind.HANDLER)
    isolated = _node("handler:isolated", ContractNodeKind.HANDLER)
    graph = SymbolicContractGraph(
        SNAPSHOT,
        (declaration, left, right, isolated),
        (
            _edge(
                declaration,
                left,
                ContractEdgeKind.HANDLED_BY,
                line=1,
            ),
            _edge(
                declaration,
                right,
                ContractEdgeKind.HANDLED_BY,
                line=2,
            ),
        ),
    )
    tracer = McpInvocationTracer(graph)

    traces = (
        tracer.trace("reachable", declaration.node_id, (left.node_id,)),
        tracer.trace("refuted", declaration.node_id, (isolated.node_id,)),
        tracer.trace(
            "ambiguous",
            declaration.node_id,
            (left.node_id, right.node_id),
        ),
        tracer.trace(
            "unsupported",
            declaration.node_id,
            (),
            supported=False,
        ),
        tracer.trace(
            "not-measured",
            declaration.node_id,
            (left.node_id,),
            measured=False,
        ),
    )

    assert {trace.terminal_state for trace in traces} == set(
        InvocationTerminalState
    )
    assert [
        trace.terminal_state.value for trace in traces
    ] == [
        "reachable",
        "refuted",
        "ambiguous",
        "unsupported",
        "not_measured",
    ]
    assert all(
        trace.to_dict()["terminal_state"]
        in {item.value for item in InvocationTerminalState}
        for trace in traces
    )


def test_only_mandatory_authoritative_allowlisted_edges_can_prove() -> None:
    declaration = _node("descriptor:strict", ContractNodeKind.METHOD)
    handler = _node("handler:strict", ContractNodeKind.HANDLER)
    nonmandatory = SymbolicContractGraph(
        SNAPSHOT,
        (declaration, handler),
        (
            _edge(
                declaration,
                handler,
                ContractEdgeKind.HANDLED_BY,
                line=11,
                mandatory=False,
            ),
        ),
    )
    unrelated = SymbolicContractGraph(
        SNAPSHOT,
        (declaration, handler),
        (
            _edge(
                declaration,
                handler,
                ContractEdgeKind.RELATED_TO,
                line=12,
                mandatory=False,
            ),
        ),
    )

    assert (
        McpInvocationTracer(nonmandatory)
        .trace("strict", declaration.node_id, (handler.node_id,))
        .terminal_state
        is InvocationTerminalState.REFUTED
    )
    assert (
        McpInvocationTracer(unrelated)
        .trace("strict", declaration.node_id, (handler.node_id,))
        .terminal_state
        is InvocationTerminalState.REFUTED
    )


def test_incomplete_graph_bounds_and_missing_provenance_fail_closed() -> None:
    graph, declaration, implementation = _direct_and_compatibility_graph()
    incomplete = replace(
        graph,
        mandatory_edge_ids=(
            *graph.mandatory_edge_ids,
            "baguqeeramissingmandatoryedge",
        ),
        graph_root_claim="",
        identity_claim=None,
    )
    incomplete_trace = McpInvocationTracer(incomplete).trace(
        "incomplete",
        declaration.node_id,
        (implementation.node_id,),
    )
    assert incomplete_trace.terminal_state is InvocationTerminalState.NOT_MEASURED
    assert incomplete_trace.complete is False
    assert incomplete_trace.reason_code == "incomplete_symbolic_contract_graph"

    bounded_trace = McpInvocationTracer(
        graph, bounds=TraceBounds(max_depth=1)
    ).trace(
        "bounded",
        declaration.node_id,
        (implementation.node_id,),
    )
    assert bounded_trace.terminal_state is InvocationTerminalState.NOT_MEASURED
    assert bounded_trace.reason_code == "max_depth_exceeded"

    source = _node("descriptor:no-span", ContractNodeKind.METHOD)
    target = _node("handler:no-span", ContractNodeKind.HANDLER)
    no_span_edge = ContractGraphEdge(
        kind=ContractEdgeKind.HANDLED_BY,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        mandatory=True,
        source_refs=("source:known-but-no-span",),
    )
    no_span = SymbolicContractGraph(
        SNAPSHOT, (source, target), (no_span_edge,)
    )
    no_span_trace = McpInvocationTracer(no_span).trace(
        "no-span", source.node_id, (target.node_id,)
    )
    assert no_span_trace.terminal_state is InvocationTerminalState.NOT_MEASURED
    assert no_span_trace.reason_code == "path_source_provenance_incomplete"
    assert no_span_trace.direct_paths[0].edge_ids == (
        no_span_edge.edge_id,
    )


def test_batch_is_deterministic_and_rejects_duplicate_operations() -> None:
    graph, declaration, implementation = _direct_and_compatibility_graph()
    requests = (
        InvocationTraceRequest(
            "z-operation",
            declaration.node_id,
            (implementation.node_id,),
        ),
        InvocationTraceRequest(
            "a-operation",
            declaration.node_id,
            (implementation.node_id,),
        ),
    )

    result = compute_mcp_invocation_traces(graph, requests)
    assert [item.operation_id for item in result] == [
        "a-operation",
        "z-operation",
    ]
    assert result == compute_mcp_invocation_traces(
        graph, tuple(reversed(requests))
    )
    with pytest.raises(McpInvocationTraceError, match="duplicate operation"):
        compute_mcp_invocation_traces(
            graph, (requests[0], requests[0])
        )


def test_trace_and_path_content_tampering_is_rejected() -> None:
    graph, declaration, implementation = _direct_and_compatibility_graph()
    trace = compute_mcp_invocation_trace(
        graph,
        "tamper-test",
        declaration.node_id,
        (implementation.node_id,),
    )

    payload = trace.to_dict()
    payload["reason_code"] = "tampered"
    with pytest.raises(McpInvocationTraceError, match="trace identity"):
        McpInvocationTrace.from_dict(payload)

    payload = trace.to_dict()
    payload["direct_paths"][0]["segments"][0]["edge_id"] = "tampered-edge"
    with pytest.raises(McpInvocationTraceError, match="segment identity"):
        McpInvocationTrace.from_dict(payload)


def test_stable_keys_are_accepted_but_similar_names_are_not_guessed() -> None:
    graph, declaration, implementation = _direct_and_compatibility_graph()
    trace = McpInvocationTracer(graph).trace(
        "stable-key",
        declaration.stable_key,
        (implementation.stable_key,),
    )
    assert trace.terminal_state is InvocationTerminalState.REACHABLE

    with pytest.raises(McpInvocationTraceError, match="not in the pinned graph"):
        McpInvocationTracer(graph).trace(
            "no-guess",
            "descriptor:ipfs",
            (implementation.stable_key,),
        )
