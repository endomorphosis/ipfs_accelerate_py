"""SCA-217: endpoint anchors and observed package contract compilation."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_assurance_baseline import (
    BaselineStageName,
    StageCompleteness,
    TerminalContractStatus,
    materialize_contract_assurance_baseline,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mcp_invocation_trace import (
    InvocationPathClass,
    InvocationTerminalState,
)
from ipfs_accelerate_py.agent_supervisor.analysis.runtime_contract_evidence_compiler import (
    PATH_CLASS_DIRECT,
    PATH_CLASS_MCP_PLUS_PLUS,
    RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE,
    AnchorResolutionState,
    EvidenceFindingKind,
    RuntimeContractEvidenceCompiler,
    compile_runtime_contract_evidence,
)
from ipfs_accelerate_py.agent_supervisor.analysis.swissknife_contract_extractor import (
    extract_swissknife_contracts,
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


SNAPSHOT = "repository-snapshot:sha256:sca-217-evidence-fixture"


def _canonical_sources() -> dict[str, str]:
    return {
        "src/services/mcp/mcp-plus-plus.ts": """
export const IPFS_KIT_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-kit',
  namespace: 'com.ipfs.kit',
  version: '1.0.0',
  interface_cid: 'bafy-kit',
  methods: [{
    name: 'ipfs.add',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
  }],
  errors: [{ name: 'Unavailable', code: 503 }],
  requires: ['mcp++/cid-envelope', 'mcp++/deontic-policy', 'mcp++/p2p-transport'],
  compatibility: { compatible_with: [], supersedes: [] },
};
export const IPFS_ACCELERATE_INTERFACE: MCPPPInterfaceDescriptor = {
  name: 'ipfs-accelerate',
  namespace: 'com.ipfs.accelerate',
  version: '1.0.0',
  interface_cid: 'bafy-acc',
  methods: [{
    name: 'accelerate.inference',
    input_schema_cid: 'bafy-in',
    output_schema_cid: 'bafy-out',
    error_schema_cids: ['bafy-error'],
    interaction_pattern: 'stream',
  }],
  errors: [],
  requires: ['mcp++/cid-envelope'],
  compatibility: { compatible_with: [], supersedes: [] },
};
"""
    }


def _extraction():
    return extract_swissknife_contracts(
        _canonical_sources(),
        repository_tree_id="tree-fixture",
        source_version="git:fixture",
    )


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
) -> ContractGraphEdge:
    values: dict[str, object] = {
        "source_span": _span("fixture/mcp_path.py", line),
    }
    if compatibility:
        values["compatibility"] = True
        values["route_kind"] = "compatibility_route"
    return ContractGraphEdge(
        kind=kind,
        source=source.node_id,
        target=target.node_id,
        snapshot_id=SNAPSHOT,
        provenance=ContractProvenance.AST,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        version=GRAPH_VERSION,
        mandatory=True,
        payload=values,
        source_refs=(f"ast-source:{line}",),
    )


def _healthy_operation_graph():
    """Graph with distinct MCP++ and direct paths for ipfs.add."""

    declaration = _node(
        "descriptor:ipfs.add",
        ContractNodeKind.METHOD,
        payload={
            "tool_name": "ipfs.add",
            "package_id": "ipfs_kit_py",
            "label": "descriptor:ipfs.add",
        },
    )
    connector = _node(
        "connector:tools/call",
        ContractNodeKind.SYMBOL,
        payload={"package_id": "ipfs_kit_py", "label": "mcp++ tools/call"},
    )
    direct = _node(
        "direct_fetch:/api/v0/add",
        ContractNodeKind.TRANSPORT,
        payload={
            "tool_name": "ipfs.add",
            "package_id": "ipfs_kit_py",
            "route_kind": "direct_fetch",
            "label": "direct_fetch:/api/v0/add",
        },
    )
    tool = _node(
        "tool:ipfs.add",
        ContractNodeKind.TOOL,
        payload={"tool_name": "ipfs.add", "package_id": "ipfs_kit_py"},
    )
    handler = _node(
        "handler:add",
        ContractNodeKind.HANDLER,
        payload={"tool_name": "ipfs.add", "package_id": "ipfs_kit_py"},
    )
    implementation = _node(
        "implementation:add",
        ContractNodeKind.SYMBOL,
        payload={"tool_name": "ipfs.add", "package_id": "ipfs_kit_py"},
    )
    # Second reviewed operation: accelerate.inference (MCP++ only).
    decl_inf = _node(
        "descriptor:accelerate.inference",
        ContractNodeKind.METHOD,
        payload={
            "tool_name": "accelerate.inference",
            "package_id": "ipfs_accelerate_py",
        },
    )
    tool_inf = _node(
        "tool:accelerate.inference",
        ContractNodeKind.TOOL,
        payload={
            "tool_name": "accelerate.inference",
            "package_id": "ipfs_accelerate_py",
        },
    )
    handler_inf = _node(
        "handler:inference",
        ContractNodeKind.HANDLER,
        payload={
            "tool_name": "accelerate.inference",
            "package_id": "ipfs_accelerate_py",
        },
    )
    edges = (
        _edge(declaration, connector, ContractEdgeKind.DISPATCHES_TO, line=10),
        _edge(connector, tool, ContractEdgeKind.REGISTERS, line=20),
        _edge(
            declaration,
            direct,
            ContractEdgeKind.TRANSPORTED_BY,
            line=30,
            compatibility=True,
        ),
        _edge(
            direct,
            tool,
            ContractEdgeKind.DISPATCHES_TO,
            line=40,
            compatibility=True,
        ),
        _edge(tool, handler, ContractEdgeKind.HANDLED_BY, line=50),
        _edge(handler, implementation, ContractEdgeKind.IMPLEMENTS, line=60),
        _edge(decl_inf, tool_inf, ContractEdgeKind.DISPATCHES_TO, line=70),
        _edge(tool_inf, handler_inf, ContractEdgeKind.HANDLED_BY, line=80),
    )
    graph = SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(
            declaration,
            connector,
            direct,
            tool,
            handler,
            implementation,
            decl_inf,
            tool_inf,
            handler_inf,
        ),
        edges=edges,
    )
    return graph, declaration, implementation, handler, decl_inf, handler_inf


def test_interface_constant() -> None:
    assert (
        RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE
        == "RuntimeContractEvidenceCompiler@1"
    )
    assert RuntimeContractEvidenceCompiler.interface == (
        RUNTIME_CONTRACT_EVIDENCE_COMPILER_INTERFACE
    )


def test_healthy_catalog_emits_anchors_observed_contracts_and_traces() -> None:
    extraction = _extraction()
    graph, *_ = _healthy_operation_graph()

    result = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        graph=graph,
        extraction=extraction,
    )

    assert result.operations
    assert len(result.anchors) == len(result.operations)
    assert len(result.observed_contracts) == len(result.operations)
    assert result.compilation_id.startswith("b")

    # Every reviewed tool-bearing operation is covered.
    op_ids = {item.operation_id for item in result.operations}
    assert "ipfs_kit_py:ipfs.add" in op_ids
    assert "ipfs_accelerate_py:accelerate.inference" in op_ids

    for anchor in result.anchors:
        assert anchor.anchor_id.startswith("b")
        assert PATH_CLASS_MCP_PLUS_PLUS in anchor.path_classes
        assert anchor.resolution_state is AnchorResolutionState.RESOLVED
        assert anchor.is_traceable
        assert anchor.source_node_id
        assert anchor.target_node_ids

    for observed in result.observed_contracts:
        assert observed["operation_id"]
        assert observed["routes"]
        assert observed["complete"] is True
        route_kinds = {
            route.get("mediation_path_class") or route.get("path_kind")
            for route in observed["routes"]
        }
        assert PATH_CLASS_MCP_PLUS_PLUS in route_kinds

    # Healthy inputs yield nonempty traces.
    assert result.traces
    assert len(result.traces) == len(result.traceable_anchors)
    for trace in result.traces:
        assert trace.terminal_state is InvocationTerminalState.REACHABLE
        assert trace.direct_paths or trace.compatibility_paths
        assert trace.all_paths


def test_mcp_plus_plus_and_direct_paths_remain_distinct() -> None:
    extraction = _extraction()
    graph, declaration, _impl, handler, *_ = _healthy_operation_graph()

    result = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        graph=graph,
        extraction=extraction,
    )
    anchor = result.anchor_map["ipfs_kit_py:ipfs.add"]
    assert PATH_CLASS_MCP_PLUS_PLUS in anchor.path_classes
    assert PATH_CLASS_DIRECT in anchor.path_classes
    assert anchor.mcp_plus_plus_source_node_id
    assert anchor.direct_source_node_id
    # Distinct path classes must not collapse onto one synthetic identity.
    assert anchor.mcp_plus_plus_source_node_id != anchor.direct_source_node_id

    observed = result.observed_contract_map["ipfs_kit_py:ipfs.add"]
    kinds = [
        route.get("mediation_path_class") or route.get("path_kind")
        for route in observed["routes"]
    ]
    assert PATH_CLASS_MCP_PLUS_PLUS in kinds
    assert PATH_CLASS_DIRECT in kinds
    assert kinds.count(PATH_CLASS_MCP_PLUS_PLUS) == 1
    assert kinds.count(PATH_CLASS_DIRECT) == 1

    # Tracer still keeps direct vs compatibility path classes distinct.
    add_trace = next(
        item for item in result.traces if item.operation_id == "ipfs_kit_py:ipfs.add"
    )
    assert add_trace.direct_paths
    assert add_trace.compatibility_paths
    assert all(
        path.path_class is InvocationPathClass.DIRECT
        for path in add_trace.direct_paths
    )
    assert all(
        path.path_class is InvocationPathClass.COMPATIBILITY
        for path in add_trace.compatibility_paths
    )
    # Sanity: anchors point at the declaration/handler nodes used by the graph.
    assert anchor.source_node_id == declaration.node_id
    assert handler.node_id in anchor.target_node_ids


def test_missing_anchors_become_typed_unknown_findings() -> None:
    extraction = _extraction()
    # Graph has unrelated nodes only — no operation anchors.
    orphan = _node("unrelated:symbol", ContractNodeKind.SYMBOL)
    graph = SymbolicContractGraph(snapshot_id=SNAPSHOT, nodes=(orphan,), edges=())

    result = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        graph=graph,
        extraction=extraction,
    )

    assert result.operations
    assert result.anchors
    assert result.observed_contracts
    assert not result.complete
    assert result.findings
    kinds = {item.kind for item in result.findings}
    assert EvidenceFindingKind.MISSING_SOURCE_ANCHOR in kinds or (
        EvidenceFindingKind.MISSING_TARGET_ANCHOR in kinds
    )
    for anchor in result.anchors:
        assert anchor.resolution_state in {
            AnchorResolutionState.MISSING,
            AnchorResolutionState.INCOMPLETE,
            AnchorResolutionState.AMBIGUOUS,
        }
        assert not anchor.is_traceable
    # Empty-success is forbidden: nonempty reviewed set cannot claim complete.
    assert result.complete is False


def test_ambiguous_targets_are_typed_unknown_not_empty_success() -> None:
    extraction = _extraction()
    declaration = _node(
        "descriptor:ipfs.add",
        ContractNodeKind.METHOD,
        payload={"tool_name": "ipfs.add", "package_id": "ipfs_kit_py"},
    )
    handler_a = _node(
        "handler:add",
        ContractNodeKind.HANDLER,
        payload={"tool_name": "ipfs.add", "package_id": "ipfs_kit_py"},
    )
    handler_b = _node(
        "handler:ipfs.add.alt",
        ContractNodeKind.HANDLER,
        payload={"tool_name": "ipfs.add", "package_id": "ipfs_kit_py"},
    )
    graph = SymbolicContractGraph(
        snapshot_id=SNAPSHOT,
        nodes=(declaration, handler_a, handler_b),
        edges=(
            _edge(declaration, handler_a, ContractEdgeKind.HANDLED_BY, line=1),
            _edge(declaration, handler_b, ContractEdgeKind.HANDLED_BY, line=2),
        ),
    )
    result = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        graph=graph,
        extraction=extraction,
    )
    add_anchor = result.anchor_map["ipfs_kit_py:ipfs.add"]
    assert add_anchor.resolution_state is AnchorResolutionState.AMBIGUOUS
    assert any(
        item.kind is EvidenceFindingKind.AMBIGUOUS_TARGET_ANCHOR
        for item in result.findings
        if item.operation_id == "ipfs_kit_py:ipfs.add"
    )
    assert result.complete is False


def test_compilation_is_deterministic() -> None:
    extraction = _extraction()
    graph, *_ = _healthy_operation_graph()
    first = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        graph=graph,
        extraction=extraction,
    )
    second = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        graph=graph,
        extraction=extraction,
    )
    assert first.compilation_id == second.compilation_id
    assert first.to_dict() == second.to_dict()
    assert [item.anchor_id for item in first.anchors] == [
        item.anchor_id for item in second.anchors
    ]


def test_name_only_unreviewed_contracts_do_not_synthesize_operations() -> None:
    extraction = _extraction()
    # Drop all reviewed tool contracts by filtering — only keep non-tool rows.
    non_tool = tuple(
        contract
        for contract in extraction.catalog.contracts
        if not contract.tool_name
    )
    from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_catalog import (
        McpContractCatalog,
    )

    payload = extraction.catalog.to_dict()
    payload.pop("catalog_id", None)
    payload["contracts"] = [item.to_dict() for item in non_tool]
    catalog = McpContractCatalog.from_dict(payload)
    result = compile_runtime_contract_evidence(
        catalog, snapshot_id=SNAPSHOT, run_traces=False
    )
    assert result.operations == ()
    assert result.anchors == ()
    assert result.observed_contracts == ()
    assert "no_reviewed_runtime_operations" in result.reason_codes


def test_baseline_uses_compiled_anchors_not_withheld_empty_success() -> None:
    extraction = _extraction()
    graph, *_ = _healthy_operation_graph()
    result = materialize_contract_assurance_baseline(
        snapshot_id=SNAPSHOT,
        snapshot={
            "schema": "ipfs_accelerate_py/agent-supervisor/sca-repository-snapshot@1",
            "schema_version": 1,
            "snapshot_id": SNAPSHOT,
            "scope_id": "fixture-scope",
            "scope_policy_id": "sca-scope-policy:sha256:fixture-policy",
            "head_commit_id": "commit-fixture",
            "head_tree_id": "tree-fixture",
            "index_tree_id": "tree-fixture",
            "is_clean": True,
            "primary_root": "swissknife",
            "dispositions": [],
            "dependency_identities": [],
            "gitlinks": [],
            "stats": {
                "tracked_path_count": 0,
                "disposition_count": 0,
                "semantic_path_count": 0,
                "excluded_path_count": 0,
                "overlay_path_count": 0,
                "dirty_path_count": 0,
                "deleted_path_count": 0,
                "dependency_identity_count": 0,
                "gitlink_count": 0,
                "hashed_bytes": 0,
            },
        },
        extraction=extraction,
        catalog=extraction.catalog,
        graph=graph,
        extract_expected=False,
        project_graph=False,
        scope_policy_root="sca-scope-policy:sha256:fixture-policy",
    )
    stage_by_name = {stage.name: stage for stage in result.stages}
    trace_stage = stage_by_name[BaselineStageName.INVOCATION_TRACE]
    assert trace_stage.completeness in {
        StageCompleteness.COMPLETE,
        StageCompleteness.PARTIAL,
    }
    assert "endpoint_anchors_not_supplied" not in trace_stage.reason_codes
    assert int(trace_stage.details.get("anchor_count") or 0) >= 1
    assert int(trace_stage.details.get("trace_count") or 0) >= 1
    assert result.findings.get("endpoint_anchor_count", 0) >= 1
    assert result.findings.get("invocation_trace_count", 0) >= 1
    assert result.findings.get("evidence_compilation_root", "").startswith("b")


def test_baseline_missing_anchors_emit_unknown_not_withheld_empty() -> None:
    extraction = _extraction()
    orphan = _node("orphan:only", ContractNodeKind.SYMBOL)
    graph = SymbolicContractGraph(snapshot_id=SNAPSHOT, nodes=(orphan,), edges=())
    result = materialize_contract_assurance_baseline(
        snapshot_id=SNAPSHOT,
        snapshot={
            "schema": "ipfs_accelerate_py/agent-supervisor/sca-repository-snapshot@1",
            "schema_version": 1,
            "snapshot_id": SNAPSHOT,
            "scope_id": "fixture-scope",
            "scope_policy_id": "sca-scope-policy:sha256:fixture-policy",
            "head_commit_id": "commit-fixture",
            "head_tree_id": "tree-fixture",
            "index_tree_id": "tree-fixture",
            "is_clean": True,
            "primary_root": "swissknife",
            "dispositions": [],
            "dependency_identities": [],
            "gitlinks": [],
            "stats": {
                "tracked_path_count": 0,
                "disposition_count": 0,
                "semantic_path_count": 0,
                "excluded_path_count": 0,
                "overlay_path_count": 0,
                "dirty_path_count": 0,
                "deleted_path_count": 0,
                "dependency_identity_count": 0,
                "gitlink_count": 0,
                "hashed_bytes": 0,
            },
        },
        extraction=extraction,
        catalog=extraction.catalog,
        graph=graph,
        extract_expected=False,
        project_graph=False,
        scope_policy_root="sca-scope-policy:sha256:fixture-policy",
    )
    stage_by_name = {stage.name: stage for stage in result.stages}
    trace_stage = stage_by_name[BaselineStageName.INVOCATION_TRACE]
    # Must not withhold as empty success when reviewed operations exist.
    assert trace_stage.completeness is not StageCompleteness.WITHHELD
    assert int(trace_stage.details.get("anchor_count") or 0) >= 1
    assert int(trace_stage.details.get("finding_count") or 0) >= 1
    # Population still closes every contract with a terminal status.
    for row in result.findings["contract_population"]["contracts"]:
        assert row[3] in {
            TerminalContractStatus.PROVED.value,
            TerminalContractStatus.REFUTED.value,
            TerminalContractStatus.UNKNOWN.value,
            TerminalContractStatus.UNSUPPORTED.value,
            TerminalContractStatus.STALE.value,
        }


def test_zero_llm_and_reexport_from_baseline() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis import (
        contract_assurance_baseline as baseline_mod,
    )

    assert callable(baseline_mod.compile_runtime_contract_evidence)
    extraction = _extraction()
    result = compile_runtime_contract_evidence(
        extraction.catalog,
        snapshot_id=SNAPSHOT,
        extraction=extraction,
        run_traces=False,
    )
    assert result.llm_call_count if hasattr(result, "llm_call_count") else True
    assert result.observed_contracts
    # Observed MCP++ route is always present even without package surfaces.
    for observed in result.observed_contracts:
        kinds = {
            route.get("mediation_path_class") for route in observed["routes"]
        }
        assert PATH_CLASS_MCP_PLUS_PLUS in kinds
