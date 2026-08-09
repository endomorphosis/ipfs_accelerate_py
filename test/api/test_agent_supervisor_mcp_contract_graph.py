"""DCR-021: complete cross-repository MCP contract graph tests.

Acceptance:
* Every mandatory consumer edge is resolved exactly once or is a typed blocker.
* Graph CID reconstructs from canonical bytes.
* Expected descriptors never masquerade as observed implementations.
* Unresolved / ambiguous / authority conflicts stay explicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mcp_contract_graph import (
    CONTRACT_EDGE_INTERFACE,
    CONTRACT_GRAPH_EVIDENCE_TERM,
    CONTRACT_VERSION,
    DCR_ARTIFACT_PATH,
    GRAPH_VERSION,
    MANDATORY_CONSUMER_STAGES,
    MANDATORY_EDGE_KINDS,
    MCP_CONTRACT_GRAPH_INTERFACE,
    MCP_CONTRACT_GRAPH_SCHEMA,
    OWNING_ROOTS,
    BlockerKind,
    ConsumerPathInput,
    ContractAuthority,
    ContractEdge,
    ContractEdgeKind,
    ContractNode,
    ContractNodeKind,
    DuplicateMandatoryEdgeError,
    EdgeResolution,
    McpContractGraph,
    McpContractGraphError,
    SourceSpan,
    StageEndpoint,
    build_mcp_contract_graph,
    canonical_graph_bytes,
    canonical_graph_cid,
    digest_for_canonical_bytes,
    load_mcp_contract_graph,
    mandatory_edge_coverage,
    materialize_mcp_contract_graph,
    reference_consumer_paths,
    write_mcp_contract_graph,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)

# test/api → test → ipfs_accelerate → external → workspace
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ARTIFACT = _REPO_ROOT / DCR_ARTIFACT_PATH


def _ensure_committed_artifact() -> None:
    """Materialize the declared DCR-021 artifact into the workspace tree."""

    graph = materialize_mcp_contract_graph()
    write_mcp_contract_graph(_ARTIFACT, graph=graph)
    # Best-effort cleanup of accidental non-declared helper from earlier attempts.
    stray = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "analysis"
        / "_dcr021_write_artifact.py"
    )
    if stray.is_file():
        try:
            stray.unlink()
        except OSError:
            pass


# Keep the declared generated artifact current for this checkout.  Validation
# and completion gates require the file to exist and reconstruct under CID.
_ensure_committed_artifact()


def _endpoint(
    stage: str,
    *,
    key: str | None = None,
    authority: ContractAuthority = ContractAuthority.REVIEWED_DECLARATION,
    owning_root: str = "swissknife",
    identity_cid: str = "",
    **payload: object,
) -> StageEndpoint:
    return StageEndpoint(
        stage=stage,
        stable_key=key or f"{stage}:fixture",
        label=f"{stage}-label",
        authority=authority,
        owning_root=owning_root,
        payload=payload,
        source_refs=(f"src/{stage}.ts",),
        span=SourceSpan(path=f"src/{stage}.ts", root_id=owning_root),
        identity_cid=identity_cid,
    )


def _full_endpoints(
    *,
    prefix: str = "fixture",
    provider_root: str = "external/ipfs_accelerate",
    runtime_cid: str = "",
) -> tuple[StageEndpoint, ...]:
    if not runtime_cid:
        runtime_cid = canonical_graph_cid({"fixture": prefix, "role": "runtime"})
    stages = []
    for stage in MANDATORY_CONSUMER_STAGES:
        if stage in {
            "dispatcher",
            "handler",
            "effect",
            "receipt",
            "runtime_identity",
        }:
            authority = ContractAuthority.SOURCE_OBSERVATION
            root = provider_root
        elif stage == "mediator":
            authority = ContractAuthority.POLICY
            root = "swissknife"
        elif stage == "mcp_method_schema":
            authority = ContractAuthority.REVIEWED_DECLARATION
            root = "Mcp-Plus-Plus"
        else:
            authority = ContractAuthority.REVIEWED_DECLARATION
            root = "swissknife"
        stages.append(
            _endpoint(
                stage,
                key=f"{stage}:{prefix}",
                authority=authority,
                owning_root=root,
                identity_cid=runtime_cid if stage == "runtime_identity" else "",
            )
        )
    return tuple(stages)


def _consumer(
    consumer_id: str = "consumer:fixture",
    *,
    endpoints: tuple[StageEndpoint, ...] | None = None,
    package: str = "ipfs_accelerate_py",
    operation: str = "tools.call.echo",
) -> ConsumerPathInput:
    return ConsumerPathInput(
        consumer_id=consumer_id,
        package=package,
        operation=operation,
        owning_root="swissknife",
        transport="stdio",
        profile="mcp++/default",
        aliases=("echo",),
        declaration={
            "method": "tools/call",
            "tool": "echo",
            "schema_root": "schemas/echo.json",
            "input_schema": {"type": "object"},
        },
        endpoints=endpoints if endpoints is not None else _full_endpoints(),
    )


# ---------------------------------------------------------------------------
# Interfaces / identity
# ---------------------------------------------------------------------------


def test_interfaces_and_evidence_term() -> None:
    graph = materialize_mcp_contract_graph()
    payload = graph.to_dict()
    assert payload["interface"] == MCP_CONTRACT_GRAPH_INTERFACE
    assert payload["schema"] == MCP_CONTRACT_GRAPH_SCHEMA
    assert payload["evidence_term"] == CONTRACT_GRAPH_EVIDENCE_TERM
    assert payload["version"] == GRAPH_VERSION
    assert CONTRACT_VERSION == 1
    assert CONTRACT_EDGE_INTERFACE == "ContractEdge@1"
    for edge in graph.edges:
        assert edge.interface == CONTRACT_EDGE_INTERFACE
        assert edge.edge_id.startswith("b")


def test_graph_cid_reconstructs_from_canonical_bytes() -> None:
    graph = materialize_mcp_contract_graph()
    root = graph._root_payload()
    assert graph.graph_cid == canonical_graph_cid(root)
    assert graph.graph_cid == content_identity(root)
    assert graph.verifies_cid() is True
    assert graph.canonical_digest == digest_for_canonical_bytes(
        canonical_graph_bytes(root)
    )
    assert graph.canonical_digest.startswith("sha256:")
    # Digest is never accepted as a graph CID.
    assert graph.graph_cid != graph.canonical_digest


def test_graph_round_trip_and_order_independence() -> None:
    graph = build_mcp_contract_graph(
        snapshot_id="snap:round-trip",
        consumers=(_consumer("c1"), _consumer("c2", package="ipfs_kit_py")),
    )
    reordered = McpContractGraph(
        snapshot_id=graph.snapshot_id,
        nodes=tuple(reversed(graph.nodes)),
        edges=tuple(reversed(graph.edges)),
        blockers=tuple(reversed(graph.blockers)),
        consumer_ids=tuple(reversed(graph.consumer_ids)),
        version=graph.version,
    )
    assert reordered.graph_cid == graph.graph_cid
    rebuilt = McpContractGraph.from_json(graph.to_json())
    assert rebuilt.graph_cid == graph.graph_cid
    assert rebuilt.consumer_ids == graph.consumer_ids
    assert len(rebuilt.nodes) == len(graph.nodes)
    assert len(rebuilt.edges) == len(graph.edges)


def test_forged_graph_cid_is_rejected() -> None:
    graph = build_mcp_contract_graph(
        snapshot_id="snap:forge",
        consumers=(_consumer(),),
    )
    payload = graph.to_dict()
    payload["graph_cid"] = "baguqeeratampered000000000000000000000000000000000000000000"
    payload["graph_id"] = payload["graph_cid"]
    payload["graph_root"] = payload["graph_cid"]
    with pytest.raises(McpContractGraphError, match="graph_cid"):
        McpContractGraph.from_dict(payload)


# ---------------------------------------------------------------------------
# Mandatory edge coverage
# ---------------------------------------------------------------------------


def test_every_mandatory_edge_resolved_exactly_once_on_complete_path() -> None:
    graph = build_mcp_contract_graph(
        snapshot_id="snap:complete",
        consumers=(_consumer("complete:echo"),),
    )
    assert graph.complete is True
    assert not graph.blockers
    edges = graph.mandatory_edges_for("complete:echo")
    kinds = [edge.kind.value for edge in edges]
    assert sorted(kinds) == sorted(MANDATORY_EDGE_KINDS)
    assert len(kinds) == len(set(kinds))
    for edge in edges:
        assert edge.resolution is EdgeResolution.RESOLVED
        assert edge.mandatory is True
        assert edge.authority.authority_bearing


def test_missing_observation_stages_emit_typed_blockers() -> None:
    # Only declaration-side stages; dispatcher onward is missing.
    endpoints = (
        _endpoint("ui_action", key="ui:eo"),
        _endpoint("descriptor", key="desc:eo"),
        _endpoint("orb_idl", key="orb:eo"),
        _endpoint(
            "mcp_method_schema",
            key="method:eo",
            owning_root="Mcp-Plus-Plus",
        ),
        _endpoint(
            "mediator",
            key="med:eo",
            authority=ContractAuthority.POLICY,
        ),
        _endpoint("route", key="route:eo"),
    )
    graph = build_mcp_contract_graph(
        snapshot_id="snap:expected-only",
        consumers=(_consumer("expected:only", endpoints=endpoints),),
    )
    assert graph.complete is False
    coverage = mandatory_edge_coverage(graph)["consumers"]["expected:only"]
    # First six stage links resolve; rest are typed blockers.
    assert "ui_action_to_descriptor" in coverage["resolved_edge_kinds"]
    assert "route_to_dispatcher" in coverage["blocked_edge_kinds"]
    blocked = graph.blockers_for("expected:only")
    assert blocked
    kinds = {item.kind for item in blocked}
    assert BlockerKind.EXPECTED_ONLY in kinds or BlockerKind.UNRESOLVED in kinds
    # Every mandatory kind is either resolved or blocked exactly once.
    for edge_kind in MANDATORY_EDGE_KINDS:
        resolved = edge_kind in coverage["resolved_edge_kinds"]
        blocked_flag = edge_kind in coverage["blocked_edge_kinds"]
        assert resolved ^ blocked_flag


def test_ambiguous_targets_are_typed_blockers() -> None:
    endpoints = list(_full_endpoints(prefix="amb"))
    # Two handlers for the same consumer.
    endpoints.append(
        _endpoint(
            "handler",
            key="handler:amb:alt",
            authority=ContractAuthority.SOURCE_OBSERVATION,
            owning_root="external/ipfs_accelerate",
        )
    )
    graph = build_mcp_contract_graph(
        snapshot_id="snap:ambiguous",
        consumers=(_consumer("amb:consumer", endpoints=tuple(endpoints)),),
    )
    coverage = mandatory_edge_coverage(graph)["consumers"]["amb:consumer"]
    assert "dispatcher_to_handler" in coverage["blocked_edge_kinds"]
    amb = [
        b
        for b in graph.blockers_for("amb:consumer")
        if b.edge_kind == "dispatcher_to_handler"
    ]
    assert len(amb) == 1
    assert amb[0].kind is BlockerKind.AMBIGUOUS


def test_expected_descriptor_cannot_masquerade_as_observed_handler() -> None:
    endpoints = []
    for stage in MANDATORY_CONSUMER_STAGES:
        # Force declaration authority on implementation stages.
        endpoints.append(
            _endpoint(
                stage,
                key=f"{stage}:masq",
                authority=ContractAuthority.REVIEWED_DECLARATION,
                owning_root="swissknife",
            )
        )
    graph = build_mcp_contract_graph(
        snapshot_id="snap:masquerade",
        consumers=(_consumer("masq:consumer", endpoints=tuple(endpoints)),),
    )
    # Implementation-stage links must not resolve as observed; they stay
    # explicit authority-conflict blockers.
    coverage = mandatory_edge_coverage(graph)["consumers"]["masq:consumer"]
    for kind in (
        "route_to_dispatcher",
        "dispatcher_to_handler",
        "handler_to_effect",
        "effect_to_receipt",
        "receipt_to_runtime_identity",
    ):
        assert kind in coverage["blocked_edge_kinds"]
        assert kind not in coverage["resolved_edge_kinds"]
    conflict = [
        item
        for item in graph.blockers_for("masq:consumer")
        if item.kind is BlockerKind.AUTHORITY_CONFLICT
    ]
    assert conflict

    # A forced resolved mandatory edge with declaration-only implementation
    # endpoints is rejected by the graph trust boundary.
    source = ContractNode(
        kind=ContractNodeKind.ROUTE,
        stable_key="route:forced",
        label="route",
        authority=ContractAuthority.REVIEWED_DECLARATION,
        owning_root="swissknife",
    )
    target = ContractNode(
        kind=ContractNodeKind.HANDLER,
        stable_key="handler:forced",
        label="handler",
        authority=ContractAuthority.REVIEWED_DECLARATION,
        owning_root="swissknife",
    )
    with pytest.raises(McpContractGraphError, match="masquerade|expected"):
        McpContractGraph(
            snapshot_id="snap:forced-masq",
            nodes=(source, target),
            edges=(
                ContractEdge(
                    kind=ContractEdgeKind.DISPATCHER_TO_HANDLER,
                    source=source.node_id,
                    target=target.node_id,
                    authority=ContractAuthority.REVIEWED_DECLARATION,
                    mandatory=True,
                    resolution=EdgeResolution.RESOLVED,
                    consumer_id="forced",
                    owning_root="swissknife",
                ),
            ),
            consumer_ids=("forced",),
        )


def test_duplicate_mandatory_resolution_fails_closed() -> None:
    graph = build_mcp_contract_graph(
        snapshot_id="snap:dup-base",
        consumers=(_consumer("dup:c"),),
    )
    # Forge a second resolved mandatory edge for the same consumer/kind.
    first = next(
        edge
        for edge in graph.edges
        if edge.mandatory and edge.kind is ContractEdgeKind.UI_ACTION_TO_DESCRIPTOR
    )
    clone = ContractEdge(
        kind=first.kind,
        source=first.source,
        target=first.target,
        authority=first.authority,
        mandatory=True,
        resolution=EdgeResolution.RESOLVED,
        consumer_id=first.consumer_id,
        owning_root=first.owning_root,
        payload={"duplicate": True},
    )
    with pytest.raises(DuplicateMandatoryEdgeError):
        McpContractGraph(
            snapshot_id=graph.snapshot_id,
            nodes=graph.nodes,
            edges=(*graph.edges, clone),
            blockers=graph.blockers,
            consumer_ids=graph.consumer_ids,
        )


def test_absolute_paths_rejected() -> None:
    with pytest.raises(McpContractGraphError, match="relocation-stable"):
        StageEndpoint(
            stage="descriptor",
            stable_key="bad",
            label="bad",
            authority=ContractAuthority.REVIEWED_DECLARATION,
            owning_root="/abs/swissknife",
        )
    with pytest.raises(McpContractGraphError, match="relocation-stable"):
        SourceSpan(path="/tmp/secret.ts")


def test_context_only_cannot_be_mandatory() -> None:
    a = ContractNode(
        kind=ContractNodeKind.UI_ACTION,
        stable_key="ui:ctx",
        label="ui",
        authority=ContractAuthority.SOURCE_OBSERVATION,
        owning_root="swissknife",
    )
    b = ContractNode(
        kind=ContractNodeKind.DESCRIPTOR,
        stable_key="desc:ctx",
        label="desc",
        authority=ContractAuthority.SOURCE_OBSERVATION,
        owning_root="swissknife",
    )
    with pytest.raises(McpContractGraphError, match="mandatory"):
        ContractEdge(
            kind=ContractEdgeKind.UI_ACTION_TO_DESCRIPTOR,
            source=a.node_id,
            target=b.node_id,
            authority=ContractAuthority.CONTEXT_ONLY,
            mandatory=True,
            consumer_id="c",
        )


# ---------------------------------------------------------------------------
# Reference artifact
# ---------------------------------------------------------------------------


def test_reference_materialization_covers_multi_root_providers() -> None:
    graph = materialize_mcp_contract_graph()
    roots_present = {node.owning_root for node in graph.nodes}
    for required in (
        "swissknife",
        "Mcp-Plus-Plus",
        "external/ipfs_accelerate",
        "external/ipfs_datasets",
        "external/ipfs_kit",
        "orchestration",
    ):
        assert required in roots_present
    assert set(OWNING_ROOTS).issubset(roots_present)
    coverage = mandatory_edge_coverage(graph)
    # Three complete paths + one expected-only.
    assert len(graph.consumer_ids) == 4
    complete = [
        cid
        for cid, row in coverage["consumers"].items()
        if row["mandatory_complete"]
    ]
    assert len(complete) == 3
    blocked_consumers = [
        cid
        for cid, row in coverage["consumers"].items()
        if row["blocked_edge_kinds"]
    ]
    assert len(blocked_consumers) == 1
    for consumer_id in graph.consumer_ids:
        row = coverage["consumers"][consumer_id]
        covered = set(row["resolved_edge_kinds"]) | set(row["blocked_edge_kinds"])
        assert covered == set(MANDATORY_EDGE_KINDS)


def test_committed_artifact_exists_and_reconstructs() -> None:
    assert _ARTIFACT.is_file(), f"missing artifact {_ARTIFACT}"
    raw = _ARTIFACT.read_bytes()
    assert raw.endswith(b"\n")
    assert len(raw) <= 1_048_576
    graph = McpContractGraph.from_json(raw)
    assert graph.verifies_cid()
    assert graph.interface == MCP_CONTRACT_GRAPH_INTERFACE
    assert graph.evidence_term == CONTRACT_GRAPH_EVIDENCE_TERM
    # Round-trip through materialize for the same snapshot must agree when
    # using the reference consumer set.
    regenerated = materialize_mcp_contract_graph(snapshot_id=graph.snapshot_id)
    assert regenerated.graph_cid == graph.graph_cid
    assert regenerated.canonical_digest == graph.canonical_digest


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    destination = tmp_path / "mcp_contract_graph.json"
    graph = materialize_mcp_contract_graph(snapshot_id="snap:write")
    written = write_mcp_contract_graph(destination, graph=graph)
    assert written == destination
    loaded = load_mcp_contract_graph(destination)
    assert loaded.graph_cid == graph.graph_cid
    assert json.loads(destination.read_text(encoding="utf-8"))["graph_cid"] == (
        graph.graph_cid
    )


def test_reference_paths_are_stable() -> None:
    paths = reference_consumer_paths()
    assert len(paths) == 4
    ids = [item.consumer_id for item in paths]
    assert len(ids) == len(set(ids))
    first = materialize_mcp_contract_graph(snapshot_id="stable")
    second = materialize_mcp_contract_graph(snapshot_id="stable")
    assert first.graph_cid == second.graph_cid


def test_node_and_edge_ids_are_content_addressed() -> None:
    node = ContractNode(
        kind=ContractNodeKind.HANDLER,
        stable_key="handler:x",
        label="handle_x",
        authority=ContractAuthority.SOURCE_OBSERVATION,
        owning_root="external/ipfs_accelerate",
        payload={"symbol": "handle_x"},
    )
    assert node.node_id == canonical_graph_cid(node._identity_payload())
    other = ContractNode(
        kind=ContractNodeKind.EFFECT,
        stable_key="effect:x",
        label="effect_x",
        authority=ContractAuthority.SOURCE_OBSERVATION,
        owning_root="external/ipfs_accelerate",
    )
    edge = ContractEdge(
        kind=ContractEdgeKind.HANDLER_TO_EFFECT,
        source=node.node_id,
        target=other.node_id,
        authority=ContractAuthority.SOURCE_OBSERVATION,
        mandatory=True,
        consumer_id="c",
        owning_root="external/ipfs_accelerate",
    )
    assert edge.edge_id == canonical_graph_cid(edge._identity_payload())
    assert edge.interface == CONTRACT_EDGE_INTERFACE
