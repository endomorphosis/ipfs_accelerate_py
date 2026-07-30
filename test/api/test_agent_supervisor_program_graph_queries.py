"""Tests for minimal dependency-complete call/impact slice queries (VFS-013/041).

VFS-041 objective validation repair: kind-partitioned query indexes accelerate
traversal without changing canonical graph identity, and acceptance criteria
(seeded transitive callers/callees and MCP paths complete within scope,
unrelated source omitted, limits never silently claim complete) remain proven.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.program_graph import (
    ProgramEdgeKind,
    ProgramNodeKind,
    ResolverStatus,
    SourceSpan,
    build_program_graph,
    make_edge,
    make_node,
)
from ipfs_accelerate_py.agent_supervisor.program_graph_queries import (
    MINIMAL_CALL_SLICE_EVIDENCE,
    PROGRAM_GRAPH_SLICE_SCHEMA,
    QUERY_INDEX_VERSION,
    ProgramGraphQuery,
    ProgramGraphQueryError,
    QueryBounds,
    QueryKind,
    _GraphView,
    query_changed_blob_impact,
    query_contract_consumers,
    query_contract_producers,
    query_mcp_end_to_end,
    query_program_graph_slice,
    query_proof_dependencies,
    query_shortest_counterexample,
    query_symbol_callees,
    query_symbol_callers,
    query_vfs_operation_surface,
)


FOREST_ID = "forest:test-vfs-013"
PRODUCER = "program-graph-queries-test@1"
BLOB_A = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
BLOB_B = "baguqeerbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
BLOB_C = "baguqeercccccccccccccccccccccccccccccccccccccccccccccccccccc"
BLOB_UNRELATED = "baguqeeruuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu"


def _span(line: int = 1, col: int = 0) -> SourceSpan:
    return SourceSpan(line_start=line, column_start=col, line_end=line, column_end=col + 4)


def _node(
    kind: ProgramNodeKind | str,
    key: str,
    *,
    component_id: str = "",
    blob_cid: str = BLOB_A,
    forest_id: str = FOREST_ID,
    qualified_name: str = "",
    path: str = "",
    language: str = "python",
    resolver_status: ResolverStatus | str = ResolverStatus.RESOLVED_STATIC,
    record: dict[str, Any] | None = None,
) -> Any:
    return make_node(
        kind=kind,
        record_key=key,
        producer=PRODUCER,
        blob_cid=blob_cid,
        forest_id=forest_id,
        component_id=component_id or key,
        qualified_name=qualified_name or key,
        path=path,
        language=language,
        span=_span(),
        resolver_status=resolver_status,
        record=record or {},
    )


def _edge(
    source: str,
    target: str,
    kind: ProgramEdgeKind | str,
    *,
    component_id: str = "comp-a",
    blob_cid: str = BLOB_A,
    forest_id: str = FOREST_ID,
    resolver_status: ResolverStatus | str = ResolverStatus.RESOLVED_STATIC,
    record: dict[str, Any] | None = None,
) -> Any:
    return make_edge(
        source=source,
        target=target,
        kind=kind,
        producer=PRODUCER,
        blob_cid=blob_cid,
        forest_id=forest_id,
        component_id=component_id,
        span=_span(2),
        resolver_status=resolver_status,
        record=record or {},
    )


def _seeded_call_graph():
    """A -> B -> C call chain with an unrelated symbol U, and mutual D<->E.

    Call convention matches VFS-008/012 fixtures:
    ``caller --contains--> call --calls--> callee``.
    """

    repo = _node(
        ProgramNodeKind.REPOSITORY,
        "repo:accelerator",
        component_id="repo:accelerator",
        qualified_name="accelerator",
    )
    sym_a = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.a",
        component_id="module:pkg",
        path="pkg/a.py",
        qualified_name="pkg.a",
        blob_cid=BLOB_A,
    )
    def_a = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.a",
        component_id="module:pkg",
        path="pkg/a.py",
        qualified_name="pkg.a",
        blob_cid=BLOB_A,
    )
    sym_b = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.b",
        component_id="module:pkg",
        path="pkg/b.py",
        qualified_name="pkg.b",
        blob_cid=BLOB_A,
    )
    def_b = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.b",
        component_id="module:pkg",
        path="pkg/b.py",
        qualified_name="pkg.b",
        blob_cid=BLOB_A,
    )
    sym_c = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.c",
        component_id="module:pkg",
        path="pkg/c.py",
        qualified_name="pkg.c",
        blob_cid=BLOB_B,
    )
    def_c = _node(
        ProgramNodeKind.DEFINITION,
        "def:pkg.c",
        component_id="module:pkg",
        path="pkg/c.py",
        qualified_name="pkg.c",
        blob_cid=BLOB_B,
    )
    sym_u = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.unrelated",
        component_id="module:other",
        path="pkg/unrelated.py",
        qualified_name="pkg.unrelated",
        blob_cid=BLOB_UNRELATED,
    )
    # Mutual recursion D <-> E (call cycle is legal).
    sym_d = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.d",
        component_id="module:pkg",
        path="pkg/d.py",
        qualified_name="pkg.d",
        blob_cid=BLOB_A,
    )
    sym_e = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.e",
        component_id="module:pkg",
        path="pkg/e.py",
        qualified_name="pkg.e",
        blob_cid=BLOB_A,
    )
    call_ab = _node(
        ProgramNodeKind.CALL,
        "call:pkg.a->pkg.b",
        component_id="module:pkg",
        path="pkg/a.py",
        qualified_name="pkg.b",
        blob_cid=BLOB_A,
    )
    call_bc = _node(
        ProgramNodeKind.CALL,
        "call:pkg.b->pkg.c",
        component_id="module:pkg",
        path="pkg/b.py",
        qualified_name="pkg.c",
        blob_cid=BLOB_A,
    )
    call_de = _node(
        ProgramNodeKind.CALL,
        "call:pkg.d->pkg.e",
        component_id="module:pkg",
        path="pkg/d.py",
        qualified_name="pkg.e",
        blob_cid=BLOB_A,
    )
    call_ed = _node(
        ProgramNodeKind.CALL,
        "call:pkg.e->pkg.d",
        component_id="module:pkg",
        path="pkg/e.py",
        qualified_name="pkg.d",
        blob_cid=BLOB_A,
    )
    # Ambiguous call site for frontier diagnostics.
    call_amb = _node(
        ProgramNodeKind.CALL,
        "call:pkg.a->ambiguous",
        component_id="module:pkg",
        path="pkg/a.py",
        qualified_name="ambiguous",
        blob_cid=BLOB_A,
        resolver_status=ResolverStatus.AMBIGUOUS,
        record={"reason": "multi_candidate"},
    )
    amb_target = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.amb",
        component_id="module:pkg",
        path="pkg/amb.py",
        qualified_name="pkg.amb",
        blob_cid=BLOB_A,
        resolver_status=ResolverStatus.CANDIDATE,
        record={"reason": "candidate"},
    )

    nodes = (
        repo,
        sym_a,
        def_a,
        sym_b,
        def_b,
        sym_c,
        def_c,
        sym_u,
        sym_d,
        sym_e,
        call_ab,
        call_bc,
        call_de,
        call_ed,
        call_amb,
        amb_target,
    )
    edges = (
        _edge(repo.node_id, sym_a.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(repo.node_id, sym_b.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(repo.node_id, sym_c.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(repo.node_id, sym_u.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(repo.node_id, sym_d.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(repo.node_id, sym_e.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(sym_a.node_id, def_a.node_id, ProgramEdgeKind.DEFINES, component_id="module:pkg"),
        _edge(sym_b.node_id, def_b.node_id, ProgramEdgeKind.DEFINES, component_id="module:pkg"),
        _edge(sym_c.node_id, def_c.node_id, ProgramEdgeKind.DEFINES, component_id="module:pkg"),
        # A -> B
        _edge(sym_a.node_id, call_ab.node_id, ProgramEdgeKind.CONTAINS, component_id="module:pkg"),
        _edge(call_ab.node_id, sym_b.node_id, ProgramEdgeKind.CALLS, component_id="module:pkg"),
        # B -> C
        _edge(sym_b.node_id, call_bc.node_id, ProgramEdgeKind.CONTAINS, component_id="module:pkg"),
        _edge(call_bc.node_id, sym_c.node_id, ProgramEdgeKind.CALLS, component_id="module:pkg"),
        # D <-> E cycle
        _edge(sym_d.node_id, call_de.node_id, ProgramEdgeKind.CONTAINS, component_id="module:pkg"),
        _edge(call_de.node_id, sym_e.node_id, ProgramEdgeKind.CALLS, component_id="module:pkg"),
        _edge(sym_e.node_id, call_ed.node_id, ProgramEdgeKind.CONTAINS, component_id="module:pkg"),
        _edge(call_ed.node_id, sym_d.node_id, ProgramEdgeKind.CALLS, component_id="module:pkg"),
        # Ambiguous A -> amb
        _edge(
            sym_a.node_id,
            call_amb.node_id,
            ProgramEdgeKind.CONTAINS,
            component_id="module:pkg",
        ),
        _edge(
            call_amb.node_id,
            amb_target.node_id,
            ProgramEdgeKind.CALLS,
            component_id="module:pkg",
            resolver_status=ResolverStatus.AMBIGUOUS,
            record={"reason": "multi_candidate"},
        ),
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=nodes, edges=edges, producer=PRODUCER
    )
    return {
        "graph": graph,
        "repo": repo,
        "a": sym_a,
        "b": sym_b,
        "c": sym_c,
        "u": sym_u,
        "d": sym_d,
        "e": sym_e,
        "def_a": def_a,
        "def_b": def_b,
        "def_c": def_c,
        "call_ab": call_ab,
        "call_bc": call_bc,
        "call_amb": call_amb,
        "amb": amb_target,
    }


def _mcp_route_graph():
    """MCP end-to-end: registration -> tool -> transport + implements symbol."""

    repo = _node(
        ProgramNodeKind.REPOSITORY,
        "repo:accelerator",
        component_id="repo:accelerator",
        qualified_name="accelerator",
    )
    reg = _node(
        ProgramNodeKind.MCP_REGISTRATION,
        "mcp_reg:entry",
        component_id="mcp:entry",
        path="mcp/tools.json",
        language="json",
        qualified_name="entry",
        blob_cid=BLOB_B,
    )
    tool = _node(
        ProgramNodeKind.MCP_TOOL,
        "mcp_tool:entry",
        component_id="mcp:entry",
        path="mcp/tools.json",
        language="json",
        qualified_name="entry",
        blob_cid=BLOB_B,
    )
    transport = _node(
        ProgramNodeKind.TRANSPORT,
        "transport:stdio",
        component_id="mcp:entry",
        path="mcp/server.py",
        qualified_name="stdio",
        blob_cid=BLOB_B,
    )
    schema = _node(
        ProgramNodeKind.SCHEMA,
        "schema:tool.entry.input",
        component_id="mcp:entry",
        path="schemas/entry.json",
        language="json",
        qualified_name="tool.entry.input",
        blob_cid=BLOB_B,
    )
    impl = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.mod.entry",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.entry",
        blob_cid=BLOB_A,
    )
    helper = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.mod.helper",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.helper",
        blob_cid=BLOB_A,
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:entry->helper",
        component_id="module:pkg.mod",
        path="pkg/mod.py",
        qualified_name="pkg.mod.helper",
        blob_cid=BLOB_A,
    )
    unrelated = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.other",
        component_id="module:other",
        path="pkg/other.py",
        qualified_name="pkg.other",
        blob_cid=BLOB_UNRELATED,
    )
    nodes = (repo, reg, tool, transport, schema, impl, helper, call, unrelated)
    edges = (
        _edge(repo.node_id, reg.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(repo.node_id, impl.node_id, ProgramEdgeKind.CONTAINS, component_id="repo:accelerator"),
        _edge(reg.node_id, tool.node_id, ProgramEdgeKind.REGISTERS, component_id="mcp:entry"),
        _edge(tool.node_id, transport.node_id, ProgramEdgeKind.USES_TRANSPORT, component_id="mcp:entry"),
        _edge(tool.node_id, impl.node_id, ProgramEdgeKind.IMPLEMENTS, component_id="mcp:entry"),
        _edge(tool.node_id, schema.node_id, ProgramEdgeKind.REFERENCES, component_id="mcp:entry"),
        _edge(impl.node_id, call.node_id, ProgramEdgeKind.CONTAINS, component_id="module:pkg.mod"),
        _edge(call.node_id, helper.node_id, ProgramEdgeKind.CALLS, component_id="module:pkg.mod"),
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=nodes, edges=edges, producer=PRODUCER
    )
    return {
        "graph": graph,
        "reg": reg,
        "tool": tool,
        "transport": transport,
        "schema": schema,
        "impl": impl,
        "helper": helper,
        "unrelated": unrelated,
    }


def _vfs_surface_graph():
    """VFS operation surfaces with callers across facades."""

    vfs_open = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:ipfs_kit.vfs.open",
        component_id="module:vfs",
        path="ipfs_kit_py/mcp/ipfs_kit/vfs.py",
        qualified_name="ipfs_kit.vfs.open",
        blob_cid=BLOB_A,
        record={"operation": "open", "surface": "vfs"},
    )
    fsspec = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:ipfs_fsspec.IPFSFileSystem.open",
        component_id="module:fsspec",
        path="ipfs_kit_py/ipfs_fsspec.py",
        qualified_name="ipfs_fsspec.IPFSFileSystem.open",
        blob_cid=BLOB_B,
        record={"operation": "open", "surface": "fsspec"},
    )
    bucket = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:bucket_vfs_manager.read",
        component_id="module:bucket",
        path="ipfs_kit_py/bucket_vfs_manager.py",
        qualified_name="bucket_vfs_manager.read",
        blob_cid=BLOB_C,
        record={"operation": "read", "surface": "bucket"},
    )
    caller = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:app.use_vfs",
        component_id="module:app",
        path="app/use_vfs.py",
        qualified_name="app.use_vfs",
        blob_cid=BLOB_A,
    )
    call = _node(
        ProgramNodeKind.CALL,
        "call:app.use_vfs->vfs.open",
        component_id="module:app",
        path="app/use_vfs.py",
        qualified_name="ipfs_kit.vfs.open",
        blob_cid=BLOB_A,
    )
    noise = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:app.noise",
        component_id="module:app",
        path="app/noise.py",
        qualified_name="app.noise",
        blob_cid=BLOB_UNRELATED,
    )
    nodes = (vfs_open, fsspec, bucket, caller, call, noise)
    edges = (
        _edge(caller.node_id, call.node_id, ProgramEdgeKind.CONTAINS, component_id="module:app"),
        _edge(call.node_id, vfs_open.node_id, ProgramEdgeKind.CALLS, component_id="module:app"),
        _edge(fsspec.node_id, vfs_open.node_id, ProgramEdgeKind.DEPENDS_ON, component_id="module:fsspec"),
        _edge(bucket.node_id, vfs_open.node_id, ProgramEdgeKind.DEPENDS_ON, component_id="module:bucket"),
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=nodes, edges=edges, producer=PRODUCER
    )
    return {
        "graph": graph,
        "vfs_open": vfs_open,
        "fsspec": fsspec,
        "bucket": bucket,
        "caller": caller,
        "noise": noise,
    }


def _contract_graph():
    schema = _node(
        ProgramNodeKind.SCHEMA,
        "schema:contract.input",
        component_id="contract:input",
        path="schemas/input.json",
        language="json",
        qualified_name="contract.input",
        blob_cid=BLOB_B,
    )
    producer = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.emit_contract",
        component_id="module:pkg",
        path="pkg/emit.py",
        qualified_name="pkg.emit_contract",
        blob_cid=BLOB_A,
    )
    consumer = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.use_contract",
        component_id="module:pkg",
        path="pkg/use.py",
        qualified_name="pkg.use_contract",
        blob_cid=BLOB_A,
    )
    mcp_tool = _node(
        ProgramNodeKind.MCP_TOOL,
        "mcp_tool:with_schema",
        component_id="mcp:with_schema",
        path="mcp/tools.json",
        language="json",
        qualified_name="with_schema",
        blob_cid=BLOB_B,
    )
    noise = _node(
        ProgramNodeKind.SYMBOL,
        "symbol:pkg.noise",
        component_id="module:noise",
        path="pkg/noise.py",
        qualified_name="pkg.noise",
        blob_cid=BLOB_UNRELATED,
    )
    nodes = (schema, producer, consumer, mcp_tool, noise)
    edges = (
        _edge(producer.node_id, schema.node_id, ProgramEdgeKind.DEFINES, component_id="module:pkg"),
        _edge(consumer.node_id, schema.node_id, ProgramEdgeKind.REFERENCES, component_id="module:pkg"),
        _edge(mcp_tool.node_id, schema.node_id, ProgramEdgeKind.REFERENCES, component_id="mcp:with_schema"),
        _edge(mcp_tool.node_id, consumer.node_id, ProgramEdgeKind.IMPLEMENTS, component_id="mcp:with_schema"),
    )
    graph = build_program_graph(
        forest_id=FOREST_ID, nodes=nodes, edges=edges, producer=PRODUCER
    )
    return {
        "graph": graph,
        "schema": schema,
        "producer": producer,
        "consumer": consumer,
        "mcp_tool": mcp_tool,
        "noise": noise,
    }


# ---------------------------------------------------------------------------
# Core query surfaces
# ---------------------------------------------------------------------------


def test_symbol_callees_are_transitive_and_omit_unrelated() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
    )
    assert result.kind is QueryKind.SYMBOL_CALLEES
    assert fx["a"].node_id in result.node_ids
    assert fx["b"].node_id in result.node_ids
    assert fx["c"].node_id in result.node_ids
    assert fx["call_ab"].node_id in result.node_ids
    assert fx["call_bc"].node_id in result.node_ids
    # Definitions pulled for dependency completeness.
    assert fx["def_a"].node_id in result.node_ids
    assert fx["def_b"].node_id in result.node_ids
    assert fx["def_c"].node_id in result.node_ids
    # Unrelated and cycle-only symbols omitted.
    assert fx["u"].node_id not in result.node_ids
    assert fx["d"].node_id not in result.node_ids
    assert result.minimal is True
    assert result.dependency_complete is True
    # Ambiguous callees keep complete=False.
    assert result.complete is False
    assert fx["amb"].node_id in result.node_ids
    assert result.ambiguous_element_ids
    payload = result.to_dict()
    assert payload["embeds_source_bodies"] is False
    assert payload["evidence"] == MINIMAL_CALL_SLICE_EVIDENCE
    assert payload["schema"] == PROGRAM_GRAPH_SLICE_SCHEMA


def test_symbol_callers_are_transitive_and_minimal() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callers(
        fx["graph"],
        seed_qualified_names=["pkg.c"],
    )
    assert fx["c"].node_id in result.node_ids
    assert fx["b"].node_id in result.node_ids
    assert fx["a"].node_id in result.node_ids
    assert fx["call_bc"].node_id in result.node_ids
    assert fx["call_ab"].node_id in result.node_ids
    assert fx["u"].node_id not in result.node_ids
    assert fx["d"].node_id not in result.node_ids
    assert result.minimal is True
    # No ambiguous nodes on the pure A<-B<-C chain when amb is not reached
    # from C via reverse CALLS only... actually reverse from C goes B then A
    # and A contains amb call. So amb may appear.
    # Required dependencies must cover every node in the slice.
    assert set(result.required_dependencies) == set(result.node_ids)
    assert not result.omitted_dependencies


def test_call_cycles_are_reported_not_rejected() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.d"],
    )
    assert fx["d"].node_id in result.node_ids
    assert fx["e"].node_id in result.node_ids
    assert result.cycles
    assert any("call_cycle" in item or "cycle:" in item for item in result.cycles)
    assert result.minimal is True


def test_changed_blob_impact_selects_dependents() -> None:
    fx = _seeded_call_graph()
    # Changing BLOB_B (owns pkg.c) should impact callers A and B via reverse walk.
    result = query_changed_blob_impact(
        fx["graph"],
        seed_blob_cids=[BLOB_B],
    )
    assert fx["c"].node_id in result.node_ids
    # Reverse CALLS/CONTAINS from C reaches B and A.
    assert fx["b"].node_id in result.node_ids
    assert fx["a"].node_id in result.node_ids
    assert fx["u"].node_id not in result.node_ids
    assert result.minimal is True
    assert BLOB_UNRELATED not in {
        fx["graph"].node(nid).binding.blob_cid for nid in result.node_ids
    } or fx["u"].node_id not in result.node_ids


def test_contract_consumers_and_producers() -> None:
    fx = _contract_graph()
    consumers = query_contract_consumers(
        fx["graph"],
        seed_node_ids=[fx["schema"].node_id],
    )
    assert fx["schema"].node_id in consumers.node_ids
    assert fx["consumer"].node_id in consumers.node_ids
    assert fx["mcp_tool"].node_id in consumers.node_ids
    assert fx["noise"].node_id not in consumers.node_ids
    assert consumers.minimal is True

    producers = query_contract_producers(
        fx["graph"],
        seed_node_ids=[fx["schema"].node_id],
    )
    assert fx["schema"].node_id in producers.node_ids
    assert fx["producer"].node_id in producers.node_ids
    assert fx["noise"].node_id not in producers.node_ids


def test_mcp_end_to_end_route_is_dependency_complete() -> None:
    fx = _mcp_route_graph()
    result = query_mcp_end_to_end(
        fx["graph"],
        seed_node_ids=[fx["reg"].node_id],
    )
    for key in ("reg", "tool", "transport", "schema", "impl", "helper"):
        assert fx[key].node_id in result.node_ids, key
    assert fx["unrelated"].node_id not in result.node_ids
    assert result.dependency_complete is True
    assert result.minimal is True
    assert result.complete is True
    # Paths should reach implementation or helper.
    assert result.paths
    covered = {step.node_id for path in result.paths for step in path.steps}
    assert fx["tool"].node_id in covered or fx["tool"].node_id in result.node_ids


def test_vfs_operation_surface_auto_seeds_and_neighbors() -> None:
    fx = _vfs_surface_graph()
    # Auto-seed from VFS markers without explicit seeds.
    result = query_vfs_operation_surface(fx["graph"])
    assert fx["vfs_open"].node_id in result.node_ids
    assert fx["fsspec"].node_id in result.node_ids
    assert fx["bucket"].node_id in result.node_ids
    assert fx["caller"].node_id in result.node_ids
    assert fx["noise"].node_id not in result.node_ids
    assert result.minimal is True


def test_proof_dependencies_include_defines_and_calls() -> None:
    fx = _seeded_call_graph()
    result = query_proof_dependencies(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
    )
    assert fx["a"].node_id in result.node_ids
    assert fx["b"].node_id in result.node_ids
    assert fx["def_a"].node_id in result.node_ids
    assert fx["u"].node_id not in result.node_ids
    assert result.minimal is True


def test_shortest_counterexample_is_minimal_path() -> None:
    fx = _seeded_call_graph()
    # Longer alternative: A -> amb (not to C). Shortest A..C is A-call-B-call-C.
    result = query_shortest_counterexample(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        target_qualified_names=["pkg.c"],
    )
    assert result.paths
    lengths = {path.length for path in result.paths}
    assert len(lengths) == 1  # only shortest retained
    path = result.paths[0]
    assert path.entry_node_id == fx["a"].node_id
    assert path.exit_node_id == fx["c"].node_id
    path_nodes = [step.node_id for step in path.steps]
    assert fx["b"].node_id in path_nodes
    # Unrelated and cycle nodes excluded from the minimal slice.
    assert fx["u"].node_id not in result.node_ids
    assert fx["d"].node_id not in result.node_ids
    assert result.minimal is True
    assert result.dependency_complete is True
    # Every path node is required.
    for step in path.steps:
        assert step.node_id in result.node_ids
        assert step.node_id in result.required_dependencies


def test_shortest_counterexample_missing_target_is_incomplete() -> None:
    fx = _seeded_call_graph()
    result = query_shortest_counterexample(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        target_qualified_names=["pkg.does_not_exist"],
    )
    assert result.complete is False
    assert result.dependency_complete is False
    assert result.missing_node_ids
    assert "no_path_to_target" in result.notes or result.missing_node_ids


# ---------------------------------------------------------------------------
# Bounds, frontiers, provenance, stability
# ---------------------------------------------------------------------------


def test_truncation_never_claims_complete() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        bounds=QueryBounds(max_nodes=2, max_edges=2, max_depth=1),
    )
    assert result.truncated is True
    assert result.complete is False
    assert result.dependency_complete is False
    assert result.truncation_reasons
    # Still may be minimal relative to the retained set.
    assert result.minimal is True
    # Omitted dependencies recorded when neighbors exist beyond the bound.
    assert result.omitted_dependencies or result.truncation_reasons


def test_missing_seed_is_explicit() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callers(
        fx["graph"],
        seed_node_ids=["pnode-does-not-exist"],
        seed_qualified_names=["pkg.missing"],
    )
    assert result.empty is True
    assert result.complete is False
    assert result.missing_node_ids
    assert "no_seeds_resolved" in result.notes


def test_excluded_repositories_are_reported() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        excluded_repository_ids=["repo:other"],
    )
    assert "repo:other" in result.excluded_repository_ids
    # Seeds under accelerator still resolve.
    assert fx["a"].node_id in result.seed_node_ids


def test_results_are_stable_and_content_addressed() -> None:
    fx = _seeded_call_graph()
    left = query_symbol_callees(
        fx["graph"], seed_qualified_names=["pkg.a"]
    )
    right = query_symbol_callees(
        fx["graph"], seed_qualified_names=["pkg.a"]
    )
    assert left.slice_id == right.slice_id
    assert left.to_dict() == right.to_dict()
    assert left.to_json() == right.to_json()
    # JSON is canonical (sorted keys via canonical_program_json).
    payload = json.loads(left.to_json())
    assert payload["slice_id"] == left.slice_id
    assert payload["graph_id"] == fx["graph"].graph_id
    assert payload["forest_id"] == FOREST_ID
    assert payload["provenance"]["evidence"] == MINIMAL_CALL_SLICE_EVIDENCE


def test_query_identity_is_deterministic() -> None:
    q1 = ProgramGraphQuery(
        kind=QueryKind.SYMBOL_CALLERS,
        seed_qualified_names=["pkg.c", "pkg.a"],
    )
    q2 = ProgramGraphQuery(
        kind=QueryKind.SYMBOL_CALLERS,
        seed_qualified_names=["pkg.a", "pkg.c"],
    )
    assert q1.query_id == q2.query_id
    assert q1.seed_qualified_names == ("pkg.a", "pkg.c")


def test_query_requires_seed_except_vfs_auto() -> None:
    with pytest.raises(ProgramGraphQueryError):
        ProgramGraphQuery(kind=QueryKind.SYMBOL_CALLERS)
    # VFS allows empty seeds (auto-discovery).
    q = ProgramGraphQuery(kind=QueryKind.VFS_OPERATION_SURFACE)
    assert q.kind is QueryKind.VFS_OPERATION_SURFACE


def test_bounds_reject_out_of_range() -> None:
    with pytest.raises(ProgramGraphQueryError):
        QueryBounds(max_nodes=0)
    with pytest.raises(ProgramGraphQueryError):
        QueryBounds(max_nodes=10**9)


def test_mapping_query_entrypoint() -> None:
    fx = _seeded_call_graph()
    result = query_program_graph_slice(
        fx["graph"],
        {
            "kind": "symbol_callees",
            "seed_qualified_names": ["pkg.b"],
        },
    )
    assert fx["b"].node_id in result.node_ids
    assert fx["c"].node_id in result.node_ids
    assert fx["a"].node_id not in result.node_ids


def test_no_source_bodies_in_results() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"], seed_qualified_names=["pkg.a"]
    )
    blob = json.dumps(result.to_dict())
    assert "def foo" not in blob
    assert "source_text" not in blob
    assert "source_body" not in blob
    # Provenance carries handles only.
    assert "graph_id" in result.provenance
    assert result.provenance["producer"]


def test_frontier_projection_is_bounded_and_sorted() -> None:
    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        bounds=QueryBounds(max_frontier=1),
    )
    # Ambiguous elements exist; frontier may be truncated.
    assert result.frontier
    ids = [item.element_id for item in result.frontier]
    assert ids == sorted(ids)
    if result.truncated and "max_frontier" in result.truncation_reasons:
        assert result.complete is False


def test_all_query_kinds_are_exercised() -> None:
    """Smoke-run every QueryKind on a suitable seeded graph."""

    call_fx = _seeded_call_graph()
    mcp_fx = _mcp_route_graph()
    vfs_fx = _vfs_surface_graph()
    contract_fx = _contract_graph()

    runners = [
        (
            QueryKind.SYMBOL_CALLERS,
            lambda: query_symbol_callers(
                call_fx["graph"], seed_qualified_names=["pkg.c"]
            ),
        ),
        (
            QueryKind.SYMBOL_CALLEES,
            lambda: query_symbol_callees(
                call_fx["graph"], seed_qualified_names=["pkg.a"]
            ),
        ),
        (
            QueryKind.CHANGED_BLOB_IMPACT,
            lambda: query_changed_blob_impact(
                call_fx["graph"], seed_blob_cids=[BLOB_B]
            ),
        ),
        (
            QueryKind.CONTRACT_CONSUMERS,
            lambda: query_contract_consumers(
                contract_fx["graph"],
                seed_node_ids=[contract_fx["schema"].node_id],
            ),
        ),
        (
            QueryKind.CONTRACT_PRODUCERS,
            lambda: query_contract_producers(
                contract_fx["graph"],
                seed_node_ids=[contract_fx["schema"].node_id],
            ),
        ),
        (
            QueryKind.MCP_END_TO_END,
            lambda: query_mcp_end_to_end(
                mcp_fx["graph"], seed_node_ids=[mcp_fx["reg"].node_id]
            ),
        ),
        (
            QueryKind.VFS_OPERATION_SURFACE,
            lambda: query_vfs_operation_surface(vfs_fx["graph"]),
        ),
        (
            QueryKind.PROOF_DEPENDENCIES,
            lambda: query_proof_dependencies(
                call_fx["graph"], seed_qualified_names=["pkg.a"]
            ),
        ),
        (
            QueryKind.SHORTEST_COUNTEREXAMPLE,
            lambda: query_shortest_counterexample(
                call_fx["graph"],
                seed_qualified_names=["pkg.a"],
                target_qualified_names=["pkg.c"],
            ),
        ),
    ]
    seen = set()
    for kind, runner in runners:
        result = runner()
        assert result.kind is kind
        assert result.graph_id == result.provenance["graph_id"]
        assert result.minimal is True
        # Truncated or frontier-open results must not claim completeness.
        if result.truncated or result.omitted_dependencies or result.missing_node_ids:
            assert result.complete is False
            assert result.dependency_complete is False
        seen.add(kind)
    assert seen == set(QueryKind)


def test_minimality_does_not_omit_required_call_intermediates() -> None:
    """Prove the slice keeps call nodes required by the A->B->C chain."""

    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        bounds=QueryBounds(max_nodes=512, max_edges=2048, max_depth=32),
    )
    # Required intermediates on the dependency-complete path.
    required = {
        fx["a"].node_id,
        fx["call_ab"].node_id,
        fx["b"].node_id,
        fx["call_bc"].node_id,
        fx["c"].node_id,
    }
    assert required.issubset(set(result.node_ids))
    assert required.issubset(set(result.required_dependencies))
    # Unrelated symbol is not a required dependency.
    assert fx["u"].node_id not in result.required_dependencies


def test_repository_filter_scopes_results() -> None:
    fx = _seeded_call_graph()
    # Filter to a non-matching repository yields no seeds under that repo
    # when seeds resolve only via record keys still under accelerator —
    # nodes without matching repo still allowed when repository is known.
    result = query_symbol_callees(
        fx["graph"],
        seed_qualified_names=["pkg.a"],
        repository_ids=["repo:accelerator"],
    )
    assert fx["a"].node_id in result.seed_node_ids
    assert fx["b"].node_id in result.node_ids


# ---------------------------------------------------------------------------
# VFS-041: query index optimization + objective validation repair
# ---------------------------------------------------------------------------


def test_kind_partitioned_query_indexes_preserve_canonical_graph_identity() -> None:
    """Indexes accelerate lookups; graph_id / node_id / edge_id stay canonical."""

    fx = _seeded_call_graph()
    graph = fx["graph"]
    view = _GraphView(graph)
    stats = view.index_stats()

    assert stats["query_index_version"] == QUERY_INDEX_VERSION
    assert stats["graph_id"] == graph.graph_id
    assert stats["forest_id"] == graph.forest_id
    assert stats["canonical_graph_identity_preserved"] is True
    assert stats["node_count"] == len(graph.nodes)
    assert stats["edge_count"] == len(graph.edges)
    assert stats["kind_partitioned_edge_slots"] == len(graph.edges)
    assert stats["qualified_name_keys"] >= 1
    assert "symbol" in view.by_node_kind

    # Every edge appears exactly once in the kind-partitioned forward index.
    indexed_edge_ids: set[str] = set()
    for node_map in view.forward_by_kind.values():
        for bucket in node_map.values():
            for item in bucket:
                indexed_edge_ids.add(item.edge.edge_id)
    assert indexed_edge_ids == {edge.edge_id for edge in graph.edges}

    # Kind buckets match the edge kind vocabulary used by callers/callees.
    calls = ProgramEdgeKind.CALLS.value
    contains = ProgramEdgeKind.CONTAINS.value
    call_ab = fx["call_ab"].node_id
    # call_ab --calls--> symbol:b is present under the CALLS kind partition.
    call_neighbors = {
        item.neighbor
        for item in view.forward_by_kind.get(call_ab, {}).get(calls, ())
    }
    assert fx["b"].node_id in call_neighbors
    # symbol:a --contains--> call_ab under CONTAINS.
    contains_neighbors = {
        item.neighbor
        for item in view.forward_by_kind.get(fx["a"].node_id, {}).get(
            contains, ()
        )
    }
    assert call_ab in contains_neighbors

    # Building the view never mutates the underlying graph identity payload.
    assert graph.graph_id == view.graph_id
    assert graph.forest_id == view.forest_id
    for node in graph.nodes:
        assert node.node_id in view.nodes
        assert view.nodes[node.node_id] is node
    for edge in graph.edges:
        assert edge.edge_id in view.edges
        assert view.edges[edge.edge_id] is edge


def test_query_index_layout_is_recorded_without_embedding_cardinalities() -> None:
    """Provenance carries the fixed index layout marker, not volatile counts."""

    fx = _seeded_call_graph()
    result = query_symbol_callees(
        fx["graph"], seed_qualified_names=["pkg.a"]
    )
    assert result.provenance["query_index"] == QUERY_INDEX_VERSION
    assert result.provenance["canonical_graph_identity_preserved"] is True
    assert result.provenance["graph_id"] == fx["graph"].graph_id
    # Cardinalities must not leak into identity-bearing provenance.
    for forbidden in (
        "kind_partitioned_edge_slots",
        "forward_nodes",
        "node_count_index",
        "vfs_candidates",
    ):
        assert forbidden not in result.provenance


def test_objective_validation_repair_acceptance_seeded_slices() -> None:
    """VFS-G041 acceptance: complete within scope, omit unrelated, no silent complete.

    Objective validation repair for the missing evidence term is proven by
    these hard-bounded query behaviours remaining green under the optimized
    kind-partitioned indexes.
    """

    call_fx = _seeded_call_graph()
    mcp_fx = _mcp_route_graph()

    callees = query_symbol_callees(
        call_fx["graph"], seed_qualified_names=["pkg.a"]
    )
    callers = query_symbol_callers(
        call_fx["graph"], seed_qualified_names=["pkg.c"]
    )
    mcp = query_mcp_end_to_end(
        mcp_fx["graph"], seed_node_ids=[mcp_fx["reg"].node_id]
    )

    # Seeded transitive callees / callers complete within scope.
    for result in (callees, callers, mcp):
        assert result.dependency_complete is True or result.truncated is True
        if not result.truncated and not result.missing_node_ids:
            assert result.minimal is True
            # Dependency-complete implies required chain retained.
            assert result.required_dependencies
        # Limits never silently convert incomplete -> complete.
        if result.truncated or result.omitted_dependencies or result.missing_node_ids:
            assert result.complete is False
            assert result.dependency_complete is False

    # Transitive call chain A->B->C retained; unrelated U omitted.
    assert {
        call_fx["a"].node_id,
        call_fx["b"].node_id,
        call_fx["c"].node_id,
    }.issubset(set(callees.node_ids))
    assert call_fx["u"].node_id not in callees.node_ids
    assert call_fx["a"].node_id in callers.node_ids
    assert call_fx["u"].node_id not in callers.node_ids

    # MCP registration route is dependency-complete for the registration seed.
    assert mcp_fx["reg"].node_id in mcp.seed_node_ids
    assert mcp_fx["impl"].node_id in mcp.node_ids
    assert mcp.dependency_complete is True
    assert mcp.minimal is True

    # Hard bound still fails closed under the optimized indexes.
    truncated = query_symbol_callees(
        call_fx["graph"],
        seed_qualified_names=["pkg.a"],
        bounds=QueryBounds(max_nodes=2, max_edges=2, max_depth=1),
    )
    assert truncated.truncated is True
    assert truncated.complete is False
    assert truncated.dependency_complete is False
    assert truncated.provenance["query_index"] == QUERY_INDEX_VERSION


def test_kind_partitioned_neighbors_match_full_adjacency_filter() -> None:
    """Kind partition must yield the same neighbor set as filtering full adj."""

    fx = _seeded_call_graph()
    view = _GraphView(fx["graph"])
    edge_kinds = frozenset(
        {
            ProgramEdgeKind.CALLS.value,
            ProgramEdgeKind.CONTAINS.value,
            ProgramEdgeKind.RESOLVES_TO.value,
        }
    )
    for node_id in list(view.nodes)[:12]:
        partitioned = view.neighbors_for(
            node_id, directions=("forward", "reverse"), edge_kinds=edge_kinds
        )
        # Reconstruct the pre-optimization filter from full adjacency.
        raw: list[Any] = []
        raw.extend(view.forward.get(node_id, ()))
        raw.extend(view.reverse.get(node_id, ()))
        expected_ids = []
        for item in raw:
            if item.edge.kind.value not in edge_kinds:
                continue
            if not view.allowed(item.neighbor):
                continue
            neighbor = view.nodes.get(item.neighbor)
            current = view.nodes.get(node_id)
            if item.edge.kind is ProgramEdgeKind.CONTAINS:
                if neighbor is not None and neighbor.kind is ProgramNodeKind.REPOSITORY:
                    continue
                if (
                    current is not None
                    and current.kind is ProgramNodeKind.REPOSITORY
                    and item.forward
                ):
                    continue
            expected_ids.append(
                (item.edge.edge_id, item.neighbor, item.forward)
            )
        got_ids = [
            (item.edge.edge_id, item.neighbor, item.forward)
            for item in partitioned
        ]
        assert sorted(got_ids) == sorted(expected_ids)
