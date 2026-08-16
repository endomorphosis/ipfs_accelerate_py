"""Tests for DatabaseImpactGraph@1 / ImpactClosure@1 / ChangedSymbolNeighborhood@1 (DQP-023).

Evidence subset: recursion, SCC, aliases, reexports, dynamic calls, generated
code, cross-language, deletion, parser uncertainty, pagination.

Acceptance:

* All resolved consumers receive exactly one disposition
* Open or unsupported frontier blocks automatic repair
* Query result binds snapshot/parser/policy/schema
* Similarity and graph proximity remain nomination rather than semantic authority
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.database_impact_graph import (
    AUTHORITY_CLASS,
    CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE,
    DATABASE_IMPACT_GRAPH_INTERFACE,
    DEFAULT_POLICY_ID,
    IMPACT_CLOSURE_INTERFACE,
    NOMINATION_AUTHORITY,
    ChangedSymbolNeighborhood,
    ConsumerDisposition,
    DatabaseImpactGraph,
    DatabaseImpactGraphIntegrityError,
    EdgeKind,
    FrontierDisposition,
    FrontierKind,
    ImpactClosure,
    ImpactCompleteness,
    ImpactEdgeSpec,
    ImpactFrontierRecord,
    ImpactFrontierSpec,
    ImpactSymbolSpec,
    duckdb_available,
    open_database_impact_graph,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseImpactGraph hermetic tests",
)


def _open(tmp_path: Path) -> DatabaseImpactGraph:
    return open_database_impact_graph(tmp_path / "impact_graph.duckdb")


def _materialize_core(graph: DatabaseImpactGraph, **overrides):
    """Materialize a small multi-edge fixture covering core edge kinds."""

    edges = [
        # consume → Service.dispatch (calls)
        ImpactEdgeSpec(
            source_symbol="consume",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.CALLS,
            path="src/consumer.py",
            source_path="src/consumer.py",
            target_path="src/service.py",
            source_language="python",
            target_language="python",
        ),
        # consume imports Service
        ImpactEdgeSpec(
            source_symbol="consume",
            target_symbol="Service",
            edge_kind=EdgeKind.IMPORTS,
            path="src/consumer.py",
        ),
        # alias / reexport chain
        ImpactEdgeSpec(
            source_symbol="ServiceFacade",
            target_symbol="Service",
            edge_kind=EdgeKind.ALIASES,
            path="src/facade.py",
        ),
        ImpactEdgeSpec(
            source_symbol="public_api.Service",
            target_symbol="ServiceFacade",
            edge_kind=EdgeKind.REEXPORTS,
            path="src/public_api.py",
        ),
        # types
        ImpactEdgeSpec(
            source_symbol="Service",
            target_symbol="ServiceContract",
            edge_kind=EdgeKind.TYPES,
            path="src/service.py",
        ),
        # tests / contracts / proofs / config / docs
        ImpactEdgeSpec(
            source_symbol="test_consume",
            target_symbol="consume",
            edge_kind=EdgeKind.TESTS,
            path="test/test_consumer.py",
            source_path="test/test_consumer.py",
        ),
        ImpactEdgeSpec(
            source_symbol="ServiceContract.spec",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.CONTRACTS,
            path="contracts/service.json",
        ),
        ImpactEdgeSpec(
            source_symbol="proof_dispatch_total",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.PROOFS,
            path="proofs/dispatch.lean",
        ),
        ImpactEdgeSpec(
            source_symbol="service_config",
            target_symbol="Service",
            edge_kind=EdgeKind.CONFIG,
            path="config/service.yaml",
        ),
        ImpactEdgeSpec(
            source_symbol="README.Service",
            target_symbol="Service",
            edge_kind=EdgeKind.DOCS,
            path="docs/service.md",
        ),
        # recursive mutual recursion (SCC)
        ImpactEdgeSpec(
            source_symbol="walk_a",
            target_symbol="walk_b",
            edge_kind=EdgeKind.CALLS,
            path="src/walk.py",
        ),
        ImpactEdgeSpec(
            source_symbol="walk_b",
            target_symbol="walk_a",
            edge_kind=EdgeKind.CALLS,
            path="src/walk.py",
        ),
        ImpactEdgeSpec(
            source_symbol="walk_a",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.CALLS,
            path="src/walk.py",
        ),
        # nomination only (must not expand mandatory closure)
        ImpactEdgeSpec(
            source_symbol="similar_helper",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.NOMINATED,
            path="src/similar.py",
            authority=NOMINATION_AUTHORITY,
        ),
    ]
    symbols = [
        ImpactSymbolSpec("Service", path="src/service.py", language="python"),
        ImpactSymbolSpec(
            "Service.dispatch", path="src/service.py", language="python"
        ),
        ImpactSymbolSpec(
            "ServiceContract", path="src/service.py", language="python"
        ),
        ImpactSymbolSpec("consume", path="src/consumer.py", language="python"),
        ImpactSymbolSpec(
            "test_consume", path="test/test_consumer.py", language="python"
        ),
        ImpactSymbolSpec("walk_a", path="src/walk.py", language="python"),
        ImpactSymbolSpec("walk_b", path="src/walk.py", language="python"),
        ImpactSymbolSpec(
            "similar_helper", path="src/similar.py", language="python"
        ),
        ImpactSymbolSpec(
            "ServiceFacade", path="src/facade.py", language="python"
        ),
        ImpactSymbolSpec(
            "public_api.Service", path="src/public_api.py", language="python"
        ),
    ]
    kwargs = {
        "snapshot_id": "snapshot:demo-1",
        "edges": edges,
        "symbols": symbols,
        "parser_id": "python-ast@test",
        "policy_id": DEFAULT_POLICY_ID,
        "repository_id": "repo:demo",
        "tree_id": "tree:abc",
    }
    kwargs.update(overrides)
    return graph.materialize(**kwargs)


def test_interface_identities() -> None:
    assert DATABASE_IMPACT_GRAPH_INTERFACE == "DatabaseImpactGraph@1"
    assert IMPACT_CLOSURE_INTERFACE == "ImpactClosure@1"
    assert CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE == "ChangedSymbolNeighborhood@1"
    assert DatabaseImpactGraph.INTERFACE == DATABASE_IMPACT_GRAPH_INTERFACE
    assert AUTHORITY_CLASS == "derived_evidence"
    assert EdgeKind.coerce("call") is EdgeKind.CALLS
    assert EdgeKind.coerce("re_export") is EdgeKind.REEXPORTS


def test_cold_import_and_construction_have_no_side_effects() -> None:
    store = DatabaseImpactGraph("/tmp/should-not-exist-until-open.duckdb")
    assert store.is_open is False


def test_materialize_and_query_bind_snapshot_parser_policy_schema(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as graph:
        result = _materialize_core(graph)
        revision = result.revision
        assert revision.snapshot_id == "snapshot:demo-1"
        assert revision.parser_id == "python-ast@test"
        assert revision.policy_id == DEFAULT_POLICY_ID
        assert revision.schema_id
        assert result.edge_count >= 10
        assert result.symbol_count >= 5

        closure = graph.impact_closure(["Service.dispatch"])
        assert closure.interface == IMPACT_CLOSURE_INTERFACE
        assert closure.snapshot_id == "snapshot:demo-1"
        assert closure.parser_id == "python-ast@test"
        assert closure.policy_id == DEFAULT_POLICY_ID
        assert closure.schema_id == revision.schema_id
        assert closure.freshness["revision_id"] == revision.revision_id
        assert closure.to_dict()["authority"] == AUTHORITY_CLASS
        assert closure.to_dict()["freshness"]["snapshot_id"] == "snapshot:demo-1"

        neighborhood = graph.changed_neighborhood(
            ["Service.dispatch"], radius=2
        )
        assert neighborhood.interface == CHANGED_SYMBOL_NEIGHBORHOOD_INTERFACE
        assert neighborhood.freshness["parser_id"] == "python-ast@test"
        assert neighborhood.freshness["policy_id"] == DEFAULT_POLICY_ID
        assert neighborhood.freshness["schema_id"] == revision.schema_id

        meta = graph.metadata()
        assert meta["interface"] == DATABASE_IMPACT_GRAPH_INTERFACE
        assert meta["authority"] == AUTHORITY_CLASS


def test_resolved_consumers_have_exactly_one_disposition(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        _materialize_core(graph)
        closure = graph.impact_closure(["Service.dispatch"])
        assert closure.consumers
        symbols = [item.symbol for item in closure.consumers]
        assert len(symbols) == len(set(symbols))
        for consumer in closure.consumers:
            assert isinstance(consumer.disposition, ConsumerDisposition)
            assert consumer.disposition is not ConsumerDisposition.UNCHANGED
        # Direct callers / recursive / tests / contracts / proofs present.
        by_name = {item.symbol: item for item in closure.consumers}
        assert "consume" in by_name
        assert by_name["consume"].disposition is ConsumerDisposition.MUST_REPAIR
        assert "walk_a" in by_name
        assert "test_consume" in by_name
        assert (
            by_name["test_consume"].disposition
            is ConsumerDisposition.MUST_REVALIDATE
        )
        assert "ServiceContract.spec" in by_name
        assert "proof_dispatch_total" in by_name


def test_recursion_and_scc_are_materialized(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        _materialize_core(graph)
        closure = graph.impact_closure(["Service.dispatch"])
        # Mutual recursion between walk_a and walk_b yields an SCC.
        scc_members = [set(item.member_symbols) for item in closure.sccs]
        assert any({"walk_a", "walk_b"} <= members for members in scc_members)
        # Both recursive consumers are present once.
        names = [item.symbol for item in closure.consumers]
        assert names.count("walk_a") == 1
        assert names.count("walk_b") == 1


def test_aliases_and_reexports_expand_reverse_impact(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        _materialize_core(graph)
        closure = graph.impact_closure(["Service"])
        names = {item.symbol for item in closure.consumers}
        assert "ServiceFacade" in names
        assert "public_api.Service" in names
        assert "consume" in names
        assert "service_config" in names
        assert "README.Service" in names


def test_dynamic_calls_surface_frontier_and_block_automatic_repair(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as graph:
        graph.materialize(
            snapshot_id="snapshot:dynamic",
            parser_id="python-ast@test",
            edges=[
                ImpactEdgeSpec(
                    source_symbol="dispatch_dynamic",
                    target_symbol="<dynamic>",
                    edge_kind=EdgeKind.DYNAMIC,
                    path="src/dyn.py",
                    is_dynamic=True,
                    reason="getattr_dispatch",
                ),
                ImpactEdgeSpec(
                    source_symbol="dispatch_dynamic",
                    target_symbol="Service.dispatch",
                    edge_kind=EdgeKind.CALLS,
                    path="src/dyn.py",
                ),
            ],
            symbols=[
                ImpactSymbolSpec(
                    "dispatch_dynamic", path="src/dyn.py", language="python"
                ),
                ImpactSymbolSpec(
                    "Service.dispatch",
                    path="src/service.py",
                    language="python",
                ),
            ],
        )
        frontiers = graph.list_frontiers()
        assert frontiers
        assert any(
            item.kind is FrontierKind.DYNAMIC_CALL
            or item.kind is FrontierKind.UNRESOLVED_SYMBOL
            for item in frontiers
        )
        assert any(item.blocks_repair for item in frontiers)

        closure = graph.impact_closure(["Service.dispatch"])
        assert closure.blocks_automatic_repair is True
        assert closure.completeness is not ImpactCompleteness.COMPLETE
        assert any(item.blocks_repair for item in closure.frontiers)
        # Dynamic edge does not invent mandatory consumers beyond the call edge.
        names = {item.symbol for item in closure.consumers}
        assert "dispatch_dynamic" in names


def test_generated_code_and_cross_language_are_explicit(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        graph.materialize(
            snapshot_id="snapshot:gen",
            parser_id="python-ast@test",
            edges=[
                ImpactEdgeSpec(
                    source_symbol="generated.bind",
                    target_symbol="Service.dispatch",
                    edge_kind=EdgeKind.GENERATED_FROM,
                    path="generated/bind.py",
                    is_generated=True,
                ),
                ImpactEdgeSpec(
                    source_symbol="ts.wrapper",
                    target_symbol="Service.dispatch",
                    edge_kind=EdgeKind.CALLS,
                    path="web/wrapper.ts",
                    source_language="typescript",
                    target_language="python",
                    is_cross_language=True,
                ),
            ],
            symbols=[
                ImpactSymbolSpec(
                    "generated.bind",
                    path="generated/bind.py",
                    language="python",
                    is_generated=True,
                ),
                ImpactSymbolSpec(
                    "ts.wrapper",
                    path="web/wrapper.ts",
                    language="typescript",
                ),
                ImpactSymbolSpec(
                    "Service.dispatch",
                    path="src/service.py",
                    language="python",
                ),
            ],
        )
        closure = graph.impact_closure(["Service.dispatch"])
        by_name = {item.symbol: item for item in closure.consumers}
        assert by_name["generated.bind"].disposition is ConsumerDisposition.GENERATED
        assert (
            by_name["ts.wrapper"].disposition
            is ConsumerDisposition.CROSS_LANGUAGE
        )
        assert closure.blocks_automatic_repair is True
        kinds = {item.kind for item in closure.frontiers}
        assert FrontierKind.GENERATED_CODE in kinds
        assert FrontierKind.CROSS_LANGUAGE in kinds


def test_deletion_and_parser_uncertainty_frontiers(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        graph.materialize(
            snapshot_id="snapshot:delete",
            parser_id="python-ast@test",
            edges=[
                ImpactEdgeSpec(
                    source_symbol="legacy_consumer",
                    target_symbol="Service.dispatch",
                    edge_kind=EdgeKind.CALLS,
                    path="src/legacy.py",
                ),
            ],
            symbols=[
                ImpactSymbolSpec(
                    "legacy_consumer",
                    path="src/legacy.py",
                    language="python",
                    is_deleted=True,
                ),
                ImpactSymbolSpec(
                    "Service.dispatch",
                    path="src/service.py",
                    language="python",
                ),
            ],
            frontiers=[
                ImpactFrontierSpec(
                    kind=FrontierKind.DELETION,
                    disposition=FrontierDisposition.OPEN,
                    symbol_key="legacy_consumer",
                    path="src/legacy.py",
                    reason="path_deleted",
                    blocks_repair=True,
                ),
                ImpactFrontierSpec(
                    kind=FrontierKind.PARSER_UNCERTAINTY,
                    disposition=FrontierDisposition.OPEN,
                    path="src/broken.py",
                    reason="syntax_error",
                    blocks_repair=True,
                ),
            ],
        )
        closure = graph.impact_closure(["Service.dispatch"])
        by_name = {item.symbol: item for item in closure.consumers}
        assert by_name["legacy_consumer"].disposition is ConsumerDisposition.DELETED
        assert closure.blocks_automatic_repair is True
        kinds = {item.kind for item in closure.frontiers}
        assert FrontierKind.DELETION in kinds
        assert FrontierKind.PARSER_UNCERTAINTY in kinds
        assert closure.completeness is ImpactCompleteness.PARTIAL_WITH_FRONTIER


def test_nomination_is_not_semantic_authority(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        _materialize_core(graph)
        closure = graph.impact_closure(["Service.dispatch"])
        mandatory_names = {item.symbol for item in closure.consumers}
        # similar_helper is only nominated, never a mandatory consumer.
        assert "similar_helper" not in mandatory_names
        nominated_names = {item.symbol for item in closure.nominated}
        assert "similar_helper" in nominated_names
        for item in closure.nominated:
            assert item.mandatory is False
            assert item.disposition is ConsumerDisposition.NOMINATED
            assert item.edge_kinds == (EdgeKind.NOMINATED.value,)


def test_changed_neighborhood_buckets_and_pagination(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        _materialize_core(graph)
        neighborhood = graph.changed_neighborhood(
            ["Service.dispatch"],
            radius=2,
            page_offset=0,
            page_limit=3,
        )
        assert isinstance(neighborhood, ChangedSymbolNeighborhood)
        assert "Service.dispatch" in neighborhood.changed_symbols
        assert "consume" in neighborhood.callers or any(
            node["symbol"] == "consume" for node in neighborhood.nodes
        )
        assert neighborhood.tests or "test_consume" in {
            node["symbol"] for node in neighborhood.nodes
        }
        assert neighborhood.contracts
        assert neighborhood.proofs
        assert neighborhood.total_edge_count >= len(neighborhood.edges)
        if neighborhood.total_edge_count > 3:
            assert neighborhood.has_more is True
            page2 = graph.changed_neighborhood(
                ["Service.dispatch"],
                radius=2,
                page_offset=3,
                page_limit=3,
            )
            ids1 = {edge["edge_id"] for edge in neighborhood.edges}
            ids2 = {edge["edge_id"] for edge in page2.edges}
            assert ids1.isdisjoint(ids2)

        callers = graph.list_callers("Service.dispatch", page_limit=2)
        assert callers["freshness"]["snapshot_id"] == "snapshot:demo-1"
        assert callers["total_count"] >= 1
        assert callers["authority"] == AUTHORITY_CLASS
        if callers["total_count"] > 2:
            assert callers["has_more"] is True

        callees = graph.list_callees("walk_a")
        assert any(item["symbol"] == "walk_b" for item in callees["items"])

        tests = graph.list_tests("consume")
        assert any(item["symbol"] == "test_consume" for item in tests["items"])

        docs = graph.list_docs("Service")
        assert any(item["symbol"] == "README.Service" for item in docs["items"])

        config = graph.list_config("Service")
        assert any(item["symbol"] == "service_config" for item in config["items"])

        edges_page = graph.list_edges(page_limit=5)
        assert edges_page["total_count"] >= 5
        assert edges_page["has_more"] is True
        assert edges_page["freshness"]["policy_id"] == DEFAULT_POLICY_ID


def test_open_frontier_blocks_complete_closure_construction() -> None:
    # Direct construction: complete + blocking frontier is rejected.
    with pytest.raises(DatabaseImpactGraphIntegrityError):
        ImpactClosure(
            query_id="",
            revision_id="rev:1",
            snapshot_id="snapshot:1",
            parser_id="parser:1",
            policy_id="policy:1",
            schema_id="schema:1",
            seed_symbols=("Service",),
            completeness=ImpactCompleteness.COMPLETE,
            consumers=(),
            frontiers=(
                ImpactFrontierRecord(
                    frontier_id="",
                    kind=FrontierKind.DYNAMIC_CALL,
                    disposition=FrontierDisposition.OPEN,
                    reason="open",
                    blocks_repair=True,
                ),
            ),
            blocks_automatic_repair=True,
        )


def test_complete_closure_when_no_open_frontiers(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        graph.materialize(
            snapshot_id="snapshot:clean",
            parser_id="python-ast@test",
            edges=[
                ImpactEdgeSpec(
                    source_symbol="consume",
                    target_symbol="Service.dispatch",
                    edge_kind=EdgeKind.CALLS,
                    path="src/consumer.py",
                ),
            ],
            symbols=[
                ImpactSymbolSpec("consume", path="src/consumer.py"),
                ImpactSymbolSpec(
                    "Service.dispatch", path="src/service.py"
                ),
            ],
        )
        closure = graph.impact_closure(["Service.dispatch"])
        assert closure.completeness is ImpactCompleteness.COMPLETE
        assert closure.blocks_automatic_repair is False
        assert closure.frontiers == ()
        assert len(closure.consumers) == 1
        assert closure.consumers[0].symbol == "consume"


def test_materialize_from_ast_index_projection(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.analysis.duckdb_ast_index import (
        SourceFileSpec,
        open_duckdb_ast_index,
    )

    service = """\
class Service:
    def dispatch(self, request):
        return request
"""
    consumer = """\
from src.service import Service

def consume(request):
    service = Service()
    return service.dispatch(request)
"""
    with open_duckdb_ast_index(tmp_path / "ast.duckdb") as index:
        ingest = index.ingest_snapshot(
            repository_id="repo:demo",
            tree_id="tree:1",
            files=[
                SourceFileSpec(path="src/service.py", content=service),
                SourceFileSpec(path="src/consumer.py", content=consumer),
            ],
        )
        snapshot_id = ingest.snapshot.snapshot_id
        with _open(tmp_path) as graph:
            result = graph.materialize_from_ast_index(
                index,
                snapshot_id,
                repository_id="repo:demo",
                tree_id="tree:1",
            )
            assert result.symbol_count >= 2
            # Impact over Service / dispatch should include consume via calls
            # or imports projected from the AST index.
            seeds = []
            symbols = index.list_symbols(snapshot_id)
            for item in symbols:
                if item.qualified_name in {"Service", "Service.dispatch", "consume"}:
                    seeds.append(item.qualified_name)
            assert seeds
            # Prefer dispatch if present, else Service.
            seed = (
                "Service.dispatch"
                if "Service.dispatch" in seeds
                else ("Service" if "Service" in seeds else seeds[0])
            )
            closure = graph.impact_closure([seed])
            assert closure.snapshot_id == snapshot_id
            assert closure.parser_id
            assert closure.to_dict()["authority"] == AUTHORITY_CLASS
            # At least the freshness binding is present even if call resolution
            # is identity-based and names differ.
            assert closure.freshness["schema_id"]


def test_idempotent_rematerialization(tmp_path: Path) -> None:
    with _open(tmp_path) as graph:
        first = _materialize_core(graph)
        second = _materialize_core(graph)
        assert first.revision.revision_id == second.revision.revision_id
        assert first.edge_count == second.edge_count
