"""Native in-memory source/link controls; no provider or task acceptance claims."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_dependency_proposal import (
    nominate_typed_semantic_dependencies,
)
from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes,
    cid_for_bytes,
    cid_for_structured,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.identity import (
    stable_symbol_id,
    symbol_version_cid,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
    AnalysisConfidence,
    ArtifactRecord,
    DependencyEdge,
    RelationType,
    RepositoryState,
    SymbolKind,
    SymbolRecord,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.python_analysis import (
    analyze_python_source,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.symbol_graph import (
    resolve_edge_targets,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.api import (
    build_semantic_state,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SemanticLinkNode,
    SemanticStateBundle,
    SortedPairIndex,
    SymbolFactNode,
    SymbolMerkleNode,
)

REPOSITORY = "repo:typed-dependency-fixture"
GRAPH_ROOT = "typed-dependency-graph:fixture"
SOURCES = {
    "pkg/a.py": (
        "def helper(value):\n    return value + 1\n\n"
        "def target(value):\n    return helper(value)\n\n"
        "def incoming_caller(value):\n    return target(value)\n"
    ),
    "pkg/b.py": (
        "def helper(value):\n    return value + 99\n\n"
        "def other_target(value):\n    return helper(value)\n"
    ),
}


def _cid(label):
    return cid_for_structured({"schema": "authored-typed-dependency-fixture@1", "label": label})


def _aux_symbol(name, kind=SymbolKind.FUNCTION):
    source = f"def {name}(value):\n    return value\n"
    node = ast.parse(source).body[0]
    qualified = "checks." + name
    stable = stable_symbol_id(REPOSITORY, "python", "checks.py", qualified, kind, "checks")
    signature = {"parameters": ["value"]}
    version = symbol_version_cid(stable, node, signature, [], {})
    return SymbolRecord(
        stable, version, REPOSITORY, "python", "checks.py", qualified, kind, "checks",
        cid_for_bytes(source.encode()), None, AnalysisConfidence.EXACT,
        signature, [], {}, {}, normalized_ast=node,
    )


def _edge(source, target, relation):
    return DependencyEdge(source.stable_id, target.stable_id, relation, "authored-static",
                          AnalysisConfidence.EXACT, "authored@1")


def _build_source_bundle(sources, *, repository=REPOSITORY, reverse=False, extras=(), extra_edges=()):
    analyses = [analyze_python_source(text, path, repository)
                for path, text in sorted(sources.items(), reverse=reverse)]
    symbols = [symbol for analysis in analyses for symbol in analysis.symbol_records] + list(extras)
    edges = [edge for analysis in analyses for edge in analysis.edges] + list(extra_edges)
    if reverse:
        symbols.reverse()
        edges.reverse()
    resolved = resolve_edge_targets(edges, symbols)
    bundle = build_semantic_state(RepositoryState(repository, tuple(symbols), edges=resolved))
    return bundle, {symbol.qualified_name: symbol for symbol in symbols}


def _fixture(*, sources=None, reverse=False):
    captured = SOURCES if sources is None else sources
    proof = _aux_symbol("proof")
    public_test = _aux_symbol("public_test", SymbolKind.TEST)
    fixture = _aux_symbol("fixture", SymbolKind.FIXTURE)
    # Resolve the authored auxiliary edges against the same AST identity used
    # by the final build, without assembling an unused preliminary bundle.
    target = next(symbol for symbol in analyze_python_source(
        captured["pkg/a.py"], "pkg/a.py", REPOSITORY).symbol_records
        if symbol.qualified_name == "pkg.a.target")
    extras = (proof, public_test, fixture)
    relations = (
        _edge(target, proof, RelationType.PROOF_DEPENDS_ON),
        _edge(target, public_test, RelationType.TESTED_BY),
        _edge(public_test, fixture, RelationType.USES_FIXTURE),
    )
    return _build_source_bundle(captured, reverse=reverse, extras=extras, extra_edges=relations)


@pytest.fixture(scope="module")
def native():
    return _fixture()


def _nominate(bundle, seeds, **limits):
    return nominate_typed_semantic_dependencies(
        bundle=bundle, root_cid=bundle.root.root_cid, graph_root_id=GRAPH_ROOT,
        seed_symbol_ids=tuple(seeds), **limits,
    )


def _index(bundle, field):
    index_cid = getattr(bundle.root, field)
    return SortedPairIndex.from_dict({**json.loads(bundle.blocks[index_cid]), "index_cid": index_cid})


def _record(bundle, field, key, cls, identity_field):
    record_cid = dict(_index(bundle, field).pairs)[key]
    return cls.from_dict({**json.loads(bundle.blocks[record_cid]), identity_field: record_cid})


def _links(bundle):
    return [_record(bundle, "semantic_link_index_cid", key, SemanticLinkNode, "link_cid")
            for key, _ in _index(bundle, "semantic_link_index_cid").pairs]


def _node(bundle, stable):
    return _record(bundle, "symbol_node_index_cid", stable, SymbolMerkleNode, "node_cid")


def _fact(bundle, stable):
    return _record(bundle, "symbol_fact_index_cid", stable, SymbolFactNode, "fact_cid")


def _with_indexes(bundle, changes, records=()):
    """Rehash canonical records/indexes so deeper binding checks are exercised."""
    blocks = dict(bundle.blocks)
    root_changes = {}
    for field, pairs in changes.items():
        index = SortedPairIndex(pairs)
        blocks[index.index_cid] = canonical_dag_json_bytes(index.identity_payload())
        root_changes[field] = index.index_cid
    for record, identity_field in records:
        blocks[getattr(record, identity_field)] = canonical_dag_json_bytes(record.identity_payload())
    return SemanticStateBundle(replace(bundle.root, **root_changes), blocks)


def _with_link(bundle, old, changed):
    pairs = dict(_index(bundle, "semantic_link_index_cid").pairs)
    del pairs[old.edge_id]
    pairs[changed.edge_id] = changed.link_cid
    node_pairs = dict(_index(bundle, "symbol_node_index_cid").pairs)
    records = [(changed, "link_cid")]
    for stable in node_pairs:
        node = _node(bundle, stable)
        changed_node = replace(
            node,
            incoming_link_cids=[changed.link_cid if value == old.link_cid else value
                                for value in node.incoming_link_cids],
            outgoing_link_cids=[changed.link_cid if value == old.link_cid else value
                                for value in node.outgoing_link_cids],
        )
        node_pairs[stable] = changed_node.node_cid
        records.append((changed_node, "node_cid"))
    return _with_indexes(bundle, {"semantic_link_index_cid": list(pairs.items()),
                                  "symbol_node_index_cid": list(node_pairs.items())}, records)


def test_module_collision_and_symbolic_global_read_do_not_change_native_call_target(native):
    bundle, names = native
    target, helper = names["pkg.a.target"].stable_id, names["pkg.a.helper"].stable_id
    report = _nominate(bundle, [target])
    assert target in report["nominee_symbol_ids"] and helper in report["nominee_symbol_ids"]
    assert names["pkg.b.helper"].stable_id not in report["nominee_symbol_ids"]
    assert names["pkg.b.other_target"].stable_id not in report["nominee_symbol_ids"]
    assert names["pkg.a.incoming_caller"].stable_id not in report["nominee_symbol_ids"]
    observations = [node["record"] for node in report["graph"]["nodes"]
                    if node["record"].get("source_stable_id") == target]
    calls = [row for row in observations if row.get("relation") == "calls"]
    reads = [row for row in observations if row.get("relation") == "reads_state"]
    assert len(calls) == len(reads) == 1
    assert calls[0]["target_stable_id"] == helper and calls[0]["target_kind"] == "symbol"
    assert reads[0]["target_stable_id"] is None and reads[0]["target_kind"] == "unresolved"
    assert reads[0]["unresolved_target"] == "global:helper"
    assert reads[0]["edge_id"] != calls[0]["edge_id"]
    unresolved = report["unresolved_state_binding_rows"]
    assert any(row["source_stable_id"] == target and row["unresolved_target"] == "global:helper"
               for row in unresolved)
    bindings = [node for node in report["graph"]["nodes"]
                if node["record"].get("obligation_kind") == "unresolved_state_binding"
                and node["record"].get("recorded_target") == "global:helper"]
    assert {node["record"]["source_module_path"] for node in bindings} == {"pkg/a.py", "pkg/b.py"}
    assert len({node["node_id"] for node in bindings}) == 2
    assert all(node["record"]["binding_scope"] == "module_global"
               and node["record"]["binding_name"] == "helper"
               and node["record"]["binding_value_verified"] is False for node in bindings)


def test_proof_test_and_fixture_links_join_the_proposal_without_semantic_authority(native):
    bundle, names = native
    report = _nominate(bundle, [names["pkg.a.target"].stable_id])
    expected = {names[name].stable_id for name in (
        "pkg.a.target", "pkg.a.helper", "checks.proof", "checks.public_test", "checks.fixture")}
    assert set(report["nominee_symbol_ids"]) == expected
    relations = {node["record"]["relation"] for node in report["graph"]["nodes"]
                 if "relation" in node["record"]}
    assert {"calls", "reads_state", "proof_depends_on", "tested_by", "uses_fixture"} <= relations
    assert report["nomination_available"] is True
    assert report["nomination_status"] == "experimental_only"
    assert report["nominee_relation_audit"]["selected_symbol_ids"] == sorted(expected)
    assert report["graph"]["root_id"] == GRAPH_ROOT
    for node in report["graph"]["nodes"]:
        assert node["authority"] == "proposal_only" and node["authoritative"] is False
    for edge in report["graph"]["edges"]:
        assert edge["authority"] == "proposal_only" and edge["authoritative"] is False
        assert edge["mandatory"] is False
    for flag in ("source_freshness_verified", "dependency_semantics_complete",
                 "full_program_completeness_verified", "source_substitution_authority",
                 "proof_authority", "execution_authority", "completion_authority", "acceptance_verified"):
        assert report[flag] is False
    encoded = json.dumps(report)
    assert "normalized_ast" not in encoded and "return value + 99" not in encoded


def test_native_edge_wrapper_ids_and_symbol_facts_are_preserved_in_graph(native):
    bundle, names = native
    report = _nominate(bundle, [names["pkg.a.target"].stable_id])
    observations = {node["record"]["edge_id"]: node["record"] for node in report["graph"]["nodes"]
                    if "edge_id" in node["record"]}
    links = {link.edge_id: link for link in _links(bundle)}
    assert set(observations) == set(links)
    for edge_id, native_link in links.items():
        assert observations[edge_id]["link_cid"] == native_link.link_cid
        assert observations[edge_id]["relation"] == native_link.relation
        assert observations[edge_id]["source_fact_cid"] == native_link.source_fact_cid
    graph_symbols = {node["record"]["native_symbol_id"]: node["record"]
                     for node in report["graph"]["nodes"] if "native_symbol_id" in node["record"]}
    assert set(graph_symbols) == set(report["full_symbol_ids"])
    for stable, row in graph_symbols.items():
        fact = _fact(bundle, stable)
        assert row["version_cid"] == fact.version_cid and row["fact_cid"] == fact.fact_cid


def test_input_order_and_shared_seeds_are_deterministic_without_mutating_bundle(native):
    bundle, names = native
    second, _ = _fixture(reverse=True)
    before = dict(bundle.blocks)
    target, helper = names["pkg.a.target"].stable_id, names["pkg.a.helper"].stable_id
    left = _nominate(bundle, [target, helper])
    right = _nominate(second, [helper, target])
    assert left == right
    assert left["seed_symbol_ids"] == sorted([target, helper])
    assert len(left["nominee_symbol_ids"]) == len(set(left["nominee_symbol_ids"]))
    assert dict(bundle.blocks) == before and type(bundle.blocks) is MappingProxyType


@pytest.mark.parametrize("source,expected", [
    ("def left(value):\n    return right(value)\n\ndef right(value):\n    return left(value)\n", {"cycles.left", "cycles.right"}),
    ("def left(value):\n    return left(value)\n", {"cycles.left"}),
])
def test_recursive_calls_terminate_without_authority_or_self_edges(source, expected):
    bundle, names = _build_source_bundle({"cycles.py": source})
    report = _nominate(bundle, [names["cycles.left"].stable_id])
    assert set(report["nominee_symbol_ids"]) == {names[name].stable_id for name in expected}
    assert report == _nominate(bundle, [names["cycles.left"].stable_id])
    assert all(edge["source"] != edge["target"] and edge["mandatory"] is False
               for edge in report["graph"]["edges"])
    assert report["execution_authority"] is False


@pytest.mark.parametrize("limits", [
    {"max_selected_symbols": 1}, {"max_graph_nodes": 1},
    {"max_graph_edges": 1}, {"max_report_bytes": 1},
])
def test_budget_overflow_cannot_return_a_partial_success_or_drop_obligations(native, limits):
    bundle, names = native
    with pytest.raises(ValueError):
        _nominate(bundle, [names["pkg.a.target"].stable_id], **limits)


@pytest.mark.parametrize("limits", [
    {"max_selected_symbols": True}, {"max_selected_symbols": 0},
    {"max_selected_symbols": 33}, {"max_graph_nodes": 2049},
    {"max_graph_edges": False}, {"max_graph_edges": 8193},
    {"max_report_bytes": 2_000_001},
])
def test_bounds_are_literal_and_cannot_expand_hard_limits(native, limits):
    bundle, names = native
    with pytest.raises(ValueError):
        _nominate(bundle, [names["pkg.a.target"].stable_id], **limits)


@pytest.mark.parametrize("root_cid", [True, "", "not-a-cid", _cid("foreign-root")])
def test_native_root_binding_cannot_be_relabelled(native, root_cid):
    bundle, names = native
    with pytest.raises(ValueError):
        nominate_typed_semantic_dependencies(
            bundle=bundle, root_cid=root_cid, graph_root_id=GRAPH_ROOT,
            seed_symbol_ids=(names["pkg.a.target"].stable_id,),
        )


@pytest.mark.parametrize("damage", ["list", "empty", "duplicate", "foreign", "bool"])
def test_explicit_seed_tuple_cannot_be_inferred_or_relaxed(native, damage):
    bundle, names = native
    target = names["pkg.a.target"].stable_id
    seeds = {"list": [target], "empty": (), "duplicate": (target, target),
             "foreign": (_cid("foreign-seed"),), "bool": (True,)}[damage]
    with pytest.raises(ValueError):
        nominate_typed_semantic_dependencies(
            bundle=bundle, root_cid=bundle.root.root_cid, graph_root_id=GRAPH_ROOT,
            seed_symbol_ids=seeds,
        )


@pytest.mark.parametrize("field", ["source_version_cid", "source_fact_cid",
                                  "target_version_cid", "target_fact_cid"])
def test_rehashed_typed_link_cannot_reuse_stale_source_or_target_fact(native, field):
    bundle, names = native
    target = names["pkg.a.target"].stable_id
    old = next(link for link in _links(bundle)
               if link.source_stable_id == target and link.relation == "calls")
    changed = _with_link(bundle, old, replace(old, **{field: _cid("stale:" + field)}))
    with pytest.raises(ValueError):
        _nominate(changed, [target])


def test_valid_link_bytes_rebound_to_wrong_index_key_are_rejected(native):
    bundle, names = native
    pairs = list(_index(bundle, "semantic_link_index_cid").pairs)
    pairs[0] = (_cid("wrong-edge-index-key"), pairs[0][1])
    changed = _with_indexes(bundle, {"semantic_link_index_cid": pairs})
    with pytest.raises(ValueError):
        _nominate(changed, [names["pkg.a.target"].stable_id])


def test_rebound_source_cannot_borrow_another_current_symbol_fact(native):
    bundle, names = native
    target = names["pkg.a.target"].stable_id
    old = next(link for link in _links(bundle)
               if link.source_stable_id == target and link.relation == "calls")
    foreign_fact = _fact(bundle, names["pkg.b.other_target"].stable_id)
    changed = _with_link(bundle, old, replace(
        old, source_stable_id=foreign_fact.stable_symbol_id,
        source_version_cid=foreign_fact.version_cid, source_fact_cid=foreign_fact.fact_cid))
    with pytest.raises(ValueError):
        _nominate(changed, [target])


def test_foreign_repository_link_cannot_be_inserted_into_current_root(native):
    bundle, names = native
    other, _ = _build_source_bundle(SOURCES, repository="repo:foreign-typed-dependency")
    injected = next(link for link in _links(other) if link.relation == "calls")
    pairs = [*_index(bundle, "semantic_link_index_cid").pairs,
             (injected.edge_id, injected.link_cid)]
    changed = _with_indexes(bundle, {"semantic_link_index_cid": pairs}, [(injected, "link_cid")])
    with pytest.raises(ValueError):
        _nominate(changed, [names["pkg.a.target"].stable_id])


@pytest.mark.parametrize("damage", ["missing_link_index", "missing_link", "tampered_link"])
def test_missing_or_tampered_root_reachable_links_are_rejected(native, damage):
    bundle, names = native
    blocks = dict(bundle.blocks)
    target = bundle.root.semantic_link_index_cid if damage == "missing_link_index" else _links(bundle)[0].link_cid
    if damage == "tampered_link":
        blocks[target] = b"{}"
    else:
        del blocks[target]
    changed = object.__new__(SemanticStateBundle)
    object.__setattr__(changed, "root", bundle.root)
    object.__setattr__(changed, "blocks", MappingProxyType(blocks))
    with pytest.raises(ValueError):
        _nominate(changed, [names["pkg.a.target"].stable_id])


def test_mutable_subclassed_and_injected_bundle_readers_are_rejected_without_callbacks(native):
    bundle, names = native
    def forbidden(*_args):
        raise AssertionError("untrusted callback must not run")
    class InjectedBundle(SemanticStateBundle):
        get_block = forbidden
    subclassed = InjectedBundle(bundle.root, bundle.blocks)
    mutable = SemanticStateBundle(bundle.root, bundle.blocks)
    object.__setattr__(mutable, "blocks", dict(mutable.blocks))
    injected = SimpleNamespace(root=bundle.root, blocks=bundle.blocks, get_block=forbidden)
    for candidate in (subclassed, mutable, injected):
        with pytest.raises(ValueError):
            _nominate(candidate, [names["pkg.a.target"].stable_id])


def test_changed_source_generation_preserves_stable_target_but_rebinds_facts(native):
    bundle, names = native
    sources = dict(SOURCES)
    sources["pkg/a.py"] = sources["pkg/a.py"].replace("value + 1", "value + 7")
    changed, changed_names = _fixture(sources=sources)
    target = names["pkg.a.target"].stable_id
    assert changed_names["pkg.a.target"].stable_id == target
    assert changed_names["pkg.a.helper"].stable_id == names["pkg.a.helper"].stable_id
    assert changed.root.root_cid != bundle.root.root_cid
    old, new = _nominate(bundle, [target]), _nominate(changed, [target])
    assert old["nominee_symbol_ids"] == new["nominee_symbol_ids"]
    assert old["root_cid"] != new["root_cid"]
    helper = names["pkg.a.helper"].stable_id
    def graph_helper(report):
        return next(node["record"] for node in report["graph"]["nodes"]
                    if node["record"].get("native_symbol_id") == helper)
    assert graph_helper(old)["version_cid"] != graph_helper(new)["version_cid"]
    assert graph_helper(old)["fact_cid"] != graph_helper(new)["fact_cid"]
    assert new["source_freshness_verified"] is False
    with pytest.raises(ValueError):
        nominate_typed_semantic_dependencies(
            bundle=changed, root_cid=bundle.root.root_cid, graph_root_id=GRAPH_ROOT,
            seed_symbol_ids=(target,),
        )


@pytest.mark.parametrize("seed_names", [
    ("pkg.a.target",),
    ("pkg.a.target", "pkg.a.helper"),
    ("pkg.b.other_target",),
])
def test_inventory_reuse_and_opt_out_have_identical_canonical_reports(native, monkeypatch, seed_names):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_dependency_audit import _encoded

    bundle, names = native
    seeds = [names[name].stable_id for name in seed_names]
    before = dict(bundle.blocks)
    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", raising=False)
    optimized = _nominate(bundle, seeds)
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", "0")
    original_traversal = _nominate(bundle, seeds)
    assert _encoded(optimized) == _encoded(original_traversal)
    assert bundle.blocks == before
    assert optimized["execution_authority"] is False
    assert optimized["current_source_wrapper_required"] is True


def test_inventory_reuse_verifies_the_whole_root_once_and_opt_out_rewalks_it(native, monkeypatch):
    import ipfs_datasets_py.logic.software_contracts.semantic_state.api as native_api

    bundle, names = native
    verify, opened = native_api.verify_semantic_state_bundle, native_api.open_semantic_state
    typed_links = native_api.VerifiedSemanticStateView.semantic_links_for_symbol
    counts = {"verify": 0, "open": 0, "outgoing": 0}

    def counted_verify(candidate):
        counts["verify"] += 1
        return verify(candidate)

    def counted_open(root_cid, get_block):
        counts["open"] += 1
        return opened(root_cid, get_block)

    def counted_typed_links(view, symbol_id, direction="both"):
        counts["outgoing"] += 1
        return typed_links(view, symbol_id, direction)

    monkeypatch.setattr(native_api, "verify_semantic_state_bundle", counted_verify)
    monkeypatch.setattr(native_api, "open_semantic_state", counted_open)
    monkeypatch.setattr(native_api.VerifiedSemanticStateView, "semantic_links_for_symbol", counted_typed_links)
    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", raising=False)
    optimized = _nominate(bundle, [names["pkg.a.target"].stable_id])
    assert counts == {"verify": 1, "open": 1, "outgoing": 0}
    counts.update(verify=0, open=0, outgoing=0)
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", "0")
    original_traversal = _nominate(bundle, [names["pkg.a.target"].stable_id])
    assert counts == {"verify": 2, "open": 3, "outgoing": len(names)}
    assert optimized == original_traversal


@pytest.mark.parametrize("change", [
    "root_replaced", "blocks_replaced", "mutable_proxy_backing_changed",
    "equal_bytearray", "equal_bytes_subclass", "equal_key_subclass",
])
def test_reuse_refuses_inventory_change_between_seed_and_graph(native, monkeypatch, change):
    import ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_dependency_proposal as proposal
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_relation_audit import (
        _relation_audit_from_inventory,
    )

    original, names = native
    candidate = SemanticStateBundle(original.root, original.blocks)
    backing = dict(candidate.blocks)
    if change in {"mutable_proxy_backing_changed", "equal_bytearray",
                  "equal_bytes_subclass", "equal_key_subclass"}:
        # Deliberately bypass the normal bundle's owned immutable map to model
        # an illicit mutable-backed proxy. The captured inventory remains sealed.
        object.__setattr__(candidate, "blocks", MappingProxyType(backing))
    calls = []

    def change_after_seed(inventory, selected):
        report = _relation_audit_from_inventory(inventory, selected)
        calls.append(report["selected_symbol_ids"])
        if len(calls) == 1:
            if change == "root_replaced":
                object.__setattr__(candidate, "root", replace(candidate.root))
            elif change == "blocks_replaced":
                object.__setattr__(candidate, "blocks", MappingProxyType(dict(candidate.blocks)))
            elif change == "equal_bytearray":
                key = candidate.root.symbol_fact_index_cid
                backing[key] = bytearray(backing[key])
                assert backing == dict(original.blocks)
            elif change == "equal_bytes_subclass":
                class EqualBytes(bytes):
                    pass
                key = candidate.root.symbol_fact_index_cid
                backing[key] = EqualBytes(backing[key])
                assert backing == dict(original.blocks)
            elif change == "equal_key_subclass":
                class EqualCID(str):
                    pass
                key = candidate.root.symbol_fact_index_cid
                raw = backing.pop(key)
                backing[EqualCID(key)] = raw
                assert backing == dict(original.blocks)
            else:
                backing[candidate.root.symbol_fact_index_cid] = b"{}"
        return report

    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", raising=False)
    monkeypatch.setattr(proposal, "_relation_audit_from_inventory", change_after_seed)
    with pytest.raises(ValueError, match="native relation inventory changed"):
        _nominate(candidate, [names["pkg.a.target"].stable_id])
    assert len(calls) == 1
    assert original.blocks != {} and original.root is not None


@pytest.mark.parametrize("change", [
    "root_replaced", "blocks_replaced", "mutable_proxy_backing_changed",
    "equal_bytearray", "equal_bytes_subclass", "equal_key_subclass",
])
def test_reuse_refuses_inventory_change_after_graph_projection(native, monkeypatch, change):
    import ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph as graph_module

    original, names = native
    candidate = SemanticStateBundle(original.root, original.blocks)
    backing = dict(candidate.blocks)
    if change in {"mutable_proxy_backing_changed", "equal_bytearray",
                  "equal_bytes_subclass", "equal_key_subclass"}:
        object.__setattr__(candidate, "blocks", MappingProxyType(backing))
    graph_type = graph_module.SemanticDependencyGraph
    projected = []

    def change_after_graph(*args, **kwargs):
        graph = graph_type(*args, **kwargs)
        projected.append(graph)
        if change == "root_replaced":
            object.__setattr__(candidate, "root", replace(candidate.root))
        elif change == "blocks_replaced":
            object.__setattr__(candidate, "blocks", MappingProxyType(dict(candidate.blocks)))
        elif change == "equal_bytearray":
            key = candidate.root.symbol_fact_index_cid
            backing[key] = bytearray(backing[key])
            assert backing == dict(original.blocks)
        elif change == "equal_bytes_subclass":
            class EqualBytes(bytes):
                pass
            key = candidate.root.symbol_fact_index_cid
            backing[key] = EqualBytes(backing[key])
            assert backing == dict(original.blocks)
        elif change == "equal_key_subclass":
            class EqualCID(str):
                pass
            key = candidate.root.symbol_fact_index_cid
            raw = backing.pop(key)
            backing[EqualCID(key)] = raw
            assert backing == dict(original.blocks)
        else:
            backing[candidate.root.symbol_fact_index_cid] = b"{}"
        return graph

    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", raising=False)
    monkeypatch.setattr(graph_module, "SemanticDependencyGraph", change_after_graph)
    with pytest.raises(ValueError, match="native relation inventory changed"):
        _nominate(candidate, [names["pkg.a.target"].stable_id])
    assert len(projected) == 1


@pytest.mark.parametrize("reuse", ["1", "0"])
def test_reuse_and_opt_out_both_refuse_a_rehashed_stale_native_link(native, monkeypatch, reuse):
    bundle, names = native
    old = next(link for link in _links(bundle) if link.relation == "calls"
               and link.target_kind == "symbol")
    changed = _with_link(bundle, old, replace(old, source_fact_cid=_cid("wrong-source-fact")))
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", reuse)
    with pytest.raises(ValueError):
        _nominate(changed, [names["pkg.a.target"].stable_id])


# Scoped graph controls retain complete native verification and explicit
# omission diagnostics. These tests make no inference/provider/token claims.
def _scope_digest(value):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_dependency_audit import _encoded

    return hashlib.sha256(_encoded(value)).hexdigest()


def _scope_population(values):
    return {"count": len(values), "sha256": _scope_digest(values)}


def test_full_root_scope_keeps_exact_legacy_report_without_scoped_fields(native):
    bundle, names = native
    seeds = [names["pkg.a.target"].stable_id]
    implicit = _nominate(bundle, seeds)
    explicit = _nominate(bundle, seeds, graph_scope="full_root")
    assert implicit == explicit
    assert explicit["schema"] == "supervisor-semantic-typed-dependency-proposal@1"
    assert explicit["adapter_version"] == "native-typed-dependency-proposal@1"
    assert not any(key in explicit for key in (
        "graph_scope", "graph_population", "graph_scope_commitment", "root_symbol_records"))
    assert all(node["version"] == "native-typed-dependency-proposal@1"
               for node in explicit["graph"]["nodes"])


@pytest.mark.parametrize("scope", [None, True, "", "full", "task", "TASK_CLOSURE"])
def test_graph_scope_requires_one_explicit_supported_literal(native, scope):
    bundle, names = native
    with pytest.raises(ValueError, match="graph_scope must be the literal"):
        _nominate(bundle, [names["pkg.a.target"].stable_id], graph_scope=scope)


def test_graph_scope_rejects_equal_string_subclass(native):
    class Scope(str):
        pass

    bundle, names = native
    with pytest.raises(ValueError, match="graph_scope must be the literal"):
        _nominate(bundle, [names["pkg.a.target"].stable_id], graph_scope=Scope("task_closure"))


def test_task_closure_retains_complete_root_bindings_and_all_outgoing_obligations(native):
    bundle, names = native
    seeds = [names["pkg.a.target"].stable_id]
    full = _nominate(bundle, seeds)
    scoped = _nominate(bundle, seeds, graph_scope="task_closure")
    selected = set(full["nominee_symbol_ids"])
    all_symbols = full["full_symbol_ids"]
    omitted = sorted(set(all_symbols) - selected)
    all_links = _links(bundle)
    chosen_links = [link for link in all_links if link.source_stable_id in selected]
    omitted_links = [link for link in all_links if link.source_stable_id not in selected]
    assert scoped["schema"] == "supervisor-semantic-typed-dependency-proposal@2"
    assert scoped["adapter_version"] == "native-typed-dependency-proposal@2"
    assert scoped["graph_scope"] == "task_closure"
    assert scoped["full_symbol_ids"] == all_symbols
    assert scoped["materialized_symbol_ids"] == scoped["nominee_symbol_ids"] == sorted(selected)
    assert scoped["omitted_root_symbol_ids"] == omitted
    assert scoped["full_root_graph_materialized"] is False
    for audit in ("seed_relation_audit", "nominee_relation_audit"):
        assert scoped[audit] == full[audit]
    symbol_rows = [node["record"] for node in full["graph"]["nodes"]
                   if "native_symbol_id" in node["record"]]
    symbol_rows.sort(key=lambda record: record["native_symbol_id"])
    assert scoped["root_symbol_records"] == symbol_rows
    assert scoped["root_symbol_records_count"] == len(all_symbols)
    assert scoped["root_symbol_records_sha256"] == _scope_digest(symbol_rows)
    assert {node["record"]["native_symbol_id"] for node in scoped["graph"]["nodes"]
            if "native_symbol_id" in node["record"]} == selected
    observations = [node["record"] for node in scoped["graph"]["nodes"]
                    if "relation" in node["record"]]
    assert {row["edge_id"] for row in observations} == {link.edge_id for link in chosen_links}
    assert len(scoped["graph"]["edges"]) == 2 * len(chosen_links)
    assert len(observations) == len(chosen_links)
    expected = {
        "symbols": {
            "full_root": _scope_population(all_symbols),
            "materialized": _scope_population(sorted(selected)),
            "omitted_root": _scope_population(omitted),
        },
        "relations": {
            "full_root": _scope_population(sorted([[link.edge_id, link.link_cid] for link in all_links])),
            "materialized": _scope_population(sorted([[link.edge_id, link.link_cid] for link in chosen_links])),
            "omitted_root": _scope_population(sorted([[link.edge_id, link.link_cid] for link in omitted_links])),
        },
        "referenced_artifact_endpoints": {name: _scope_population([])
                                          for name in ("full_root", "materialized", "omitted_root")},
    }
    assert scoped["graph_population"] == expected
    assert scoped["graph_sha256"] == _scope_digest(scoped["graph"])
    assert scoped["graph_scope_commitment_sha256"] == _scope_digest(scoped["graph_scope_commitment"])
    commitment = scoped["graph_scope_commitment"]
    assert commitment["root_cid"] == bundle.root.root_cid
    assert commitment["index_cids"] == full["seed_relation_audit"]["index_cids"]
    assert commitment["graph_population"] == expected
    assert commitment["root_symbol_records_count"] == len(symbol_rows)
    assert commitment["root_symbol_records_sha256"] == scoped["root_symbol_records_sha256"]
    assert commitment["full_relation_ledger_sha256"] == full["seed_relation_audit"]["full_relation_ledger_sha256"]
    assert commitment["full_capsule_identities_sha256"] == full["seed_relation_audit"]["full_capsule_identities_sha256"]
    assert all(node["version"] == "native-typed-dependency-proposal@2"
               and node["authoritative"] is False for node in scoped["graph"]["nodes"])
    assert all(edge["version"] == "native-typed-dependency-proposal@2"
               and edge["authoritative"] is False and edge["mandatory"] is False
               for edge in scoped["graph"]["edges"])
    assert scoped["unresolved_state_binding_rows"]
    graph_obligations = {node["node_id"] for node in scoped["graph"]["nodes"]
                         if "obligation_kind" in node["record"]}
    assert all(row["obligation_node_id"] in graph_obligations
               for row in scoped["unresolved_state_binding_rows"])
    for flag in ("source_freshness_verified", "dependency_semantics_complete",
                 "full_program_completeness_verified", "effective_task_coverage_verified",
                 "source_substitution_authority", "proof_authority", "execution_authority",
                 "completion_authority", "acceptance_verified", "report_is_provider_input"):
        assert scoped[flag] is False


def test_task_closure_incoming_boundary_stays_diagnostic_and_unselected(native):
    bundle, names = native
    report = _nominate(bundle, [names["pkg.a.target"].stable_id], graph_scope="task_closure")
    rows = report["nominee_relation_audit"]["incoming_omitted_relation_rows"]
    assert rows
    assert report["incoming_boundary_relation_rows"] == rows
    assert report["incoming_boundary_relation_count"] == len(rows)
    assert report["incoming_boundary_relations_sha256"] == _scope_digest(rows)
    caller = names["pkg.a.incoming_caller"].stable_id
    assert caller in report["omitted_root_symbol_ids"] and caller not in report["materialized_symbol_ids"]
    assert all(row["source_stable_id"] in report["omitted_root_symbol_ids"]
               and row["target_stable_id"] in report["materialized_symbol_ids"] for row in rows)
    observed = {node["record"]["edge_id"] for node in report["graph"]["nodes"]
                if "relation" in node["record"]}
    assert not observed.intersection(row["edge_id"] for row in rows)
    assert report["incoming_relations_automatically_selected"] is False
    assert "incoming_caller_or_relation_omitted" in report["selection_refusal_reasons"]


def test_task_closure_inventory_reuse_and_opt_out_have_identical_canonical_reports(native, monkeypatch):
    bundle, names = native
    seeds = [names["pkg.a.target"].stable_id]
    monkeypatch.delenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", raising=False)
    reused = _nominate(bundle, seeds, graph_scope="task_closure")
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", "0")
    independent = _nominate(bundle, seeds, graph_scope="task_closure")
    assert reused == independent
    assert reused["graph_scope_commitment_sha256"] == independent["graph_scope_commitment_sha256"]


def test_task_closure_graph_bounds_do_not_expand_or_drop_selected_obligations(native):
    bundle, names = native
    seeds = [names["pkg.a.target"].stable_id]
    scoped = _nominate(bundle, seeds, graph_scope="task_closure")
    exact_nodes, exact_edges = len(scoped["graph"]["nodes"]), len(scoped["graph"]["edges"])
    bounded = _nominate(bundle, seeds, graph_scope="task_closure",
                        max_graph_nodes=exact_nodes, max_graph_edges=exact_edges)
    assert bounded["graph"] == scoped["graph"]
    assert bounded["materialized_symbol_ids"] == scoped["materialized_symbol_ids"]
    with pytest.raises(ValueError, match="typed proposal graph exceeds node bound"):
        _nominate(bundle, seeds, graph_scope="full_root", max_graph_nodes=exact_nodes)
    with pytest.raises(ValueError, match="typed proposal graph exceeds node bound"):
        _nominate(bundle, seeds, graph_scope="task_closure", max_graph_nodes=exact_nodes - 1)
    with pytest.raises(ValueError, match="typed proposal graph exceeds edge bound"):
        _nominate(bundle, seeds, graph_scope="task_closure", max_graph_edges=exact_edges - 1)
    with pytest.raises(ValueError, match="nomination exceeds symbol bound"):
        _nominate(bundle, seeds, graph_scope="task_closure", max_selected_symbols=1)
    assert scoped["bounds"]["max_selected_symbols"] == 32
    assert scoped["bounds"]["max_graph_nodes"] == 2048
    assert scoped["bounds"]["max_graph_edges"] == 8192


@pytest.mark.parametrize("reuse", ["1", "0"])
def test_task_closure_still_verifies_rehashed_links_outside_materialized_scope(native, monkeypatch, reuse):
    bundle, names = native
    source = names["pkg.b.other_target"].stable_id
    old = next(link for link in _links(bundle)
               if link.source_stable_id == source and link.target_kind == "symbol")
    damaged = _with_link(bundle, old, replace(old, source_fact_cid=_cid("omitted-source-stale")))
    monkeypatch.setenv("IPFS_ACCELERATE_TYPED_INVENTORY_REUSE", reuse)
    with pytest.raises(ValueError):
        _nominate(damaged, [names["pkg.a.target"].stable_id], graph_scope="task_closure")


def test_task_closure_still_hashes_block_of_omitted_root_symbol(native):
    bundle, names = native
    omitted = names["pkg.b.other_target"].stable_id
    fact_cid = dict(_index(bundle, "symbol_fact_index_cid").pairs)[omitted]
    blocks = dict(bundle.blocks)
    blocks[fact_cid] = b"{}"
    damaged = object.__new__(SemanticStateBundle)
    object.__setattr__(damaged, "root", bundle.root)
    object.__setattr__(damaged, "blocks", MappingProxyType(blocks))
    with pytest.raises(ValueError):
        _nominate(damaged, [names["pkg.a.target"].stable_id], graph_scope="task_closure")


def test_task_closure_entire_root_membership_does_not_claim_semantic_completeness():
    source = "def left(value):\n    return right(value)\n\ndef right(value):\n    return value\n"
    bundle, names = _build_source_bundle({"whole.py": source})
    partial = _nominate(bundle, [names["whole.left"].stable_id], graph_scope="task_closure")
    assert partial["full_root_graph_materialized"] is False
    seeds = [sid for sid, _ in _index(bundle, "symbol_fact_index_cid").pairs]
    report = _nominate(bundle, seeds, graph_scope="task_closure")
    assert report["full_root_graph_materialized"] is True
    assert report["materialized_symbol_ids"] == report["full_symbol_ids"]
    assert report["omitted_root_symbol_ids"] == []
    assert report["graph_population"]["relations"]["omitted_root"] == _scope_population([])
    assert report["dependency_semantics_complete"] is False
    assert report["full_program_completeness_verified"] is False
    assert report["effective_task_coverage_verified"] is False
    assert report["source_substitution_authority"] is False


def test_task_closure_artifact_endpoints_include_selected_and_disclose_omitted_members():
    target, other = _aux_symbol("target"), _aux_symbol("other")
    shared = ArtifactRecord("artifact:shared", "authored-data", "shared.json", cid_for_bytes(b"{}"))
    omitted = ArtifactRecord("artifact:omitted", "authored-data", "omitted.json", cid_for_bytes(b"[]"))
    unreferenced = ArtifactRecord("artifact:unreferenced", "authored-data", "unused.json", cid_for_bytes(b"null"))
    def artifact_edge(source, artifact):
        return DependencyEdge(source.stable_id, artifact.artifact_id, RelationType.CALLS,
                              "authored-static", AnalysisConfidence.EXACT, "authored@1",
                              metadata={"resolution": "definite", "unresolved_target": artifact.artifact_id})
    bundle = build_semantic_state(RepositoryState(
        REPOSITORY, (target, other), (shared, omitted, unreferenced),
        (artifact_edge(target, shared), artifact_edge(other, shared), artifact_edge(other, omitted))))
    report = _nominate(bundle, [target.stable_id], graph_scope="task_closure")
    artifact_nodes = [node["record"] for node in report["graph"]["nodes"]
                      if "native_artifact_id" in node["record"]]
    assert len(artifact_nodes) == 1 and artifact_nodes[0]["native_artifact_id"] == shared.artifact_id
    assert artifact_nodes[0]["effective_coverage_verified"] is False
    links = _links(bundle)
    full = sorted({(link.target_stable_id, link.target_fact_cid) for link in links})
    chosen = [binding for binding in full if binding[0] == shared.artifact_id]
    remaining = [binding for binding in full if binding[0] == omitted.artifact_id]
    expected = {"full_root": _scope_population([list(value) for value in full]),
                "materialized": _scope_population([list(value) for value in chosen]),
                "omitted_root": _scope_population([list(value) for value in remaining])}
    assert report["graph_population"]["referenced_artifact_endpoints"] == expected
    assert unreferenced.artifact_id not in json.dumps(report["graph"])
    assert len(report["resolved_artifact_relation_rows"]) == 1
    assert report["resolved_artifact_relation_rows"][0]["target_stable_id"] == shared.artifact_id
    assert report["effective_task_coverage_verified"] is False
