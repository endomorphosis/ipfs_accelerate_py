"""Root-bound native relation diagnostics; no source or task completeness claim."""

import ast
import json
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_relation_audit import (
    audit_selected_capsule_relations,
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
from ipfs_datasets_py.logic.software_contracts.semantic_index.symbol_graph import (
    resolve_edge_targets,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state import build_semantic_state
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    ArtifactFactNode,
    SemanticCapsule,
    SemanticLinkNode,
    SemanticStateBundle,
    SortedPairIndex,
    SymbolMerkleNode,
)

REPOSITORY = "repo:authored-relation-audit"


def cid(label):
    return cid_for_structured({"schema": "authored-relation-audit@1", "label": label})


def symbol(name, *, kind=SymbolKind.FUNCTION, body=None, metadata=None):
    source = body or f"def {name}(value):\n    return value + 1\n"
    node = ast.parse(source).body[0]
    stable = stable_symbol_id(REPOSITORY, "python", "operations.py", name, kind, "operations")
    signature = {"parameters": ["value"]}
    version = symbol_version_cid(stable, node, signature, [], {})
    return SymbolRecord(
        stable, version, REPOSITORY, "python", "operations.py", name, kind, "operations",
        cid_for_bytes(source.encode()), None, AnalysisConfidence.EXACT,
        signature, [], {}, metadata or {}, normalized_ast=node,
    )


def edge(source, target, relation=RelationType.CALLS, *, method="static"):
    target_id = target.stable_id if isinstance(target, SymbolRecord) else target
    return DependencyEdge(source.stable_id, target_id, relation, method,
                          AnalysisConfidence.EXACT, "authored@1")


def build(symbols, edges=(), *, resolve=True):
    # This is the same finite native resolution step that records the original
    # symbolic spelling before native link wrappers omit unresolved identity.
    if resolve:
        edges = resolve_edge_targets(edges, symbols)
    return build_semantic_state(RepositoryState(REPOSITORY, tuple(symbols), edges=edges))


@pytest.fixture(scope="module")
def native():
    symbols = {name: symbol(name, kind=SymbolKind.TEST if name == "public_test"
                            else SymbolKind.FUNCTION)
               for name in ("target", "helper", "caller", "public_test", "irrelevant")}
    edges = [edge(symbols["target"], symbols["helper"]),
             edge(symbols["target"], "global:helper", RelationType.READS_STATE,
                  method="global_binding"),
             edge(symbols["caller"], symbols["target"]),
             edge(symbols["target"], symbols["public_test"], RelationType.TESTED_BY)]
    return build(list(symbols.values()), edges), symbols


def audit(bundle, selected, **limits):
    return audit_selected_capsule_relations(
        bundle=bundle, root_cid=bundle.root.root_cid,
        selected_symbol_ids=selected, **limits,
    )


def index(bundle, field):
    index_cid = getattr(bundle.root, field)
    return SortedPairIndex.from_dict({**json.loads(bundle.blocks[index_cid]),
                                      "index_cid": index_cid})


def record(bundle, field, key, cls, identity_field):
    record_cid = dict(index(bundle, field).pairs)[key]
    return cls.from_dict({**json.loads(bundle.blocks[record_cid]),
                          identity_field: record_cid})


def links(bundle):
    return [record(bundle, "semantic_link_index_cid", key, SemanticLinkNode, "link_cid")
            for key, _ in index(bundle, "semantic_link_index_cid").pairs]


def capsule(bundle, stable):
    return record(bundle, "capsule_index_cid", stable, SemanticCapsule, "capsule_cid")


def node(bundle, stable):
    return record(bundle, "symbol_node_index_cid", stable, SymbolMerkleNode, "node_cid")


def with_indexes(bundle, changes, records=()):
    """Author a distinct native root; keep bytes canonical and CIDs valid."""
    blocks = dict(bundle.blocks)
    root_changes = {}
    for field, pairs in changes.items():
        changed = SortedPairIndex(pairs)
        blocks[changed.index_cid] = canonical_dag_json_bytes(changed.identity_payload())
        root_changes[field] = changed.index_cid
    for value, identity_field in records:
        blocks[getattr(value, identity_field)] = canonical_dag_json_bytes(value.identity_payload())
    return SemanticStateBundle(replace(bundle.root, **root_changes), blocks)


def with_capsule(bundle, stable, **changes):
    changed = replace(capsule(bundle, stable), **changes)
    changed_node = replace(node(bundle, stable), capsule_cid=changed.capsule_cid)
    capsule_pairs = dict(index(bundle, "capsule_index_cid").pairs)
    capsule_pairs[stable] = changed.capsule_cid
    node_pairs = dict(index(bundle, "symbol_node_index_cid").pairs)
    node_pairs[stable] = changed_node.node_cid
    return with_indexes(bundle, {"capsule_index_cid": list(capsule_pairs.items()),
                                 "symbol_node_index_cid": list(node_pairs.items())},
                        [(changed, "capsule_cid"), (changed_node, "node_cid")])


def with_link(bundle, old, changed):
    """Update wrapper and adjacency so fact/edge drift is tested directly."""
    pairs = dict(index(bundle, "semantic_link_index_cid").pairs)
    del pairs[old.edge_id]
    pairs[changed.edge_id] = changed.link_cid
    node_pairs = dict(index(bundle, "symbol_node_index_cid").pairs)
    records = [(changed, "link_cid")]
    for stable in node_pairs:
        existing = node(bundle, stable)
        changed_node = replace(
            existing,
            incoming_link_cids=[changed.link_cid if value == old.link_cid else value
                                for value in existing.incoming_link_cids],
            outgoing_link_cids=[changed.link_cid if value == old.link_cid else value
                                for value in existing.outgoing_link_cids],
        )
        node_pairs[stable] = changed_node.node_cid
        records.append((changed_node, "node_cid"))
    return with_indexes(bundle, {"semantic_link_index_cid": list(pairs.items()),
                                 "symbol_node_index_cid": list(node_pairs.items())}, records)


def test_known_call_and_symbolic_state_binding_are_distinct(native):
    bundle, symbols = native
    target, helper = symbols["target"].stable_id, symbols["helper"].stable_id
    report = audit(bundle, [target])
    assert report["missing_known_call_dependency_symbol_ids"] == [helper]
    assert report["declared_known_call_closure_selected"] is False
    state_rows = report["unresolved_state_binding_rows"]
    assert len(state_rows) == 1
    assert state_rows[0]["relation"] == RelationType.READS_STATE.value
    assert state_rows[0]["extraction_method"] == "global_binding"
    assert state_rows[0]["unresolved_target"] == "global:helper"
    assert state_rows[0]["target_stable_id"] is None
    assert report["unresolved_other_reference_rows"] == []
    call_rows = [row for row in report["outgoing_relation_rows"]
                 if row["relation"] == RelationType.CALLS.value]
    assert len(call_rows) == 1 and call_rows[0]["target_stable_id"] == helper
    assert state_rows[0]["edge_id"] != call_rows[0]["edge_id"]
    assert "global:helper" in capsule(bundle, target).dependency_stable_ids
    assert "global:helper" not in report["missing_known_call_dependency_symbol_ids"]


def test_known_helper_closure_does_not_cover_incoming_callers_or_tests(native):
    bundle, symbols = native
    target, helper = symbols["target"].stable_id, symbols["helper"].stable_id
    report = audit(bundle, [target, helper])
    assert report["missing_known_call_dependency_symbol_ids"] == []
    assert report["declared_known_call_closure_selected"] is True
    assert report["omitted_outgoing_test_symbol_ids"] == [symbols["public_test"].stable_id]
    incoming = report["incoming_omitted_relation_rows"]
    assert len(incoming) == 1
    assert incoming[0]["source_stable_id"] == symbols["caller"].stable_id
    assert incoming[0]["target_stable_id"] == target
    assert incoming[0]["relation"] == RelationType.CALLS.value
    assert report["selected_declared_relation_coverage_complete"] is False
    assert report["selection_eligibility"] == "experimental_only"


def test_edge_identity_and_wrapper_identity_are_both_root_bound(native):
    bundle, symbols = native
    report = audit(bundle, [symbols["target"].stable_id])
    by_edge = dict(index(bundle, "semantic_link_index_cid").pairs)
    for row in report["outgoing_relation_rows"] + report["incoming_omitted_relation_rows"]:
        assert row["edge_id"] in by_edge
        assert row["link_cid"] == by_edge[row["edge_id"]]
        assert row["edge_id"] != row["link_cid"]
    assert report["root_cid"] == bundle.root.root_cid


def test_all_symbols_selected_still_has_no_semantic_or_execution_authority(native):
    bundle, symbols = native
    report = audit(bundle, [value.stable_id for value in symbols.values()])
    assert report["missing_known_call_dependency_symbol_ids"] == []
    assert report["incoming_omitted_relation_rows"] == []
    assert report["omitted_outgoing_test_symbol_ids"] == []
    assert report["selection_eligibility"] == "experimental_only"
    for flag in ("source_freshness_verified", "dependency_semantics_complete",
                 "full_program_completeness_verified", "execution_authority",
                 "proof_authority", "deployment_ready", "activation_authority",
                 "runtime_activation"):
        assert report[flag] is False


def test_raw_native_unresolved_links_without_recovery_metadata_refuse():
    target = symbol("target")
    bundle = build([target], [edge(target, "global:helper", RelationType.READS_STATE,
                                  method="global_binding")], resolve=False)
    assert links(bundle)[0].metadata == MappingProxyType({})
    with pytest.raises(ValueError):
        audit(bundle, [target.stable_id])


def test_unresolved_calls_are_not_state_bindings_or_resolved_by_name():
    target, helper = symbol("target"), symbol("helper")
    bundle = build([target, helper], [edge(target, "lexical:elsewhere.helper")])
    report = audit(bundle, [target.stable_id])
    assert report["missing_known_call_dependency_symbol_ids"] == []
    assert report["unresolved_state_binding_rows"] == []
    rows = report["unresolved_other_reference_rows"]
    assert len(rows) == 1 and rows[0]["relation"] == RelationType.CALLS.value
    assert rows[0]["target_stable_id"] is None
    assert rows[0]["unresolved_target"] == "lexical:elsewhere.helper"
    assert report["declared_known_call_closure_selected"] is False


def test_noncall_capsule_coverage_fields_remain_independently_visible():
    target, related = symbol("target"), symbol("related")
    relations = (RelationType.CALLS, RelationType.USES_FIXTURE, RelationType.TESTED_BY,
                 RelationType.PROOF_DEPENDS_ON, RelationType.VALIDATES,
                 RelationType.SERIALIZES)
    bundle = build([target, related], [edge(target, related, relation)
                                      for relation in relations])
    report = audit(bundle, [target.stable_id, related.stable_id])
    counts = report["capsule_coverage_field_counts"][target.stable_id]
    for field in ("fixture_refs", "test_refs", "proof_obligation_refs", "schema_relations",
                  "serialization_relations"):
        assert counts[field] == 1
    assert report["relation_counts"]["selected_outgoing"] == {
        relation.value: 1 for relation in relations}


def test_sealed_metadata_effects_and_exceptions_merge_with_declared_relations():
    target = symbol("target", metadata={
        "effects": ["author_declared_io"],
        "exception_behavior": {"raises": ["exception:AuthorError"], "policy": "authored"},
    })
    bundle = build([target], [edge(target, "global:helper", RelationType.READS_STATE,
                                  method="global_binding"),
                              edge(target, "exception:RecordedError", RelationType.RAISES)])
    report = audit(bundle, [target.stable_id])
    native_capsule = capsule(bundle, target.stable_id)
    assert set(native_capsule.effects) == {"author_declared_io", "reads_state:global:helper"}
    assert native_capsule.to_dict()["exception_behavior"] == {
        "raises": ["exception:AuthorError", "exception:RecordedError"], "policy": "authored"}
    assert report["capsule_coverage_field_counts"][target.stable_id]["effects"] == 2
    assert report["dependency_semantics_complete"] is False


@pytest.mark.parametrize("damage", ["extra_effect", "drop_metadata_effect",
                                   "extra_exception", "drop_metadata_exception"])
def test_rehashed_capsule_effect_exception_metadata_drift_refuses(damage):
    target = symbol("target", metadata={
        "effects": ["author_declared_io"],
        "exception_behavior": {"raises": ["exception:AuthorError"]},
    })
    bundle = build([target], [edge(target, "global:helper", RelationType.READS_STATE,
                                  method="global_binding"),
                              edge(target, "exception:RecordedError", RelationType.RAISES)])
    native_capsule = capsule(bundle, target.stable_id)
    if damage == "extra_effect":
        changes = {"effects": [*native_capsule.effects, "invented_effect"]}
    elif damage == "drop_metadata_effect":
        changes = {"effects": ["reads_state:global:helper"]}
    elif damage == "extra_exception":
        changes = {"exception_behavior": {
            "raises": ["exception:AuthorError", "exception:RecordedError", "exception:Invented"]}}
    else:
        changes = {"exception_behavior": {"raises": ["exception:RecordedError"]}}
    changed = with_capsule(bundle, target.stable_id, **changes)
    with pytest.raises(ValueError):
        audit(changed, [target.stable_id])


@pytest.mark.parametrize("field", ["source_version_cid", "source_fact_cid",
                                  "target_version_cid", "target_fact_cid"])
def test_native_rehashed_link_cannot_drift_from_current_facts(native, field):
    bundle, symbols = native
    old = next(value for value in links(bundle) if value.relation == "calls"
               and value.source_stable_id == symbols["target"].stable_id)
    forged = with_link(bundle, old, replace(old, **{field: cid("foreign:" + field)}))
    with pytest.raises(ValueError):
        audit(forged, [symbols["target"].stable_id])


def test_native_rehashed_link_edge_id_must_recompute(native):
    bundle, symbols = native
    old = links(bundle)[0]
    forged = with_link(bundle, old, replace(old, edge_id=cid("foreign-edge")))
    with pytest.raises(ValueError):
        audit(forged, [symbols["target"].stable_id])


@pytest.mark.parametrize("damage", ["missing", "wrapper_cid", "foreign_source"])
def test_capsule_dependency_edge_ids_match_native_outgoing_projection(native, damage):
    bundle, symbols = native
    target = symbols["target"].stable_id
    if damage == "missing":
        ids = []
    elif damage == "wrapper_cid":
        ids = [value.link_cid for value in links(bundle) if value.source_stable_id == target
               and value.relation in ("calls", "reads_state")]
    else:
        ids = [value.edge_id for value in links(bundle)
               if value.source_stable_id == symbols["caller"].stable_id]
    changed = with_capsule(bundle, target, dependency_link_ids=ids)
    with pytest.raises(ValueError):
        audit(changed, [target])


@pytest.mark.parametrize("field", ["dependency_stable_ids", "dependency_version_cids",
                                  "dependency_fact_cids", "test_refs"])
def test_capsule_native_relation_projection_cannot_silently_drop_fields(native, field):
    bundle, symbols = native
    target = symbols["target"].stable_id
    assert getattr(capsule(bundle, target), field)
    changed = with_capsule(bundle, target, **{field: []})
    with pytest.raises(ValueError):
        audit(changed, [target])


def test_root_link_index_cannot_omit_known_node_adjacency(native):
    bundle, symbols = native
    changed = with_indexes(bundle, {"semantic_link_index_cid": []})
    with pytest.raises(ValueError):
        audit(changed, [symbols["target"].stable_id])


def test_link_index_key_must_equal_authoritative_edge_id(native):
    bundle, symbols = native
    pairs = list(index(bundle, "semantic_link_index_cid").pairs)
    pairs[0] = (cid("wrong-index-key"), pairs[0][1])
    changed = with_indexes(bundle, {"semantic_link_index_cid": pairs})
    with pytest.raises(ValueError):
        audit(changed, [symbols["target"].stable_id])


@pytest.mark.parametrize("field", ["incoming_link_cids", "outgoing_link_cids"])
def test_root_node_adjacency_cannot_drop_link_membership(native, field):
    bundle, symbols = native
    target = symbols["target"].stable_id
    changed_node = replace(node(bundle, target), **{field: []})
    pairs = dict(index(bundle, "symbol_node_index_cid").pairs)
    pairs[target] = changed_node.node_cid
    changed = with_indexes(bundle, {"symbol_node_index_cid": list(pairs.items())},
                           [(changed_node, "node_cid")])
    with pytest.raises(ValueError):
        audit(changed, [target])


@pytest.mark.parametrize("field", ["symbol_fact_index_cid", "symbol_node_index_cid",
                                  "capsule_index_cid"])
def test_root_fact_node_and_capsule_inventory_memberships_are_exact(native, field):
    bundle, symbols = native
    pairs = list(index(bundle, field).pairs)
    changed = with_indexes(bundle, {field: pairs[:-1]})
    with pytest.raises(ValueError):
        audit(changed, [symbols["target"].stable_id])


def test_missing_root_reachable_block_refuses(native):
    bundle, symbols = native
    blocks = dict(bundle.blocks)
    del blocks[bundle.root.semantic_link_index_cid]
    changed = SemanticStateBundle(bundle.root, blocks)
    with pytest.raises(ValueError):
        audit(changed, [symbols["target"].stable_id])


def test_selection_duplicate_foreign_and_nonstrings_refuse(native):
    bundle, symbols = native
    target = symbols["target"].stable_id
    for selected in ([target, target], [cid("foreign")], [True], [""], [target + "\n"],
                     tuple([target])):
        with pytest.raises(ValueError):
            audit(bundle, selected)


def test_mutable_subclassed_and_injected_bundle_inputs_refuse(native):
    bundle, symbols = native
    target = symbols["target"].stable_id

    class SubclassedBundle(SemanticStateBundle):
        pass

    derived = SubclassedBundle(bundle.root, bundle.blocks)
    mutable = SemanticStateBundle(bundle.root, bundle.blocks)
    object.__setattr__(mutable, "blocks", dict(mutable.blocks))
    injected = SimpleNamespace(root=bundle.root, blocks=bundle.blocks, get_block=bundle.get_block)
    for candidate in (derived, mutable, injected):
        with pytest.raises(ValueError):
            audit_selected_capsule_relations(bundle=candidate, root_cid=bundle.root.root_cid,
                                            selected_symbol_ids=[target])


@pytest.mark.parametrize("root_cid", [True, "", "not-a-cid", cid("different-root")])
def test_caller_root_binding_must_match_exact_native_root(native, root_cid):
    bundle, symbols = native
    with pytest.raises(ValueError):
        audit_selected_capsule_relations(bundle=bundle, root_cid=root_cid,
                                        selected_symbol_ids=[symbols["target"].stable_id])


@pytest.mark.parametrize("limits", [
    {"max_symbols": True}, {"max_links": False}, {"max_bundle_bytes": True},
    {"max_report_bytes": False}, {"max_symbols": 0}, {"max_links": 0},
    {"max_bundle_bytes": 0}, {"max_report_bytes": 0}, {"max_symbols": 1025},
    {"max_links": 8193}, {"max_bundle_bytes": 32_000_001},
    {"max_report_bytes": 2_000_001}, {"max_links": 1.5},
])
def test_bounds_are_literal_positive_and_cannot_expand_hard_limits(native, limits):
    bundle, symbols = native
    with pytest.raises(ValueError):
        audit(bundle, [symbols["target"].stable_id], **limits)


@pytest.mark.parametrize("limits", [{"max_symbols": 1}, {"max_links": 1},
                                  {"max_bundle_bytes": 1}, {"max_report_bytes": 1}])
def test_actual_inventory_bundle_and_report_limits_refuse_without_partial_rows(native, limits):
    bundle, symbols = native
    with pytest.raises(ValueError):
        audit(bundle, [symbols["target"].stable_id], **limits)


def test_empty_selection_is_explicitly_ineligible(native):
    bundle, _ = native
    report = audit(bundle, [])
    assert report["selected_symbol_ids"] == []
    assert report["selection_eligibility"] == "refused_empty_selection"
    assert report["selection_refusal_reasons"]
    assert report["deployment_ready"] is False


def test_cycles_and_order_permutations_are_finite_without_mutating_input():
    left, right, unused = symbol("left"), symbol("right"), symbol("unused")
    edges = [edge(left, right), edge(right, left), edge(unused, unused)]
    first = build([left, right, unused], edges)
    second = build([unused, right, left], list(reversed(edges)))
    before = dict(first.blocks)
    report = audit(first, [left.stable_id, right.stable_id])
    assert report == audit(second, [right.stable_id, left.stable_id])
    assert report["declared_known_call_closure_selected"] is True
    assert report["missing_known_call_dependency_symbol_ids"] == []
    assert len(report["outgoing_relation_rows"]) == 2
    assert first.blocks == before
    assert type(first.blocks) is MappingProxyType


def test_diagnostic_report_omits_symbol_ast_source_and_capsule_bodies(native):
    bundle, symbols = native
    report = audit(bundle, [symbols["target"].stable_id])
    encoded = json.dumps(report, sort_keys=True)
    assert "FunctionDef" not in encoded
    assert "return value + 1" not in encoded
    assert "capsules" not in report
    assert "normalized_ast" not in report
    assert report["full_capsule_count"] == len(symbols)
    assert report["full_link_count"] == len(links(bundle))


@pytest.mark.parametrize("field,value", [
    ("resolution", {"body": "private source body"}),
    ("resolution", ["private source body"]),
    ("resolution", True),
    ("resolution", "unexpected_resolution"),
    ("unresolved_target", {"body": "private source body"}),
    ("unresolved_target", ["private source body"]),
    ("unresolved_target", True),
    ("unresolved_target", ""),
    ("unresolved_target", "lexical:helper\n"),
    ("unresolved_target", "lexical:cafe\u0301"),
    ("unresolved_target", "x" * 16_385),
])
def test_recorded_resolution_and_original_target_have_closed_literal_shapes(field, value):
    target, helper = symbol("target"), symbol("helper")
    metadata = {"resolution": "definite", "unresolved_target": "lexical:helper", field: value}
    authored_edge = replace(edge(target, helper), metadata=metadata)
    bundle = build([target, helper], [authored_edge], resolve=False)
    assert links(bundle)[0].target_kind == "symbol"
    with pytest.raises(ValueError):
        audit(bundle, [target.stable_id, helper.stable_id])


def test_native_finite_may_original_spelling_does_not_turn_known_targets_unresolved():
    target = symbol("target")
    helper = symbol("helper")
    fixture = symbol("helper", kind=SymbolKind.FIXTURE)
    bundle = build([target, helper, fixture], [edge(target, "lexical:helper")])
    report = audit(bundle, [target.stable_id, helper.stable_id, fixture.stable_id])
    rows = report["outgoing_relation_rows"]
    assert len(rows) == 2
    assert {row["target_stable_id"] for row in rows} == {helper.stable_id, fixture.stable_id}
    assert all(row["target_kind"] == "symbol" and row["resolution"] == "finite_may"
               and row["unresolved_target"] == "lexical:helper" for row in rows)
    assert report["unresolved_state_binding_rows"] == []
    assert report["unresolved_other_reference_rows"] == []
    assert report["selection_eligibility"] == "experimental_only"
    assert report["dependency_semantics_complete"] is False


def without_incoming_links(bundle, stable):
    changed_node = replace(node(bundle, stable), incoming_link_cids=[])
    pairs = dict(index(bundle, "symbol_node_index_cid").pairs)
    pairs[stable] = changed_node.node_cid
    return with_indexes(bundle, {"symbol_node_index_cid": list(pairs.items())},
                        [(changed_node, "node_cid")])


def test_known_symbol_wrapper_cannot_downgrade_to_unresolved():
    target, helper = symbol("target"), symbol("helper")
    authored_edge = replace(edge(target, helper), metadata={
        "resolution": "definite", "unresolved_target": helper.stable_id})
    bundle = build([target, helper], [authored_edge], resolve=False)
    old = links(bundle)[0]
    wrong = replace(old, target_kind="unresolved", target_stable_id=None,
                    target_version_cid=None, target_fact_cid=None)
    changed = with_link(bundle, old, wrong)
    # Preserve adjacency for the forged unresolved wrapper, so the audited
    # disagreement is its inventory classification rather than stale pointers.
    changed = without_incoming_links(changed, helper.stable_id)
    assert capsule(changed, target.stable_id).dependency_stable_ids == (helper.stable_id,)
    with pytest.raises(ValueError):
        audit(changed, [target.stable_id, helper.stable_id])


def test_native_known_artifact_relation_is_typed_and_never_authorizes_coverage():
    target = symbol("target")
    artifact = ArtifactRecord("artifact:configuration", "authored-data", "configuration.json",
                              cid_for_bytes(b'{"answer":1}'))
    authored_edge = replace(edge(target, artifact.artifact_id), metadata={
        "resolution": "definite", "unresolved_target": artifact.artifact_id})
    bundle = build_semantic_state(RepositoryState(REPOSITORY, (target,), (artifact,), (authored_edge,)))
    report = audit(bundle, [target.stable_id])
    rows = report["resolved_artifact_relation_rows"]
    assert len(rows) == 1 and rows[0]["target_kind"] == "artifact"
    assert rows[0]["target_stable_id"] == artifact.artifact_id
    assert rows[0]["target_fact_cid"] is not None
    assert report["unresolved_other_reference_rows"] == []
    assert report["unresolved_state_binding_rows"] == []
    assert report["selected_declared_relation_coverage_complete"] is False
    assert report["selection_eligibility"] == "experimental_only"


def test_known_artifact_wrapper_cannot_downgrade_to_unresolved():
    target = symbol("target")
    artifact = ArtifactRecord("artifact:configuration", "authored-data", "configuration.json",
                              cid_for_bytes(b'{"answer":1}'))
    authored_edge = replace(edge(target, artifact.artifact_id), metadata={
        "resolution": "definite", "unresolved_target": artifact.artifact_id})
    bundle = build_semantic_state(RepositoryState(REPOSITORY, (target,), (artifact,), (authored_edge,)))
    old = links(bundle)[0]
    wrong = replace(old, target_kind="unresolved", target_stable_id=None,
                    target_version_cid=None, target_fact_cid=None)
    changed = with_link(bundle, old, wrong)
    with pytest.raises(ValueError):
        audit(changed, [target.stable_id])


def test_symbol_and_artifact_identity_collision_keeps_native_symbol_priority():
    target, helper = symbol("target"), symbol("helper")
    artifact = ArtifactRecord(helper.stable_id, "authored-data", "configuration.json",
                              cid_for_bytes(b'{"answer":1}'))
    authored_edge = replace(edge(target, helper), metadata={
        "resolution": "definite", "unresolved_target": helper.stable_id})
    bundle = build_semantic_state(RepositoryState(
        REPOSITORY, (target, helper), (artifact,), (authored_edge,)))
    old = links(bundle)[0]
    assert old.target_kind == "symbol"
    report = audit(bundle, [target.stable_id, helper.stable_id])
    assert report["outgoing_relation_rows"][0]["target_kind"] == "symbol"
    artifact_fact = record(bundle, "artifact_fact_index_cid", artifact.artifact_id,
                           ArtifactFactNode, "fact_cid")
    wrong = replace(old, target_kind="artifact", target_version_cid=None,
                    target_fact_cid=artifact_fact.fact_cid)
    changed = with_link(bundle, old, wrong)
    changed = without_incoming_links(changed, helper.stable_id)
    with pytest.raises(ValueError):
        audit(changed, [target.stable_id, helper.stable_id])


def test_private_inventory_reports_are_fresh_and_cannot_contaminate_later_selection(native):
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_relation_audit import (
        _relation_audit_from_inventory, _verified_relation_inventory,
    )

    bundle, symbols = native
    target = symbols["target"].stable_id
    inventory = _verified_relation_inventory(
        bundle=bundle, root_cid=bundle.root.root_cid, selected_symbol_ids=[target],
    )
    clean = _relation_audit_from_inventory(inventory, [target])
    changed = _relation_audit_from_inventory(inventory, [target])
    changed["outgoing_relation_rows"][0]["relation"] = "forged"
    changed["outgoing_relation_rows"][0]["source_span"] = {"forged": True}
    changed["capsule_coverage_field_counts"][target]["effects"] = 999
    changed["root_index_and_fact_bindings_verified"] = False
    assert _relation_audit_from_inventory(inventory, [target]) == clean
    with pytest.raises(TypeError):
        inventory.coverage_counts[target]["effects"] = 999
    with pytest.raises(TypeError):
        inventory.outgoing[target] = ()
