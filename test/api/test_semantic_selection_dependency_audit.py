"""Authored native-schema graphs; no source freshness or semantic authority."""

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_selection_dependency_audit import (
    audit_selected_capsule_dependencies,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
    SEMANTIC_INDEX_SCHEMA,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
    SemanticCapsule,
)


def identity(name):
    return cid_for_structured({"schema": "authored-dependency-audit-fixture@1", "name": name})


def capsule(name, dependencies=(), **fields):
    return SemanticCapsule(
        stable_symbol_id=identity(name), version_cid=identity("version:" + name),
        semantic_index_schema=SEMANTIC_INDEX_SCHEMA,
        extractor_version="authored-fixture@1",
        dependency_stable_ids=[identity(value) for value in dependencies],
        **fields,
    ).to_dict()


def audit(capsules, selected, **bounds):
    return audit_selected_capsule_dependencies(
        full_capsules=capsules,
        selected_symbol_ids=[identity(name) for name in selected],
        **bounds,
    )


def test_declared_transitive_gap_keeps_known_targets_and_no_authority():
    report = audit([capsule("target", ["helper"]), capsule("helper", ["leaf"]),
                    capsule("leaf"), capsule("irrelevant")], ["target"])
    assert report["known_transitive_closure_symbol_ids"] == sorted(
        identity(name) for name in ("target", "helper", "leaf"))
    assert report["missing_selected_dependency_symbol_ids"] == sorted(
        identity(name) for name in ("helper", "leaf"))
    assert report["selected_dependency_symbol_ids"] == []
    assert report["unresolved_dependency_symbol_ids"] == []
    assert report["declared_graph_closure_selected"] is False
    assert report["cycle_present"] is False
    assert (report["full_capsule_count"], report["reachable_known_count"],
            report["declared_edge_count"], report["reachable_edge_count"]) == (4, 3, 2, 2)
    for flag in ("source_freshness_verified", "dependency_semantics_complete",
                 "full_program_completeness_verified", "source_substitution_authority",
                 "execution_authority", "proof_authority", "completion_authority",
                 "deployment_ready"):
        assert report[flag] is False


def test_selected_declared_closure_is_exact_without_claiming_semantic_closure():
    report = audit([capsule("target", ["helper"]), capsule("helper")], ["target", "helper"])
    assert report["declared_graph_closure_selected"] is True
    assert report["selected_dependency_symbol_ids"] == [identity("helper")]
    assert report["missing_selected_dependency_symbol_ids"] == []
    assert report["dependency_semantics_complete"] is False
    assert report["deployment_ready"] is False


def test_missing_target_stays_unresolved_and_unreachable_edges_do_not_enter_closure():
    report = audit([capsule("target", ["external"]), capsule("irrelevant", ["other"])],
                   ["target"])
    assert report["known_transitive_closure_symbol_ids"] == [identity("target")]
    assert report["unresolved_dependency_symbol_ids"] == [identity("external")]
    assert report["unresolved_dependency_count"] == 1
    assert report["declared_graph_closure_selected"] is False
    assert report["reachable_edge_count"] == 1 and report["declared_edge_count"] == 2


def test_cycles_terminate_deterministically_and_only_reachable_cycles_count():
    records = [capsule("target", ["helper"]), capsule("helper", ["target"]),
               capsule("unrelated", ["unrelated"])]
    report = audit(records, ["target", "helper"])
    assert report["cycle_present"] is True
    assert report["declared_graph_closure_selected"] is True
    assert report["selected_dependency_symbol_ids"] == sorted(
        [identity("target"), identity("helper")])
    no_cycle = audit([capsule("target"), capsule("unrelated", ["unrelated"])], ["target"])
    assert no_cycle["cycle_present"] is False


def test_self_cycle_and_empty_selection_are_explicit():
    assert audit([capsule("target", ["target"])], ["target"])["cycle_present"] is True
    report = audit([], [])
    assert report["known_transitive_closure_symbol_ids"] == []
    assert report["declared_graph_closure_selected"] is True
    assert report["selected_capsule_count"] == 0 and report["deployment_ready"] is False


def test_permutations_do_not_change_report_and_input_is_not_mutated():
    first = [capsule("target", ["left", "right"]), capsule("left"), capsule("right")]
    before = deepcopy(first)
    second = deepcopy(list(reversed(first)))
    second[-1]["dependency_stable_ids"].reverse()
    assert audit(first, ["target", "left"]) == audit(second, ["left", "target"])
    assert first == before


def test_no_arbitrary_cid_scan_and_no_raw_body_in_report():
    secret = "authored source body must stay outside graph diagnostics"
    report = audit([capsule("target", signature={"reference": identity("decoy")},
                            metadata={"body": secret, "dependency": identity("other")})], ["target"])
    assert report["known_transitive_closure_symbol_ids"] == [identity("target")]
    assert report["unresolved_dependency_symbol_ids"] == []
    assert secret not in str(report) and identity("decoy") not in str(report)
    assert all(len(report[key]) == 64 for key in (
        "full_capsule_identities_sha256", "declared_graph_sha256", "selection_sha256"))


def test_duplicate_inventory_and_duplicate_or_foreign_seeds_refuse():
    record = capsule("target")
    for records, selected in (([record, record], ["target"]),
                              ([record], ["target", "target"]),
                              ([record], ["foreign"])):
        with pytest.raises(ValueError):
            audit(records, selected)


@pytest.mark.parametrize("damage", ["schema", "identity", "missing", "unknown"])
def test_native_schema_identity_and_required_fields_revalidate(damage):
    record = capsule("target")
    if damage == "schema":
        record["schema"] = "unsupported@9"
    elif damage == "identity":
        record["capsule_cid"] = identity("forged")
    elif damage == "missing":
        record.pop("version_cid")
    else:
        record["unknown"] = "ignored?"
    with pytest.raises(ValueError):
        audit([record], ["target"])


@pytest.mark.parametrize("dependencies", ["not-a-list", (), [True], [""],
                                             [identity("helper"), identity("helper")]])
def test_malformed_declared_edges_refuse(dependencies):
    record = capsule("target")
    record["dependency_stable_ids"] = dependencies
    with pytest.raises(ValueError):
        audit([record], ["target"])


def test_literal_container_shapes_and_invalid_json_refuse():
    record = capsule("target")
    for records, selected in ((tuple([record]), [identity("target")]),
                              ([record], tuple([identity("target")])),
                              ([record], [True])):
        with pytest.raises(ValueError):
            audit_selected_capsule_dependencies(full_capsules=records,
                                               selected_symbol_ids=selected)
    record["metadata"] = {"value": float("nan")}
    with pytest.raises(ValueError, match="finite JSON"):
        audit([record], ["target"])


@pytest.mark.parametrize("bounds", [dict(max_nodes=True), dict(max_edges=False),
                                    dict(max_nodes=0), dict(max_nodes=1025),
                                    dict(max_edges=8193), dict(max_edges=1.5)])
def test_bounds_are_strict_and_cannot_expand_hard_limits(bounds):
    with pytest.raises(ValueError):
        audit([capsule("target")], ["target"], **bounds)


def test_known_unresolved_node_and_total_edge_limits_refuse():
    with pytest.raises(ValueError, match="inventory.*node bound"):
        audit([capsule("target"), capsule("helper")], ["target"], max_nodes=1)
    with pytest.raises(ValueError, match="reachable.*node bound"):
        audit([capsule("target", ["external"])], ["target"], max_nodes=1)
    with pytest.raises(ValueError, match="edge bound"):
        audit([capsule("target", ["left", "right"])], ["target"], max_edges=1)


def test_inventory_and_graph_hash_bind_unreachable_native_changes():
    first = audit([capsule("target"), capsule("irrelevant")], ["target"])
    second = audit([capsule("target"), capsule("irrelevant", ["external"])], ["target"])
    assert first["known_transitive_closure_symbol_ids"] == second["known_transitive_closure_symbol_ids"]
    assert first["selection_sha256"] == second["selection_sha256"]
    assert first["full_capsule_identities_sha256"] != second["full_capsule_identities_sha256"]
    assert first["declared_graph_sha256"] != second["declared_graph_sha256"]


def test_native_symbolic_dependency_remains_unresolved_without_name_inference():
    symbolic = "global:helper_operation"
    target = SemanticCapsule(
        stable_symbol_id=identity("target"), version_cid=identity("version:target"),
        semantic_index_schema=SEMANTIC_INDEX_SCHEMA, extractor_version="authored-fixture@1",
        dependency_stable_ids=[symbolic],
    ).to_dict()
    report = audit([target, capsule("helper_operation")], ["target"])
    assert report["known_transitive_closure_symbol_ids"] == [identity("target")]
    assert report["unresolved_dependency_symbol_ids"] == [symbolic]
    assert report["missing_selected_dependency_symbol_ids"] == []
    assert report["unresolved_dependency_count"] == report["reachable_edge_count"] == 1
    assert report["declared_graph_closure_selected"] is False
    assert report["dependency_semantics_complete"] is report["deployment_ready"] is False


@pytest.mark.parametrize("dependency", ["global:helper\n", "x" * 16_385])
def test_native_dependency_text_grammar_rejects_invalid_or_oversized_text(dependency):
    # Keep the capsule CID intact: native parsing rejects the field grammar
    # before a repaired content identity could mask malformed dependency text.
    record = capsule("target")
    record["dependency_stable_ids"] = [dependency]
    with pytest.raises(ValueError, match="invalid text|trimmed NFC"):
        audit([record], ["target"])
