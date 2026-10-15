"""Whole authored native capsule closure nominations, never semantic authority."""

import hashlib
import json
from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_capsule_closure_nomination import (
    nominate_declared_capsule_closure,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
    SEMANTIC_INDEX_SCHEMA,
)
from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SemanticCapsule


def identity(name):
    return cid_for_structured({"schema": "authored-closure-nomination-fixture@1", "name": name})


def capsule(name, dependencies=(), **fields):
    return SemanticCapsule(
        stable_symbol_id=identity(name), version_cid=identity("version:" + name),
        semantic_index_schema=SEMANTIC_INDEX_SCHEMA,
        extractor_version="authored-fixture@1",
        dependency_stable_ids=[identity(value) for value in dependencies],
        **fields,
    ).to_dict()


def nominate(records, seeds, **bounds):
    return nominate_declared_capsule_closure(
        full_capsules=records, seed_symbol_ids=[identity(name) for name in seeds], **bounds,
    )


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def test_transitive_nomination_copies_whole_known_native_bodies():
    records = [capsule("target", ["helper"], metadata={"risk": ["literal detail"]}),
               capsule("helper", ["leaf"], contracts={"requires": "exact assertion"}),
               capsule("leaf"), capsule("unrelated")]
    report = nominate(records, ["target"])
    wanted = sorted(identity(name) for name in ("target", "helper", "leaf"))
    assert report["nomination_status"] == "nomination_available_declared_graph_closed"
    assert report["nomination_available"] is True
    assert report["nominee_symbol_ids"] == report["known_closure_symbol_ids"] == wanted
    assert report["initial_seed_symbol_ids"] == [identity("target")]
    assert report["added_known_symbol_ids"] == sorted(identity(name) for name in ("helper", "leaf"))
    assert report["nominee_packet"]["capsules"] == sorted(records[:3], key=lambda item: item["stable_symbol_id"])
    assert report["known_closure_capsule_count"] == 3
    assert report["known_closure_capsules_sha256"] == hashlib.sha256(encoded(report["nominee_packet"]["capsules"])).hexdigest()
    assert report["graph_diagnostics"]["declared_graph_closure_selected"] is False
    assert report["nomination_graph_diagnostics"]["declared_graph_closure_selected"] is True
    assert report["declared_graph_closed"] is True


def test_multiple_seeds_do_not_duplicate_shared_dependencies():
    report = nominate([capsule("left", ["leaf"]), capsule("right", ["leaf"]),
                       capsule("leaf")], ["left", "right", "leaf"])
    assert len(report["nominee_packet"]["capsules"]) == report["known_closure_capsule_count"] == 3
    assert report["added_known_symbol_ids"] == []
    assert report["declared_graph_closed"] is True


def test_cycles_are_finite_and_unreachable_cycles_are_excluded():
    records = [capsule("target", ["helper"]), capsule("helper", ["target"]),
               capsule("unrelated", ["unrelated"])]
    report = nominate(records, ["target"])
    assert report["cycle_present"] is True
    assert report["cycle_analysis_scope"] == "reachable_known_declared_graph"
    assert report["known_closure_capsule_count"] == 2
    assert report["declared_graph_closed"] is True
    report = nominate([capsule("target"), records[-1]], ["target"])
    assert report["cycle_present"] is False


def test_known_dependency_added_and_native_symbolic_name_kept_unresolved():
    target = capsule("target", ["helper"])
    target["dependency_stable_ids"].append("global:helper_operation")
    target.pop("capsule_cid")
    # Rebuild the authored native record, preserving the exact native symbolic edge.
    target.pop("schema")
    target = SemanticCapsule(**target).to_dict()
    report = nominate([target, capsule("helper"), capsule("helper_operation")], ["target"])
    assert report["nomination_status"] == "nomination_available_with_unresolved"
    assert report["added_known_symbol_ids"] == [identity("helper")]
    assert report["unresolved_dependency_symbol_ids"] == ["global:helper_operation"]
    assert report["declared_graph_closed"] is False
    assert identity("helper_operation") not in report["nominee_symbol_ids"]
    assert "global:helper_operation" in report["nominee_packet"]["capsules"][report["nominee_symbol_ids"].index(identity("target"))]["dependency_stable_ids"]


def test_unknown_cid_remains_unresolved_without_a_fabricated_body():
    report = nominate([capsule("target", ["foreign"])], ["target"])
    assert report["nominee_symbol_ids"] == [identity("target")]
    assert report["unresolved_dependency_symbol_ids"] == [identity("foreign")]
    assert report["nomination_graph_diagnostics"]["unresolved_dependency_count"] == 1


def test_seed_and_inventory_permutations_are_deterministic_without_input_mutation():
    records = [capsule("left", ["leaf"]), capsule("right", ["leaf"]), capsule("leaf")]
    before = deepcopy(records)
    first = nominate(records, ["left", "right"])
    second = nominate(list(reversed(records)), ["right", "left"])
    assert first == second
    assert records == before


def test_returned_receipt_is_isolated_from_later_nested_input_mutation():
    records = [capsule("target", metadata={"nested": {"values": ["original"]}})]
    seeds = [identity("target")]
    report = nominate_declared_capsule_closure(full_capsules=records, seed_symbol_ids=seeds)
    original = encoded(report)
    records[0]["metadata"]["nested"]["values"].append("changed")
    records[0]["dependency_stable_ids"].append(identity("foreign"))
    seeds.clear()
    assert encoded(report) == original


def test_capsule_budget_refuses_the_entire_closure_without_partial_candidate():
    records = [capsule("target", ["helper"]), capsule("helper")]
    report = nominate(records, ["target"], max_capsules=1)
    assert report["nomination_status"] == "nomination_refused_budget"
    assert report["nomination_available"] is False
    assert report["nominee_symbol_ids"] == []
    assert report["nominee_packet"] is report["nominee_packet_bytes"] is report["nominee_packet_sha256"] is None
    assert report["known_closure_capsule_count"] == 2
    assert report["added_known_symbol_ids"] == [identity("helper")]
    assert report["declared_graph_closed"] is True  # Declared graph only; no available candidate.
    assert report["nomination_refusal_reasons"] == ["known_closure_exceeds_capsule_budget"]
    assert len(report["attempted_candidate_packet_sha256"]) == 64


def test_full_packet_overhead_is_budgeted_even_when_bodies_fit():
    records = [capsule("target")]
    bodies_only = len(encoded(records))
    report = nominate(records, ["target"], max_packet_bytes=bodies_only + 10)
    assert report["nomination_available"] is False
    assert report["nominee_packet"] is None
    assert report["nomination_refusal_reasons"] == ["full_candidate_packet_exceeds_byte_budget"]
    assert report["attempted_candidate_packet_bytes"] > bodies_only + 10


def test_exact_full_packet_byte_boundary_and_digest_are_literal():
    records = [capsule("target")]
    report = nominate(records, ["target"])
    packet = encoded(report["nominee_packet"])
    limit = len(packet)
    exact = nominate(records, ["target"], max_packet_bytes=limit)
    assert exact["nomination_available"] is True
    assert exact["nominee_packet_bytes"] == limit
    assert exact["nominee_packet_sha256"] == hashlib.sha256(packet).hexdigest()
    assert encoded(exact["nominee_packet"]) == packet
    assert exact["packet_budget_scope"] == "nominee_packet_only"
    assert exact["report_is_provider_input"] is False
    assert len(encoded(exact)) > limit
    refused = nominate(records, ["target"], max_packet_bytes=limit - 1)
    assert refused["nomination_available"] is False
    assert refused["attempted_candidate_packet_bytes"] == limit
    assert refused["attempted_candidate_packet_sha256"] == hashlib.sha256(packet).hexdigest()
    assert refused["nominee_packet"] is refused["nominee_packet_bytes"] is refused["nominee_packet_sha256"] is None


def test_empty_seeds_are_explicit_non_candidates_and_native_records_still_validate():
    report = nominate([capsule("target")], [])
    assert report["nomination_status"] == "nomination_refused_empty_seeds"
    assert report["nomination_available"] is report["declared_graph_closed"] is False
    assert report["nominee_symbol_ids"] == []
    assert report["nominee_packet"] is report["nominee_packet_bytes"] is report["nominee_packet_sha256"] is None
    damaged = capsule("target")
    damaged["capsule_cid"] = identity("forged")
    with pytest.raises(ValueError):
        nominate([damaged], [])


@pytest.mark.parametrize("damage", ["schema", "capsule_cid", "source", "body", "unknown"])
def test_native_schema_identity_source_claim_and_body_tampering_refuse(damage):
    record = capsule("target", source_cid=identity("source"), source_slice_path="operations.py")
    if damage == "schema":
        record["schema"] = "foreign-schema@1"
    elif damage == "capsule_cid":
        record["capsule_cid"] = identity("forged")
    elif damage == "source":
        record["source_cid"] = identity("changed-source")
    elif damage == "body":
        record["metadata"]["body"] = "changed without repairing native capsule identity"
    else:
        record["unknown"] = "outside native closed schema"
    with pytest.raises(ValueError):
        nominate([record], ["target"])


def test_invalid_inventory_seeds_and_nonliteral_json_refuse():
    record = capsule("target")
    for records, seeds in (([record, record], [identity("target")]),
                           ([record], [identity("target"), identity("target")]),
                           ([record], [identity("foreign")]),
                           (tuple([record]), [identity("target")]),
                           ([record], tuple([identity("target")])), ([record], [True]),
                           ([record], ["x" * 16_385])):
        with pytest.raises(ValueError):
            nominate_declared_capsule_closure(full_capsules=records, seed_symbol_ids=seeds)
    record["metadata"] = {"value": float("nan")}
    with pytest.raises(ValueError, match="finite JSON"):
        nominate([record], ["target"])


@pytest.mark.parametrize("bounds", [dict(max_capsules=True), dict(max_capsules=33),
                                    dict(max_capsules=0), dict(max_packet_bytes=False),
                                    dict(max_packet_bytes=65_537), dict(max_packet_bytes=0),
                                    dict(max_nodes=1025), dict(max_nodes=True),
                                    dict(max_edges=8193), dict(max_edges=1.5)])
def test_bounds_are_strict_and_cannot_expand_existing_limits(bounds):
    with pytest.raises(ValueError):
        nominate([capsule("target")], ["target"], **bounds)


def test_graph_limits_bound_all_inventory_edges_and_reachable_unresolved_nodes():
    with pytest.raises(ValueError, match="node bound"):
        nominate([capsule("target"), capsule("helper")], ["target"], max_nodes=1)
    with pytest.raises(ValueError, match="reachable.*node bound"):
        nominate([capsule("target", ["foreign"])], ["target"], max_nodes=1)
    with pytest.raises(ValueError, match="edge bound"):
        nominate([capsule("target"), capsule("unrelated", ["left", "right"])],
                 ["target"], max_edges=1)


def test_unreachable_inventory_changes_bind_graph_but_never_enter_nominee_bodies():
    first = nominate([capsule("target"), capsule("unrelated")], ["target"])
    second = nominate([capsule("target"), capsule("unrelated", ["foreign"])], ["target"])
    assert first["nominee_packet"]["capsules"] == second["nominee_packet"]["capsules"]
    assert first["known_closure_capsules_sha256"] == second["known_closure_capsules_sha256"]
    assert first["graph_diagnostics"]["declared_graph_sha256"] != second["graph_diagnostics"]["declared_graph_sha256"]


def test_nomination_never_claims_source_freshness_or_semantic_and_execution_authority():
    report = nominate([capsule("target", source_cid=identity("historical-source"))], ["target"])
    assert report["declared_graph_closed"] is True
    for flag in ("source_freshness_verified", "dependency_semantics_complete",
                 "full_program_completeness_verified", "test_task_impact_relevance_complete",
                 "source_substitution_authority",
                 "execution_authority", "proof_authority", "completion_authority",
                 "acceptance_verified", "deployment_ready", "activation_authority"):
        assert report[flag] is False


def test_refusal_diagnostics_have_an_independent_hard_byte_bound():
    # A native graph can contain long bounded symbolic targets. Reporting each
    # one across audit phases is still subject to the separate diagnostic cap.
    record = capsule("target")
    record.pop("schema")
    record.pop("capsule_cid")
    record["dependency_stable_ids"] = [f"global:unknown_{index}:" + "x" * 15_980 for index in range(64)]
    record = SemanticCapsule(**record).to_dict()
    with pytest.raises(ValueError, match="hard report byte bound"):
        nominate([record], ["target"], max_packet_bytes=1)
