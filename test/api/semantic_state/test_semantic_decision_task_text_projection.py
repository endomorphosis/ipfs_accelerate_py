"""Exact task-text references preserve independently bound required material.

The parent and source inventory below are authored pure-build fixtures. These
checks establish representation custody and exact reconstruction, not current
source admission, semantic equivalence, provider usage or measured savings.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import (
    SemanticDecisionPolicy,
    build_semantic_decision_packet,
    expand_semantic_decision_packet,
    restore_semantic_decision_required_context,
    semantic_decision_packet_references,
    verify_semantic_decision_packet,
)
from test.api.semantic_state.test_semantic_decision_packet import (
    OPTIONAL, REQUIRED, SELECTED, bind, encoded, parent as original_parent,
)

SOURCE = ".supervisor-instruction.md"
INSTRUCTION = "Keep café and literal s0.  \nPreserve the requested output\tand every trailing space.  \n"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def parent(original_parent):
    value = copy.deepcopy(original_parent)
    value["raw_sources"][SOURCE] = INSTRUCTION
    value["required_raw_paths"].append(SOURCE)
    value["manifest"][SOURCE] = {"sha256": digest(INSTRUCTION), "source_cid": "authored-instruction-source"}
    # Whitespace normalization occurred in this authored producer, never in
    # projection. The two retained roots are intentionally different strings.
    value["objective"] = " ".join(INSTRUCTION.split())
    value["pack"]["pack"]["objective"] = value["objective"]
    value["pack"]["pack"]["interface_cids"] = ["interface-current"]
    value["pack"]["pack"]["route"] = "model"
    value["pack"]["pack"]["escalation_recommendation"] = "Preserve unique security constraints"
    value["manifest"]["dependency.py"] = {"sha256": "d" * 64, "source_cid": "dependency-source"}
    value["worker_projection"] = {
        "schema": "supervisor-semantic-worker-projection@1",
        "selector": "lexical-query-symbol-name@1", "query": INSTRUCTION,
        "raw_source_fetch_required": {"dependency.py": copy.deepcopy(value["manifest"]["dependency.py"])},
        "raw_required_caveat": "Read exact current source before editing; references grant no substitutions.",
        "semantic_equivalence_claimed": False, "completion_authority": False,
        "future_required_condition": {"instruction": "Retain the unique global prohibition", "required": True},
    }
    return value


def policy(**changes):
    return SemanticDecisionPolicy(**{
        "required_capsule_ids": (REQUIRED,), "optional_capsule_ids": (OPTIONAL, SELECTED),
        "selected_optional_capsule_ids": (SELECTED,), "task_text_source_path": SOURCE, **changes,
    })


def case(parent, selected_policy=None):
    text = encoded(parent)
    bindings = bind(text)
    chosen = selected_policy or policy()
    view = build_semantic_decision_packet(original_text=text, bindings=bindings, policy=chosen)
    return text, bindings, chosen, view


def selected_required(parent):
    expected = copy.deepcopy(parent)
    expected["capsules"] = expected["capsules"][:2]
    del expected["pack"]["token_estimate"]
    del expected["pack"]["pack"]["token_totals"]
    return expected


def test_explicit_profile_keeps_two_distinct_literal_roots_and_reconstructs_exact_selected_required(parent):
    before = copy.deepcopy(parent)
    text, bindings, chosen, view = case(parent)
    assert view["schema"] == "supervisor-semantic-decision-packet@2"
    assert view["projection_revision"] == "explicit-capsule-selection+exact-task-text-references@2"
    required = view["required_context"]
    assert required["raw_sources"][SOURCE] == INSTRUCTION
    assert required["objective"] == before["objective"]
    assert required["objective"] != required["raw_sources"][SOURCE]
    assert "query" not in required["worker_projection"]
    assert "objective" not in required["pack"]["pack"]
    references = {tuple(row["field_selector"]): row for row in view["required_text_references"]}
    assert references == {
        ("worker_projection", "query"): {
            "field_selector": ["worker_projection", "query"],
            "target_selector": ["raw_sources", SOURCE], "sha256": digest(INSTRUCTION),
        },
        ("pack", "pack", "objective"): {
            "field_selector": ["pack", "pack", "objective"],
            "target_selector": ["objective"], "sha256": digest(before["objective"]),
        },
    }
    expected = selected_required(before)
    assert view["canonical_required_coverage_sha256"] == digest(encoded(expected))
    assert view["required_coverage_sha256"] == digest(encoded({
        "required_context": required, "required_text_references": view["required_text_references"],
    }))
    assert restore_semantic_decision_required_context(
        view=view, original_text=text, bindings=bindings, policy=chosen) == expected
    assert parent == before and encoded(parent) == text
    assert verify_semantic_decision_packet(view=view, original_text=text, bindings=bindings, policy=chosen) == view
    assert view["execution_authority"] is view["completion_authority"] is view["semantic_equivalence_claimed"] is False


def test_all_admissions_unique_constraints_manifest_sources_and_fetch_caveats_remain_literal(parent):
    _, _, _, view = case(parent)
    required = view["required_context"]
    for name in ("admissions", "manifest", "raw_sources", "required_raw_paths", "preparation_bounds",
                 "doctor_snapshot_id", "doctor_manifest_cid", "reconstruction", "refresh_lineage"):
        assert required[name] == parent[name]
    for name, value in parent["pack"]["pack"].items():
        if name not in {"objective", "token_totals"}:
            assert required["pack"]["pack"][name] == value
    assert {k: v for k, v in required["worker_projection"].items()} == {
        k: v for k, v in parent["worker_projection"].items() if k != "query"}
    assert required["pack"]["references"] == parent["pack"]["references"]
    assert len(required["admissions"]) == 3 and len(required["capsules"]) == 2


def test_v1_default_policy_keeps_existing_three_fields_and_literal_duplicates(parent):
    legacy = policy(task_text_source_path=None)
    text, bindings, _, view = case(parent, legacy)
    assert legacy.schema_revision == "1" and policy().schema_revision == "2"
    assert legacy.to_dict() == {"required_capsule_ids": [REQUIRED],
        "optional_capsule_ids": [OPTIONAL, SELECTED], "selected_optional_capsule_ids": [SELECTED]}
    assert view["schema"] == "supervisor-semantic-decision-packet@1"
    assert view["projection_revision"] == "explicit-capsule-selection@1"
    assert "required_text_references" not in view and "canonical_required_coverage_sha256" not in view
    assert view["required_context"] == selected_required(parent)
    assert restore_semantic_decision_required_context(
        view=view, original_text=text, bindings=bindings, policy=legacy) == selected_required(parent)
    assert policy().to_dict() == {**legacy.to_dict(), "task_text_source_path": SOURCE}


@pytest.mark.parametrize("path", [True, 1, [], {}, "", "../instruction.md", "/instruction.md",
                                      "./instruction.md", "a/../instruction.md", "a//instruction.md", "a\x00b",
                                      "a\nb", "a\rb", "x" * 4097])
def test_policy_rejects_nonliteral_noncanonical_source_path(path):
    with pytest.raises(ValueError):
        policy(task_text_source_path=path)


@pytest.mark.parametrize("selected", [None, SOURCE])
def test_policy_closed_wire_roundtrip_preserves_explicit_revision(selected):
    chosen = policy(task_text_source_path=selected)
    assert SemanticDecisionPolicy.from_dict(chosen.to_dict()) == chosen


@pytest.mark.parametrize("mutation", ["null_profile", "extra_field", "missing_partition", "tuple_partition"])
def test_policy_wire_refuses_ambiguous_profile_or_partition(mutation):
    value = policy().to_dict()
    if mutation == "null_profile":
        value["task_text_source_path"] = None
    elif mutation == "extra_field":
        value["skip_unique_constraints"] = True
    elif mutation == "missing_partition":
        del value["required_capsule_ids"]
    else:
        value["optional_capsule_ids"] = (OPTIONAL, SELECTED)
    with pytest.raises(ValueError):
        SemanticDecisionPolicy.from_dict(value)


@pytest.mark.parametrize("mutation", [
    "query_distinct", "query_whitespace_only_difference", "pack_objective_distinct",
    "normalized_query", "query_missing", "projection_missing", "source_not_required",
    "source_missing", "manifest_missing", "manifest_sha_changed", "manifest_sha_missing",
    "manifest_binding_nonobject", "source_nonstring", "query_nonstring", "objective_nonstring",
    "pack_objective_nonstring", "empty_text", "query_empty", "objective_empty", "pack_objective_empty",
])
def test_profile_refuses_distinct_text_missing_coverage_and_bad_source_pins(parent, mutation):
    if mutation == "query_distinct":
        parent["worker_projection"]["query"] += "Unique extra requirement"
    elif mutation == "query_whitespace_only_difference":
        parent["worker_projection"]["query"] = INSTRUCTION.rstrip()
    elif mutation == "pack_objective_distinct":
        parent["pack"]["pack"]["objective"] += " Unique output constraint"
    elif mutation == "normalized_query":
        parent["worker_projection"]["query"] = parent["objective"]
    elif mutation == "query_missing":
        del parent["worker_projection"]["query"]
    elif mutation == "projection_missing":
        del parent["worker_projection"]
    elif mutation == "source_not_required":
        parent["required_raw_paths"].remove(SOURCE)
    elif mutation == "source_missing":
        del parent["raw_sources"][SOURCE]
    elif mutation == "manifest_missing":
        del parent["manifest"][SOURCE]
    elif mutation == "manifest_sha_changed":
        parent["manifest"][SOURCE]["sha256"] = "f" * 64
    elif mutation == "manifest_sha_missing":
        del parent["manifest"][SOURCE]["sha256"]
    elif mutation == "manifest_binding_nonobject":
        parent["manifest"][SOURCE] = "not a binding"
    elif mutation == "source_nonstring":
        parent["raw_sources"][SOURCE] = {"text": INSTRUCTION}
    elif mutation == "query_nonstring":
        parent["worker_projection"]["query"] = [INSTRUCTION]
    elif mutation == "objective_nonstring":
        parent["objective"] = [parent["objective"]]
    elif mutation == "pack_objective_nonstring":
        parent["pack"]["pack"]["objective"] = {"text": parent["objective"]}
    elif mutation == "empty_text":
        parent["raw_sources"][SOURCE] = parent["worker_projection"]["query"] = ""
        parent["manifest"][SOURCE]["sha256"] = digest("")
    elif mutation == "query_empty":
        parent["worker_projection"]["query"] = ""
    elif mutation == "objective_empty":
        parent["objective"] = parent["pack"]["pack"]["objective"] = ""
    elif mutation == "pack_objective_empty":
        parent["pack"]["pack"]["objective"] = ""
    with pytest.raises(ValueError):
        case(parent)


def test_nominated_required_source_cannot_be_replaced_by_another_equal_literal(parent):
    parent["raw_sources"]["other-instruction.md"] = INSTRUCTION
    parent["manifest"]["other-instruction.md"] = copy.deepcopy(parent["manifest"][SOURCE])
    parent["required_raw_paths"].remove(SOURCE)
    parent["required_raw_paths"].append("other-instruction.md")
    with pytest.raises(ValueError):
        case(parent)


@pytest.mark.parametrize("mutation", [
    "ref_digest", "ref_target", "ref_field", "missing_ref", "duplicate_ref", "extra_ref_field",
    "literal_target", "core_objective", "restored_copy_in_core", "source_manifest", "admission",
    "unique_obligation", "source_fetch", "canonical_coverage", "rendered_coverage", "profile_revision",
])
def test_tampered_reference_required_core_or_coverage_cannot_verify_or_reconstruct(parent, mutation):
    text, bindings, chosen, view = case(parent)
    bad = copy.deepcopy(view)
    refs = bad["required_text_references"]
    query_ref = next(row for row in refs if row["field_selector"] == ["worker_projection", "query"])
    if mutation == "ref_digest":
        query_ref["sha256"] = "0" * 64
    elif mutation == "ref_target":
        query_ref["target_selector"] = ["objective"]
    elif mutation == "ref_field":
        query_ref["field_selector"] = ["pack", "pack", "future_requirement"]
    elif mutation == "missing_ref":
        refs.remove(query_ref)
    elif mutation == "duplicate_ref":
        refs.append(copy.deepcopy(query_ref))
    elif mutation == "extra_ref_field":
        query_ref["required"] = True
    elif mutation == "literal_target":
        bad["required_context"]["raw_sources"][SOURCE] += "changed"
    elif mutation == "core_objective":
        bad["required_context"]["objective"] += "changed"
    elif mutation == "restored_copy_in_core":
        bad["required_context"]["worker_projection"]["query"] = INSTRUCTION
    elif mutation == "source_manifest":
        bad["required_context"]["manifest"][SOURCE]["sha256"] = "0" * 64
    elif mutation == "admission":
        bad["required_context"]["admissions"].pop()
    elif mutation == "unique_obligation":
        bad["required_context"]["pack"]["pack"]["obligation_cids"] = []
    elif mutation == "source_fetch":
        bad["required_context"]["worker_projection"]["raw_source_fetch_required"] = {}
    elif mutation == "canonical_coverage":
        bad["canonical_required_coverage_sha256"] = "0" * 64
    elif mutation == "rendered_coverage":
        bad["required_coverage_sha256"] = "0" * 64
    elif mutation == "profile_revision":
        bad["projection_revision"] = "explicit-capsule-selection@1"
    for operation in (verify_semantic_decision_packet, restore_semantic_decision_required_context):
        with pytest.raises(ValueError):
            operation(view=bad, original_text=text, bindings=bindings, policy=chosen)


@pytest.mark.parametrize("changed", ["legacy_policy", "other_source_policy", "parent_bytes", "binding_sha"])
def test_trusted_policy_and_original_parent_remain_independently_required(parent, changed):
    text, bindings, chosen, view = case(parent)
    if changed == "legacy_policy":
        chosen = policy(task_text_source_path=None)
    elif changed == "other_source_policy":
        chosen = policy(task_text_source_path="target.py")
    elif changed == "parent_bytes":
        altered = copy.deepcopy(parent)
        altered["worker_projection"]["query"] += "unique constraint"
        text = encoded(altered)
    else:
        bindings = replace(bindings, expected_sha256="0" * 64)
    for operation in (verify_semantic_decision_packet, restore_semantic_decision_required_context):
        with pytest.raises(ValueError):
            operation(view=view, original_text=text, bindings=bindings, policy=chosen)


def test_chunked_required_packet_carries_original_coverage_and_both_reference_roots(parent):
    text, bindings, chosen, view = case(parent)
    refs = semantic_decision_packet_references(view=view, original_text=text, bindings=bindings,
        policy=chosen, coverage_ids=("ACCEPT-1", "SECURITY-1"), chunk_bytes=512)
    assert len(refs) > 1 and all(ref.required for ref in refs)
    assert json.loads("".join(ref.summary for ref in refs)) == view
    coverage = {item for ref in refs for item in ref.coverage_ids}
    assert {"ACCEPT-1", "SECURITY-1", "semantic-decision-required:" + view["required_coverage_sha256"]} <= coverage


def test_optional_capsule_expansion_stays_bound_to_projected_parent_without_deferring_task_text(parent):
    text, bindings, chosen, view = case(parent)
    calls = []

    def resolver(**request):
        calls.append(request)
        return text

    handle = view["deferred_references"][0]
    expanded = expand_semantic_decision_packet(view=view, reference_id=handle["reference_id"],
        bindings=bindings, policy=chosen, resolve_original=resolver, count_tokens=lambda _wire: 10,
        max_bytes=65536, max_tokens=100, capsule_ids=(OPTIONAL,))
    archive = json.loads(expanded.summary)
    assert archive["fields"] == [{"field_selector": ["capsules", 2], "value": parent["capsules"][2]}]
    assert expanded.metadata["required_coverage_sha256"] == view["required_coverage_sha256"]
    assert len(calls) == 2 and calls[0] == calls[1] == {
        "artifact": bindings.artifact, "expected_sha256": bindings.expected_sha256, "task_id": bindings.task_id}
    assert restore_semantic_decision_required_context(
        view=view, original_text=text, bindings=bindings, policy=chosen) == selected_required(parent)
