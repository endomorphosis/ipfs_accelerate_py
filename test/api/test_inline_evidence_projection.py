"""Authored evidence objects preserve exact native input without provider or file IO."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextReference
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as semantic
from ipfs_accelerate_py.agent_supervisor.runtime import inline_evidence_projection as projection


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def encoded_factory(native):
    root, _, prompt, source = native
    def build(bodies=None, *, mutate=None, transport_schema=semantic.TRANSPORT_SCHEMA):
        wire, end = json.JSONDecoder().raw_decode(prompt)
        if bodies is None:
            bodies = {
                "intent-world-context": {"schema": "authored-world@1", "status": "advisory",
                    "escaped": 'café 世界 \\ "\n' * 700, "nested": {"required": True}},
                "source-formalization-context": {"schema": "authored-formal@1", "proof": "unverified",
                    "equations": [{"predicate": "P", "argument": str(i)} for i in range(50)]},
                "code-retrieval-context": {"schema": "authored-retrieval@1",
                    "hits": [{"score": 0.125, "symbol": "x"}, {"score": -0.0, "symbol": "y"}]},
            }
        for kind, body in bodies.items():
            text = body if type(body) is str else canonical(body)
            prefix = projection.KINDS.get(kind, "unknown")
            wire["evidence"].extend(row.to_record() for row in build_text_context_references(
                text, reference_prefix=prefix, kind=kind, path=".runtime/" + prefix + ".json",
                repository_id=wire["repository_id"], tree_id=wire["tree_id"], required=True,
                chunk_bytes=projection.CHUNK_BYTES))
        if mutate: mutate(wire)
        value = semantic.encode_semantic_router_prompt(prompt=semantic._json(wire) + prompt[end:],
            repository=root, transport_schema=transport_schema)
        return value
    return build


def test_canonical_objects_preserve_whole_artifacts_floats_and_native_inverse(encoded_factory):
    encoded = encoded_factory()
    before = deepcopy(encoded)
    view = projection.project_inline_evidence(encoded)
    original = json.loads(encoded.provider_prompt); candidate = json.loads(view.provider_prompt)
    assert candidate["schema"] == projection.VIEW_SCHEMA
    assert candidate["source_transport_schema"] == semantic.TRANSPORT_SCHEMA
    for key in set(original) - {"schema", "native_context"}:
        assert candidate[key] == original[key]
    for key in set(original["native_context"]) - {"evidence"}:
        assert candidate["native_context"][key] == original["native_context"][key]
    refs = candidate["native_context"]["evidence"]
    assert all(row["metadata"]["required"] is True for row in refs)
    assert all(row["tier"] == "invariant" for row in refs)
    assert candidate["inline_evidence_artifacts"][2]["hits"][0]["score"] == 0.125
    assert canonical(candidate["inline_evidence_artifacts"][2]["hits"][1]["score"]) == "-0.0"
    world = [row for row in refs if row["kind"] == "intent-world-context"]
    assert len(world) > 1
    assert all(row["summary"] == {projection.REFERENCE_KEY: 0} for row in world)
    assert projection.restore_inline_evidence(view.provider_prompt, view.receipt) == encoded.provider_prompt
    assert semantic._restore(projection.restore_inline_evidence(view.provider_prompt, view.receipt),
        encoded.table) == encoded.native_prompt
    assert encoded == before
    assert view.receipt["artifact_count"] == 3
    assert view.receipt["evidence_row_count"] == len(refs)
    assert view.receipt["removed_string_overhead_bytes"] > 1000


def test_projection_is_immutable_and_export_is_closed(encoded_factory):
    encoded = encoded_factory(); view = projection.project_inline_evidence(encoded)
    with pytest.raises(FrozenInstanceError): view.provider_prompt = "changed"
    receipt = view.receipt
    assert projection.validate_projection_receipt(receipt) == receipt
    for key in projection._FALSE_FIELDS: assert receipt[key] is False
    for key in projection._TRUE_FIELDS: assert receipt[key] is True
    assert receipt["provider_token_usage"] is None
    assert receipt["native_transport_sha256"] == sha(encoded.provider_prompt)
    assert receipt["candidate_view_bytes"] == len(view.provider_prompt.encode())
    assert receipt["summary_json_string_bytes"] - receipt["summary_utf8_bytes"] == receipt["removed_string_overhead_bytes"]
    assert all(text not in view.receipt_json for text in ("café", "TASK-1", ".runtime/", "authored-world", "repo:test"))
    receipt["artifact_count"] = 999
    assert view.receipt["artifact_count"] == 3


def test_unknown_fields_and_opaque_evidence_are_preserved(encoded_factory):
    def mutate(wire):
        wire["future-required-contract"] = {"keep": "PRIVATE-UNKNOWN-CONTENT"}
        for row in wire["evidence"]:
            if row.get("kind") == "intent-world-context":
                row["future-reference-field"] = {"must_keep": True}
                row["metadata"]["future_required_flag"] = "keep"
                raw = {key: value for key, value in row.items() if key not in {"content_id", "future-reference-field"}}
                row.update(ContextReference.from_dict(raw).to_record())
        wire["evidence"].extend([
            {"kind": "future-evidence", "summary": {projection.REFERENCE_KEY: "literal"}, "required": True},
            {"kind": "future-evidence", "summary": "PRIVATE-OPAQUE"},
            "PRIVATE-UNKNOWN-ROW",
        ])
    encoded = encoded_factory(mutate=mutate); view = projection.project_inline_evidence(encoded)
    wire = json.loads(view.provider_prompt)
    assert wire["native_context"]["future-required-contract"]["keep"] == "PRIVATE-UNKNOWN-CONTENT"
    assert wire["native_context"]["evidence"][-3:] == json.loads(encoded.provider_prompt)["native_context"]["evidence"][-3:]
    for row in wire["native_context"]["evidence"]:
        if type(row) is dict and row.get("kind") == "intent-world-context":
            assert row["future-reference-field"] == {"must_keep": True}
            assert row["metadata"]["future_required_flag"] == "keep"
    assert projection.restore_inline_evidence(view.provider_prompt, view.receipt) == encoded.provider_prompt


@pytest.mark.parametrize("body", ['{"z": 1}', '{"z":1,"z":2}', 'not JSON', '[1,2]',
    'NaN', '{"x":Infinity}', '"literal"'])
def test_noncanonical_or_nonobject_evidence_stays_literal(encoded_factory, body):
    encoded = encoded_factory({"intent-world-context": body})
    view = projection.project_inline_evidence(encoded)
    assert view.receipt["artifact_count"] == 0
    assert json.loads(view.provider_prompt)["native_context"] == json.loads(encoded.provider_prompt)["native_context"]
    assert projection.restore_inline_evidence(view.provider_prompt, view.receipt) == encoded.provider_prompt
    selected = projection.select_inline_evidence(view, encoded.provider_prompt, view.provider_prompt)
    assert selected.selected_mode == "legacy" and selected.selected_prompt == encoded.provider_prompt


@pytest.mark.parametrize("changed", ["missing", "reordered", "duplicate", "chunk_index", "chunk_count",
    "digest", "artifact", "byte_count", "boolean_count", "path", "required", "priority", "tree"])
def test_known_group_contract_corruption_cannot_project(encoded_factory, changed):
    def mutate(wire):
        rows = [row for row in wire["evidence"] if row.get("kind") == "intent-world-context"]
        assert len(rows) > 1
        if changed == "missing": wire["evidence"].remove(rows[-1])
        elif changed == "reordered":
            a, b = wire["evidence"].index(rows[0]), wire["evidence"].index(rows[1])
            wire["evidence"][a], wire["evidence"][b] = rows[1], rows[0]
        elif changed == "duplicate": wire["evidence"].append(deepcopy(rows[0]))
        elif changed in {"chunk_index", "chunk_count"}: rows[0]["metadata"][changed] += 1
        elif changed == "digest": rows[0]["referenced_content_id"] = "sha256:" + "0" * 64
        elif changed == "artifact": rows[0]["metadata"]["artifact_content_id"] = "sha256:" + "0" * 64
        elif changed == "byte_count": rows[0]["byte_count"] -= 1
        elif changed == "boolean_count": rows[0]["byte_count"] = True
        elif changed == "path": rows[0]["path"] = ".runtime/foreign.json"
        elif changed == "required": rows[0]["metadata"]["required"] = False
        elif changed == "priority": rows[0]["metadata"]["priority"] = True
        else: rows[0]["tree_id"] = "tree:foreign"
    encoded = encoded_factory(mutate=mutate)
    with pytest.raises(projection.InlineEvidenceProjectionError): projection.project_inline_evidence(encoded)


@pytest.mark.parametrize("changed", ["native", "provider", "table", "historical", "bool_freshness", "compact"])
def test_only_current_exact_original_semantic_encoding_is_admitted(encoded_factory, changed):
    encoded = encoded_factory(transport_schema=semantic.COMPACT_TRANSPORT_SCHEMA
        if changed == "compact" else semantic.TRANSPORT_SCHEMA)
    if changed == "native": encoded = replace(encoded, native_prompt=encoded.native_prompt + "changed")
    elif changed == "provider": encoded = replace(encoded, provider_prompt=encoded.provider_prompt + " ")
    elif changed == "table": encoded = replace(encoded, table=None)
    elif changed == "historical": encoded = replace(encoded, freshness_checked=False)
    elif changed == "bool_freshness": encoded = replace(encoded, freshness_checked=1)
    with pytest.raises(projection.InlineEvidenceProjectionError): projection.project_inline_evidence(encoded)


def test_codec_never_reads_files_starts_processes_or_uses_network(encoded_factory, monkeypatch):
    encoded = encoded_factory()
    def forbidden(*args, **kwargs): pytest.fail("pure codec attempted external operation")
    monkeypatch.setattr(semantic, "_read", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    view = projection.project_inline_evidence(encoded)
    assert projection.restore_inline_evidence(view.provider_prompt, view.receipt) == encoded.provider_prompt
    projection.select_inline_evidence(view, "prefix" + encoded.provider_prompt + "suffix",
        "prefix" + view.provider_prompt + "suffix")


@pytest.mark.parametrize("field,value", [("artifact_count", True), ("evidence_row_count", -1),
    ("removed_string_overhead_bytes", 0), ("candidate_view_sha256", "PRIVATE-TEXT"),
    ("native_transport_bytes", True), ("proof_authority", True), ("provider_token_usage", 0),
    ("unknown-field", "PRIVATE-SECRET")])
def test_malformed_projection_receipt_is_rejected(encoded_factory, field, value):
    view = projection.project_inline_evidence(encoded_factory())
    receipt = {**view.receipt, field: value}
    with pytest.raises(projection.InlineEvidenceProjectionError) as error:
        projection.restore_inline_evidence(view.provider_prompt, receipt)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("change", ["body", "chunk_ref", "instructions", "unused_artifact", "unknown_top", "required"])
def test_candidate_mutation_fails_against_retained_receipt(encoded_factory, change):
    view = projection.project_inline_evidence(encoded_factory())
    wire = json.loads(view.provider_prompt)
    if change == "body": wire["inline_evidence_artifacts"][0]["status"] = "proved"
    elif change == "chunk_ref": wire["native_context"]["evidence"][0]["summary"][projection.REFERENCE_KEY] = 1
    elif change == "instructions": wire["inline_evidence_instructions"] = "PRIVATE changed"
    elif change == "unused_artifact": wire["inline_evidence_artifacts"].append({})
    elif change == "required": wire["native_context"]["evidence"][0]["metadata"]["required"] = False
    else: wire["unknown"] = True
    with pytest.raises(projection.InlineEvidenceProjectionError):
        projection.restore_inline_evidence(canonical(wire), view.receipt)


def test_complete_prompt_selection_and_accounting_use_actual_wrappers(encoded_factory):
    encoded = encoded_factory(); view = projection.project_inline_evidence(encoded)
    prefix, suffix = 'Workspace λ and policy:\n', '\nOriginal public instruction and Doctor constraints remain literal.'
    original = prefix + encoded.provider_prompt + suffix
    candidate = prefix + view.provider_prompt + suffix
    result = projection.select_inline_evidence(view, original, candidate)
    assert result.selected_mode == projection.MODE and result.selected_prompt == candidate
    receipt = result.receipt
    assert receipt["native_complete_bytes"] == len(original.encode())
    assert receipt["candidate_complete_bytes"] == len(candidate.encode())
    assert receipt["candidate_complete_bytes"] < receipt["native_complete_bytes"]
    assert receipt["provider_token_usage"] is None
    assert projection.validate_selection_receipt(receipt) == receipt
    with pytest.raises(FrozenInstanceError): result.selected_mode = "changed"


@pytest.mark.parametrize("change", ["prefix", "suffix", "missing", "duplicate", "candidate_duplicate"])
def test_changed_or_ambiguous_complete_wrappers_are_rejected(encoded_factory, change):
    encoded = encoded_factory(); view = projection.project_inline_evidence(encoded)
    original, candidate = "before" + encoded.provider_prompt + "after", "before" + view.provider_prompt + "after"
    if change == "prefix": candidate = "changed" + candidate
    elif change == "suffix": candidate += "changed"
    elif change == "missing": original = "missing transport"
    elif change == "duplicate": original += encoded.provider_prompt
    else: candidate += view.provider_prompt
    with pytest.raises(projection.InlineEvidenceProjectionError):
        projection.select_inline_evidence(view, original, candidate)


@pytest.mark.parametrize("changed", ["mode", "selected_hash", "wrapper_size", "authority", "extra"])
def test_selection_receipt_cannot_invent_savings(encoded_factory, changed):
    encoded = encoded_factory(); view = projection.project_inline_evidence(encoded)
    value = projection.select_inline_evidence(view, encoded.provider_prompt, view.provider_prompt).receipt
    if changed == "mode": value["selected_mode"] = "legacy"
    elif changed == "selected_hash": value["selected_complete_sha256"] = "0" * 64
    elif changed == "wrapper_size": value["candidate_complete_bytes"] += 1
    elif changed == "authority": value["completion_authority"] = True
    else: value["private"] = "PRIVATE-SECRET"
    with pytest.raises(projection.InlineEvidenceProjectionError): projection.validate_selection_receipt(value)


def test_known_reserved_summary_marker_cannot_collide(encoded_factory):
    def mutate(wire):
        row = next(row for row in wire["evidence"] if row["kind"] == "code-retrieval-context")
        row["summary"] = {projection.REFERENCE_KEY: 0}
    with pytest.raises(projection.InlineEvidenceProjectionError, match="marker collision"):
        projection.project_inline_evidence(encoded_factory(mutate=mutate))


def test_bounded_json_and_no_artifact_candidate_keep_native_fallback(encoded_factory, monkeypatch):
    encoded = encoded_factory({})
    view = projection.project_inline_evidence(encoded)
    assert view.receipt["artifact_count"] == 0
    selected = projection.select_inline_evidence(view, encoded.provider_prompt, view.provider_prompt)
    assert selected.selected_mode == "legacy"
    monkeypatch.setattr(projection, "MAX_BYTES", 10)
    with pytest.raises(projection.InlineEvidenceProjectionError): projection.project_inline_evidence(encoded)
