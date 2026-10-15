"""Authored lossless candidate views; no provider, filesystem or admission."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_prompt_projection as codec


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def authored(*, version=3, repeated=500, suffix="\n\nLiteral suffix: café 😀 \\\"keep\\\"\r\n", public=True):
    native = dict(contract_version=1, objective_id="AUTHORED-TASK", goal={"todo": "Preserve all behavior"},
        scope={"paths": ["a.py"]}, acceptance={"criteria": ["Do not weaken checks"]},
        authority={"completion_authority": False, "unknown_authority": {"requires": "owner"}},
        evidence=[dict(kind="semantic-context", tier="invariant", metadata={"required": True},
            summary=canonical({"source": 'print("café")\n' * repeated,
                "unknown_behavior": {"do_not_drop": ["§", 0, False, None]}})),
            dict(kind="unknown-evidence", unknown={"required": "retain exactly"})],
        native_context={"collision": "This is an ordinary unknown native field"})
    original = dict(schema="supervisor-owner-candidate-router-input@" + str(version),
        native_prompt=canonical(native) + suffix,
        original_public_instruction_block="\nPUBLIC: Keep all acceptance criteria.\n" if public else "",
        instructions="Exact original guidance", proof_authority=False,
        candidate_binding={"workspace_generation": "a" * 64},
        original_requirements_binding={"native_prompt_sha256": sha(canonical(native) + suffix)},
        unknown_outer={"native_prompt": "An unknown nested native_prompt is not the selected input"},
        native_context={"collision": "An unknown outer field must remain intact"},
        native_suffix="Unknown outer suffix is independent of the selected native suffix",
        original_candidate={"collision": "Unknown outer field remains intact"})
    if version >= 2:
        original["doctor_advisory"] = "Do not grant extra calls. Pending proof remains pending."
    if version == 3:
        original["public_validation"] = {"feedback": {"status": "inconclusive", "authority": False}}
    return canonical(original), original, native


@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("public", [True, False])
def test_exact_round_trip_preserves_all_native_outer_unknown_and_required_fields(version, public):
    prompt, original, native = authored(version=version, public=public)
    view = codec.project_candidate_prompt(prompt)
    wire = json.loads(view.provider_prompt)
    assert codec.restore_candidate_prompt(view.provider_prompt, view.receipt) == prompt
    assert wire["native_context"] == native
    assert wire["original_candidate"] == {key: value for key, value in original.items() if key != "native_prompt"}
    assert wire["native_context"]["evidence"][0]["metadata"]["required"] is True
    assert wire["native_context"]["evidence"][0]["summary"] == native["evidence"][0]["summary"]
    assert wire["native_suffix"] == original["native_prompt"][len(canonical(native)):]
    assert view.receipt["original_input_sha256"] == sha(prompt)
    assert view.receipt["public_instruction_block_bytes"] == len(original["original_public_instruction_block"].encode())
    assert view.receipt["native_prompt_sha256"] == original["original_requirements_binding"]["native_prompt_sha256"]
    assert "PUBLIC:" not in view.receipt_json and "print(" not in view.receipt_json
    assert all(view.receipt[key] is False for key in codec._FALSE_FIELDS)
    assert view.receipt["representation_lossless"] is True


@pytest.mark.parametrize("suffix", ["", " ", "\n\n", "\r\n\t", "{}", "[1,2]", "\n😀\\x\"tail\"\n", "\0"])
def test_suffix_is_opaque_exact_text_including_json_looking_and_control_text(suffix):
    prompt, _, _ = authored(suffix=suffix)
    view = codec.project_candidate_prompt(prompt)
    assert json.loads(view.provider_prompt)["native_suffix"] == suffix
    assert codec.restore_candidate_prompt(provider_prompt=view.provider_prompt, receipt=view.receipt) == prompt


def test_view_is_immutable_and_receipt_is_a_detached_copy():
    view = codec.project_candidate_prompt(authored()[0])
    with pytest.raises(FrozenInstanceError):
        view.original_prompt = "changed"
    receipt = view.receipt
    receipt["proof_authority"] = True
    assert view.receipt["proof_authority"] is False


def test_optional_evidence_keeps_original_requirement_status():
    _, original, native = authored()
    native["evidence"].append(dict(kind="advisory-example", tier="evidence", metadata={"required": False},
        summary="Optional example; not an acceptance criterion."))
    original["native_prompt"] = canonical(native) + "\nLiteral task suffix.\n"
    prompt = canonical(original)
    view = codec.project_candidate_prompt(prompt)
    wire = json.loads(view.provider_prompt)
    assert wire["native_context"]["evidence"][0]["metadata"]["required"] is True
    assert wire["native_context"]["evidence"][-1]["metadata"]["required"] is False
    assert wire["native_context"]["evidence"][-1]["tier"] == "evidence"
    assert "original requirement status" in wire["instructions"]
    assert "inline and required" not in wire["instructions"]
    assert codec.restore_candidate_prompt(view.provider_prompt, view.receipt) == prompt


@pytest.mark.parametrize("number", [0.0, -0.0, 0.125, -1.75, 1e-100, 1e100, 5e-324, 1.7976931348623157e308])
def test_finite_public_timing_and_unknown_numbers_match_owner_json_exactly(number):
    _, original, native = authored()
    original["public_validation"]["feedback"]["measurements"] = {
        "seconds": number, "median": 0.001345, "stable": False}
    original["unknown_outer"]["number"] = number
    native["unknown_finite_number"] = number
    original["native_prompt"] = canonical(native) + "\nKeep the literal native suffix.\n"
    prompt = canonical(original)
    view = codec.project_candidate_prompt(prompt)
    wire = json.loads(view.provider_prompt)
    observed = wire["original_candidate"]["public_validation"]["feedback"]["measurements"]["seconds"]
    assert type(observed) is float and observed == number
    assert math.copysign(1, observed) == math.copysign(1, number)
    assert wire["native_context"]["unknown_finite_number"] == number
    assert codec.restore_candidate_prompt(view.provider_prompt, view.receipt) == prompt
    selected = codec.select_candidate_prompt(view, prompt, view.provider_prompt)
    assert selected.selected_mode == codec.MODE


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999", "1e-999"])
@pytest.mark.parametrize("position", ["outer", "native"])
def test_nonfinite_overflow_and_noncanonical_underflow_numbers_refuse_projection(number, position):
    prompt, original, native = authored(repeated=0)
    if position == "outer":
        prompt = prompt[:-1] + ',"unknown_number":' + number + "}"
    else:
        original["native_prompt"] = canonical(native)[:-1] + ',"unknown_number":' + number + "}"
        prompt = canonical(original)
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.project_candidate_prompt(prompt)


@pytest.mark.parametrize("position", ["outer", "native"])
def test_node_limit_refuses_large_shallow_json_within_byte_bound(position):
    _, original, native = authored(repeated=0)
    nodes = [0] * 100_001
    if position == "outer":
        original["unknown_nodes"] = nodes
    else:
        native["unknown_nodes"] = nodes
        original["native_prompt"] = canonical(native)
    prompt = canonical(original)
    assert len(prompt.encode()) < codec.MAX_BYTES
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.project_candidate_prompt(prompt)


def test_complete_input_selector_keeps_literal_wrappers_and_records_no_token_claims():
    view = codec.project_candidate_prompt(authored()[0])
    native = "allocated checkout\n" + view.original_prompt + "\nreply grammar\n"
    candidate = "allocated checkout\n" + view.provider_prompt + "\nreply grammar\n"
    result = codec.select_candidate_prompt(view, native, candidate)
    assert result.selected_prompt == candidate and result.selected_mode == codec.MODE
    receipt = codec.validate_selection_receipt(result.receipt)
    assert receipt["selected_complete_sha256"] == sha(candidate)
    assert receipt["native_complete_bytes"] == len(native.encode())
    assert receipt["candidate_complete_bytes"] < receipt["native_complete_bytes"]
    assert receipt["view_receipt"] == view.receipt
    assert receipt["provider_token_usage_verified"] is receipt["complete_run_token_savings_claimed"] is False
    assert receipt["selection_basis"] == "complete_utf8_bytes"
    assert not any("token" in key for key in receipt if key not in codec._FALSE_FIELDS)


@pytest.mark.parametrize("extra", [0, 1, 100])
def test_equal_or_larger_complete_input_falls_back_exactly(monkeypatch, extra):
    prompt = authored()[0]
    initial = codec.project_candidate_prompt(prompt)
    savings = len(prompt.encode()) - len(initial.provider_prompt.encode())
    assert savings > 0
    # Authored fixed instruction text sets an exact equality/larger boundary.
    monkeypatch.setattr(codec, "INSTRUCTIONS", codec.INSTRUCTIONS + "x" * (savings + extra))
    view = codec.project_candidate_prompt(prompt)
    assert len(view.provider_prompt.encode()) == len(prompt.encode()) + extra
    result = codec.select_candidate_prompt(view, "prefix" + prompt + "suffix",
        "prefix" + view.provider_prompt + "suffix")
    assert result.selected_prompt == "prefix" + prompt + "suffix"
    assert result.selected_mode == "legacy"
    assert result.receipt["fallback_reason"] == "complete_input_not_smaller"
    codec.validate_selection_receipt(result.receipt)


def test_small_context_uses_unchanged_legacy_input_when_wrapper_overhead_wins():
    original = dict(schema="supervisor-owner-candidate-router-input@1",
        native_prompt=canonical(dict(contract_version=1, evidence=[], goal={}, scope={}, acceptance={}, authority={})),
        original_public_instruction_block="")
    prompt = canonical(original)
    view = codec.project_candidate_prompt(prompt)
    result = codec.select_candidate_prompt(view, prompt, view.provider_prompt)
    assert result.selected_mode == "legacy" and result.selected_prompt == prompt


@pytest.mark.parametrize("fault", ["changed_prefix", "changed_suffix", "missing_original", "duplicate_original",
    "changed_original_field", "changed_view_field", "untyped_view"])
def test_selector_refuses_changed_wrappers_or_unbound_views(fault):
    view = codec.project_candidate_prompt(authored()[0])
    native, candidate = "before" + view.original_prompt + "after", "before" + view.provider_prompt + "after"
    if fault == "changed_prefix": candidate = "foreign" + view.provider_prompt + "after"
    if fault == "changed_suffix": candidate += "changed"
    if fault == "missing_original": native = "no admitted candidate"
    if fault == "duplicate_original": native += view.original_prompt
    if fault == "changed_original_field": view = replace(view, original_prompt=view.original_prompt + " ")
    if fault == "changed_view_field": view = replace(view, provider_prompt=view.provider_prompt + " ")
    if fault == "untyped_view": view = {"provider_prompt": view.provider_prompt}
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.select_candidate_prompt(view, native, candidate)


@pytest.mark.parametrize("fault", ["unknown_schema", "missing_native", "native_not_string", "missing_public",
    "duplicate_outer", "duplicate_native", "noncanonical_outer", "noncanonical_native", "native_array",
    "missing_evidence", "nonlist_evidence", "missing_goal", "wrong_contract", "bool_contract",
    "nonfinite", "invalid_utf8", "deep_outer", "deep_native", "too_big", "malformed_native"])
def test_invalid_or_unsupported_projection_never_weakens_the_input(fault):
    prompt, original, native = authored(repeated=0)
    if fault == "unknown_schema": original["schema"] = "supervisor-owner-candidate-router-input@4"
    if fault == "missing_native": del original["native_prompt"]
    if fault == "native_not_string": original["native_prompt"] = native
    if fault == "missing_public": del original["original_public_instruction_block"]
    if fault == "noncanonical_outer": prompt = json.dumps(original, ensure_ascii=False)
    elif fault == "duplicate_outer": prompt = prompt[:-1] + ',"schema":"' + original["schema"] + '"}'
    elif fault == "duplicate_native": original["native_prompt"] = canonical(native)[:-1] + ',"evidence":[]}'
    elif fault == "noncanonical_native": original["native_prompt"] = json.dumps(native, ensure_ascii=False)
    elif fault == "native_array": original["native_prompt"] = "[]"
    elif fault == "missing_evidence": del native["evidence"]; original["native_prompt"] = canonical(native)
    elif fault == "nonlist_evidence": native["evidence"] = {}; original["native_prompt"] = canonical(native)
    elif fault == "missing_goal": del native["goal"]; original["native_prompt"] = canonical(native)
    elif fault == "wrong_contract": native["contract_version"] = 2; original["native_prompt"] = canonical(native)
    elif fault == "bool_contract": native["contract_version"] = True; original["native_prompt"] = canonical(native)
    elif fault == "nonfinite": original["native_prompt"] = canonical(native)[:-1] + ',"nan":NaN}'
    elif fault == "invalid_utf8": original["unknown"] = "\ud800"
    elif fault in {"deep_outer", "deep_native"}:
        nested = 0
        for _ in range(45): nested = [nested]
        if fault == "deep_outer": original["unknown"] = nested
        else: native["unknown"] = nested; original["native_prompt"] = canonical(native)
    elif fault == "too_big": original["unknown"] = "x" * codec.MAX_BYTES
    elif fault == "malformed_native": original["native_prompt"] = '{"unfinished":'
    if fault not in {"duplicate_outer", "noncanonical_outer"}:
        prompt = canonical(original)
    with pytest.raises(codec.CandidatePromptProjectionError) as error:
        codec.project_candidate_prompt(prompt)
    assert "print(" not in str(error.value) and "PUBLIC:" not in str(error.value)


@pytest.mark.parametrize("location", ["native_context", "native_suffix", "original_candidate", "instructions",
    "schema", "unexpected", "nested_collision"])
def test_tampered_view_cannot_restore_against_retained_original_receipt(location):
    view = codec.project_candidate_prompt(authored()[0])
    wire = json.loads(view.provider_prompt)
    if location == "native_context": wire[location]["evidence"][0]["summary"] = "omitted"
    elif location == "original_candidate": wire[location]["original_public_instruction_block"] += "weakened"
    elif location == "nested_collision": wire["original_candidate"]["native_prompt"] = "injected duplicate role"
    elif location == "unexpected": wire[location] = False
    else: wire[location] += "changed"
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.restore_candidate_prompt(canonical(wire), view.receipt)


def test_rehashing_projected_view_does_not_replace_independently_retained_original_digest():
    view = codec.project_candidate_prompt(authored()[0])
    wire = json.loads(view.provider_prompt)
    wire["original_candidate"]["instructions"] = "unauthorized changed instruction"
    changed = canonical(wire)
    receipt = dict(view.receipt, candidate_view_sha256=sha(changed), candidate_view_bytes=len(changed.encode()))
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.restore_candidate_prompt(changed, receipt)


@pytest.mark.parametrize("fault", ["missing", "extra", "schema", "authority", "byte_bool", "byte_negative",
    "digest", "lossless", "source_schema", "prefix_sum"])
def test_projection_receipt_is_closed_and_strictly_typed(fault):
    view = codec.project_candidate_prompt(authored()[0])
    receipt = view.receipt
    if fault == "missing": del receipt["native_prompt_sha256"]
    elif fault == "extra": receipt["invented_authority"] = True
    elif fault == "schema": receipt["schema"] += "changed"
    elif fault == "authority": receipt["proof_authority"] = True
    elif fault == "byte_bool": receipt["native_prompt_bytes"] = True
    elif fault == "byte_negative": receipt["native_prompt_bytes"] = -1
    elif fault == "digest": receipt["native_prompt_sha256"] = "g" * 64
    elif fault == "lossless": receipt["representation_lossless"] = 1
    elif fault == "source_schema": receipt["source_schema"] = "unknown"
    elif fault == "prefix_sum": receipt["native_suffix_bytes"] += 1
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.restore_candidate_prompt(view.provider_prompt, receipt)


@pytest.mark.parametrize("fault", ["missing", "extra", "schema", "basis", "authority", "view_hash",
    "selected_mode", "fallback", "selected_hash", "selected_bytes", "wrapper_bytes", "byte_bool",
    "negative_bytes", "false_savings_claim", "nested_receipt"])
def test_selection_receipt_rejects_false_claims_and_inconsistent_joins(fault):
    view = codec.project_candidate_prompt(authored()[0])
    receipt = codec.select_candidate_prompt(view, view.original_prompt, view.provider_prompt).receipt
    if fault == "missing": del receipt["view_receipt"]
    elif fault == "extra": receipt["provider_tokens"] = 0
    elif fault == "schema": receipt["schema"] += "changed"
    elif fault == "basis": receipt["selection_basis"] = "verified_tokens"
    elif fault == "authority": receipt["completion_authority"] = True
    elif fault == "view_hash": receipt["view_receipt_sha256"] = "0" * 64
    elif fault == "selected_mode": receipt["selected_mode"] = "legacy"
    elif fault == "fallback": receipt["fallback_reason"] = "none"
    elif fault == "selected_hash": receipt["selected_complete_sha256"] = "0" * 64
    elif fault == "selected_bytes": receipt["selected_complete_bytes"] += 1
    elif fault == "wrapper_bytes": receipt["candidate_complete_bytes"] += 1
    elif fault == "byte_bool": receipt["native_complete_bytes"] = True
    elif fault == "negative_bytes": receipt["candidate_complete_bytes"] = -1
    elif fault == "false_savings_claim": receipt["complete_run_token_savings_claimed"] = True
    elif fault == "nested_receipt": receipt["view_receipt"]["unknown_fields_preserved"] = False
    with pytest.raises(codec.CandidatePromptProjectionError):
        codec.validate_selection_receipt(receipt)


def test_selection_receipt_validation_does_not_mutate_caller_data():
    view = codec.project_candidate_prompt(authored()[0])
    receipt = codec.select_candidate_prompt(view, view.original_prompt, view.provider_prompt).receipt
    before = copy.deepcopy(receipt)
    validated = codec.validate_selection_receipt(receipt)
    validated["view_receipt"]["source_schema"] = "changed copy"
    assert receipt == before
