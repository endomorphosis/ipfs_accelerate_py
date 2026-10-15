"""Pack factoring preserves actual producer contracts and exact native bytes."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_pack_census import _authored_extension
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as semantic
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_pack_projection as projection


def _encoded(native):
    root, _, prompt, _ = native
    return semantic.encode_semantic_router_prompt(prompt=prompt, repository=root)


def _pack(text):
    return json.loads(text)["translated_semantic"]["pack"]


def _inverse(encoded, view):
    restored = projection.restore_semantic_pack(view.provider_prompt, view.receipt)
    assert restored == encoded.provider_prompt
    assert semantic._restore(restored, encoded.table) == encoded.native_prompt
    assert projection.validate_projection_receipt(view.receipt) == view.receipt


def test_actual_producer_pack_preserves_roles_identifiers_and_all_remaining_fields(native):
    encoded = _encoded(native); before = deepcopy(encoded)
    original = json.loads(encoded.provider_prompt)
    view = projection.project_semantic_pack(encoded)
    candidate = json.loads(view.provider_prompt)
    old, new = original["translated_semantic"]["pack"], candidate["translated_semantic"]["pack"]
    defaults = candidate["semantic_pack_recipe"]["reference_defaults"]
    assert defaults == projection.DEFAULT_FIELDS
    assert view.receipt["token_estimate_factored"] is True
    assert "token_estimate" not in new
    assert new["pack"] == old["pack"] and new["pack_cid"] == old["pack_cid"]
    assert len(old["references"]) == len(new["references"])
    assert len({row["kind"] for row in new["references"]} &
        {"target_source", "surrounding_source", "test_source"}) == 3
    for source, target in zip(old["references"], new["references"]):
        assert target == {key: value for key, value in source.items() if key not in defaults}
        assert {**target, **defaults} == source
    # Inner native CIDs and translated typed reference aliases retain their
    # distinct representations; no new substitution or reference merging occurs.
    assert old["pack"]["target_source_cid"] == old["pack"]["test_source_cid"]
    for key in set(old) - {"references", "token_estimate"}: assert new[key] == old[key]
    for key in set(original) - {"schema", "translated_semantic"}: assert candidate[key] == original[key]
    for key in set(original["translated_semantic"]) - {"pack"}:
        assert candidate["translated_semantic"][key] == original["translated_semantic"][key]
    _inverse(encoded, view)
    assert encoded == before
    with pytest.raises(FrozenInstanceError): view.provider_prompt = "changed"
    receipt = view.receipt
    for key in projection._FALSE_FIELDS: assert receipt[key] is False
    for key in projection._TRUE_FIELDS: assert receipt[key] is True
    assert receipt["provider_token_usage"] is None
    assert all(text not in view.receipt_json for text in (str(native[0]), "mod.py", "TASK-1", "Inspect both"))


@pytest.mark.parametrize("confidence,admission", [("exact", "exact"),
    ("conservative", "conservative"), ("heuristic", None)])
def test_real_context_packer_outputs_keep_assurance_caveats_and_raw_requirements(native, confidence, admission):
    from test.api.semantic_state.test_context_pack import _admit, _base_kwargs
    from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import pack_context
    from ipfs_accelerate_py.agent_supervisor.semantic_state.capsules import ADMISSION_EXACT, ADMISSION_CONSERVATIVE
    admission = {"exact": ADMISSION_EXACT, "conservative": ADMISSION_CONSERVATIVE}.get(admission)
    dependency = _admit(label=confidence, confidence=confidence, assessment_admission=admission)
    actual = pack_context(**_base_kwargs(dependency_admissions=[dependency])).to_dict()
    encoded = _authored_extension(native, lambda pack: pack.update(actual))
    view = projection.project_semantic_pack(encoded)
    original, projected = _pack(encoded.provider_prompt), _pack(view.provider_prompt)
    assert projected["pack"] == original["pack"]
    assert projected["decisions"] == original["decisions"]
    assert [row["metadata"] for row in projected["references"]] == [row["metadata"] for row in original["references"]]
    if confidence == "heuristic":
        raw = [row for row in projected["references"] if row["kind"] == "raw_dependency_source"]
        assert len(raw) == 1 and raw[0]["metadata"]["raw_source_required"] is True
    if confidence == "conservative":
        assert any("conservative" in text for text in projected["pack"]["assumptions"])
    _inverse(encoded, view)


def test_unknown_fields_remain_literal_and_never_enter_receipt(native):
    def mutate(pack):
        pack["PRIVATE-EXTRA"] = {"semantic_pack_recipe": {"reference_defaults": {"path": "literal"}}}
        pack["pack"]["PRIVATE-INNER"] = 'λ 世界 \\ "\n'
        pack["references"][0]["PRIVATE-REFERENCE"] = ["semantic-pack-view@1", 0]
        pack["references"][0]["metadata"]["PRIVATE-METADATA"] = {"required": True}
        pack["references"][0]["summary"] = 'PRIVATE-SUMMARY λ \\ "\n'
    encoded = _authored_extension(native, mutate)
    view = projection.project_semantic_pack(encoded)
    original, candidate = _pack(encoded.provider_prompt), _pack(view.provider_prompt)
    assert candidate["PRIVATE-EXTRA"] == original["PRIVATE-EXTRA"]
    assert candidate["pack"] == original["pack"]
    assert candidate["references"][0]["PRIVATE-REFERENCE"] == original["references"][0]["PRIVATE-REFERENCE"]
    assert candidate["references"][0]["metadata"] == original["references"][0]["metadata"]
    assert candidate["references"][0]["summary"] == original["references"][0]["summary"]
    assert "PRIVATE" not in view.receipt_json
    _inverse(encoded, view)


@pytest.mark.parametrize("key", sorted(projection.DEFAULT_FIELDS))
def test_missing_defaults_are_never_synthesized(native, key):
    encoded = _authored_extension(native, lambda pack: pack["references"][0].pop(key))
    view = projection.project_semantic_pack(encoded)
    candidate = json.loads(view.provider_prompt)
    assert key not in candidate["semantic_pack_recipe"]["reference_defaults"]
    assert key not in candidate["translated_semantic"]["pack"]["references"][0]
    _inverse(encoded, view)


def test_valid_nondefault_envelopes_stay_literal(native):
    different = dict(schema="future-context-reference@2", contract_version=2,
        repository_id="repo:another", tree_id="tree:another", path="literal/path.py", byte_count=1)
    encoded = _authored_extension(native, lambda pack: pack["references"][0].update(different))
    view = projection.project_semantic_pack(encoded)
    assert json.loads(view.provider_prompt)["semantic_pack_recipe"]["reference_defaults"] == {}
    assert all(_pack(view.provider_prompt)["references"][0][key] == value for key, value in different.items())
    _inverse(encoded, view)


@pytest.mark.parametrize("change", ["extra", "different", "missing-input", "unknown-category"])
def test_estimate_recipe_requires_exact_three_fields_and_matching_inner_values(native, change):
    def mutate(pack):
        if change == "extra": pack["token_estimate"]["PRIVATE-ESTIMATE"] = "preserve"
        elif change == "different": pack["token_estimate"] = dict(totals={"future": 19}, estimator_version="other@2", total=19)
        elif change == "missing-input": del pack["pack"]["estimator_version"]
        else:
            pack["pack"]["token_totals"]["PRIVATE-CATEGORY"] = 17
            pack["token_estimate"]["totals"]["PRIVATE-CATEGORY"] = 17
            pack["token_estimate"]["total"] += 17
    encoded = _authored_extension(native, mutate)
    view = projection.project_semantic_pack(encoded)
    assert view.receipt["token_estimate_factored"] is (change == "unknown-category")
    if change != "unknown-category":
        assert _pack(view.provider_prompt)["token_estimate"] == _pack(encoded.provider_prompt)["token_estimate"]
    assert "PRIVATE" not in view.receipt_json
    _inverse(encoded, view)


@pytest.mark.parametrize("change", ["bool-byte", "float-byte", "bool-version", "negative-category", "bool-total", "sum-mismatch"])
def test_malformed_known_numeric_contracts_reject_without_echo(native, change):
    if change == "float-byte":
        # Native proof JSON rejects floats before encoding. Exercise the pack
        # decoder directly with independently malformed incoming view bytes.
        view = projection.project_semantic_pack(_encoded(native))
        wire = json.loads(view.provider_prompt)
        wire["translated_semantic"]["pack"]["references"][0]["byte_count"] = 0.0
        text = json.dumps(wire, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        receipt = {**view.receipt, "candidate_view_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "candidate_view_bytes": len(text.encode())}
        with pytest.raises(projection.SemanticPackProjectionError, match="envelope integer differs"):
            projection.restore_semantic_pack(text, receipt)
        return
    def mutate(pack):
        if change == "bool-byte": pack["references"][0]["byte_count"] = False
        elif change == "bool-version": pack["references"][0]["contract_version"] = True
        elif change == "negative-category": pack["pack"]["token_totals"]["PRIVATE-CATEGORY"] = -1
        elif change == "bool-total": pack["token_estimate"]["total"] = True
        else: pack["token_estimate"]["total"] += 1
    encoded = _authored_extension(native, mutate)
    with pytest.raises(projection.SemanticPackProjectionError) as error:
        projection.project_semantic_pack(encoded)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("count", [0, 1])
def test_small_reference_sets_fall_back_to_complete_original_input(native, count):
    encoded = _authored_extension(native, lambda pack: pack.update(references=pack["references"][:count]))
    view = projection.project_semantic_pack(encoded)
    _inverse(encoded, view)
    if count == 0: assert view.receipt["reference_default_fields"] == 0
    prefix, suffix = "literal prefix λ\n", "\nunchanged native task and policy"
    selected = projection.select_semantic_pack(view, prefix+encoded.provider_prompt+suffix,
        prefix+view.provider_prompt+suffix)
    assert selected.selected_mode == "legacy"
    assert selected.selected_prompt == prefix+encoded.provider_prompt+suffix
    assert selected.receipt["fallback_reason"] == "complete_input_not_smaller"
    assert projection.validate_selection_receipt(selected.receipt) == selected.receipt


def test_complete_selection_measures_recipe_instructions_and_identical_wrappers(native):
    encoded = _encoded(native); view = projection.project_semantic_pack(encoded)
    prefix, suffix = "Policy and workspace λ\n", "\nNative Doctor and public instruction."
    old, new = prefix+encoded.provider_prompt+suffix, prefix+view.provider_prompt+suffix
    selected = projection.select_semantic_pack(view, old, new)
    assert selected.selected_mode == projection.MODE and selected.selected_prompt == new
    assert selected.receipt["native_complete_bytes"] == len(old.encode())
    assert selected.receipt["candidate_complete_bytes"] == len(new.encode())
    assert projection.validate_selection_receipt(selected.receipt) == selected.receipt
    for wrong in ("changed"+new, new+"changed", new+view.provider_prompt):
        with pytest.raises(projection.SemanticPackProjectionError): projection.select_semantic_pack(view, old, wrong)


@pytest.mark.parametrize("change", ["default", "collision", "estimate", "inner", "instructions", "schema"])
def test_recipe_tampering_rejects_even_with_recomputed_candidate_digest(native, change):
    view = projection.project_semantic_pack(_encoded(native)); wire = json.loads(view.provider_prompt)
    pack = wire["translated_semantic"]["pack"]
    if change == "default": wire["semantic_pack_recipe"]["reference_defaults"]["path"] = "PRIVATE-CHANGED"
    elif change == "collision": pack["references"][0]["byte_count"] = 0
    elif change == "estimate": pack["token_estimate"] = dict(totals={}, estimator_version="other", total=0)
    elif change == "inner": pack["pack"]["objective"] = "PRIVATE-CHANGED"
    elif change == "instructions": wire["semantic_pack_instructions"] = "PRIVATE-CHANGED"
    else: wire["schema"] = "unsupported@999"
    text = semantic._json(wire)
    receipt = {**view.receipt, "candidate_view_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "candidate_view_bytes": len(text.encode())}
    with pytest.raises(projection.SemanticPackProjectionError) as error:
        projection.restore_semantic_pack(text, receipt)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("change", ["native", "table", "historical", "integer-freshness", "compact", "noncanonical"])
def test_only_exact_current_original_transport_is_admitted(native, change):
    root, _, prompt, _ = native
    encoded = semantic.encode_semantic_router_prompt(prompt=prompt, repository=root,
        transport_schema=semantic.COMPACT_TRANSPORT_SCHEMA if change == "compact" else semantic.TRANSPORT_SCHEMA)
    if change == "native": encoded = replace(encoded, native_prompt="PRIVATE-CHANGED")
    elif change == "table": encoded = replace(encoded, table=None)
    elif change == "historical": encoded = replace(encoded, freshness_checked=False)
    elif change == "integer-freshness": encoded = replace(encoded, freshness_checked=1)
    elif change == "noncanonical": encoded = replace(encoded, provider_prompt=encoded.provider_prompt+" ")
    with pytest.raises(projection.SemanticPackProjectionError) as error: projection.project_semantic_pack(encoded)
    assert "PRIVATE" not in str(error.value)


def test_closed_receipt_types_counts_authorities_and_selection_are_checked(native):
    encoded = _encoded(native); view = projection.project_semantic_pack(encoded)
    changes = [("reference_count", True), ("reference_fields_factored", 1), ("reference_default_fields", 7),
        ("candidate_view_sha256", "PRIVATE-DIGEST"), ("native_transport_bytes", True),
        ("proof_authority", True), ("provider_token_usage", 0), ("unknown-field", "PRIVATE-CONTENT")]
    for key, value in changes:
        with pytest.raises(projection.SemanticPackProjectionError) as error:
            projection.validate_projection_receipt({**view.receipt, key: value})
        assert "PRIVATE" not in str(error.value)
    selected = projection.select_semantic_pack(view, encoded.provider_prompt, view.provider_prompt)
    for key, value in [("schema", "unknown@1"), ("selected_mode", "legacy"),
        ("view_receipt_sha256", "0"*64), ("native_complete_bytes", True), ("unknown", "PRIVATE")]:
        with pytest.raises(projection.SemanticPackProjectionError):
            projection.validate_selection_receipt({**selected.receipt, key: value})


def test_projection_restore_and_selection_perform_no_external_operations(native, monkeypatch):
    encoded = _encoded(native)
    def forbidden(*args, **kwargs): pytest.fail("pure pack codec attempted external operation")
    monkeypatch.setattr(semantic, "_read", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    view = projection.project_semantic_pack(encoded)
    _inverse(encoded, view)
    projection.select_semantic_pack(view, encoded.provider_prompt, view.provider_prompt)
