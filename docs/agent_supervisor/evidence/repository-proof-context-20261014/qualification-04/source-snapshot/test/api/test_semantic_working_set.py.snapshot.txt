"""Pure provenance deferral over real, authored semantic producer output.

No provider, benchmark task, owner database or broker connection is used.
The extension fixture only exercises future transport-field preservation;
it does not assert new producer or proof authority.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import build_text_context_references
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set as working
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
from ipfs_accelerate_py.agent_supervisor.runtime.worker_semantic_evidence_client import guidance
from test.api.test_semantic_metadata_view import native, native_factory  # noqa: F401
from test.api.test_worker_source_client import descriptor as source_descriptor


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def descriptor(encoded):
    return source_descriptor(source.semantic_binding(encoded, "identity@1")) | dict(
        schema=working.DESCRIPTOR_SCHEMA, source_edit_mode="disabled",
        view_policy_binding_sha256=None, source_scope_sha256="f" * 64,
        semantic_evidence_mode=working.EVIDENCE_MODE)


def native_capsules(encoded):
    return json.loads(codec._prompt_parts(encoded.native_prompt)[4])["capsules"]


def resolver(encoded, calls):
    capsules = native_capsules(encoded)
    def resolve(ref):
        assert set(ref) == {"schema", "capsule_index", "capsule_sha256"}
        assert ref["schema"] == working.REFERENCE_SCHEMA
        calls.append(ref)
        return canonical(capsules[ref["capsule_index"]]).encode()
    return resolve


def selected(encoded, desc=None):
    view = working.project_semantic_working_set(encoded, desc or descriptor(encoded))
    prefix = "Workspace and security contract: café.\n"
    suffix = "\nPublic task, validation obligations, and reply contract remain exact."
    result = working.select_working_set(view, prefix + encoded.provider_prompt + suffix,
        prefix + view.provider_prompt + suffix)
    return view, result


def test_real_producer_exact_inverse_uses_native_capsule_bytes(native):
    repository, encoded, _ = native
    desc = descriptor(encoded)
    view, _ = selected(encoded, desc)
    calls = []
    restored = working.restore_semantic_working_set(provider_prompt=view.provider_prompt,
        receipt=view.receipt, encoded=encoded, descriptor=desc, resolver=resolver(encoded, calls))
    assert restored == encoded.provider_prompt
    assert codec.restore_semantic_router_prompt(provider_prompt=restored,
        table=encoded.table, repository=repository) == encoded.native_prompt
    capsules = native_capsules(encoded)
    assert [call["capsule_index"] for call in calls] == list(range(len(capsules)))
    assert [call["capsule_sha256"] for call in calls] == [sha(canonical(row)) for row in capsules]
    original = json.loads(encoded.provider_prompt)["translated_semantic"]["capsules"]
    assert any(sha(canonical(row)) != call["capsule_sha256"] for row, call in zip(original, calls))


def test_only_closed_provenance_fields_deferred_and_all_other_values_preserved(native):
    _repository, encoded, raw_source = native
    view = working.project_semantic_working_set(encoded, descriptor(encoded))
    original, projected = json.loads(encoded.provider_prompt), json.loads(view.provider_prompt)
    assert projected["schema"] == working.VIEW_SCHEMA
    assert projected["source_transport_schema"] == codec.TRANSPORT_SCHEMA
    assert guidance() in projected["working_set_instructions"]
    assert "--expected-capsule-sha256 <capsule_sha256>" in projected["working_set_instructions"]
    for key in set(original) - {"schema", "translated_semantic"}:
        assert projected[key] == original[key]
    before, after = original["translated_semantic"], projected["translated_semantic"]
    for key in set(before) - {"capsules"}:
        assert after[key] == before[key]
    assert after["raw_sources"]["module.py"] == raw_source
    deferred = 0
    for original_row, row in zip(before["capsules"], after["capsules"]):
        assert row == {key: value for key, value in original_row.items() if key not in working.DEFERRED_FIELDS}
        deferred += len(set(original_row) & working.DEFERRED_FIELDS)
        for key in ("stable_symbol_id", "capsule_cid", "source_slice_path", "signature",
                    "metadata", "confidence", "docstring_hint"):
            assert row[key] == original_row[key]
    assert view.receipt["deferred_field_count"] == deferred > 0
    assert view.dispatch_usable and not view.historical_replay
    with pytest.raises(codec.SemanticTranslationError):
        codec._restore(view.provider_prompt, encoded.table)


def test_complete_prompt_byte_accounting_and_unknown_provider_tokens(native):
    _repository, encoded, _ = native
    view, selection = selected(encoded)
    receipt = selection.receipt
    assert selection.selected_mode == working.PROFILE
    assert receipt["selected"] is True
    assert receipt["candidate_complete_bytes"] < receipt["native_complete_bytes"]
    assert receipt["selected_complete_sha256"] == sha(selection.provider_prompt)
    assert receipt["selected_complete_bytes"] == len(selection.provider_prompt.encode())
    assert receipt["native_complete_bytes"] > len(encoded.provider_prompt.encode())
    assert receipt["candidate_complete_bytes"] > len(view.provider_prompt.encode())
    assert receipt["view_receipt"] == view.receipt
    assert receipt["selection_basis"] == "complete_prompt_utf8_bytes"
    for obj in (receipt, receipt["view_receipt"]):
        assert all(obj[key] is False for key in working._FLAGS)
        assert all(obj[key] is None for key in ("native_provider_input_tokens",
            "candidate_provider_input_tokens", "provider_token_usage"))
    assert receipt["selected_provider_input_tokens"] is None
    assert working.validate_working_set_selection_receipt(receipt) == receipt
    copied = working.validate_working_set_selection_receipt(receipt)
    copied["view_receipt"]["binding"]["semantic_binding"]["task_id"] = "mutated"
    assert copied != receipt


def test_small_module_preserves_larger_native_fallback(native_factory):
    _repository, encoded, _ = native_factory(0)
    view, result = selected(encoded)
    assert view.receipt["capsule_count"] == 1  # The module itself remains a capsule.
    assert view.receipt["deferred_field_count"] > 0
    assert result.selected_mode == "legacy"
    assert result.receipt["fallback_reason"] == "complete_input_not_strictly_smaller"
    assert result.receipt["candidate_complete_bytes"] > result.receipt["native_complete_bytes"]
    assert encoded.provider_prompt in result.provider_prompt
    assert view.provider_prompt not in result.provider_prompt


@pytest.mark.parametrize("where", ["prefix", "suffix", "missing", "duplicate", "candidate_duplicate"])
def test_complete_prompt_selection_refuses_changed_or_ambiguous_wrappers(native, where):
    _repository, encoded, _ = native
    view = working.project_semantic_working_set(encoded, descriptor(encoded))
    original = "prefix" + encoded.provider_prompt + "suffix"
    candidate = "prefix" + view.provider_prompt + "suffix"
    if where == "prefix": candidate = "changed" + view.provider_prompt + "suffix"
    elif where == "suffix": candidate += "changed"
    elif where == "missing": original = "no encoded transport"
    elif where == "duplicate": original += encoded.provider_prompt
    else: candidate += view.provider_prompt
    with pytest.raises(working.SemanticWorkingSetError):
        working.select_working_set(view, original, candidate)


@pytest.mark.parametrize("mutation", ["old_schema", "mode", "foreign_task", "foreign_digest",
    "extra", "bool_count", "authority"])
def test_source_descriptor_binding_and_capability_are_required(native, mutation):
    _repository, encoded, _ = native
    desc = descriptor(encoded)
    if mutation == "old_schema": desc["schema"] = source.DESCRIPTOR_SCHEMA_V3
    elif mutation == "mode": desc["semantic_evidence_mode"] = "disabled"
    elif mutation == "foreign_task": desc["task_id"] = "other-task"
    elif mutation == "foreign_digest": desc["semantic_sha256"] = "0" * 64
    elif mutation == "extra": desc["extra"] = False
    elif mutation == "bool_count": desc["max_requests"] = True
    else: desc["execution_authority"] = True
    with pytest.raises(working.SemanticWorkingSetError):
        working.project_semantic_working_set(encoded, desc)


def test_historical_replay_is_exact_without_operational_freshness(native):
    repository, encoded, _ = native
    desc = descriptor(encoded)
    view, selection = selected(encoded, desc)
    (repository / "module.py").write_text("# authored source edit\n")
    historical = codec.replay_semantic_router_prompt_for_audit(prompt=encoded.native_prompt,
        repository=repository)
    assert historical.freshness_checked is False
    replay = working.replay_semantic_working_set_for_audit(historical, view.receipt)
    assert replay.provider_prompt == view.provider_prompt
    assert replay.receipt == view.receipt
    assert replay.dispatch_usable is False and replay.historical_replay is True
    before = selection.provider_prompt.replace(view.provider_prompt, encoded.provider_prompt)
    selected_replay = working.select_working_set(replay, before, selection.provider_prompt)
    assert selected_replay.receipt == selection.receipt
    assert selected_replay.dispatch_usable is False and selected_replay.historical_replay is True
    with pytest.raises(working.SemanticWorkingSetError):
        working.project_semantic_working_set(historical, desc)


@pytest.mark.parametrize("mutation", ["hash", "task", "deferred", "authority", "extra", "counter"])
def test_historical_projection_receipt_cannot_be_rebound(native, mutation):
    _repository, encoded, _ = native
    view = working.project_semantic_working_set(encoded, descriptor(encoded))
    receipt = view.receipt
    if mutation == "hash": receipt["candidate_view_sha256"] = "0" * 64
    elif mutation == "task": receipt["binding"]["semantic_binding"]["task_id"] = "foreign"
    elif mutation == "deferred": receipt["deferred_field_allowlist"].append("contracts")
    elif mutation == "authority": receipt["proof_authority"] = True
    elif mutation == "extra": receipt["extra"] = 0
    else: receipt["provider_token_usage"] = 0
    with pytest.raises(working.SemanticWorkingSetError):
        working.replay_semantic_working_set_for_audit(encoded, receipt)


@pytest.mark.parametrize("mutation", ["behavior", "raw_source", "native", "translation", "reference", "extra"])
def test_tampered_view_refuses_before_capsule_resolution(native, mutation):
    _repository, encoded, _ = native
    desc = descriptor(encoded)
    view = working.project_semantic_working_set(encoded, desc)
    altered = json.loads(view.provider_prompt)
    if mutation == "behavior": altered["translated_semantic"]["capsules"][0]["signature"] = "changed"
    elif mutation == "raw_source": altered["translated_semantic"]["raw_sources"]["module.py"] = "changed"
    elif mutation == "native": altered["native_context"]["scope"] = {}
    elif mutation == "translation": altered["translation_table"] = {}
    elif mutation == "reference": altered["working_set"]["capsule_references"].reverse()
    else: altered["extra"] = False
    calls = []
    with pytest.raises(working.SemanticWorkingSetError):
        working.restore_semantic_working_set(provider_prompt=canonical(altered), receipt=view.receipt,
            encoded=encoded, descriptor=desc, resolver=resolver(encoded, calls))
    assert calls == []


@pytest.mark.parametrize("mutation", ["wrong_digest", "aliases", "noncanonical", "string", "wrong_capsule"])
def test_expansion_requires_exact_native_capsule_bytes(native, mutation):
    _repository, encoded, _ = native
    desc = descriptor(encoded)
    view = working.project_semantic_working_set(encoded, desc)
    good = resolver(encoded, [])
    def bad(ref):
        if mutation == "wrong_digest": return b"{}"
        if mutation == "aliases":
            return canonical(json.loads(encoded.provider_prompt)["translated_semantic"]["capsules"][ref["capsule_index"]]).encode()
        if mutation == "noncanonical": return good(ref) + b"\n"
        if mutation == "string": return good(ref).decode()
        return good(ref | {"capsule_index": (ref["capsule_index"] + 1) % len(native_capsules(encoded))})
    with pytest.raises(working.SemanticWorkingSetError):
        working.restore_semantic_working_set(provider_prompt=view.provider_prompt, receipt=view.receipt,
            encoded=encoded, descriptor=desc, resolver=bad)


@pytest.mark.parametrize("mutation", ["extra", "bool_bytes", "bool_count", "counter", "authority",
    "selection", "selected_hash", "wrapper", "view_extra", "binding_extra", "digest", "allowlist",
    "capsule_count_limit", "capsule_bytes_limit", "capsule_bytes_bool"])
def test_closed_selection_receipt_validation_rejects_forged_accounting(native, mutation):
    _repository, encoded, _ = native
    _, result = selected(encoded)
    receipt = result.receipt
    if mutation == "extra": receipt["extra"] = 0
    elif mutation == "bool_bytes": receipt["selected_complete_bytes"] = True
    elif mutation == "bool_count": receipt["view_receipt"]["capsule_count"] = True
    elif mutation == "counter": receipt["selected_provider_input_tokens"] = 0
    elif mutation == "authority": receipt["view_receipt"]["omission_authority"] = True
    elif mutation == "selection": receipt["selected"] = 1
    elif mutation == "selected_hash": receipt["selected_complete_sha256"] = "0" * 64
    elif mutation == "wrapper": receipt["native_complete_bytes"] += 1
    elif mutation == "view_extra": receipt["view_receipt"]["extra"] = 0
    elif mutation == "binding_extra": receipt["view_receipt"]["binding"]["socket"] = "/owner"
    elif mutation == "digest": receipt["view_receipt"]["binding"]["descriptor_sha256"] = "Q" * 64
    elif mutation == "allowlist": receipt["view_receipt"]["deferred_field_allowlist"].append("signature")
    elif mutation == "capsule_count_limit": receipt["view_receipt"]["capsule_count"] = working.MAX_SELECTED_CAPSULES + 1
    elif mutation == "capsule_bytes_limit": receipt["view_receipt"]["capsule_native_bytes"] = working.MAX_SELECTED_CAPSULE_BYTES + 1
    else: receipt["view_receipt"]["capsule_native_bytes"] = True
    with pytest.raises(working.SemanticWorkingSetError):
        working.validate_working_set_selection_receipt(receipt)


def test_projection_selection_replay_and_inverse_do_no_io(native, monkeypatch):
    _repository, encoded, _ = native
    desc = descriptor(encoded)
    resolve = resolver(encoded, [])
    guidance()  # Load the fixed command before prohibiting IO.
    def forbidden(*args, **kwargs):
        raise AssertionError("pure projection attempted IO")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    view, result = selected(encoded, desc)
    replay = working.replay_semantic_working_set_for_audit(encoded, view.receipt)
    assert replay.provider_prompt == view.provider_prompt
    assert working.validate_working_set_selection_receipt(result.receipt) == result.receipt
    assert working.restore_semantic_working_set(provider_prompt=view.provider_prompt,
        receipt=view.receipt, encoded=encoded, descriptor=desc, resolver=resolve) == encoded.provider_prompt


def with_future_transport_fields(encoded, fields=None):
    """An authored future schema extension, not a new source admission."""
    original, suffix, _, old_refs, semantic_text, artifact = codec._prompt_parts(encoded.native_prompt)
    semantic = json.loads(semantic_text)
    wire = json.loads(encoded.provider_prompt)
    fields = fields if fields is not None else dict(future_behavior={"source_cid": "literal", "schema": "literal", "empty": [],
        "proof": {"unknown": None}, "effects": ["writes:result"]}, future_empty=[],
        future_null=None, future_false=False)
    semantic["capsules"][0].update(deepcopy(fields))
    wire["translated_semantic"]["capsules"][0].update(deepcopy(fields))
    semantic["future_root"] = wire["translated_semantic"]["future_root"] = {"source_cid": "literal"}
    references = build_text_context_references(canonical(semantic), reference_prefix="semantic-context",
        kind="semantic-context", path=artifact, repository_id="repo:authored", tree_id="tree:authored",
        required=True, chunk_bytes=1201)
    assert len(original["evidence"]) == len(old_refs)
    original["evidence"] = [row.to_dict() for row in references]
    native_prompt = canonical(original) + suffix
    table = json.loads(encoded.table.payload_json)
    table.update(semantic_sha256=sha(canonical(semantic)),
        translated_semantic_sha256=sha(canonical(wire["translated_semantic"])),
        native_prompt_sha256=sha(native_prompt), semantic_positions=list(range(len(references))),
        reference_templates=[{key: value for key, value in row.to_dict().items() if key != "summary"}
            for row in references])
    identity = codec.cid_for_payload(table)
    table = codec.SemanticTranslationTable(canonical(table), identity)
    wire["translation_cid"] = identity
    provider_prompt = canonical(wire)
    receipt = encoded.receipt | dict(native_prompt_sha256=sha(native_prompt),
        provider_prompt_sha256=sha(provider_prompt))
    return replace(encoded, native_prompt=native_prompt, provider_prompt=provider_prompt,
        table=table, receipt_json=canonical(receipt)), fields


def test_unknown_and_empty_behavioral_fields_remain_exact(native):
    _repository, initial, _ = native
    encoded, future = with_future_transport_fields(initial)
    desc = descriptor(encoded)
    view = working.project_semantic_working_set(encoded, desc)
    projected = json.loads(view.provider_prompt)["translated_semantic"]
    for key, value in future.items():
        assert key in projected["capsules"][0] and projected["capsules"][0][key] == value
    assert projected["future_root"] == {"source_cid": "literal"}
    assert working.restore_semantic_working_set(provider_prompt=view.provider_prompt,
        receipt=view.receipt, encoded=encoded, descriptor=desc, resolver=resolver(encoded, [])) == encoded.provider_prompt


def test_large_selected_capsule_bytes_refuse_before_provenance_deferral(native):
    _repository, initial, _ = native
    encoded, _ = with_future_transport_fields(initial,
        dict(future_behavior="x" * working.MAX_SELECTED_CAPSULE_BYTES))
    with pytest.raises(working.SemanticWorkingSetError, match="bounded expansion eligibility"):
        working.project_semantic_working_set(encoded, descriptor(encoded))


def test_selected_population_bound_precedes_projection(native, monkeypatch):
    _repository, encoded, _ = native
    desc = descriptor(encoded)
    wire, capsules = working._encoded(encoded, current=True)
    # A unit contract input for the population guard; real-producer identity and
    # exact round trips are independently exercised by the tests above.
    capsules = [{} for _ in range(working.MAX_SELECTED_CAPSULES + 1)]
    assert sum(len(canonical(row).encode()) for row in capsules) < working.MAX_SELECTED_CAPSULE_BYTES
    wire["translated_semantic"]["capsules"] = deepcopy(capsules)
    monkeypatch.setattr(working, "_encoded", lambda *args, **kwargs: (wire, capsules))
    with pytest.raises(working.SemanticWorkingSetError, match="bounded expansion eligibility"):
        working.project_semantic_working_set(encoded, desc)
