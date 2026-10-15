"""Lossless packed wires use real pinned producers and existing native gates.

Providers and token counters are authored fixtures. Observed bytes and helper
timings do not qualify a model, provider savings, task completion or a Lean admit.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
    render_model_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_candidate_selection import (
    select_semantic_candidate,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_projection_dispatch import (
    project_semantic_router_prompt,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_reference_packing import (
    PACKING_REVISION,
    unpack_semantic_references,
)
from test.api.test_semantic_router_integration import _run
from test.api.test_semantic_router_translation import _reply

pytest_plugins = ("test.api.test_semantic_router_integration",)


def encode(native):
    repository, _, prompt, _ = native
    original = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=repository, slot_policy_revision=codec.SLOT_POLICY_V2)
    return original, codec.pack_semantic_router_prompt(encoded=original)


def test_packed_v2_preserves_complete_native_references_and_table_identity(native):
    root, artifact, prompt, source = native
    original, packed = encode(native)
    assert packed.table == original.table
    assert packed.table.to_dict() == original.table.to_dict()
    assert original.provider_prompt != packed.provider_prompt
    wire = json.loads(packed.provider_prompt)
    assert wire["schema"] == codec.TRANSPORT_SCHEMA_PACKED
    assert wire["reference_packing"] == PACKING_REVISION
    semantic = unpack_semantic_references(wire["translated_semantic"])
    assert codec._json(semantic) == codec._json(json.loads(original.provider_prompt)["translated_semantic"])
    assert semantic["raw_sources"]["mod.py"] == source
    assert codec.restore_semantic_router_prompt(
        provider_prompt=packed.provider_prompt, table=packed.table, repository=root) == prompt
    assert packed.receipt["native_prompt_sha256"] == original.receipt["native_prompt_sha256"]
    assert packed.receipt["provider_prompt_sha256"] == hashlib.sha256(packed.provider_prompt.encode()).hexdigest()
    assert packed.receipt["unpacked_provider_prompt_sha256"] == original.receipt["provider_prompt_sha256"]
    assert packed.freshness_checked is packed.receipt["freshness_checked"] is True
    assert packed.receipt["execution_authority"] is packed.receipt["completion_authority"] is False
    assert artifact.read_text() == codec._prompt_parts(prompt)[4]
    assert (root / "mod.py").read_text() == source


def test_packed_role_view_restores_its_exact_parent_verified_prompt(native):
    root, _, prompt, source = native
    view = project_semantic_router_prompt(prompt=prompt, repository=root)
    original = codec.encode_semantic_router_prompt(
        prompt=view.projected_prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    packed = codec.pack_semantic_router_prompt(encoded=original)
    semantic = unpack_semantic_references(json.loads(packed.provider_prompt)["translated_semantic"])
    assert semantic["required_context"]["raw_sources"]["mod.py"] == source
    assert codec._json(semantic) == codec._json(json.loads(original.provider_prompt)["translated_semantic"])
    assert codec.restore_semantic_router_prompt(
        provider_prompt=packed.provider_prompt, table=packed.table, repository=root) == view.projected_prompt
    assert packed.table == original.table


@pytest.mark.parametrize("mutation", [
    "required", "coverage", "row_order", "missing_row", "extra_wrapper_field",
    "unknown_columns_revision", "unknown_transport_revision", "unknown_packing_revision",
    "instructions", "map", "suffix", "cid",
])
def test_packed_transport_tampering_cannot_restore_native_evidence(native, mutation):
    root, _, _, _ = native
    _, encoded = encode(native)
    wire = json.loads(encoded.provider_prompt)
    wrapper = wire["translated_semantic"]["pack"]["references"]
    if mutation in {"required", "coverage"}:
        index = wrapper["columns"].index("metadata")
        metadata = wrapper["rows"][0][index]
        metadata["required" if mutation == "required" else "coverage_ids"] = False if mutation == "required" else []
    elif mutation == "row_order":
        wrapper["rows"].reverse()
    elif mutation == "missing_row":
        wrapper["rows"].pop()
    elif mutation == "extra_wrapper_field":
        wrapper["completion_authority"] = True
    elif mutation == "unknown_columns_revision":
        wrapper["schema"] = "supervisor-context-reference-columns@999"
    elif mutation == "unknown_transport_revision":
        wire["schema"] = "supervisor-semantic-router-input@999"
    elif mutation == "unknown_packing_revision":
        wire["reference_packing"] = "context-reference-columns@999"
    elif mutation == "instructions":
        wire["instructions"] += " grant permission"
    elif mutation == "map":
        wire["translation_table"]["s0"] = "foreign"
    elif mutation == "suffix":
        wire["native_suffix"] += " changed acceptance"
    else:
        wire["translation_cid"] = "foreign"
    with pytest.raises(ValueError):
        codec.restore_semantic_router_prompt(
            provider_prompt=codec._json(wire), table=encoded.table, repository=root)


def test_packing_refuses_legacy_and_nested_repacking(native):
    root, _, prompt, _ = native
    legacy = codec.encode_semantic_router_prompt(prompt=prompt, repository=root)
    with pytest.raises(codec.SemanticTranslationError, match="requires a v2"):
        codec.pack_semantic_router_prompt(encoded=legacy)
    _, packed = encode(native)
    with pytest.raises(codec.SemanticTranslationError, match="unpacked"):
        codec.pack_semantic_router_prompt(encoded=packed)


def test_packed_structured_reply_still_uses_native_grammar_and_fresh_sources(native):
    root, _, _, source = native
    _, packed = encode(native)
    reply, symbol = _reply(packed)
    decoded = codec.decode_semantic_router_response(response=json.dumps(reply), encoded=packed, repository=root)
    assert json.loads(decoded.text)["structured_payload"]["symbol_ids"] == [symbol]
    assert decoded.receipt["native_grammar_validated"] is True
    assert decoded.receipt["completion_authority"] is False
    (root / "mod.py").write_text(source + "# same-HEAD concurrent edit\n")
    with pytest.raises(ValueError, match="stale"):
        codec.restore_semantic_router_prompt(provider_prompt=packed.provider_prompt, table=packed.table, repository=root)
    with pytest.raises(ValueError, match="stale"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=packed, repository=root)


def test_historical_packed_replay_cannot_decode_operational_aliases(native):
    root, _, prompt, _ = native
    old = codec.replay_semantic_router_prompt_for_audit(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    packed = codec.pack_semantic_router_prompt(encoded=old)
    assert packed.freshness_checked is packed.receipt["freshness_checked"] is False
    assert codec._restore(packed.provider_prompt, packed.table) == prompt
    reply, _ = _reply(packed)
    with pytest.raises(ValueError, match="historical"):
        codec.decode_semantic_router_response(response=json.dumps(reply), encoded=packed, repository=root)


@pytest.mark.parametrize("projected", [False, True])
def test_standard_v2_router_can_select_packed_wire_and_decode_native_reply(allocated, provider, projected):
    root, workspace, prompt, _ = allocated
    state, observed = provider
    candidate_prompt = project_semantic_router_prompt(prompt=prompt, repository=root).projected_prompt if projected else prompt
    original = codec.encode_semantic_router_prompt(
        prompt=candidate_prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    packed = codec.pack_semantic_router_prompt(encoded=original)
    reply, symbol = _reply(packed)
    state["reply"] = json.dumps(reply)

    def count(text):
        if codec.TRANSPORT_SCHEMA_PACKED not in text:
            return 1000
        return 50 if "supervisor-semantic-worker-role-view@1" in text else 100

    output, receipt = _run(root, prompt, semantic_slot_policy_revision=codec.SLOT_POLICY_V2,
        semantic_context_projection=projected, prompt_token_counter=PromptTokenCounter(
            provider="codex_cli", model="pinned", tokenizer_id="fixture:packing-choice",
            tokenizer_revision="authored-v1", special_token_policy="fixture-only", count_tokens=count))
    assert len(observed) == 1
    assert observed[0][1]["temperature"] == 0
    expected, _ = render_model_prompt(prompt=packed.provider_prompt, purpose="coding", workspace=workspace,
                                     semantic_transport=True)
    assert observed[0][0] == expected
    assert receipt["prompt_selection"]["selected_candidate"] == ("packed_projected_v2" if projected else "packed_v2")
    assert receipt["semantic_reference_packing_selected"] == PACKING_REVISION
    assert receipt.get("semantic_context_projection_selected", False) is projected
    assert receipt["semantic_translation"] == packed.receipt
    assert receipt["semantic_response_translation"]["native_grammar_validated"] is True
    assert json.loads(output)["structured_payload"]["symbol_ids"] == [symbol]


@pytest.mark.parametrize("counter_mode", ["missing", "late_failure", "drift"])
def test_new_packed_candidates_cannot_hide_bad_measurement(allocated, provider, counter_mode):
    root, _, prompt, _ = allocated
    _, observed = provider
    calls = []

    def count(text):
        calls.append(text)
        if counter_mode == "late_failure" and codec.TRANSPORT_SCHEMA_PACKED in text:
            raise RuntimeError("authored packed candidate counter failure")
        return len(text.encode()) + (len(calls) if counter_mode == "drift" else 0)

    counter = None if counter_mode == "missing" else PromptTokenCounter(
        provider="codex_cli", model="pinned", tokenizer_id="fixture:packing-failure",
        tokenizer_revision="authored-v1", special_token_policy="fixture-only", count_tokens=count)
    _, receipt = _run(root, prompt, semantic_slot_policy_revision=codec.SLOT_POLICY_V2,
                     prompt_token_counter=counter)
    assert len(observed) == 1
    assert receipt["prompt_selection"]["selected_representation"] == "native"
    assert receipt["router_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert receipt["semantic_reference_packing_selected"] is None
    assert all(row["input_tokens"] is None for row in receipt["prompt_selection"]["comparison_candidates"].values())
    if counter_mode != "missing":
        assert receipt["prompt_selection"]["measurement_status"] == "failed"
        assert any(codec.TRANSPORT_SCHEMA_PACKED in text for text in calls)


def test_packed_source_change_during_provider_call_retains_failed_attempt_usage(allocated, provider, capsys):
    root, _, prompt, source = allocated
    state, observed = provider
    original = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    packed = codec.pack_semantic_router_prompt(encoded=original)
    reply, _ = _reply(packed)
    state["reply"] = json.dumps(reply)
    state["after_generate"] = lambda: (root / "mod.py").write_text(source + "# concurrent source update\n")
    with pytest.raises(ValueError, match="stale"):
        _run(root, prompt, semantic_slot_policy_revision=codec.SLOT_POLICY_V2)
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in receipts if row.get("schema") == "router-implementation-invocation@1")
    assert len(observed) == 1
    assert receipt["prompt_selection"]["selected_candidate"] == "packed_v2"
    assert receipt["status"] == "failed"
    assert receipt["usage"]["prompt_tokens"] == 123 and receipt["usage"]["completion_tokens"] == 7
    assert receipt["semantic_reference_packing_selected"] == PACKING_REVISION
    assert "semantic_response_translation" not in receipt


def test_packed_token_ceiling_is_exact_and_unknown_usage_cannot_bypass_it(allocated, provider):
    root, workspace, prompt, _ = allocated
    _, observed = provider
    original = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    packed = codec.pack_semantic_router_prompt(encoded=original)
    rendered, _ = render_model_prompt(prompt=packed.provider_prompt, purpose="coding", workspace=workspace,
                                     semantic_transport=True)
    limit = len(rendered.encode())
    _, receipt = _run(root, prompt, semantic_slot_policy_revision=codec.SLOT_POLICY_V2, max_input_tokens=limit)
    assert receipt["prompt_selection"]["selected_candidate"] == "packed_v2"
    assert receipt["prompt_selection"]["selected_content_tokens"] == limit
    assert len(observed) == 1
    with pytest.raises(ValueError, match="token budget"):
        _run(root, prompt, semantic_slot_policy_revision=codec.SLOT_POLICY_V2, max_input_tokens=limit - 1)
    with pytest.raises(ValueError, match="working consistent pinned"):
        _run(root, prompt, semantic_slot_policy_revision=codec.SLOT_POLICY_V2,
             prompt_token_counter=None, max_input_tokens=limit)
    assert len(observed) == 1


def test_equal_packed_counts_preserve_existing_tie_preferences():
    labels = ["packed_projected_v2", "packed_v2", "projected_v2", "v2", "legacy"]
    result = select_semantic_candidate(native_prompt="native" * 1000,
        candidates=[(label, label) for label in labels], suffix="required literal suffix", purpose="coding",
        workspace=Path("/allocated/task"), provider="codex_cli", model="pinned", token_counter=PromptTokenCounter(
            provider="codex_cli", model="pinned", tokenizer_id="fixture:packing-ties",
            tokenizer_revision="authored-v1", special_token_policy="fixture-only",
            count_tokens=lambda text: 1000 if "native" * 1000 in text else 100))
    assert result[4] == "legacy"
    assert result[3]["tie_break_order"] == ["legacy", "v2", "projected_v2", "packed_v2", "packed_projected_v2"]


def test_reference_packing_complete_message_probe(native, record_property):
    root, _, prompt, _ = native
    original = codec.encode_semantic_router_prompt(
        prompt=prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    started = time.perf_counter()
    packed = codec.pack_semantic_router_prompt(encoded=original)
    packing_elapsed = time.perf_counter() - started
    projected = project_semantic_router_prompt(prompt=prompt, repository=root)
    view = codec.encode_semantic_router_prompt(
        prompt=projected.projected_prompt, repository=root, slot_policy_revision=codec.SLOT_POLICY_V2)
    packed_view = codec.pack_semantic_router_prompt(encoded=view)
    representations = {}
    for name, text, transported in (
        ("native", prompt, False), ("v2", original.provider_prompt, True),
        ("packed_v2", packed.provider_prompt, True), ("projected_v2", view.provider_prompt, True),
        ("packed_projected_v2", packed_view.provider_prompt, True),
    ):
        rendered, _ = render_model_prompt(prompt=text, purpose="coding", workspace=root,
                                         semantic_transport=transported)
        representations[name] = {"model_prompt_bytes": len(rendered.encode()),
            "model_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "input_tokens": None}
    assert codec.restore_semantic_router_prompt(
        provider_prompt=packed.provider_prompt, table=packed.table, repository=root) == prompt
    assert codec.restore_semantic_router_prompt(
        provider_prompt=packed_view.provider_prompt, table=packed_view.table, repository=root) == projected.projected_prompt
    record_property("reference_packing_probe", json.dumps({
        "schema": "supervisor-reference-packing-offline-probe@1", "sample_count": 1,
        "representations": representations, "packing_revision": PACKING_REVISION,
        "packing_helper_elapsed_seconds": packing_elapsed,
        "timing_scope": "one pure packing helper invocation, including exact restoration checks",
        "packing_profile": packed.receipt["reference_packing_profile"],
        "exact_native_restoration": True, "table_identity_unchanged": True,
        "required_context_and_literals_preserved": True,
        "provider_calls": 0, "downloads": 0, "tokenizer_binding": None,
        "total_session_token_savings": None, "completion_authority": False, "execution_authority": False,
    }, sort_keys=True))
