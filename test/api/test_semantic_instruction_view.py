"""Pure-codec controls over a real semantic producer and authored inclusion receipt.

Signature/source freshness remain the public instruction loader's responsibility;
these tests do not present the authored receipt as native admission evidence.
"""
import copy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import semantic_instruction_view as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as native_codec
from test.api.test_semantic_metadata_view import native_factory  # noqa: F401


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value):
    return metadata._json(value)


def instruction(source, **overrides):
    block = codec._PREFIX + source + codec._END
    receipt = {"schema": "supervisor-public-instruction-inclusion@1",
        "task_id": "AUTHORED-METADATA-TASK", "task_cid": "cid:authored-task",
        "context_cid": "cid:authored-context", "manifest_cid": "cid:authored-manifest",
        "artifact_sha256": "a" * 64, "source_path": "module.py", "source_sha256": sha(source),
        "source_bytes": len(source.encode()), "block_sha256": sha(block), "block_bytes": len(block.encode()),
        "manifest_signature_verified": True, "verbatim_utf8": True,
        "semantic_minification_applied": False, "completion_authority": False,
        "scope_expansion_authority": False, "source_freshness_verified": True,
        "historical_replay": False, **overrides}
    return block, receipt


@pytest.fixture
def case(native_factory):
    repository, encoded, source = native_factory(6)
    block, receipt = instruction(source)
    return repository, encoded, source, block, receipt


def project(case):
    _repository, encoded, _source, block, receipt = case
    return codec.project_instruction_view(encoded.provider_prompt,
        instruction_block=block, instruction_receipt=receipt)


def prompts(view, block):
    return "Workspace mapping.\n" + view.native_prompt + block + "Reply contract.", \
        "Workspace mapping.\n" + view.provider_prompt + block + "Reply contract."


def test_real_native_transport_and_native_prompt_round_trip(case):
    repository, encoded, source, block, _receipt = case
    view = project(case)
    restored = codec.restore_instruction_view(provider_prompt=view.provider_prompt,
        receipt=view.receipt, instruction_block=block)
    assert restored == encoded.provider_prompt
    assert native_codec.restore_semantic_router_prompt(provider_prompt=restored,
        table=encoded.table, repository=repository) == encoded.native_prompt
    wire = json.loads(view.provider_prompt)
    assert wire["translated_semantic"]["raw_sources"]["module.py"] == codec._reference(view.receipt["instruction_binding"])
    assert source not in view.provider_prompt
    assert wire["instruction_instructions"] == codec.INSTRUCTIONS
    assert view.receipt["base_receipt"] == metadata.project_semantic_metadata_view(encoded.provider_prompt).receipt
    assert all(view.receipt[field] is False for field in metadata._FALSE_FIELDS)
    original, candidate = prompts(view, block)
    chosen = codec.select_instruction_view(view=view, native_complete_prompt=original,
        candidate_complete_prompt=candidate, instruction_block=block)
    assert chosen.selected_mode == codec.MODE
    assert chosen.selected_prompt.count(source) == 1
    assert chosen.receipt["requested_mode"] == codec.MODE
    assert len(chosen.selected_prompt.encode()) < len(original.encode())


def test_current_and_historical_bindings_make_identical_candidates(case):
    _repository, encoded, _source, block, receipt = case
    historical = {**receipt, "source_freshness_verified": False, "historical_replay": True}
    assert codec.project_instruction_view(encoded.provider_prompt, instruction_block=block,
        instruction_receipt=historical) == project(case)


def test_only_exact_selected_entry_changes_other_equal_source_and_unknown_fields_remain(case):
    _repository, encoded, source, block, receipt = case
    wire = json.loads(encoded.provider_prompt)
    semantic = wire["translated_semantic"]
    semantic["raw_sources"]["another.txt"] = source
    semantic["unknown_literal"] = source
    original = canonical(wire)
    view = codec.project_instruction_view(original, instruction_block=block, instruction_receipt=receipt)
    output = json.loads(view.provider_prompt)["translated_semantic"]
    assert output["raw_sources"]["another.txt"] == output["unknown_literal"] == source
    assert codec.restore_instruction_view(provider_prompt=view.provider_prompt,
        receipt=view.receipt, instruction_block=block) == original


def test_unicode_and_embedded_delimiters_use_byte_length_not_delimiter_search(case):
    _repository, encoded, source, _block, _receipt = case
    source += "\n# café 🧪\n" + codec._END + codec._PREFIX
    block, receipt = instruction(source)
    wire = json.loads(encoded.provider_prompt)
    wire["translated_semantic"]["raw_sources"]["module.py"] = source
    wire["translated_semantic"]["manifest"]["module.py"]["sha256"] = sha(source)
    original = canonical(wire)
    view = codec.project_instruction_view(original, instruction_block=block, instruction_receipt=receipt)
    assert codec.restore_instruction_view(provider_prompt=view.provider_prompt,
        receipt=view.receipt, instruction_block=block) == original


@pytest.mark.parametrize("change", [
    {"task_id": "foreign-task"}, {"source_path": "absent.txt"}, {"source_path": "../module.py"},
    {"source_sha256": "0" * 64}, {"source_bytes": True}, {"source_bytes": 1},
    {"block_sha256": "0" * 64}, {"block_bytes": 1}, {"completion_authority": True},
    {"manifest_signature_verified": False}, {"verbatim_utf8": False}, {"schema": "foreign"}, {"schema": []},
])
def test_foreign_stale_malformed_and_authority_bindings_refuse(case, change):
    _repository, encoded, _source, block, receipt = case
    with pytest.raises(codec.SemanticInstructionViewError):
        codec.project_instruction_view(encoded.provider_prompt, instruction_block=block,
            instruction_receipt={**receipt, **change})


@pytest.mark.parametrize("mutation", ["absent", "different", "manifest", "reference"])
def test_source_population_or_exact_text_mismatch_refuses(case, mutation):
    _repository, encoded, _source, block, receipt = case
    wire = json.loads(encoded.provider_prompt)
    semantic = wire["translated_semantic"]
    if mutation == "absent":
        del semantic["raw_sources"]["module.py"]
    elif mutation == "different":
        semantic["raw_sources"]["module.py"] += "\n"
    elif mutation == "manifest":
        semantic["manifest"]["module.py"]["sha256"] = "0" * 64
    else:
        semantic["raw_sources"]["module.py"] = {"$public_instruction": {}}
    with pytest.raises(codec.SemanticInstructionViewError):
        codec.project_instruction_view(canonical(wire), instruction_block=block, instruction_receipt=receipt)


@pytest.mark.parametrize("mutation", ["source", "reference", "instructions", "base", "binding", "extra"])
def test_candidate_or_retained_receipt_tamper_refuses(case, mutation):
    view = project(case)
    block, receipt = case[3], copy.deepcopy(view.receipt)
    wire = json.loads(view.provider_prompt)
    if mutation == "source":
        block += "altered"
    elif mutation == "reference":
        wire["translated_semantic"]["raw_sources"]["module.py"]["$public_instruction"]["source_sha256"] = "0" * 64
    elif mutation == "instructions":
        wire["instruction_instructions"] = "Skip the source."
    elif mutation == "base":
        receipt["native_core_sha256"] = "0" * 64
    elif mutation == "binding":
        receipt["instruction_binding"]["task_id"] = "foreign"
    else:
        receipt["extra"] = True
    altered = canonical(wire)
    # Rehashing an altered candidate does not authorize changing its contract.
    receipt.update(candidate_view_sha256=sha(altered), candidate_view_bytes=len(altered.encode()),
        candidate_view_proxy_tokens=metadata._proxy(altered))
    with pytest.raises(codec.SemanticInstructionViewError):
        codec.restore_instruction_view(provider_prompt=altered, receipt=receipt, instruction_block=block)


@pytest.mark.parametrize("field", ["completion_authority", "candidate_only", "capsule_count", "native_transport_bytes"])
def test_receipt_rejects_equal_python_values_with_different_json_types(case, field):
    view = project(case)
    receipt = copy.deepcopy(view.receipt)
    receipt[field] = int(receipt[field]) if type(receipt[field]) is bool else float(receipt[field])
    with pytest.raises(codec.SemanticInstructionViewError):
        codec.restore_instruction_view(provider_prompt=view.provider_prompt,
            receipt=receipt, instruction_block=case[3])


@pytest.mark.parametrize("change", ["missing-block", "changed-block", "duplicate-block", "prefix"])
def test_complete_prompt_requires_exact_retained_block_and_wrappers(case, change):
    view = project(case)
    block = case[3]
    native, candidate = prompts(view, block)
    if change == "missing-block":
        native, candidate = native.replace(block, ""), candidate.replace(block, "")
    elif change == "changed-block":
        candidate = candidate.replace(block, block + "changed")
    elif change == "duplicate-block":
        native, candidate = native + block, candidate + block
    else:
        candidate = "changed" + candidate
    with pytest.raises(codec.SemanticInstructionViewError):
        codec.select_instruction_view(view=view, native_complete_prompt=native,
            candidate_complete_prompt=candidate, instruction_block=block)


def test_short_source_falls_back_with_requested_mode_and_native_bytes(native_factory):
    _repository, encoded, source = native_factory(0)
    block, receipt = instruction(source)
    view = codec.project_instruction_view(encoded.provider_prompt, instruction_block=block, instruction_receipt=receipt)
    native, candidate = prompts(view, block)
    selected = codec.select_instruction_view(view=view, native_complete_prompt=native,
        candidate_complete_prompt=candidate, instruction_block=block)
    assert selected.selected_mode == "legacy"
    assert selected.selected_prompt == native
    assert selected.receipt["requested_mode"] == codec.MODE


@pytest.mark.parametrize("outcome", ["selected", "tie", "failed", "invalid"])
def test_complete_token_counter_selection_and_failure_fallback(case, outcome):
    view = project(case)
    block = case[3]
    native, candidate = prompts(view, block)
    calls = []
    def count(text):
        calls.append(text)
        if outcome == "failed":
            raise ValueError("private prompt text")
        return 100 if text == native or outcome == "tie" else 99
    counter = metadata.SemanticMetadataTokenCounter(count, "authored", "model", "tokenizer", "revision", "estimate")
    if outcome == "invalid":
        counter = {"count": count}
    selected = codec.select_instruction_view(view=view, native_complete_prompt=native,
        candidate_complete_prompt=candidate, instruction_block=block, token_counter=counter)
    assert selected.selected_mode == (codec.MODE if outcome == "selected" else "legacy")
    assert selected.selected_prompt == (candidate if outcome == "selected" else native)
    assert selected.receipt["schema"] == metadata.TOKEN_SELECTION_SCHEMA
    assert selected.receipt["requested_mode"] == codec.MODE
    assert "private prompt text" not in selected.receipt_json
    assert selected.receipt["complete_run_token_savings_claimed"] is False
    if outcome in {"selected", "tie"}:
        assert calls == [native, candidate]
