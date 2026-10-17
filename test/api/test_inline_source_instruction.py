"""Pure real-codec boundaries with authored transport and inclusion receipts.

These fixtures execute no native index, source admission, filesystem lookup or
provider. Authored signature flags satisfy the representation codec's input
contract; they are not signature verification or delivery evidence.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_instruction as carrier
from ipfs_accelerate_py.agent_supervisor.runtime import inline_source_observation as observer
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_instruction_view as codec
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata


INSTRUCTION_PATH = "instructions.txt"
OTHER_SOURCE = "retained = 'λ'\n"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instruction_case(*, source=None, remaining=None, block_extra=""):
    """Return native text, actual projected view, block and authored receipt."""
    if source is None:
        source = "Retain the declared outputs and verify the result. Ω\n" * 80
    sources = {INSTRUCTION_PATH: source, **({"other.py": OTHER_SOURCE} if remaining is None else remaining)}
    task = "AUTHORED-INSTRUCTION-TASK"
    native = metadata._json(dict(schema=metadata.TRANSPORT_SCHEMA, translation_cid="cid:authored-translation",
        translation_table={}, native_context={"objective_id": task, "evidence": []},
        translated_semantic=dict(task_id=task, scope_cid="cid:authored-scope", semantic_root_cid="cid:authored-root",
            capsules=[], admissions=[], raw_sources=sources,
            manifest={path: {"sha256": sha(content)} for path, content in sources.items()}),
        native_suffix="", instructions="Authored typed transport.", execution_authority=False, completion_authority=False))
    block = codec._PREFIX + source + codec._END + block_extra
    receipt = dict(schema="supervisor-public-instruction-inclusion@1", task_id=task,
        task_cid="cid:authored-task", context_cid="cid:authored-context", manifest_cid="cid:authored-manifest",
        artifact_sha256="a" * 64, source_path=INSTRUCTION_PATH, source_sha256=sha(source),
        source_bytes=len(source.encode()), block_sha256=sha(block), block_bytes=len(block.encode()),
        manifest_signature_verified=True, verbatim_utf8=True, semantic_minification_applied=False,
        completion_authority=False, scope_expansion_authority=False)
    view = codec.project_instruction_view(native, instruction_block=block, instruction_receipt=receipt)
    return native, view, block, receipt


def _selected(case):
    native, view, block, _receipt = case
    advisory, tail = "Authored workspace.\n", "\nReturn the requested report."
    selection = codec.select_instruction_view(view=view, native_complete_prompt=advisory + native + block + tail,
        candidate_complete_prompt=advisory + view.provider_prompt + block + tail, instruction_block=block)
    return selection, advisory


def test_actual_selected_codec_preserves_remaining_literals_and_counts_relocated_source_once(monkeypatch):
    case = instruction_case()
    native, view, block, _receipt = case
    selection, advisory = _selected(case)
    assert selection.selected_mode == codec.MODE
    wire, before = json.loads(view.provider_prompt), deepcopy(json.loads(view.provider_prompt))
    expected = json.loads(native)["translated_semantic"]["raw_sources"]
    # Fixtures are prepared first. The observer must not restore any prompt or
    # source after selection, even though the representation codec can do so.
    def forbidden(**kwargs):
        raise AssertionError("source restoration was invoked")
    monkeypatch.setattr(codec, "restore_instruction_view", forbidden)
    monkeypatch.setattr(metadata, "restore_semantic_metadata_view", forbidden)
    result = carrier.instruction_sources(wire, block + "Unrelated reply text")
    assert result == expected and wire == before
    result["other.py"] = "caller mutation"
    assert wire == before
    observed = observer.observe_inline_sources(model_prompt=selection.selected_prompt,
        workspace_advisory=advisory, invocation_id="authored-selected")
    assert observed["schema"] == "inline-source-observation@2"
    assert observed["scope"] == "translated_semantic.raw_sources_and_bound_public_instruction"
    assert observed["status"] == "observed" and observed["source_count"] == 2
    assert observed["logical_source_bytes"] == sum(len(text.encode()) for text in expected.values())
    assert {row["content_sha256"] for row in observed["entries"]} == {sha(text) for text in expected.values()}
    assert observed["wire_overlap_bytes"] is observed["provider_tokens"] is None
    assert observed["full_prompt_source_coverage"] is observed["provider_ingestion_verified"] is None
    exported = json.dumps(observed)
    assert INSTRUCTION_PATH not in exported and "other.py" not in exported and OTHER_SOURCE not in exported


def test_actual_short_source_fallback_observes_original_native_slots_without_candidate_restoration(monkeypatch):
    case = instruction_case(source="Do it.\n")
    native, view, _block, _receipt = case
    selection, advisory = _selected(case)
    assert selection.selected_mode == "legacy"
    assert selection.selected_prompt.startswith(advisory + native)
    assert view.provider_prompt not in selection.selected_prompt
    def forbidden(*args, **kwargs):
        raise AssertionError("unselected instruction candidate was inspected")
    monkeypatch.setattr(carrier, "instruction_sources", forbidden)
    observed = observer.observe_inline_sources(model_prompt=selection.selected_prompt,
        workspace_advisory=advisory, invocation_id="authored-fallback")
    assert observed["schema"] == "inline-source-observation@1"
    assert observed["scope"] == "translated_semantic.raw_sources"
    assert observed["transport_schema"] == metadata.TRANSPORT_SCHEMA
    assert observed["source_count"] == 2
    assert observed["logical_source_bytes"] == len("Do it.\n".encode()) + len(OTHER_SOURCE.encode())


def test_utf8_fixed_byte_span_ignores_marker_like_text_within_source():
    source = "雪 Ω\r\n" + codec._END + codec._PREFIX + "literal marker text remains source\n"
    native, view, block, _receipt = instruction_case(source=source)
    assert carrier.instruction_sources(json.loads(view.provider_prompt), block) == json.loads(native)["translated_semantic"]["raw_sources"]


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("schema", []), ("source_path", "foreign.txt"),
    ("source_sha256", "f" * 64), ("source_sha256", []), ("block_sha256", "f" * 64),
    ("source_bytes", True), ("source_bytes", 0), ("source_bytes", 32_769), ("source_bytes", 1),
])
def test_reference_changes_refuse_with_closed_reason(field, value):
    _native, view, block, _receipt = instruction_case()
    wire = json.loads(view.provider_prompt)
    wire["translated_semantic"]["raw_sources"][INSTRUCTION_PATH]["$public_instruction"][field] = value
    with pytest.raises(ValueError, match="^invalid_instruction_carrier$"):
        carrier.instruction_sources(wire, block)


@pytest.mark.parametrize("mutation", [
    "outer_extra", "outer_missing", "instructions", "metadata_instructions", "transport", "proof_extra",
    "execution", "completion", "boolean_authority", "common_type", "semantic_type", "manifest",
    "reference_extra", "wrapper_extra", "multiple_refs", "unknown_nonliteral", "no_reference", "path",
])
def test_closed_carrier_and_single_reference_are_required(mutation):
    _native, view, block, _receipt = instruction_case()
    wire = json.loads(view.provider_prompt)
    sources = wire["translated_semantic"]["raw_sources"]
    if mutation == "outer_extra": wire["PRIVATE"] = "untrusted"
    elif mutation == "outer_missing": wire.pop("translation_cid")
    elif mutation == "instructions": wire["instruction_instructions"] += "changed"
    elif mutation == "metadata_instructions": wire["metadata_instructions"] = "changed"
    elif mutation == "transport": wire["source_transport_schema"] = "foreign"
    elif mutation == "proof_extra": wire["proof_authority"] = True
    elif mutation == "execution": wire["execution_authority"] = True
    elif mutation == "completion": wire["completion_authority"] = True
    elif mutation == "boolean_authority": wire["execution_authority"] = 0
    elif mutation == "common_type": wire["common_bindings"] = []
    elif mutation == "semantic_type": wire["translated_semantic"] = []
    elif mutation == "manifest": wire["translated_semantic"]["manifest"][INSTRUCTION_PATH]["sha256"] = "f" * 64
    elif mutation == "reference_extra": sources[INSTRUCTION_PATH]["$public_instruction"]["PRIVATE"] = "untrusted"
    elif mutation == "wrapper_extra": sources[INSTRUCTION_PATH]["PRIVATE"] = "untrusted"
    elif mutation == "multiple_refs": sources["other.py"] = deepcopy(sources[INSTRUCTION_PATH])
    elif mutation == "unknown_nonliteral": sources["other.py"] = {"$unknown": "PRIVATE"}
    elif mutation == "no_reference": sources[INSTRUCTION_PATH] = "literal instead"
    else:
        reference = sources.pop(INSTRUCTION_PATH)
        reference["$public_instruction"]["source_path"] = "../instructions.txt"
        sources["../instructions.txt"] = reference
    with pytest.raises(ValueError, match="^invalid_instruction_carrier$"):
        carrier.instruction_sources(wire, block)


@pytest.mark.parametrize("mutation", ["absent", "guidance", "json_decoy", "changed_body", "changed_end", "truncated", "unicode", "oversized"])
def test_only_exact_suffix_prefix_and_declared_boundaries_are_accepted(mutation):
    _native, view, block, _receipt = instruction_case()
    if mutation == "absent": suffix = ""
    elif mutation == "guidance": suffix = "Guidance first.\n" + block
    elif mutation == "json_decoy": suffix = json.dumps({"quoted_block": block})
    elif mutation == "changed_body": suffix = block[:len(codec._PREFIX)] + "changed" + block[len(codec._PREFIX):]
    elif mutation == "changed_end": suffix = block[:-len(codec._END)] + "wrong boundary"
    elif mutation == "truncated": suffix = block[:-1]
    elif mutation == "unicode": suffix = block + "\ud800"
    else: suffix = block + "x" * carrier.MAX_SUFFIX_BYTES
    with pytest.raises(ValueError, match="^invalid_instruction_carrier$"):
        carrier.instruction_sources(json.loads(view.provider_prompt), suffix)


def test_enriched_instruction_block_is_not_mistaken_for_simple_source_carrier():
    _native, view, block, _receipt = instruction_case(block_extra="\nVerified task-specific IntentIR bindings: {}\n")
    with pytest.raises(ValueError, match="^invalid_instruction_carrier$"):
        carrier.instruction_sources(json.loads(view.provider_prompt), block)


def test_parent_observer_still_validates_remaining_literal_source_hashes():
    _native, view, block, _receipt = instruction_case()
    wire = json.loads(view.provider_prompt)
    wire["translated_semantic"]["raw_sources"]["other.py"] = "changed literal body\n"
    assert carrier.instruction_sources(wire, block)["other.py"] == "changed literal body\n"
    observed = observer.observe_inline_sources(model_prompt=metadata._json(wire) + block,
        workspace_advisory="", invocation_id="authored-stale-literal")
    assert observed["status"] == "unavailable"
    assert observed["entries"] is observed["source_count"] is observed["logical_source_bytes"] is None
