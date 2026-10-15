"""Observe exact representation changes without inventing omission or billing.

The semantic producer and allocated Git worktree are real. Coding output and
token counters are authored fixtures; they provide no live provider evidence.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
import subprocess
import sys

import pytest

from test.api.test_semantic_metadata_view import native_factory  # noqa: F401
from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke  # noqa: F401
from test.api.test_semantic_instruction_view import instruction
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_instruction_view as instructions
from ipfs_accelerate_py.agent_supervisor.runtime import representation_displacement as displacement


PROVIDER = "PRIVATE_AUTHORED_PROVIDER"
MODEL = "PRIVATE_AUTHORED_MODEL"
PREFIX = "Private workspace café 🧪.\n"
SUFFIX = "\nPrivate requirements remain inline.\nPrivate reply contract."


def canonical(value):
    return metadata._json(value)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def pure_case(native_factory):
    _repository, encoded, source = native_factory(6)
    view = metadata.project_semantic_metadata_view(encoded.provider_prompt)
    return dict(view=view, source=source, native=PREFIX + view.native_prompt + SUFFIX,
        candidate=PREFIX + view.provider_prompt + SUFFIX)


def counter(count, **overrides):
    values = dict(count=count, provider=PROVIDER, model=MODEL,
        tokenizer="PRIVATE_AUTHORED_TOKENIZER", tokenizer_revision="PRIVATE_AUTHORED_REVISION",
        qualification="estimate")
    values.update(overrides)
    return metadata.SemanticMetadataTokenCounter(**values)


def select(case, token_counter=None):
    return metadata.select_semantic_metadata_view(view=case["view"],
        native_complete_prompt=case["native"], candidate_complete_prompt=case["candidate"],
        token_counter=token_counter)


def observe(case, selection=None, **options):
    return displacement.observe_representation_displacement(view=case["view"],
        selection=selection if selection is not None else select(case),
        native_complete_prompt=case["native"], candidate_complete_prompt=case["candidate"],
        provider=options.pop("provider", PROVIDER), model=options.pop("model", MODEL), **options)


def validate(receipt, selection):
    return displacement.validate_representation_displacement(receipt,
        selected_input_sha256=sha(selection.selected_prompt),
        selected_input_bytes=len(selection.selected_prompt.encode("utf-8")))


def changed_selection(selection, **changes):
    receipt = selection.receipt
    receipt.update(changes)
    return metadata.SemanticMetadataSelection(selection.selected_prompt,
        selection.selected_mode, canonical(receipt))


def dispatch_counter(count):
    return counter(count, provider="codex_cli", model="authored-model")


def test_exact_selected_input_is_observed_without_exporting_source_or_owner_text(pure_case):
    selection = select(pure_case)
    receipt = observe(pure_case, selection)
    assert receipt["schema"] == "supervisor-representation-displacement@1"
    assert receipt["treatment"] == "rendered_replacement"
    assert receipt["observation_stage"] == "router_prepared_input"
    assert receipt["source_preservation"] == "all_raw_sources_unchanged_inline"
    assert receipt["selected_complete_sha256"] == sha(selection.selected_prompt)
    assert receipt["selected_complete_bytes"] == len(selection.selected_prompt.encode())
    assert receipt["view_receipt_sha256"] == sha(pure_case["view"].receipt_json)
    assert receipt["selection_receipt_sha256"] == sha(selection.receipt_json)
    assert receipt["candidate_complete_byte_reduction"] == (
        len(pure_case["native"].encode()) - len(pure_case["candidate"].encode()))
    assert receipt["selected_complete_byte_reduction"] == receipt["candidate_complete_byte_reduction"] > 0
    assert receipt["token_counter_status"] == "not_supplied"
    assert receipt["source_guidance"] is None and receipt["proposed_avoided_reads"] == []
    validate(receipt, selection)
    serialized = canonical(receipt)
    for marker in (PROVIDER, MODEL, PREFIX.strip(), SUFFIX.strip(), pure_case["source"],
                   "module.py", "AUTHORED-METADATA-TASK", "ROOT_TEXT", "authored_function_0"):
        assert marker not in serialized


def test_utf8_replacement_offsets_address_bytes_and_preserve_literal_wrappers(pure_case):
    selection = select(pure_case)
    receipt = observe(pure_case, selection)
    assert len(PREFIX.encode()) > len(PREFIX)
    for arm, text, transport in (("native", pure_case["native"], pure_case["view"].native_prompt),
            ("candidate", pure_case["candidate"], pure_case["view"].provider_prompt),
            ("selected", selection.selected_prompt, pure_case["view"].provider_prompt)):
        span = receipt[arm + "_transport_span"]
        assert span == dict(start_byte=len(PREFIX.encode()),
            end_byte=len(PREFIX.encode()) + len(transport.encode()), sha256=sha(transport))
        assert text.encode()[span["start_byte"]:span["end_byte"]] == transport.encode()
    assert receipt["unchanged_prefix_sha256"] == sha(PREFIX)
    assert receipt["unchanged_prefix_bytes"] == len(PREFIX.encode())
    assert receipt["unchanged_suffix_sha256"] == sha(SUFFIX)
    assert receipt["unchanged_suffix_bytes"] == len(SUFFIX.encode())


@pytest.mark.parametrize("candidate_tokens", [100, 101])
def test_counter_fallback_preserves_exact_native_arm_and_no_displacement(pure_case, candidate_tokens):
    calls = []
    def count(text):
        calls.append(text)
        return 100 if text == pure_case["native"] else candidate_tokens
    selection = select(pure_case, counter(count))
    receipt = observe(pure_case, selection)
    assert calls == [pure_case["native"], pure_case["candidate"]]
    assert receipt["treatment"] == "native_fallback"
    assert selection.selected_prompt == pure_case["native"]
    assert receipt["selected_transport_span"] == receipt["native_transport_span"]
    assert receipt["selected_complete_byte_reduction"] == 0
    assert receipt["selected_complete_tokens"] == 100
    assert receipt["selection_basis"] == "recorded_token_counter"
    validate(receipt, selection)


@pytest.mark.parametrize("status", ["counted", "invalid", "failed"])
def test_observer_never_reinvokes_recorded_counter_and_never_claims_provider_usage(pure_case, status):
    calls = []
    def count(text):
        calls.append(text)
        if status == "failed":
            raise RuntimeError("PRIVATE_COUNTER_FAILURE " + text)
        return 100 if text == pure_case["native"] else 90
    token_counter = counter(count, qualification="foreign" if status == "invalid" else "estimate")
    selection = select(pure_case, token_counter)
    before = tuple(calls)
    receipt = observe(pure_case, selection)
    assert tuple(calls) == before
    assert receipt["token_counter_status"] == status
    assert receipt["token_counter_qualification"] == (None if status == "invalid" else "estimate")
    if status != "counted":
        assert receipt["selected_mode"] == "legacy"
        assert all(receipt[arm + "_complete_tokens"] is None for arm in ("native", "candidate", "selected"))
    else:
        assert receipt["native_complete_tokens"] == 100
        assert receipt["selected_complete_tokens"] == receipt["candidate_complete_tokens"] == 90
    for marker in ("PRIVATE_COUNTER_FAILURE", "PRIVATE_AUTHORED_TOKENIZER", "PRIVATE_AUTHORED_REVISION"):
        assert marker not in canonical(receipt)
    validate(receipt, selection)


def test_source_reader_guidance_remains_explicitly_additive(pure_case):
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_client import guidance
    advice = guidance("python-whitespace@1")
    case = {**pure_case, "native": pure_case["native"] + advice,
        "candidate": pure_case["candidate"] + advice}
    receipt = observe(case, source_guidance=advice)
    assert receipt["treatment"] == "rendered_replacement"
    assert receipt["source_guidance"] == dict(treatment="additive_retained_guidance", sha256=sha(advice),
        bytes=len(advice.encode()), suffix_start_byte=len(SUFFIX.encode()),
        suffix_end_byte=len(SUFFIX.encode()) + len(advice.encode()))
    assert receipt["proposed_avoided_reads"] == []


@pytest.mark.parametrize("change", ["absent", "duplicate"])
def test_source_guidance_requires_exact_unique_retained_suffix(pure_case, change):
    advice = "Private exact reader guidance.\n"
    copies = 0 if change == "absent" else 2
    case = {**pure_case, "native": pure_case["native"] + advice * copies,
        "candidate": pure_case["candidate"] + advice * copies}
    with pytest.raises(ValueError):
        observe(case, source_guidance=advice)


@pytest.mark.parametrize("candidate_tokens", [None, 200])
def test_instruction_duplicate_reference_retains_complete_verbatim_source(native_factory, candidate_tokens):
    _repository, encoded, source = native_factory(6)
    block, instruction_receipt = instruction(source)
    view = instructions.project_instruction_view(encoded.provider_prompt,
        instruction_block=block, instruction_receipt=instruction_receipt)
    case = dict(view=view, source=source, native=PREFIX + view.native_prompt + block + SUFFIX,
        candidate=PREFIX + view.provider_prompt + block + SUFFIX)
    selection = instructions.select_instruction_view(view=view, native_complete_prompt=case["native"],
        candidate_complete_prompt=case["candidate"], instruction_block=block,
        token_counter=(None if candidate_tokens is None else counter(
            lambda text: 100 if text == case["native"] else candidate_tokens)))
    receipt = observe(case, selection, instruction_block=block, instruction_receipt=instruction_receipt)
    assert receipt["source_preservation"] == ("signed_instruction_duplicate_referenced_verbatim_block_retained"
        if candidate_tokens is None else "all_raw_sources_unchanged_inline")
    assert receipt["instruction_source_sha256"] == sha(source)
    assert receipt["instruction_source_bytes"] == len(source.encode())
    assert receipt["instruction_block_sha256"] == sha(block)
    assert receipt["instruction_block_bytes"] == len(block.encode())
    assert receipt["instruction_receipt_sha256"] == sha(canonical(instruction_receipt))
    assert selection.selected_prompt.count(block) == 1
    assert source in selection.selected_prompt
    assert receipt["unchanged_suffix_sha256"] == sha(block + SUFFIX)
    validate(receipt, selection)


def test_self_rehashed_alternate_lossless_factoring_is_not_the_fixed_projector(pure_case):
    original = pure_case["view"]
    wire, receipt = json.loads(original.provider_prompt), original.receipt
    field = sorted(wire["common_bindings"]["capsules"])[0]
    value = wire["common_bindings"]["capsules"].pop(field)
    for row in wire["translated_semantic"]["capsules"]:
        row[field] = value
    alternate = canonical(wire)
    receipt.update(candidate_view_sha256=sha(alternate), candidate_view_bytes=len(alternate.encode()),
        candidate_view_proxy_tokens=(len(alternate.encode()) + 3) // 4,
        common_bindings_sha256=sha(canonical(wire["common_bindings"])),
        shared_field_paths=[[name, key] for name in sorted(wire["common_bindings"])
            for key in sorted(wire["common_bindings"][name])])
    view = metadata.SemanticMetadataView(original.native_prompt, alternate, canonical(receipt))
    assert metadata.restore_semantic_metadata_view(provider_prompt=alternate, receipt=receipt) == original.native_prompt
    case = {**pure_case, "view": view, "candidate": PREFIX + alternate + SUFFIX}
    with pytest.raises(ValueError):
        observe(case)


def retarget_selection(selection, case):
    """Make coherently rehashed arm claims; this is not a trusted selection."""
    chosen = case["candidate"] if selection.selected_mode != "legacy" else case["native"]
    receipt = selection.receipt
    for arm, text in (("native", case["native"]), ("candidate", case["candidate"]), ("selected", chosen)):
        receipt.update({arm + "_complete_sha256": sha(text), arm + "_complete_bytes": len(text.encode()),
            arm + "_complete_proxy_tokens": (len(text.encode()) + 3) // 4})
    return metadata.SemanticMetadataSelection(chosen, selection.selected_mode, canonical(receipt))


@pytest.mark.parametrize("change", ["native_prefix", "candidate_suffix", "duplicate_transport", "wrong_view"])
def test_rehashed_arm_or_wrapper_changes_refuse(pure_case, change):
    selection = select(pure_case)
    case = dict(pure_case)
    if change == "native_prefix":
        case["native"] = "Changed allocated workspace.\n" + case["native"]
    elif change == "candidate_suffix":
        case["candidate"] += "\nChanged requirements."
    elif change == "duplicate_transport":
        case["native"] += pure_case["view"].native_prompt
        case["candidate"] += pure_case["view"].provider_prompt
    else:
        case["view"] = replace(pure_case["view"], native_prompt=pure_case["view"].native_prompt + " ")
    rehashed = retarget_selection(selection, case)
    with pytest.raises(ValueError):
        observe(case, rehashed)


@pytest.mark.parametrize("mutation", [
    {"native_complete_sha256": "0" * 64},
    {"selected_complete_bytes": True},
    {"selected_mode": "legacy"},
    {"fallback_reason": "forged_fallback"},
    {"proof_authority": 0},
    {"unsupported_selection_field": "ignored"},
])
def test_existing_selection_mismatch_and_unknown_claims_refuse(pure_case, mutation):
    with pytest.raises(ValueError):
        observe(pure_case, changed_selection(select(pure_case), **mutation))


@pytest.mark.parametrize("mutation", [
    {"token_counter_status": "unknown"},
    {"token_counter_status": []},
    {"native_complete_tokens": True},
    {"selected_complete_tokens": 99},
    {"token_counter_identity_sha256": "0" * 64},
    {"actual_provider_usage_verified": True},
])
def test_recorded_counter_state_is_validated_without_remeasurement(pure_case, mutation):
    selection = select(pure_case, counter(lambda text: 100 if text == pure_case["native"] else 90))
    with pytest.raises(ValueError):
        observe(pure_case, changed_selection(selection, **mutation))


@pytest.mark.parametrize("caller", [{"provider": "OTHER_PROVIDER"}, {"model": "OTHER_MODEL"}])
def test_recorded_counter_must_match_current_caller_provider_and_model(pure_case, caller):
    selection = select(pure_case, counter(lambda text: 100 if text == pure_case["native"] else 90))
    with pytest.raises(ValueError):
        observe(pure_case, selection, **caller)


def test_self_rehashed_counter_identity_cannot_change_measurement_scope_or_qualification(pure_case):
    selection = select(pure_case, counter(lambda text: 100 if text == pure_case["native"] else 90))
    for field, value in (("measurement_scope", "semantic-transport-only@1"),
            ("qualification", "native_provider_usage"), ("identity_caller_supplied", 1),
            ("provider_usage_verified", 0)):
        receipt = selection.receipt
        receipt["token_counter_identity"][field] = value
        receipt["token_counter_identity_sha256"] = sha(canonical(receipt["token_counter_identity"]))
        changed = metadata.SemanticMetadataSelection(selection.selected_prompt,
            selection.selected_mode, canonical(receipt))
        with pytest.raises(ValueError):
            observe(pure_case, changed)


@pytest.mark.parametrize("field,value", [
    ("selected_complete_sha256", "0" * 64),
    ("selected_complete_bytes", True),
    ("candidate_complete_byte_reduction", True),
    ("treatment", "source_omission"),
    ("unknown_displacement_field", "ignored"),
])
def test_validator_refuses_selected_join_types_unknown_fields_and_omission_status(pure_case, field, value):
    selection = select(pure_case)
    receipt = observe(pure_case, selection)
    receipt[field] = value
    with pytest.raises(ValueError):
        validate(receipt, selection)


def test_all_authority_usage_and_semantic_verification_flags_are_strict_false(pure_case):
    selection = select(pure_case)
    receipt = observe(pure_case, selection)
    assert receipt["required_source_bytes_preserved"] is True
    for field in displacement._FALSE_FIELDS:
        assert receipt[field] is False
        for value in (0, True):
            with pytest.raises(ValueError):
                validate({**receipt, field: value}, selection)
    with pytest.raises(ValueError):
        validate({**receipt, "required_source_bytes_preserved": 1}, selection)


def test_byte_preservation_does_not_upgrade_declared_facts_or_ir_provenance(pure_case):
    selection = select(pure_case)
    receipt = observe(pure_case, selection)
    for field in ("required_fact_coverage_verified", "semantic_alignment_verified", "ir_provenance_verified",
            "program_semantics_proved", "omission_authority", "actual_avoided_reads_verified",
            "counter_identity_authenticated", "counter_measurement_replayed_here",
            "provider_input_delivery_verified", "complete_run_token_savings_claimed"):
        assert receipt[field] is False
    for field, value in (("declared_required_fact_ids", ["fact:authored"]),
            ("ir_family", "codebase_ir"), ("proposed_avoided_reads", ["path:module.py"])):
        with pytest.raises(ValueError):
            validate({**receipt, field: value}, selection)


def test_actual_router_opt_in_binds_exact_provider_argument_and_preserves_legacy(dispatch_case):
    case = dispatch_case
    _, legacy = invoke(case)
    original = case["calls"][-1][0]
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    chosen = case["calls"][-1][0]
    assert "representation_displacement" not in legacy
    assert "semantic_metadata_view" not in legacy
    assert len(case["calls"]) == 2
    observed = receipt["representation_displacement"]
    assert observed["native_complete_sha256"] == sha(original)
    assert observed["selected_complete_sha256"] == sha(chosen) == receipt["model_prompt_sha256"]
    assert observed["selected_complete_bytes"] == len(chosen.encode()) == receipt["model_prompt_bytes"]
    assert observed["selection_receipt_sha256"] == sha(canonical(receipt["semantic_metadata_view"]))
    assert receipt["usage"] == legacy["usage"]


def test_actual_router_counts_each_complete_arm_once_and_fallback_stays_native(dispatch_case):
    case = dispatch_case
    _, legacy = invoke(case)
    baseline = case["calls"][-1][0]
    calls = []
    def count(text):
        calls.append(text)
        return 100 if text == baseline else 101
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1",
        semantic_metadata_token_counter=dispatch_counter(count))
    assert len(calls) == 2 and calls[0] == baseline
    assert len(case["calls"]) == 2
    assert case["calls"][-1][0] == baseline
    assert receipt["representation_displacement"]["treatment"] == "native_fallback"
    assert receipt["representation_displacement"]["selected_complete_byte_reduction"] == 0
    assert receipt["usage"] == legacy["usage"]


@pytest.mark.parametrize("selected", ["native", "semantic", "semantic_metadata"])
def test_complete_input_selector_retains_only_selected_metadata_displacement(dispatch_case, selected):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import PromptTokenCounter

    case = dispatch_case
    invoke(case)
    semantic_input = case["calls"][-1][0]
    _, candidate = invoke(case, semantic_metadata_view="common-bindings@1")
    metadata_input = case["calls"][-1][0]
    assert candidate["semantic_metadata_view"]["selected_mode"] == "common-bindings@1"
    assert semantic_input != metadata_input
    def count(text):
        if selected == "native":
            return 1  # Exact ties must select the native capsule input.
        wanted = semantic_input if selected == "semantic" else metadata_input
        return 1 if text == wanted else 100
    tokenizer = PromptTokenCounter(provider="codex_cli", model="authored-model",
        tokenizer_id="authored", tokenizer_revision="representation-selection", special_token_policy="none",
        count_tokens=count)

    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1", prompt_token_counter=tokenizer)

    assert receipt["prompt_selection"]["selected_representation"] == selected
    assert ("representation_displacement" in receipt) is (selected == "semantic_metadata")
    assert ("semantic_metadata_view" in receipt) is (selected == "semantic_metadata")
    assert ("semantic_prompt_components" in receipt) is (selected != "native")
    assert (receipt["semantic_translation"] is not None) is (selected != "native")
    if selected != "semantic_metadata":
        assert receipt["semantic_metadata_candidate"] == candidate["semantic_metadata_view"]
        assert "semantic_metadata_native_coding_reply_contract" not in receipt
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["workspace"].parent)
    projected, _, checks = audit._model_projection(
        rendered=case["prompt"], receipt=parsed, repository=case["repository"])
    assert projected == case["calls"][-1][0] and all(checks.values()), checks
    assert ("representation_displacement_receipt" in checks) is (selected == "semantic_metadata")
    assert receipt["completion_authority"] is False
    if selected != "semantic_metadata":
        orphaned = {**receipt, "phase": "coding",
            "representation_displacement": candidate["representation_displacement"]}
        with pytest.raises(ValueError, match="requires retained metadata selection"):
            audit._receipt(orphaned, case["workspace"].parent)


def test_census_keeps_its_original_shape_and_does_not_dispatch_provider(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1", census_only=True)
    assert "representation_displacement" not in receipt
    assert case["calls"] == [] and receipt["provider_calls"] == 0


def test_actual_audit_parser_retains_and_independently_rebuilds_opt_in_receipt(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["workspace"].parent)
    assert parsed["representation_displacement"] == receipt["representation_displacement"]
    projected, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"])
    assert projected == case["calls"][-1][0]
    assert checks["representation_displacement_receipt"] is True
    assert all(checks.values()), checks


def test_historical_counter_receipt_reconstructs_delivery_without_inventing_trusted_counter(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1",
        semantic_metadata_token_counter=dispatch_counter(
            lambda text: 90 if '"schema":"supervisor-semantic-metadata-view@1"' in text else 100))
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["workspace"].parent)
    projected, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"])
    assert projected == case["calls"][-1][0]
    assert checks["representation_displacement_receipt"] is True
    assert checks["semantic_metadata_recorded_prompt_identities"] is True
    assert checks["semantic_metadata_token_counter_verified"] is False
    assert checks["semantic_metadata_selection_receipt"] is False
    assert len(case["calls"]) == 1


@pytest.mark.parametrize("field,value", [("selected_complete_sha256", "0" * 64),
    ("proof_authority", True), ("unexpected_receipt_field", "ignored")])
def test_actual_audit_parser_refuses_displacement_tampering(dispatch_case, field, value):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    changed = deepcopy(receipt)
    changed["representation_displacement"][field] = value
    with pytest.raises(ValueError):
        audit._receipt({**changed, "phase": "coding"}, case["workspace"].parent)


def test_coherently_changed_diagnostic_binding_fails_independent_historical_replay(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    changed = deepcopy(receipt)
    changed["representation_displacement"]["selection_receipt_sha256"] = "0" * 64
    parsed = audit._receipt({**changed, "phase": "coding"}, case["workspace"].parent)
    projected, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"])
    assert projected == case["calls"][-1][0]
    assert checks["representation_displacement_receipt"] is False


def test_separate_public_audit_process_preserves_displacement_and_recorded_input(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1")
    body = dict(receipt={**receipt, "phase": "coding"}, rendered=case["prompt"],
        repository=str(case["repository"]), workspace_root=str(case["workspace"].parent))
    code = """import hashlib, json, sys
from pathlib import Path
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
body = json.load(sys.stdin)
receipt = audit._receipt(body["receipt"], Path(body["workspace_root"]))
model, _, checks = audit._model_projection(rendered=body["rendered"], receipt=receipt,
    repository=Path(body["repository"]))
print(json.dumps({"sha256": hashlib.sha256(model.encode()).hexdigest(),
    "displacement_retained": "representation_displacement" in receipt, "checks": checks}))
"""
    child = subprocess.run([sys.executable, "-B", "-P", "-c", code], input=json.dumps(body),
        text=True, capture_output=True, check=True, timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    observed = json.loads(child.stdout)
    assert observed["sha256"] == receipt["model_prompt_sha256"]
    assert observed["displacement_retained"] is True
    assert observed["checks"]["representation_displacement_receipt"] is True
    assert all(observed["checks"].values()), observed["checks"]
    assert len(case["calls"]) == 1


from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_semantic_router_integration import provider  # noqa: F401
from test.api.test_semantic_instruction_dispatch import (
    expanded_selected, compose as compose_instruction, invoke as invoke_instruction,
    counter as instruction_counter,
)  # noqa: F401


@pytest.mark.parametrize("token_counted", [False, True])
def test_signed_instruction_history_preserves_original_receipt_hash_without_currentness_upgrade(
        expanded_selected, provider, monkeypatch, token_counted):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction

    case = compose_instruction(expanded_selected, monkeypatch)
    trusted = instruction_counter(90) if token_counted else None
    options = {"semantic_metadata_view": "common-bindings-instruction@1"}
    if trusted is not None:
        options["semantic_metadata_token_counter"] = trusted
    _, receipt = invoke_instruction(case, **options)
    actual = provider[1][-1][0]
    recorded = receipt["public_instruction"]
    observed = receipt["representation_displacement"]
    assert recorded["source_freshness_verified"] is True
    assert recorded["historical_replay"] is False
    assert observed["instruction_receipt_sha256"] == sha(canonical(recorded))
    assert receipt["semantic_metadata_view"]["selected_mode"] == "common-bindings-instruction@1"

    block, historical = load_public_instruction(**case["request"], require_current_source=False)
    assert historical["source_freshness_verified"] is False
    assert historical["historical_replay"] is True
    assert sha(canonical(historical)) != observed["instruction_receipt_sha256"]
    assert sha(block) == observed["instruction_block_sha256"]
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["request"]["workspace"].parent)

    # Legitimate replay uses the signed Git baseline after current text and the
    # allocated worktree are gone. It retains the original receipt's lineage.
    (case["root"] / "answer.py").write_text("def answer():\n    return 2\n")
    (case["root"] / "test_answer.py").write_text("changed after the original invocation")
    subprocess.run(["git", "-C", str(case["root"]), "worktree", "remove", "--force",
        str(case["request"]["workspace"])], check=True)
    monkeypatch.chdir(case["root"].parent)
    projected, _, checks = audit._model_projection(rendered=case["request"]["prompt"],
        receipt=parsed, repository=case["root"])
    assert projected == actual
    assert checks["representation_displacement_receipt"] is True
    assert parsed["representation_displacement"]["instruction_receipt_sha256"] == sha(canonical(recorded))
    assert parsed["representation_displacement"]["source_freshness_verified"] is False
    assert parsed["representation_displacement"]["counter_measurement_replayed_here"] is False
    assert len(provider[1]) == 1
    if token_counted:
        assert checks["semantic_metadata_recorded_prompt_identities"] is True
        assert checks["semantic_metadata_token_counter_verified"] is False
        assert checks["semantic_metadata_selection_receipt"] is False
    else:
        assert all(checks.values()), checks
