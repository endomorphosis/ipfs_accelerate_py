"""Columns presentation crosses real native boundaries with authored providers."""
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke, token_counter  # noqa: F401
from test.api.test_semantic_metadata_view import native_factory  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_capsule_columns as columns
from ipfs_accelerate_py.agent_supervisor.runtime import representation_displacement as displacement
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit


MODE = "capsule-columns@1"


def test_opt_in_replaces_transport_and_preserves_default_input_and_usage(dispatch_case):
    case = dispatch_case
    _, baseline = invoke(case)
    native_input = case["calls"][-1][0]
    _, receipt = invoke(case, semantic_metadata_view=MODE)
    selected = case["calls"][-1][0]
    metadata = receipt["semantic_metadata_view"]
    assert "semantic_metadata_view" not in baseline
    assert metadata["requested_mode"] == MODE
    assert metadata["selected_mode"] == MODE
    assert metadata["selected_complete_sha256"] == hashlib.sha256(selected.encode()).hexdigest()
    assert metadata["selected_complete_bytes"] == len(selected.encode()) < len(native_input.encode())
    assert receipt["usage"] == baseline["usage"]
    assert receipt["coding_reply_contract"]["instruction_sha256"] == baseline["coding_reply_contract"]["instruction_sha256"]
    retained_native = receipt["semantic_metadata_native_coding_reply_contract"]
    assert {key: value for key, value in retained_native.items() if key != "response_validated"} == {
        key: value for key, value in baseline["coding_reply_contract"].items() if key != "response_validated"}
    assert retained_native["response_validated"] is False
    assert receipt["coding_reply_contract"]["response_validated"] is True
    assert baseline["coding_reply_contract"]["response_validated"] is True
    assert case["calls"][-1][1]["codex_output_schema"] == case["calls"][0][1]["codex_output_schema"]
    observed = receipt["representation_displacement"]
    assert observed["requested_mode"] == observed["selected_mode"] == MODE
    assert observed["required_source_bytes_preserved"] is True
    assert observed["source_preservation"] == "all_raw_sources_unchanged_inline"
    assert observed["provider_input_delivery_verified"] is False
    assert observed["complete_run_token_savings_claimed"] is False


@pytest.mark.parametrize("status", ["counted", "equal", "failed"])
def test_counter_uses_complete_inputs_and_native_fallback_is_exact(dispatch_case, status):
    case = dispatch_case
    invoke(case)
    native_input = case["calls"][-1][0]
    measured = []
    def count(text):
        measured.append(text)
        if status == "failed":
            raise RuntimeError("PRIVATE_COUNTER_DETAIL")
        return 100 if len(measured) == 1 or status == "equal" else 90
    _, receipt = invoke(case, semantic_metadata_view=MODE,
        semantic_metadata_token_counter=token_counter(count))
    metadata = receipt["semantic_metadata_view"]
    assert measured[0] == native_input
    assert len(measured) == (1 if status == "failed" else 2)
    assert metadata["schema"] == "supervisor-semantic-metadata-selection@2"
    assert metadata["requested_mode"] == MODE
    assert metadata["selected_mode"] == (MODE if status == "counted" else "legacy")
    assert metadata["provider_usage_reported"] is False
    if status != "counted":
        assert case["calls"][-1][0] == native_input
        assert receipt["representation_displacement"]["selected_complete_byte_reduction"] == 0
    assert "PRIVATE_COUNTER_DETAIL" not in json.dumps(receipt)


@pytest.mark.parametrize("options", [
    {"purpose": "planning"}, {"semantic_transport_schema": "supervisor-semantic-router-input@2"},
    {"census_only": True}, {"source_input_policy": "source-input-raw@1"},
    {"source_input_policy": "source-input-ast-ref-compact@1"},
    {"query_reference_policy": "required-instruction-ref@1"},
])
def test_unsupported_routes_refuse_before_provider(dispatch_case, options):
    with pytest.raises(ValueError):
        invoke(dispatch_case, semantic_metadata_view=MODE, **options)
    assert dispatch_case["calls"] == []


@pytest.mark.parametrize("settings", [{"provider": "grok_cli"}, {"model": "foreign-model"}])
def test_counter_pin_mismatch_refuses_before_count_or_provider(dispatch_case, settings):
    def forbidden(_text):
        pytest.fail("mismatched token counter called")
    with pytest.raises(ValueError, match="pinned provider and model"):
        invoke(dispatch_case, semantic_metadata_view=MODE,
            semantic_metadata_token_counter=token_counter(forbidden, **settings))
    assert dispatch_case["calls"] == []


@pytest.mark.parametrize("selected", ["native", "semantic", "semantic_metadata"])
def test_final_complete_input_selection_cancels_unselected_columns_observation(dispatch_case, selected):
    case = dispatch_case
    invoke(case)
    semantic_input = case["calls"][-1][0]
    _, evaluated = invoke(case, semantic_metadata_view=MODE)
    columns_input = case["calls"][-1][0]
    assert evaluated["semantic_metadata_view"]["selected_mode"] == MODE
    def count(text):
        if selected == "native":
            return 1
        wanted = semantic_input if selected == "semantic" else columns_input
        return 1 if text == wanted else 100
    tokenizer = runner.PromptTokenCounter(provider="codex_cli", model="authored-model",
        tokenizer_id="authored", tokenizer_revision="columns-final-selection", special_token_policy="none",
        count_tokens=count)
    _, receipt = invoke(case, semantic_metadata_view=MODE, prompt_token_counter=tokenizer)
    assert receipt["prompt_selection"]["selected_representation"] == selected
    assert ("representation_displacement" in receipt) is (selected == "semantic_metadata")
    assert ("semantic_metadata_view" in receipt) is (selected == "semantic_metadata")
    if selected != "semantic_metadata":
        assert receipt["semantic_metadata_candidate"] == evaluated["semantic_metadata_view"]
        assert "semantic_metadata_native_coding_reply_contract" not in receipt
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["workspace"].parent)
    projected, _header, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"])
    assert projected == case["calls"][-1][0] and all(checks.values()), checks
    assert ("representation_displacement_receipt" in checks) is (selected == "semantic_metadata")


def test_independent_historical_audit_reconstructs_exact_columns_input(dispatch_case):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view=MODE)
    receipt["phase"] = "coding"
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _header, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"])
    assert model == case["calls"][-1][0]
    assert all(checks.values()), checks
    assert checks["semantic_metadata_selection_receipt"]
    assert checks["representation_displacement_receipt"]


def test_token_audit_keeps_measurement_unqualified_without_independent_counter(dispatch_case):
    case = dispatch_case
    count = token_counter(lambda text: len(text.encode()))
    _, receipt = invoke(case, semantic_metadata_view=MODE, semantic_metadata_token_counter=count)
    receipt["phase"] = "coding"
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _header, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"])
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_recorded_prompt_identities"]
    assert checks["representation_displacement_receipt"]
    assert checks["semantic_metadata_selection_receipt"] is False
    assert checks["semantic_metadata_token_counter_verified"] is False
    replayed, _header, replay_checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"], token_counter=count)
    assert replayed == model and all(replay_checks.values()), replay_checks


@pytest.mark.parametrize("change", ["request", "selected_hash", "boolean_bytes"])
def test_forged_columns_delivery_binding_is_rejected(dispatch_case, change):
    case = dispatch_case
    _, receipt = invoke(case, semantic_metadata_view=MODE)
    receipt = deepcopy(receipt)
    receipt["phase"] = "coding"
    metadata = receipt["semantic_metadata_view"]
    if change == "request": metadata["requested_mode"] = "common-bindings@1"
    elif change == "selected_hash": metadata["selected_complete_sha256"] = "0" * 64
    else: metadata["selected_complete_bytes"] = True
    with pytest.raises(ValueError):
        audit._receipt(receipt, case["workspace"].parent)


def test_columns_displacement_retains_sources_and_exact_utf8_wrappers(native_factory):
    _repository, encoded, source = native_factory(6)
    view = columns.project_capsule_columns(encoded.provider_prompt)
    prefix, suffix = "Workspace café 🧪.\n", "\nLiteral requirements remain inline."
    native_input, candidate = prefix + view.native_prompt + suffix, prefix + view.provider_prompt + suffix
    selected = columns.select_capsule_columns(view=view, native_complete_prompt=native_input,
        candidate_complete_prompt=candidate)
    observed = displacement.observe_representation_displacement(view=view, selection=selected,
        native_complete_prompt=native_input, candidate_complete_prompt=candidate,
        provider="authored-provider", model="authored-model")
    assert selected.selected_mode == MODE
    span = observed["selected_transport_span"]
    assert candidate.encode()[span["start_byte"]:span["end_byte"]] == view.provider_prompt.encode()
    assert span["start_byte"] == len(prefix.encode()) > len(prefix)
    assert source not in json.dumps(observed)
    assert columns.restore_capsule_columns(provider_prompt=view.provider_prompt, receipt=view.receipt) == encoded.provider_prompt
    with pytest.raises(ValueError):
        displacement.observe_representation_displacement(view=view, selection=selected,
            native_complete_prompt=native_input, candidate_complete_prompt=candidate,
            provider="authored-provider", model="authored-model", instruction_block=suffix)


def test_cli_forwards_explicit_columns_mode_without_changing_default(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"Authored prompt")))
    monkeypatch.setattr(runner, "run", lambda **kwargs: (calls.append(kwargs) or "Authored output", {}))
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--model", "authored-model",
        "--semantic-repository", str(tmp_path), "--semantic-metadata-view", MODE])
    assert runner.main() == 0
    assert calls[-1]["semantic_metadata_view"] == MODE
    monkeypatch.setattr(runner.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"Authored prompt")))
    monkeypatch.setattr(runner.sys, "argv", ["runner", "--model", "authored-model"])
    assert runner.main() == 0
    assert calls[-1]["semantic_metadata_view"] == "legacy"
