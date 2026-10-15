"""Explicit owner selections cross the bounded audit child without path grants."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import (
    compiled, metadata_dispatch,  # noqa: F401
)


def _capture_child(monkeypatch):
    calls = []

    def child(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(audit._base()).encode())

    # Keep unrelated imports and subprocess users outside this fixture's stub.
    proxy = SimpleNamespace(**vars(subprocess))
    proxy.run = child
    monkeypatch.setattr(audit, "subprocess", proxy)
    return calls


def test_collector_passes_only_explicit_owner_selection_as_child_arguments(compiled, monkeypatch):
    calls = _capture_child(monkeypatch)
    artifact = compiled["state"] / "owner-tokenizer.json"
    expected = "a" * 64
    receipts = deepcopy(compiled["receipts"])
    receipts[0]["semantic_metadata_tokenizer_source"] = {"descriptor_path": "/untrusted/from-receipt"}
    audit.collect_terminal_context_audit(state=compiled["state"], receipts=receipts,
        workspace_root=compiled["workspace_root"],
        semantic_metadata_tokenizer_artifact=artifact,
        semantic_metadata_tokenizer_sha256=expected)
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[argv.index("--semantic-metadata-tokenizer-artifact") + 1] == str(artifact)
    assert argv[argv.index("--semantic-metadata-tokenizer-sha256") + 1] == expected
    assert "/untrusted/from-receipt" not in argv
    assert kwargs["timeout"] <= audit.MAX_AUDIT_SECONDS


def test_collector_default_launch_has_no_tokenizer_arguments(compiled, monkeypatch):
    calls = _capture_child(monkeypatch)
    audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"])
    assert len(calls) == 1
    assert not any("semantic-metadata-tokenizer" in argument for argument in calls[0][0])


@pytest.mark.parametrize("artifact,expected", [
    (None, "a" * 64), (Path("/owner/descriptor.json"), None),
    (Path("/owner/descriptor.json"), ""), (Path("/owner/descriptor.json"), "A" * 64),
    (Path("/owner/descriptor.json"), True), ("../foreign.json", "a" * 64),
    ("https://untrusted/descriptor.json", "a" * 64),
])
def test_invalid_owner_selection_never_launches_audit_child(compiled, monkeypatch, artifact, expected):
    calls = _capture_child(monkeypatch)
    observed = audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"],
        semantic_metadata_tokenizer_artifact=artifact,
        semantic_metadata_tokenizer_sha256=expected)
    assert calls == []
    assert observed["reason"] == "audit_unavailable"
    assert observed["provider_calls"] == 0
    assert observed["completion_authority"] is False


def test_default_owner_selection_does_not_need_a_descriptor():
    assert audit._metadata_tokenizer_selection(None, None) is None


def _recorded_owner(case, folder, *, add_special_tokens=False):
    from test.api.test_local_prompt_tokenizer import make_binding
    from test.api.test_semantic_metadata_dispatch import invoke

    folder.mkdir()
    artifact, expected, _ = make_binding(folder, provider="codex_cli", model="authored-model",
        add_special_tokens=add_special_tokens)
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1",
        semantic_metadata_tokenizer_artifact=artifact, semantic_metadata_tokenizer_sha256=expected)
    receipt["phase"] = "coding"
    return artifact, expected, receipt


@pytest.mark.parametrize("special_tokens", [False, True])
def test_owner_selected_local_counter_replays_complete_input_and_policy(
        metadata_dispatch, tmp_path, special_tokens):
    case = metadata_dispatch
    artifact, expected, receipt = _recorded_owner(case, tmp_path / "owner", add_special_tokens=special_tokens)
    parsed = audit._receipt(receipt, case["workspace"].parent)
    selection = audit._metadata_tokenizer_selection(artifact, expected)
    counter, source = audit._metadata_counter_for_audit(parsed, selection)
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"], token_counter=counter, tokenizer_source=source)
    assert model == case["calls"][-1][0]
    assert all(checks.values()), checks
    assert checks["semantic_metadata_token_counter_verified"] is True
    assert checks["semantic_metadata_tokenizer_provenance"] is True
    assert source["qualification"] == "estimate"
    assert source["provider_tokenizer_equivalence_verified"] is False
    assert source["provider_usage_verified"] is source["completion_authority"] is False
    assert "descriptor_path" not in source and "asset_path" not in source
    assert str(artifact) not in json.dumps(source)
    assert source["special_token_policy"].endswith(str(special_tokens).lower())


def test_recorded_provenance_cannot_load_a_counter_without_owner_selection(
        metadata_dispatch, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    case = metadata_dispatch
    _, _, receipt = _recorded_owner(case, tmp_path / "owner")
    parsed = audit._receipt(receipt, case["workspace"].parent)
    monkeypatch.setattr(runner, "load_semantic_metadata_tokenizer",
        lambda **kwargs: pytest.fail("a receipt cannot authorize a tokenizer load"))
    counter, source = audit._metadata_counter_for_audit(parsed, None)
    assert counter is source is None
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"], token_counter=counter, tokenizer_source=source)
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_recorded_prompt_identities"] is True
    assert checks["semantic_metadata_token_counter_verified"] is False
    assert checks["semantic_metadata_tokenizer_provenance"] is False


def test_recorded_descriptor_pin_does_not_replace_independent_owner_pin(metadata_dispatch, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    case = metadata_dispatch
    artifact, expected, receipt = _recorded_owner(case, tmp_path / "owner")
    receipt["semantic_metadata_tokenizer_source"]["descriptor_sha256"] = "0" * 64
    parsed = audit._receipt(receipt, case["workspace"].parent)
    actual_load, loads = runner.load_semantic_metadata_tokenizer, []

    def observed_load(**kwargs):
        loads.append(kwargs)
        return actual_load(**kwargs)

    monkeypatch.setattr(runner, "load_semantic_metadata_tokenizer", observed_load)
    counter, source = audit._metadata_counter_for_audit(parsed,
        audit._metadata_tokenizer_selection(artifact, expected))
    _, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"], token_counter=counter, tokenizer_source=source)
    assert len(loads) == 1
    assert loads[0]["artifact"] == artifact and loads[0]["expected_sha256"] == expected
    assert checks["semantic_metadata_token_counter_verified"] is True
    assert checks["semantic_metadata_tokenizer_provenance"] is False


def test_independently_selected_different_policy_cannot_verify_recorded_counter(metadata_dispatch, tmp_path):
    from test.api.test_local_prompt_tokenizer import make_binding

    case = metadata_dispatch
    _, _, receipt = _recorded_owner(case, tmp_path / "owner", add_special_tokens=False)
    other = tmp_path / "other-policy"
    other.mkdir()
    artifact, expected, _ = make_binding(other, provider="codex_cli", model="authored-model",
        add_special_tokens=True)
    parsed = audit._receipt(receipt, case["workspace"].parent)
    counter, source = audit._metadata_counter_for_audit(parsed,
        audit._metadata_tokenizer_selection(artifact, expected))
    _, _, checks = audit._model_projection(rendered=case["prompt"], receipt=parsed,
        repository=case["repository"], token_counter=counter, tokenizer_source=source)
    assert checks["semantic_metadata_token_counter_verified"] is False
    assert checks["semantic_metadata_selection_receipt"] is False
    assert checks["semantic_metadata_tokenizer_provenance"] is False


@pytest.mark.parametrize("mutation", [
    {"descriptor_path": "/untrusted/receipt-file"}, {"completion_authority": True},
    {"qualification": "provider_tokenizer"}, {"special_token_policy": "unknown"},
    {"counter_tokenizer_revision": "policy-not-bound"}, {"provider_usage_verified": 0},
])
def test_receipt_provenance_rejects_paths_authority_and_policy_drift(metadata_dispatch, tmp_path, mutation):
    case = metadata_dispatch
    _, _, receipt = _recorded_owner(case, tmp_path / "owner")
    receipt["semantic_metadata_tokenizer_source"].update(mutation)
    with pytest.raises(ValueError):
        audit._receipt(receipt, case["workspace"].parent)


def test_changed_owner_asset_is_refused_without_extra_provider_dispatch(metadata_dispatch, tmp_path):
    case = metadata_dispatch
    artifact, expected, receipt = _recorded_owner(case, tmp_path / "owner")
    parsed = audit._receipt(receipt, case["workspace"].parent)
    (artifact.parent / "tokenizer.json").write_text("changed asset")
    before = len(case["calls"])
    with pytest.raises(ValueError):
        audit._metadata_counter_for_audit(parsed, audit._metadata_tokenizer_selection(artifact, expected))
    assert len(case["calls"]) == before
