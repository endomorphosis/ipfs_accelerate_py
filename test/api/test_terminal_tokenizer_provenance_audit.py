"""Retain historical tokenizer declarations without rereading private assets.

The capsule compiler and audit execute normally. Tokenizer counts, compatibility
and absent local assets are authored historical fixtures, never model evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess

import pytest
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import (
    compiled,  # noqa: F401 -- register the ordinary native context fixture
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def declared(compiled):  # noqa: F811 -- pytest injects the registered fixture
    receipt = compiled["receipts"][0]
    source = {
        "schema": "supervisor-local-prompt-tokenizer-provenance@1",
        "backend": "tokenizers-json@1",
        "descriptor_path": str(compiled["workspace_root"] / "private owner/descriptor.json"),
        "descriptor_sha256": "d" * 64,
        "asset_path": str(compiled["workspace_root"] / "private owner/tokenizer.json"),
        "asset_sha256": "a" * 64,
        "tokenizers_version": "authored-unavailable-historical-version",
        "compatibility_basis": "owner_declaration", "completion_authority": False,
    }
    binding = {
        "provider": "codex_cli", "model": "authored-fixture-model",
        "tokenizer_id": "authored-historical-tokenizer-fixture",
        "tokenizer_revision": "tokenizers/" + source["tokenizers_version"] + "/sha256:" + source["asset_sha256"],
        "special_token_policy": "tokenizers.encode:add_special_tokens=false",
    }
    # This count is explicitly a recorded fixture. The historical auditor must
    # validate its receipt consistency while never attempting to reproduce it.
    selection = {
        "schema": "supervisor-model-prompt-selection@1", "selected_representation": "native",
        "selection_reason": "semantic_not_requested", "candidates": {"native": {
            "model_prompt_sha256": receipt["model_prompt_sha256"],
            "model_prompt_bytes": receipt["model_prompt_bytes"], "input_tokens": 37}},
        "measurement_status": "measured", "measurement_error_type": None,
        "tokenizer_binding": binding, "count_scope": "rendered_user_prompt_only",
        "selected_content_tokens": 37, "max_input_tokens": 4096,
        "max_prompt_bytes": 256_000, "total_session_token_savings": None,
        "completion_authority": False,
    }
    receipt.update(
        provider=binding["provider"], model=binding["model"],
        router_prompt_sha256=receipt["native_prompt_sha256"],
        router_prompt_bytes=receipt["native_prompt_bytes"], semantic_encoding_candidate=None,
        semantic_translation=None, prompt_selection=selection, prompt_tokenizer_source=source,
    )
    return compiled


def observe(fixture):
    return audit.audit_terminal_context(state=fixture["state"], receipts=fixture["receipts"],
                                        workspace_root=fixture["workspace_root"])


def expected_export(source):
    return {
        "schema": "terminal-historical-prompt-tokenizer-provenance@1",
        "source_schema": source["schema"], "backend": source["backend"],
        "descriptor_sha256": source["descriptor_sha256"], "asset_sha256": source["asset_sha256"],
        "tokenizers_version": source["tokenizers_version"],
        "tokenizer_revision": "tokenizers/" + source["tokenizers_version"] + "/sha256:" + source["asset_sha256"],
        "descriptor_path_sha256": digest(source["descriptor_path"]),
        "asset_path_sha256": digest(source["asset_path"]),
        "compatibility_basis": "owner_declaration", "recorded_binding_validated": True,
        "source_currentness_revalidated": False, "counts_recomputed": False,
        "provider_tokenizer_compatibility_verified": False, "completion_authority": False,
    }


def assert_observation(result, fixture):
    assert result["status"] == "verified_all_observed_model_inputs", result
    source = fixture["receipts"][0]["prompt_tokenizer_source"]
    expected = expected_export(source)
    assert result["matches"][0]["prompt_tokenizer_source"] == expected
    assert result["invocations"][0]["prompt_tokenizer_source"] == expected
    assert result["provider_calls"] == 0
    assert result["completion_authority"] is result["task_correctness_established"] is False
    exported = json.dumps(result)
    for private in (source["descriptor_path"], source["asset_path"],
                    fixture["receipts"][0]["workspace"], fixture["query"]):
        assert private not in exported
    assert "private source query" not in exported
    assert "private owner" not in exported


def test_actual_context_audit_retains_historical_provenance_without_current_assets(declared, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("historical audit must not load assets, runtime or tokenizer")
    monkeypatch.setattr(local, "load_owner_prompt_tokenizer", forbidden)
    monkeypatch.setattr(local, "_bounded_local_bytes", forbidden)
    monkeypatch.setattr(local, "_runtime", forbidden)
    assert not declared["workspace_root"].exists()
    before = copy.deepcopy(declared["receipts"])
    assert_observation(observe(declared), declared)
    assert declared["receipts"] == before


def test_child_request_retains_route_and_source_then_exports_redacted_evidence(declared, monkeypatch):
    original = copy.deepcopy(declared["receipts"])
    calls = []

    def child(arguments, **kwargs):
        calls.append(arguments)
        rows = json.loads(kwargs["input"])
        assert len(rows) == 1
        for key in ("provider", "model", "prompt_tokenizer_source"):
            assert rows[0][key] == original[0][key]
        observed = audit.audit_terminal_context(state=declared["state"], receipts=rows,
                                                workspace_root=declared["workspace_root"])
        return subprocess.CompletedProcess(arguments, 0, json.dumps(observed).encode(), b"")

    monkeypatch.setattr(audit.subprocess, "run", child)
    result = audit.collect_terminal_context_audit(state=declared["state"], receipts=declared["receipts"],
        workspace_root=declared["workspace_root"], timeout_seconds=1)
    assert len(calls) == 1
    assert_observation(result, declared)
    assert declared["receipts"] == original


def test_recorded_injected_counter_without_local_provenance_stays_compatible(declared):
    del declared["receipts"][0]["prompt_tokenizer_source"]
    result = observe(declared)
    assert result["status"] == "verified_all_observed_model_inputs"
    assert "prompt_tokenizer_source" not in result["matches"][0]
    assert "prompt_tokenizer_source" not in result["invocations"][0]


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("backend", "remote"), ("compatibility_basis", "verified_remote_model"),
    ("completion_authority", True), ("completion_authority", 0),
    ("descriptor_sha256", "D" * 64), ("asset_sha256", "a" * 63),
    ("tokenizers_version", ""), ("tokenizers_version", True),
    ("tokenizers_version", "changed-version"), ("tokenizers_version", "v" * 129),
    ("tokenizers_version", "/owner/private-version"),
    ("descriptor_path", "relative/descriptor.json"),
    ("descriptor_path", "/owner/../private/descriptor.json"),
    ("descriptor_path", "/owner/./descriptor.json"),
    ("descriptor_path", "/owner//descriptor.json"),
    ("descriptor_path", "https://example.invalid/descriptor.json"),
    ("descriptor_path", "/owner/descriptor.json\x00"),
    ("descriptor_path", "/owner/descriptor.json\\other"),
    ("descriptor_path", "/"), ("descriptor_path", "/" + "x" * 4096),
    ("asset_path", "relative/tokenizer.json"), ("asset_path", None),
])
def test_malformed_source_never_becomes_a_usable_receipt(declared, field, value):
    declared["receipts"][0]["prompt_tokenizer_source"][field] = value
    result = observe(declared)
    assert result["usable_coding_receipts"] == 0
    assert result["reason"] == "no_usable_coding_receipt"
    assert result["read_errors"] == [{"kind": "receipt", "error_type": "InvalidReceipt"}]
    assert result["matches"] == result["invocations"] == []


@pytest.mark.parametrize("mutation", ["extra", "missing", "null", "revision", "policy", "selection", "provider", "model"])
def test_closed_source_and_selection_bindings_refuse_inconsistent_receipts(declared, mutation):
    receipt = declared["receipts"][0]
    source = receipt["prompt_tokenizer_source"]
    if mutation == "extra":
        source["unexpected"] = "forbidden"
    elif mutation == "missing":
        del source["descriptor_sha256"]
    elif mutation == "null":
        receipt["prompt_tokenizer_source"] = None
    elif mutation == "revision":
        receipt["prompt_selection"]["tokenizer_binding"]["tokenizer_revision"] = "foreign"
    elif mutation == "policy":
        receipt["prompt_selection"]["tokenizer_binding"]["special_token_policy"] = "guessed"
    elif mutation == "selection":
        receipt["prompt_selection"] = None
    else:
        receipt[mutation] = "foreign"
    result = observe(declared)
    assert result["usable_coding_receipts"] == 0
    assert result["reason"] == "no_usable_coding_receipt"
    assert result["read_errors"] == [{"kind": "receipt", "error_type": "InvalidReceipt"}]
