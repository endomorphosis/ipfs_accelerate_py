"""Historical audits distinguish recorded estimates from provider usage.

Real resolver/selector and capsule audit entrypoints process authored historical
receipts. Cache/encoding doubles provide declarations, not current-asset proof.
No providers, Docker tasks, hidden verifiers, downloads, or model bodies run.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.agent_supervisor.container_coding import (
    automatic_tokenizer_audit as validator,
)
from benchmarks.agent_supervisor.container_coding import (
    terminal_context_audit as audit,
)

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.runtime import automatic_prompt_tokenizer as automatic
from ipfs_accelerate_py.agent_supervisor.runtime import local_prompt_tokenizer as local
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import (
    PromptTokenCounter,
    select_model_prompt,
)
from test.api import test_terminal_tokenizer_provenance_audit as historical_fixtures

compiled = historical_fixtures.compiled
declared = historical_fixtures.declared


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _selection(fixture, counter, source=None):
    receipt = fixture["receipts"][0]
    capsule = ContextCapsule.from_dict(json.loads(fixture["capsule"].read_text()))
    prompt = render_context_capsule(capsule)
    model, router, advisory, selection = select_model_prompt(native_prompt=prompt, semantic_prompt=None,
        suffix="", purpose="coding", workspace=Path(receipt["workspace"]),
        provider=counter.provider, model=counter.model, token_counter=counter, max_input_tokens=100_000)
    receipt.update(provider=counter.provider, model=counter.model,
        router_prompt_sha256=_sha(router), router_prompt_bytes=len(router.encode()),
        model_prompt_sha256=_sha(model), model_prompt_bytes=len(model.encode()),
        workspace_advisory_sha256=_sha(advisory), workspace_advisory_bytes=len(advisory.encode()),
        semantic_encoding_candidate=None, semantic_translation=None, prompt_selection=selection)
    if source is not None:
        receipt["prompt_tokenizer_source"] = source
    return selection


def _observe(fixture):
    return audit.audit_terminal_context(state=fixture["state"], receipts=fixture["receipts"],
                                        workspace_root=fixture["workspace_root"])


def _forbid_recount(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("historical audit attempted tokenizer resolution, asset loading, or recount")

    monkeypatch.setattr(automatic, "resolve_prompt_tokenizer", forbidden)
    monkeypatch.setattr(automatic, "_load_cached_ranks", forbidden)
    monkeypatch.setattr(local, "load_owner_prompt_tokenizer", forbidden)
    monkeypatch.setattr(PromptTokenCounter, "count", forbidden)


@pytest.fixture
def estimated(compiled, monkeypatch):
    def unavailable():
        raise ValueError("authored unavailable tokenizer runtime")

    monkeypatch.setattr(automatic, "_installed_tiktoken", unavailable)
    counter, source = automatic.resolve_prompt_tokenizer(provider="codex_cli", model="authored-fixture-model")
    _selection(compiled, counter, source)
    return compiled


def _cached_counter(monkeypatch, *, provider, model):
    class AuthoredEncoding:
        def __init__(self, **_kwargs):
            pass

        def encode_ordinary(self, prompt):
            return range(max(1, (len(prompt.encode()) + 3) // 4))

    module = SimpleNamespace(Encoding=AuthoredEncoding)
    mapping = SimpleNamespace(MODEL_TO_ENCODING={"gpt-5": "o200k_base"},
                              MODEL_PREFIX_TO_ENCODING={"gpt-5-": "o200k_base"})
    monkeypatch.setattr(automatic, "_installed_tiktoken", lambda: (module, mapping, "0.13.0"))
    monkeypatch.setattr(automatic, "_load_cached_ranks", lambda name: ({}, validator._RANK_HASHES[name]))
    return automatic.resolve_prompt_tokenizer(provider=provider, model=model)


@pytest.mark.parametrize("provider,model,kind,match", [
    ("codex_cli", "gpt-5", "tokenizer", "exact"),
    ("openai", "gpt-5-authored-prefix", "tokenizer", "prefix"),
    ("codex_cli", "authored-unknown-model", "estimate", None),
    ("grok_cli", "gpt-5", "estimate", None),
])
def test_actual_resolver_and_selector_declarations_survive_historical_audit(compiled, monkeypatch, provider, model, kind, match):
    counter, source = _cached_counter(monkeypatch, provider=provider, model=model)
    selection = _selection(compiled, counter, source)
    assert selection["schema"] == "supervisor-model-prompt-selection@2"
    assert selection["measurement_status"] == ("estimated" if kind == "estimate" else "measured")
    assert source["model_mapping_match"] == match
    before = copy.deepcopy(compiled["receipts"])
    _forbid_recount(monkeypatch)
    result = _observe(compiled)
    assert result["status"] == "verified_all_observed_model_inputs", result
    summary = result["matches"][0]["prompt_tokenizer_source"]
    assert summary == result["invocations"][0]["prompt_tokenizer_source"]
    assert summary["count_kind"] == kind
    assert summary["source_schema"] == automatic.SCHEMA
    assert summary["recorded_binding_validated"] is True
    assert summary["source_currentness_revalidated"] is False
    assert summary["counts_recomputed"] is False
    assert summary["provider_tokenizer_compatibility_verified"] is False
    assert summary["remote_tokenizer_compatibility_verified"] is False
    assert result["provider_calls"] == 0
    assert result["rendered_input_census"]["observations"][0]["provider_usage_tokens"] is None
    assert compiled["receipts"] == before


def test_utf8_estimate_history_is_valid_without_recount_or_provider_usage(estimated, monkeypatch):
    selection = estimated["receipts"][0]["prompt_selection"]
    assert selection["measurement_status"] == "estimated"
    assert selection["selected_content_tokens"] is None
    assert selection["selected_estimated_content_tokens"] > 0
    assert selection["token_budget_basis"] == "estimated_content_tokens"
    assert selection["token_budget_guaranteed"] is False
    _forbid_recount(monkeypatch)
    result = _observe(estimated)
    assert result["status"] == "verified_all_observed_model_inputs", result
    summary = result["invocations"][0]["prompt_tokenizer_source"]
    assert summary["backend"] == "utf8-bytes"
    assert summary["rank_sha256"] is None
    assert summary["compatibility_basis"] == "utf8_byte_estimate"
    exported = json.dumps(result)
    assert estimated["query"] not in exported
    assert estimated["receipts"][0]["workspace"] not in exported
    assert result["completion_authority"] is result["task_correctness_established"] is False


def test_new_exact_owner_receipt_keeps_unchanged_historical_descriptor_summary(declared, monkeypatch):
    receipt = declared["receipts"][0]
    counter = PromptTokenCounter(**receipt["prompt_selection"]["tokenizer_binding"],
                                count_tokens=lambda _: 37, compatibility_basis="owner_declaration")
    _selection(declared, counter)
    _forbid_recount(monkeypatch)
    result = _observe(declared)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["matches"][0]["prompt_tokenizer_source"] == historical_fixtures.expected_export(
        receipt["prompt_tokenizer_source"])


def test_standalone_owner_child_loads_exact_helper_with_estimated_receipt(estimated, monkeypatch, tmp_path):
    foreign = tmp_path / "foreign"
    package = foreign / "benchmarks"
    package.mkdir(parents=True)
    marker = foreign / "foreign-imported"
    (package / "__init__.py").write_text(
        "from pathlib import Path\n" + f"Path({str(marker)!r}).write_text('wrong owner')\n"
        "raise RuntimeError('foreign benchmarks package')\n")
    monkeypatch.setenv("PYTHONPATH", str(foreign))
    result = audit.collect_terminal_context_audit(state=estimated["state"], receipts=estimated["receipts"],
        workspace_root=estimated["workspace_root"], timeout_seconds=10)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["invocations"][0]["prompt_tokenizer_source"]["counts_recomputed"] is False
    assert not marker.exists()


def test_collector_preserves_selection_and_automatic_source_without_normalizing(estimated, monkeypatch):
    before = copy.deepcopy(estimated["receipts"])

    def child(command, **kwargs):
        rows = json.loads(kwargs["input"])
        assert rows == before
        result = audit.audit_terminal_context(state=estimated["state"], receipts=rows,
            workspace_root=estimated["workspace_root"])
        return subprocess.CompletedProcess(command, 0, json.dumps(result).encode(), b"")

    _forbid_recount(monkeypatch)
    monkeypatch.setattr(audit.subprocess, "run", child)
    result = audit.collect_terminal_context_audit(state=estimated["state"], receipts=estimated["receipts"],
        workspace_root=estimated["workspace_root"], timeout_seconds=1)
    assert result["status"] == "verified_all_observed_model_inputs"
    assert estimated["receipts"] == before


def _invalid(fixture):
    result = _observe(fixture)
    assert result["usable_coding_receipts"] == 0
    assert result["read_errors"] == [{"kind": "receipt", "error_type": "InvalidReceipt"}]
    assert result["matches"] == result["invocations"] == []


def test_reviewed_historical_mapping_snapshot_matches_installed_primary_source():
    spec = importlib.util.find_spec("tiktoken.model")
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text())
    recorded = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.target.id in {"MODEL_TO_ENCODING", "MODEL_PREFIX_TO_ENCODING"}
    }
    assert recorded["MODEL_TO_ENCODING"] == validator._EXACT_MODELS
    assert list(recorded["MODEL_PREFIX_TO_ENCODING"].items()) == list(validator._PREFIX_MODELS.items())


@pytest.mark.parametrize("field,value", [
    ("schema", "supervisor-model-prompt-selection@1"), ("token_count_kind", "tokenizer"),
    ("measurement_status", "measured"), ("tokenizer_compatibility_basis", "verified_remote_model"),
    ("selected_content_tokens", 37), ("selected_estimated_content_tokens", True),
    ("selected_estimated_content_tokens", 1), ("token_budget_basis", "local_tokenizer_content_tokens"),
    ("token_budget_guaranteed", True), ("token_budget_guaranteed", 0),
    ("selection_reason", "semantic_strictly_fewer_content_tokens"), ("max_input_tokens", True),
    ("max_input_tokens", 1), ("max_prompt_bytes", 256001), ("measurement_error_type", "invented"),
    ("total_session_token_savings", 10), ("completion_authority", True),
])
def test_estimate_selection_tampering_never_becomes_measured_evidence(estimated, field, value):
    estimated["receipts"][0]["prompt_selection"][field] = value
    _invalid(estimated)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign"), ("backend", "remote"), ("backend_version", "2"),
    ("encoding", "o200k_base"), ("encoding_spec_revision", "invented"), ("rank_sha256", "a" * 64),
    ("count_kind", "tokenizer"), ("compatibility_basis", "verified_remote_model"),
    ("reason", "verified_remote_model"), ("model_mapping_match", "exact"), ("model_mapping_prefix", "gpt-5-"),
    ("offline", False), ("remote_tokenizer_compatibility_verified", True), ("provider_usage", True),
    ("completion_authority", True), ("completion_authority", 0),
])
def test_automatic_source_tampering_never_becomes_usable_history(estimated, field, value):
    estimated["receipts"][0]["prompt_tokenizer_source"][field] = value
    _invalid(estimated)


@pytest.mark.parametrize("mutation", ["measured_field", "missing_estimate", "extra_count", "extra_source",
                                      "missing_source", "foreign_route", "multi_selector"])
def test_closed_contract_and_binding_mutations_are_refused(estimated, mutation):
    receipt = estimated["receipts"][0]
    selection = receipt["prompt_selection"]
    candidate = selection["candidates"]["native"]
    if mutation == "measured_field":
        candidate["input_tokens"] = candidate["estimated_input_tokens"]
    elif mutation == "missing_estimate":
        candidate.pop("estimated_input_tokens")
    elif mutation == "extra_count":
        candidate["billing_tokens"] = 1
    elif mutation == "extra_source":
        receipt["prompt_tokenizer_source"]["extra"] = True
    elif mutation == "missing_source":
        receipt["prompt_tokenizer_source"].pop("rank_sha256")
    elif mutation == "foreign_route":
        receipt["provider"] = "foreign"
    else:
        selection["candidate_selection_schema"] = "supervisor-semantic-candidate-selection@1"
        selection["comparison_candidates"] = {}
    _invalid(estimated)


@pytest.mark.parametrize("kind", ["tokenizer", "estimate"])
def test_single_comparison_choice_checks_strict_counts_and_separate_reasons(kind, tmp_path):
    counter = PromptTokenCounter(provider="codex_cli", model="authored", tokenizer_id="authored",
        tokenizer_revision="authored@1", special_token_policy="literal", count_tokens=lambda text: len(text),
        count_kind=kind, compatibility_basis="caller_declaration")
    model, _, _, selection = select_model_prompt(native_prompt="native " * 300, semantic_prompt="semantic",
        suffix=" required suffix", purpose="coding", workspace=tmp_path, provider=counter.provider,
        model=counter.model, token_counter=counter, max_input_tokens=100_000)
    receipt = {"semantic_encoding_candidate": {"authored": True},
               "model_prompt_sha256": _sha(model), "model_prompt_bytes": len(model.encode())}
    original = {"provider": counter.provider, "model": counter.model}
    validator.validate_prompt_selection(selection, receipt, original)
    assert selection["selection_reason"] == (
        "semantic_strictly_fewer_estimated_content_tokens" if kind == "estimate"
        else "semantic_strictly_fewer_content_tokens")
    selection["selection_reason"] = "native_not_larger"
    with pytest.raises(ValueError, match="choice"):
        validator.validate_prompt_selection(selection, receipt, original)


@pytest.mark.parametrize("mode", ["failed_tokenizer", "failed_estimate", "unavailable"])
def test_new_schema_preserves_unavailable_and_failed_native_fallback(mode, tmp_path):
    def failed(_):
        raise ValueError("authored count failure")

    counter = None if mode == "unavailable" else PromptTokenCounter(provider="codex_cli", model="authored",
        tokenizer_id="authored", tokenizer_revision="authored@1", special_token_policy="literal",
        count_tokens=failed, count_kind="estimate" if mode == "failed_estimate" else "tokenizer")
    model, _, _, selection = select_model_prompt(native_prompt="native", semantic_prompt="semantic",
        suffix="", purpose="coding", workspace=tmp_path, provider="codex_cli", model="authored", token_counter=counter)
    validator.validate_prompt_selection(selection, {"semantic_encoding_candidate": {"authored": True},
        "model_prompt_sha256": _sha(model), "model_prompt_bytes": len(model.encode())},
        {"provider": "codex_cli", "model": "authored"})
    assert selection["selected_representation"] == "native"
    assert selection["selected_content_tokens"] is selection["selected_estimated_content_tokens"] is None
