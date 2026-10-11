"""Historical @2 audits reconstruct delivery without inventing a trusted counter."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
import subprocess
import sys

import pytest

from test.api.test_semantic_metadata_dispatch import dispatch_case, invoke  # noqa: F401
from test.api.test_semantic_router_translation import native  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_metadata_view import (
    SemanticMetadataTokenCounter,
)


def counter(*, compact_tokens=90):
    def count(text):
        return compact_tokens if '"schema":"supervisor-semantic-metadata-view@1"' in text else 100
    return SemanticMetadataTokenCounter(count=count, provider="codex_cli", model="authored-model",
        tokenizer="authored-fixture-counter", tokenizer_revision="authored-v1", qualification="estimate")


def receipt_for(case, token_counter):
    _, receipt = invoke(case, semantic_metadata_view="common-bindings@1",
                        semantic_metadata_token_counter=token_counter)
    receipt["phase"] = "coding"
    return receipt


@pytest.mark.parametrize("compact_tokens,expected_mode", [(90, "common-bindings@1"), (200, "legacy")])
def test_trusted_counter_replays_both_selection_outcomes(dispatch_case, compact_tokens, expected_mode):
    case, trusted = dispatch_case, counter(compact_tokens=compact_tokens)
    receipt = receipt_for(case, trusted)
    assert receipt["semantic_metadata_view"]["selected_mode"] == expected_mode
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"], token_counter=trusted)
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_token_counter_verified"] is True
    assert all(checks.values()), checks


@pytest.mark.parametrize("compact_tokens", [90, 200])
def test_absent_counter_reconstructs_recorded_input_without_proxy_rescue(dispatch_case, compact_tokens):
    case = dispatch_case
    receipt = receipt_for(case, counter(compact_tokens=compact_tokens))
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"])
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_token_counter_verified"] is False
    assert checks["semantic_metadata_selection_receipt"] is False
    assert checks["semantic_metadata_recorded_prompt_identities"] is True
    assert all(value for key, value in checks.items() if key not in {
        "semantic_metadata_token_counter_verified", "semantic_metadata_selection_receipt"})
    metadata = receipt["semantic_metadata_view"]
    assert metadata["candidate_complete_bytes"] < metadata["native_complete_bytes"]
    assert metadata["candidate_complete_proxy_tokens"] < metadata["native_complete_proxy_tokens"]
    if compact_tokens == 200:
        assert metadata["selected_mode"] == "legacy"  # Byte proxy would select the other arm.


@pytest.mark.parametrize("mutation", [
    {"token_counter_identity_sha256": "0" * 64},
    {"token_counter_status": "unknown"},
    {"token_counter_status": []},
    {"schema": {}},
    {"selected_mode": []},
    {"native_complete_tokens": True},
    {"selected_complete_tokens": 99},
    {"selection_basis": "token_proxy"},
    {"provider_usage_reported": True},
    {"actual_provider_usage_verified": 0},
    {"cli_scaffolding_measured": True},
    {"complete_run_token_savings_claimed": True},
    {"proof_authority": 0},
    {"fallback_reason": "proxy_rescue"},
    {"unsupported_counter_field": "ignored"},
])
def test_token_receipt_refuses_status_usage_and_selected_arithmetic_tampering(dispatch_case, mutation):
    case = dispatch_case
    receipt = receipt_for(case, counter())
    receipt["semantic_metadata_view"].update(mutation)
    with pytest.raises(ValueError):
        audit._receipt(receipt, case["workspace"].parent)


@pytest.mark.parametrize("field,value", [
    ("schema", "foreign-counter@1"), ("measurement_scope", "semantic-rows-only@1"),
    ("qualification", "unqualified"), ("identity_caller_supplied", 1),
    ("qualification", []),
    ("provider_usage_verified", True), ("provider", "foreign-provider"),
    ("model", "foreign-model"), ("tokenizer_revision", ""),
])
def test_counter_identity_refuses_rehashed_wrong_scope_or_caller(dispatch_case, field, value):
    case = dispatch_case
    receipt = receipt_for(case, counter())
    metadata = receipt["semantic_metadata_view"]
    metadata["token_counter_identity"][field] = value
    raw = json.dumps(metadata["token_counter_identity"], sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False).encode()
    metadata["token_counter_identity_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):
        audit._receipt(receipt, case["workspace"].parent)


@pytest.mark.parametrize("change", ["identity", "counts", "caller"])
def test_trusted_replay_refuses_different_counter_or_coherent_measurement_tampering(dispatch_case, change):
    case, trusted = dispatch_case, counter()
    receipt = receipt_for(case, trusted)
    if change == "identity":
        trusted = replace(trusted, tokenizer_revision="authored-v2")
    elif change == "caller":
        trusted = replace(trusted, provider="foreign-provider")
    else:
        metadata = receipt["semantic_metadata_view"]
        for key in ("native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens"):
            metadata[key] += 1
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"], token_counter=trusted)
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_selection_receipt"] is False
    assert checks["semantic_metadata_token_counter_verified"] is False


@pytest.mark.parametrize("profile", [{"provider": "foreign-provider"}, {"model": "foreign-model"}])
def test_audit_never_calls_counter_with_foreign_provider_or_model(dispatch_case, profile):
    case = dispatch_case
    receipt = receipt_for(case, counter())
    calls = []
    def forbidden(text):
        calls.append(text)
        raise AssertionError("foreign counter must not receive retained prompt text")
    foreign = replace(counter(), count=forbidden, **profile)
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"], token_counter=foreign)
    assert calls == []
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_selection_receipt"] is False
    assert checks["semantic_metadata_token_counter_verified"] is False


@pytest.mark.parametrize("status", ["failed", "invalid"])
def test_unusable_counter_retains_native_input_and_no_token_verification(dispatch_case, status):
    case = dispatch_case
    def failed(_text):
        raise RuntimeError("authored unavailable counter")
    trusted = replace(counter(), count=failed) if status == "failed" else replace(counter(), qualification="foreign")
    receipt = receipt_for(case, trusted)
    assert receipt["semantic_metadata_view"]["token_counter_status"] == status
    assert receipt["semantic_metadata_view"]["selected_mode"] == "legacy"
    projected = audit._receipt(receipt, case["workspace"].parent)
    model, _, checks = audit._model_projection(rendered=case["prompt"], receipt=projected,
        repository=case["repository"], token_counter=trusted)
    assert model == case["calls"][-1][0]
    assert checks["semantic_metadata_selection_receipt"] is True
    assert checks["semantic_metadata_token_counter_verified"] is False
    altered = deepcopy(receipt)
    altered["semantic_metadata_view"]["native_complete_tokens"] = 100
    with pytest.raises(ValueError):
        audit._receipt(altered, case["workspace"].parent)


def test_separate_public_audit_process_cannot_replay_counter_from_json(dispatch_case):
    case = dispatch_case
    receipt = receipt_for(case, counter(compact_tokens=200))
    body = {"receipt": receipt, "rendered": case["prompt"],
            "repository": str(case["repository"]), "workspace_root": str(case["workspace"].parent)}
    code = """import hashlib, json, sys
from pathlib import Path
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
body = json.load(sys.stdin)
receipt = audit._receipt(body["receipt"], Path(body["workspace_root"]))
model, _, checks = audit._model_projection(rendered=body["rendered"], receipt=receipt,
    repository=Path(body["repository"]))
print(json.dumps({"sha256": hashlib.sha256(model.encode()).hexdigest(), "checks": checks}))
"""
    child = subprocess.run([sys.executable, "-B", "-P", "-c", code], input=json.dumps(body),
        text=True, capture_output=True, check=True, timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    observed = json.loads(child.stdout)
    assert observed["sha256"] == receipt["model_prompt_sha256"]
    assert observed["checks"]["semantic_metadata_recorded_prompt_identities"] is True
    assert observed["checks"]["semantic_metadata_selection_receipt"] is False
    assert observed["checks"]["semantic_metadata_token_counter_verified"] is False
