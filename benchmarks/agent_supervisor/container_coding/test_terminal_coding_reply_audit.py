"""Historical audit includes the final completion contract on either transport."""
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from benchmarks.agent_supervisor.container_coding.test_terminal_context_audit import compiled  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import render_context_capsule
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextCapsule
from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_router_translation import encode_semantic_router_prompt


def receipt_for(native, tmp_path, transport):
    repository, _, prompt, source = native
    encoded = encode_semantic_router_prompt(prompt=prompt, repository=repository, transport_schema=transport)
    workspace = tmp_path / "allocated/removed-worktree"
    before, advisory = render_model_prompt(prompt=encoded.provider_prompt, purpose="coding",
        workspace=workspace, semantic_transport=True)
    model, contract = apply_coding_reply_contract(model_prompt=before, mode="ordinary-completion@1",
        purpose="coding", provider="codex_cli")
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    row = {"schema": "router-implementation-invocation@1", "phase": "coding", "purpose": "coding",
        "provider": "codex_cli", "invocation_id": "fixture", "workspace": str(workspace),
        "prompt_sha256": sha(prompt), "prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": sha(prompt), "native_prompt_bytes": len(prompt.encode()),
        "router_prompt_sha256": sha(encoded.provider_prompt), "router_prompt_bytes": len(encoded.provider_prompt.encode()),
        "model_prompt_sha256": sha(model), "model_prompt_bytes": len(model.encode()),
        "workspace_advisory_sha256": sha(advisory), "workspace_advisory_bytes": len(advisory.encode()),
        "semantic_translation": dict(encoded.receipt), "coding_reply_contract": contract}
    return row, model, advisory, repository, prompt, source, workspace


@pytest.mark.parametrize("transport", ["supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2"])
def test_complete_contract_input_replays_after_source_and_workspace_change(native, tmp_path, transport):
    row, model, advisory, repository, prompt, source, workspace = receipt_for(native, tmp_path, transport)
    row["coding_reply_contract"]["response_validated"] = True
    parsed = audit._receipt(row, workspace.parent)
    (repository / "mod.py").write_text(source + "# later accepted revision\n")
    actual, header, checks = audit._model_projection(rendered=prompt, receipt=parsed, repository=repository)
    assert actual == model and header == advisory and all(checks.values())
    tampered = deepcopy(parsed)
    tampered["coding_reply_contract"]["model_prompt_before_sha256"] = "0" * 64
    assert not all(audit._model_projection(rendered=prompt, receipt=tampered, repository=repository)[2].values())
    missing = deepcopy(row)
    del missing["coding_reply_contract"]
    with pytest.raises(ValueError, match="inconsistent router input"):
        audit._receipt(missing, workspace.parent)


@pytest.mark.parametrize("field,value", [
    ("owner_selected", 1), ("execution_authority", 0), ("instruction_bytes", 506.0),
    ("native_output_schema_bytes", 215.0), ("response_validated", 1),
    ("mode", "legacy"), ("instruction_sha256", "0" * 64),
    ("native_output_schema_sha256", "0" * 64),
])
def test_contract_receipt_rejects_type_and_static_binding_tampering(native, tmp_path, field, value):
    row, _, _, _, _, _, workspace = receipt_for(native, tmp_path, "supervisor-semantic-router-input@2")
    row["coding_reply_contract"][field] = value
    with pytest.raises(ValueError, match="coding reply contract"):
        audit._receipt(row, workspace.parent)


def _completion_case(compiled):
    """Use persisted native state and the actual suffix generator, without a model."""
    row = compiled["receipts"][0]
    capsule = ContextCapsule.from_dict(json.loads(compiled["capsule"].read_text()))
    prompt = render_context_capsule(capsule)
    from pathlib import Path
    before, _ = render_model_prompt(prompt=prompt, purpose="coding", workspace=Path(row["workspace"]))
    model, contract = apply_coding_reply_contract(model_prompt=before, mode="ordinary-completion@1",
        purpose="coding", provider="codex_cli")
    row.update(provider="codex_cli", coding_reply_contract=contract,
        model_prompt_sha256=hashlib.sha256(model.encode()).hexdigest(), model_prompt_bytes=len(model.encode()))
    return row


def _collect(compiled):
    return audit.collect_terminal_context_audit(state=compiled["state"], receipts=compiled["receipts"],
        workspace_root=compiled["workspace_root"], timeout_seconds=10)


def _capture_child_requests(monkeypatch):
    """Observe the real IPC input while still running the actual audit child."""
    requests = []
    original_run = audit.subprocess.run
    def recorded_run(*args, **kwargs):
        requests.append(json.loads(kwargs["input"]))
        return original_run(*args, **kwargs)
    monkeypatch.setattr(audit.subprocess, "run", recorded_run)
    return requests


def test_actual_collector_preserves_completion_provider_and_reconstructs_final_input(compiled, monkeypatch):
    row = _completion_case(compiled)
    requests = _capture_child_requests(monkeypatch)
    result = _collect(compiled)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert result["usable_coding_receipts"] == 1 and result["read_errors"] == []
    assert result["all_observed_coding_inputs_verified"] is True
    assert result["matches"][0]["model_prompt_sha256"] == row["model_prompt_sha256"]
    assert result["matches"][0]["model_prompt_bytes"] == row["model_prompt_bytes"]
    checks = result["matches"][0]["model_input_checks"]
    assert checks["coding_reply_instruction_sha256"] is True and all(checks.values())
    assert requests[0][0]["provider"] == "codex_cli"
    assert set(requests[0][0]) == set(row) & {*audit.RECEIPT_FIELDS, *audit.TRANSLATION_FIELDS, "provider"}
    assert result["provider_calls"] == 0 and result["raw_prompts_exported"] is False
    assert result["completion_authority"] is result["task_correctness_established"] is False


@pytest.mark.parametrize("provider", [None, "grok_cli"])
def test_actual_completion_collector_rejects_missing_or_foreign_provider(compiled, monkeypatch, provider):
    row = _completion_case(compiled)
    if provider is None:
        row.pop("provider")
    else:
        row["provider"] = provider
    requests = _capture_child_requests(monkeypatch)
    result = _collect(compiled)
    assert result["reason"] == "no_usable_coding_receipt", result
    assert result["coding_receipts_seen"] == 1 and result["usable_coding_receipts"] == 0
    assert result["read_errors"] == [{"kind": "receipt", "error_type": "InvalidReceipt"}]
    assert result["matches"] == [] and result["all_observed_coding_inputs_verified"] is None
    if provider is None:
        assert "provider" not in requests[0][0]
    else:
        assert requests[0][0]["provider"] == provider
    assert result["provider_calls"] == 0


def test_actual_completion_collector_checks_input_before_suffix(compiled):
    row = _completion_case(compiled)
    row["coding_reply_contract"]["model_prompt_before_sha256"] = "0" * 64
    result = _collect(compiled)
    assert result["usable_coding_receipts"] == 1 and result["read_errors"] == []
    assert result["status"] == "verified_native_input_only", result
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["matches"][0]["model_input_checks"]["coding_reply_model_prompt_before_sha256"] is False


@pytest.mark.parametrize("explicit_null_contract", [False, True])
def test_actual_legacy_collector_keeps_original_ipc_fields(compiled, monkeypatch, explicit_null_contract):
    row = compiled["receipts"][0]
    row.update(provider="codex_cli", usage={"authored_test_only": True}, unprojected="extra receipt metadata")
    if explicit_null_contract:
        row["coding_reply_contract"] = None
    expected = {key: row[key] for key in (*audit.RECEIPT_FIELDS, *audit.TRANSLATION_FIELDS) if key in row}
    requests = _capture_child_requests(monkeypatch)
    result = _collect(compiled)
    assert result["status"] == "verified_all_observed_model_inputs", result
    assert requests == [[expected]]
    assert "provider" not in requests[0][0] and "usage" not in requests[0][0]
    assert result["provider_calls"] == 0 and result["raw_prompts_exported"] is False
