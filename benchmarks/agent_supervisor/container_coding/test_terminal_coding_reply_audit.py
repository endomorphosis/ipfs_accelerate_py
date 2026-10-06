"""Historical audit includes the final completion contract on either transport."""
from copy import deepcopy
import hashlib

import pytest

from test.api.test_semantic_router_translation import native  # noqa: F401
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
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
