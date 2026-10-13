"""Actual local tokenizer and stdio delivery; no remote model equivalence claim."""
import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_token_budget as budget
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import OwnerPromptTokenizer
from test.api.test_local_prompt_tokenizer import make_binding
from test.api.test_repository_source_mcp import configuration, real_client, initialization


def token_config(tmp_path):
    config, path, _ = configuration(tmp_path)
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    artifact, digest, _ = make_binding(tokenizer_root, provider="codex_cli", model="pinned")
    config.update(schema="repository-source-mcp-config@2", tokenizer={
        "artifact": str(artifact), "expected_sha256": digest, "provider": "codex_cli", "model": "pinned"})
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    return config, path, hashlib.sha256(raw).hexdigest()


def selections(config):
    rows = [json.loads(row) for row in Path(config["audit_store"]).with_name("token-selections.jsonl").read_text().splitlines()]
    return rows


def test_actual_child_selects_complete_smaller_reply_and_preserves_explicit_original(tmp_path):
    config, path, digest = token_config(tmp_path)
    original = (Path(config["workspace"]) / "calc.py").read_bytes()
    with real_client(path, digest) as client:
        compact, failed = client.call("repository_read", generation=1, path="calc.py")
        assert not failed and compact["representation"] == "compact"
        assert "Original public comment" not in compact["content"]
        full, failed = client.call("repository_read", generation=1, path="calc.py", representation="original")
        assert not failed and full["representation"] == "original" and full["content"].encode() == original
        assert full["source_sha256"] == compact["source_sha256"] == hashlib.sha256(original).hexdigest()
        # Selection diagnostics do not expand the model-visible response.
        assert set(full) == {"generation", "path", "source_sha256", "representation", "encoding", "content", "proof_authority"}
    summary = budget.summarize_selection_audit(path.parent / "token-selections.jsonl")
    budget.validate_selection_summary(summary)
    assert summary["read_count"] == 2 and summary["compact_selected"] == 1
    assert summary["explicit_original"] == 1 and summary["measured_comparisons"] == 1
    assert summary["provider_usage_measured"] is False
    encoded = json.dumps(selections(config))
    assert "Original public comment" not in encoded and "def add" not in encoded


def test_changed_local_tokenizer_falls_back_to_exact_original_without_token_claim(tmp_path):
    config, path, digest = token_config(tmp_path)
    original = (Path(config["workspace"]) / "calc.py").read_text()
    with real_client(path, digest) as client:
        artifact = Path(config["tokenizer"]["artifact"])
        artifact.write_text(artifact.read_text() + " ")
        read, failed = client.call("repository_read", generation=1, path="calc.py")
        assert not failed and read["representation"] == "original" and read["content"] == original
    summary = budget.summarize_selection_audit(path.parent / "token-selections.jsonl")
    budget.validate_selection_summary(summary)
    assert summary["read_count"] == summary["unmeasured_fallbacks"] == 1
    assert summary["measured_comparisons"] == summary["compact_selected"] == 0
    assert summary["selected_reply_tokens"] == 0


def test_source_change_during_count_refuses_delivery_and_does_not_record_a_success(tmp_path, monkeypatch):
    config, _, _ = token_config(tmp_path)
    source = Path(config["workspace"]) / "calc.py"
    original_count = OwnerPromptTokenizer.count
    changed = False
    def count(self, prompt):
        nonlocal changed
        value = original_count(self, prompt)
        if not changed:
            source.write_text(source.read_text() + "# concurrent external change\n")
            changed = True
        return value
    monkeypatch.setattr(OwnerPromptTokenizer, "count", count)
    server = mcp.RepositoryMCPServer(config)
    try:
        assert "result" in server.handle(initialization())
        server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
        response = server.handle(dict(jsonrpc="2.0", id=2, method="tools/call",
            params={"name": "repository_read", "arguments": {"generation": 1, "path": "calc.py"}}))
        assert response["result"]["isError"] is True
        assert json.loads(response["result"]["content"][0]["text"])["error"] == "stale_source"
    finally:
        server.close()
    summary = budget.summarize_selection_audit(Path(config["audit_store"]).with_name("token-selections.jsonl"))
    assert summary["read_count"] == 0
    assert mcp.summarize_repository_audit(config["audit_store"])["successful_reads"] == 0


def test_tokenizer_config_digest_and_private_cleanup_preserve_closed_summary(tmp_path):
    config, _, _ = token_config(tmp_path)
    binding = policy.create_binding(workspace=Path(config["workspace"]), requested_policy=policy.POLICIES[2],
                                    tokenizer=config["tokenizer"])
    identity = policy.create_runtime_store(binding)
    try:
        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            read, failed = client.call("repository_read", generation=1, path="calc.py")
            assert not failed and read["representation"] == "compact"
    finally:
        usage = policy.finish_runtime_store(binding, expected_identity=identity)
    assert usage["store_cleanup_complete"] is True and not Path(binding["invocation_dir"]).exists()
    budget.validate_selection_summary(usage["token_selection"])
    assert usage["token_selection"]["read_count"] == usage["successful_reads"] == 1
    binding["usage"] = usage
    policy.validate_binding(binding)
    binding["config"]["tokenizer"]["model"] = "changed"
    with pytest.raises(ValueError, match="binding content"):
        policy.validate_binding(binding)


def test_reply_rejected_by_transport_bound_does_not_enter_selection_audit(tmp_path):
    config, _, _ = token_config(tmp_path)
    class EscapedSession:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def read(self, **kwargs):
            return dict(generation=1, path="data.txt", original_sha256="a" * 64,
                representation="original", encoding="utf-8", content="\x01" * 200_000,
                proof_authority=False)
    server = mcp.RepositoryMCPServer(config, session_factory=EscapedSession)
    try:
        assert "result" in server.handle(initialization())
        server.handle(dict(jsonrpc="2.0", method="notifications/initialized"))
        response = server.handle(dict(jsonrpc="2.0", id=2, method="tools/call",
            params={"name": "repository_read", "arguments": {
                "generation": 1, "path": "data.txt", "representation": "original"}}))
        assert response["result"]["isError"] is True
    finally:
        server.close()
    summary = budget.summarize_selection_audit(Path(config["audit_store"]).with_name("token-selections.jsonl"))
    assert summary["read_count"] == 0
    assert mcp.summarize_repository_audit(config["audit_store"])["successful_reads"] == 0
