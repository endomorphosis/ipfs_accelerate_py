"""Supervisor forwarding and observations, using an authored provider boundary."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner


@pytest.fixture
def route(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
    from ipfs_accelerate_py.llm_allocation import intelligence_index
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    (workspace / "module.py").write_text("# retained comment\ndef square(value):\n    return value * value\n")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-native-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    monkeypatch.setattr(intelligence_index, "select_efficient_route", lambda **kwargs:
        SimpleNamespace(provider="codex_cli", model_name="pinned", reasoning_effort="high", catalog_revision="test"))
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *args, **kwargs: object())
    state = {"seen": [], "observe": True}
    def generate(prompt, **kwargs):
        state["seen"].append((prompt, kwargs))
        observation = {"exit_code": 0, "prompt_tokens": 15, "completion_tokens": 4}
        binding = kwargs.get("codex_repository_source_binding")
        if binding is not None:
            policy.validate_binding(binding, workspace=workspace, materialized=True)
            if state["observe"]:
                observation.update(codex_repository_source_configured=True,
                    codex_repository_source_config_sha256=binding["config_sha256"])
        set_last_cli_observation("codex_cli", observation)
        if state.get("callback") is not None:
            state["callback"](binding)
        return "completed authored provider result"
    monkeypatch.setattr(llm_router, "generate_text", generate)
    return workspace, state


def _run(**kwargs):
    return runner.run(prompt="Repair the declared file.", provider="codex_cli", model="pinned",
                      timeout=30, max_output_tokens=128, **kwargs)


@pytest.mark.parametrize("selected", policy.POLICIES[1:])
def test_configured_native_tools_do_not_claim_adoption(route, selected):
    from benchmarks.agent_supervisor.container_coding.terminal_context_audit import _model_projection, _receipt
    workspace, state = route
    output, receipt = _run(repository_source_policy=selected)
    assert output == "completed authored provider result"
    assert len(state["seen"]) == 1
    prompt, options = state["seen"][0]
    binding = receipt["repository_source_policy"]
    assert options["codex_repository_source_binding"] is binding
    assert options["side_effecting"] is True and options["allow_cross_provider_fallback"] is False
    expected, advisory = runner.render_model_prompt(prompt="Repair the declared file.", purpose="coding",
        workspace=workspace, repository_source_binding=binding)
    assert prompt == expected and "repository_read" in advisory
    assert binding["usage"]["status"] == "not_observed"
    assert binding["usage"]["tool_calls"] == binding["usage"]["compact_reads"] == 0
    assert binding["usage"]["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()
    assert receipt["usage"]["codex_repository_source_configured"] is True
    assert binding["helper_adoption_guaranteed"] is False
    retained = _receipt({**receipt, "phase": "coding"}, workspace.parent)
    reconstructed, reconstructed_advisory, checks = _model_projection(
        rendered="Repair the declared file.", receipt=retained, repository=None)
    assert reconstructed == prompt and reconstructed_advisory == advisory and all(checks.values())
    assert checks["repository_source_helper_source"] is True


def test_runner_preserves_legacy_route_without_new_guidance(route):
    _workspace, state = route
    _, receipt = _run()
    assert "repository_source_policy" not in receipt
    assert "codex_repository_source_binding" not in state["seen"][0][1]
    assert "repository_read" not in state["seen"][0][0]


def test_owner_tokenizer_is_hash_bound_into_actual_repository_child(route, tmp_path):
    from test.api.test_local_prompt_tokenizer import make_binding
    from test.api.test_repository_source_mcp import real_client
    workspace, state = route
    artifact, digest, _ = make_binding(tmp_path, provider="codex_cli", model="pinned")
    def child(binding):
        assert binding["config"]["schema"] == "repository-source-mcp-config@2"
        assert binding["config"]["tokenizer"] == {"artifact": str(artifact), "expected_sha256": digest,
                                                 "provider": "codex_cli", "model": "pinned"}
        with real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
            value, failed = client.call("repository_read", generation=1, path="module.py")
            assert not failed and value["representation"] == "compact"
    state["callback"] = child
    _, receipt = _run(repository_source_policy=policy.POLICIES[2],
        prompt_tokenizer_artifact=artifact, prompt_tokenizer_sha256=digest)
    usage = receipt["repository_source_policy"]["usage"]
    assert usage["successful_reads"] == usage["token_selection"]["read_count"] == 1
    assert usage["token_selection"]["measured_comparisons"] == 1 and usage["store_cleanup_complete"]


def test_native_configuration_must_be_observed(route, capsys):
    _workspace, state = route
    state["observe"] = False
    with pytest.raises(RuntimeError, match="configuration observation"):
        _run(repository_source_policy=policy.POLICIES[2])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["failure_phase"] == "provider_result_validation"
    assert receipt["repository_source_policy"]["usage"]["store_cleanup_complete"]


def test_provider_failure_preserves_failure_and_cleans_invocation(route, capsys):
    _workspace, state = route
    def fail(binding):
        assert Path(binding["config_path"]).exists()
        raise TimeoutError("authored native timeout")
    state["callback"] = fail
    with pytest.raises(TimeoutError, match="authored native timeout"):
        _run(repository_source_policy=policy.POLICIES[2])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["failure_phase"] == "provider_invocation"
    assert receipt["repository_source_policy"]["usage"]["store_cleanup_complete"]


def test_real_protocol_handler_observations_are_exported_without_contents(route):
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_mcp import RepositoryMCPServer
    _workspace, state = route
    class AuthoredSession:
        def __init__(self, *args, **kwargs): pass
        def read(self, **kwargs):
            return {"generation": kwargs["generation"], "path": kwargs["path"],
                    "original_sha256": "a" * 64, "encoding": "utf-8", "proof_authority": False,
                    "representation": "compact", "content": "authored-private-source-value"}
        def close(self): pass
    def use_handler(binding):
        server = RepositoryMCPServer(binding["config"], session_factory=AuthoredSession)
        try:
            initialized = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "authored", "version": "1"}}})
            assert "result" in initialized
            server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
            reply = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "repository_read", "arguments": {"generation": 1, "path": "module.py"}}})
            assert reply["result"]["isError"] is False
        finally:
            server.close()
    state["callback"] = use_handler
    _, receipt = _run(repository_source_policy=policy.POLICIES[2])
    usage = receipt["repository_source_policy"]["usage"]
    assert usage["status"] == "observed"
    assert usage["initializations"] == usage["tool_calls"] == usage["compact_reads"] == 1
    assert usage["canonical_edits"] == usage["uncertain_edits"] == 0
    assert usage["returned_bytes"] > 0
    assert "authored-private-source-value" not in json.dumps(receipt)


@pytest.mark.parametrize("changes", [{"purpose": "planning"}, {"source_view_advisory_policy": "source-view-raw@1"},
                                     {"census_only": True}])
def test_incompatible_repository_route_never_calls_provider(route, changes):
    with pytest.raises(ValueError):
        _run(repository_source_policy=policy.POLICIES[2], **changes)
    assert not route[1]["seen"]


def test_store_creation_failure_refuses_dispatch(route, monkeypatch, capsys):
    def fail(binding):
        raise OSError("authored store failure")
    monkeypatch.setattr(policy, "create_runtime_store", fail)
    with pytest.raises(OSError, match="authored store failure"):
        _run(repository_source_policy=policy.POLICIES[2])
    assert not route[1]["seen"]
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["failure_phase"] == "repository_source_store"
    assert receipt["repository_source_policy"]["usage"]["status"] == "unavailable"
