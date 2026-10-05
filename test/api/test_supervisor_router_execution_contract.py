"""Exercise the real shared router with a local, non-model CLI process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def native_router(tmp_path, monkeypatch):
    from ipfs_accelerate_py import llm_router, router_deps
    from ipfs_accelerate_py.llm_allocation import duckdb_store, intelligence_index

    workspace = tmp_path / "work"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    program = "#!" + sys.executable + "\n" + '''
import json, pathlib, sys
args = sys.argv[1:]
path = pathlib.Path("observed-calls.jsonl")
with path.open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if "--output-last-message" in args:
    sys.stdin.read()
    pathlib.Path(args[args.index("--output-last-message") + 1]).write_text("done")
    print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 15, "output_tokens": 4}}))
else:
    print(json.dumps({"type": "message", "text": "done"}))
'''
    for name in ("codex", "grok"):
        path = binary_dir / name
        path.write_text(program)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.chdir(workspace)
    for name in (
        "ipfs_accelerate_py_CODEX_CLI_MODEL", "IPFS_ACCELERATE_PY_CODEX_CLI_MODEL",
        "IPFS_DATASETS_PY_CODEX_CLI_MODEL", "ipfs_accelerate_py_CODEX_MODEL",
        "IPFS_ACCELERATE_PY_CODEX_MODEL", "IPFS_DATASETS_PY_CODEX_MODEL",
        "ipfs_accelerate_py_GROK_CLI_MODEL", "IPFS_ACCELERATE_PY_GROK_CLI_MODEL",
        "IPFS_DATASETS_PY_GROK_CLI_MODEL", "GROK_CLI_MODEL", "GROK_MODEL",
        "IPFS_ACCELERATE_AGENT_GROK_MODEL", "ipfs_accelerate_py_XAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("IPFS_ACCELERATE_LLM_ALLOCATION_DB", str(tmp_path / "allocation.duckdb"))
    monkeypatch.setattr(duckdb_store, "_DEFAULT_STORE", None)
    monkeypatch.setenv("IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE", "1")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setattr(intelligence_index, "discover_available_providers", lambda: ["codex_cli"])
    provider = llm_router._get_codex_cli_provider()
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda *a, **kw: provider)

    class TextCache:
        def __init__(self):
            self.reads = self.writes = 0

        def get(self, _key):
            self.reads += 1
            return "old response, with no checkout effects"

        def set(self, _key, _value):
            self.writes += 1

    cache = TextCache()
    deps = router_deps.RouterDeps(remote_cache=cache)
    monkeypatch.setattr(router_deps, "RouterDeps", lambda: deps)
    return workspace, cache, provider


@pytest.mark.parametrize("purpose", ["coding", "planning"])
def test_native_invocation_uses_router_without_text_cache_replay(native_router, purpose):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner

    workspace, cache, _provider = native_router
    for _ in range(2):
        output, receipt = runner.run(prompt="inspect the allocated repository", provider="codex_cli",
            model="gpt-6.1-sol", timeout=30, max_output_tokens=128, purpose=purpose)
        assert output == "done" and receipt["usage"]["prompt_tokens"] == 15
        assert llm_router.get_last_generation_trace() == {
            "effective_provider_name": "codex_cli", "effective_model_name": "gpt-6.1-sol"}
    calls = [json.loads(line) for line in (workspace / "observed-calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert all(args[args.index("-m") + 1] == "gpt-6.1-sol" for args in calls)
    assert (cache.reads, cache.writes) == (0, 0)


@pytest.mark.parametrize("model_override", [None, "operator-selected-model"])
def test_codex_default_agrees_with_router_descriptor_cache_key_and_native_argv(native_router, monkeypatch, model_override):
    from ipfs_accelerate_py import llm_router
    workspace, _cache, provider = native_router
    if model_override:
        monkeypatch.setenv("IPFS_DATASETS_PY_CODEX_CLI_MODEL", model_override)
    expected = model_override or "gpt-6.1-sol"
    assert provider.generate("inspect", timeout=30) == "done"
    args = json.loads((workspace / "observed-calls.jsonl").read_text().splitlines()[0])
    assert args[args.index("-m") + 1] == expected
    assert llm_router._effective_model_key(provider_key="codex_cli", model_name=None, kwargs={}) == expected
    assert dict(llm_router.get_provider_descriptor("codex_cli").labels)["model_hint"] == expected


def test_grok_router_pins_current_default_in_native_argv(native_router, monkeypatch):
    from ipfs_accelerate_py import llm_router
    workspace, _cache, _provider = native_router
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: str(workspace.parent / "bin/grok"))
    assert llm_router._get_grok_cli_provider().generate("inspect", timeout=30) == "done"
    args = json.loads((workspace / "observed-calls.jsonl").read_text().splitlines()[0])
    assert args[args.index("--model") + 1] == "grok-4.7"
