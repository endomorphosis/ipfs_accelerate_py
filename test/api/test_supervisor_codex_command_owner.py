"""Shared router command ownership without replacing native process custody."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ALIASES = (
    "ipfs_accelerate_py_CODEX_CLI_MODEL", "IPFS_ACCELERATE_PY_CODEX_CLI_MODEL",
    "IPFS_DATASETS_PY_CODEX_CLI_MODEL", "ipfs_accelerate_py_CODEX_MODEL",
    "IPFS_ACCELERATE_PY_CODEX_MODEL", "IPFS_DATASETS_PY_CODEX_MODEL",
)


@pytest.fixture
def native_codex(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
    workspace = tmp_path / "work space"
    workspace.mkdir()
    binary = tmp_path / "codex"
    binary.write_text("#!" + sys.executable + "\n" + '''
import json, pathlib, sys
args = sys.argv[1:]
prompt = sys.stdin.read()
if "-C" in args:
    import os
    os.chdir(args[args.index("-C") + 1])
with pathlib.Path("calls.jsonl").open("a") as stream:
    stream.write(json.dumps({"argv":args,"prompt":prompt,"cwd":str(pathlib.Path.cwd())}) + "\\n")
if "--output-last-message" in args:
    pathlib.Path(args[args.index("--output-last-message") + 1]).write_text("done")
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":13,"output_tokens":2}}))
''')
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.chdir(workspace)
    for key in (*ALIASES, daemon._CODEX_MODEL_ENV, daemon._CODEX_CONTEXT_WINDOW_ENV,
                daemon._CODEX_REASONING_EFFORT_ENV, daemon._CODEX_MAX_THREADS_ENV,
                daemon._CODEX_MAX_DEPTH_ENV, "ipfs_accelerate_py_CODEX_SANDBOX"):
        monkeypatch.delenv(key, raising=False)
    return workspace, binary


@pytest.mark.parametrize("alias", ALIASES)
def test_ordinary_daemon_honors_shared_model_alias_in_native_argv(native_codex, monkeypatch, alias):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
    workspace, binary = native_codex
    monkeypatch.setenv(alias, "operator-selected-model")
    command = daemon._codex_implementation_command(codex=str(binary), workspace_path=workspace)
    subprocess.run(command, input="literal $(unexecuted)", text=True, check=True, capture_output=True)
    receipt = json.loads((workspace / "calls.jsonl").read_text())
    assert receipt["argv"][receipt["argv"].index("-m") + 1] == "operator-selected-model"
    assert receipt["cwd"] == str(workspace) and receipt["prompt"] == "literal $(unexecuted)"
    assert llm_router._effective_model_key(provider_key="codex_cli", model_name=None, kwargs={}) == "operator-selected-model"


def test_provider_and_daemon_call_same_router_command_owner(native_codex, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
    workspace, binary = native_codex
    build = llm_router.build_codex_cli_command
    calls = []
    def observed(**kwargs):
        calls.append(kwargs)
        return build(**kwargs)
    monkeypatch.setattr(llm_router, "build_codex_cli_command", observed)
    command = daemon._codex_implementation_command(codex=str(binary), workspace_path=workspace)
    subprocess.run(command, input="daemon", text=True, check=True, capture_output=True)
    assert llm_router._get_codex_cli_provider().generate("provider", reasoning_effort="high") == "done"
    assert len(calls) == 2
    receipts = [json.loads(line) for line in (workspace / "calls.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in receipts] == ["daemon", "provider"]
    for row in receipts:
        assert row["argv"][row["argv"].index("-m") + 1] == "gpt-6.1-sol"
    assert "--dangerously-bypass-approvals-and-sandbox" in receipts[0]["argv"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in receipts[1]["argv"]


def test_explicit_daemon_policy_overrides_shared_model_default(native_codex, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
    workspace, binary = native_codex
    monkeypatch.setenv(ALIASES[0], "shared-model")
    monkeypatch.setenv(daemon._CODEX_MODEL_ENV, "daemon-model")
    for override, expected in ((None, "daemon-model"), ("signed-explicit-model", "signed-explicit-model")):
        command = daemon._codex_implementation_command(codex=str(binary), workspace_path=workspace,
            model_override=override, reasoning_effort_override="medium", codex_context_window=4096)
        assert command[command.index("-m") + 1] == expected
        assert 'model_reasoning_effort="medium"' in command
        assert "model_context_window=4096" in command
        assert command[-1] == "-"


def test_native_daemon_argv_preserves_explicit_policy_order(native_codex, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
    workspace, binary = native_codex
    monkeypatch.setenv(daemon._CODEX_MAX_THREADS_ENV, "7")
    monkeypatch.setenv(daemon._CODEX_MAX_DEPTH_ENV, "0")
    command = daemon._codex_implementation_command(codex=str(binary), workspace_path=workspace,
        model_override="explicit-model", reasoning_effort_override="high", codex_context_window=4096)
    assert command == [str(binary), "exec", "--dangerously-bypass-approvals-and-sandbox",
        "-C", str(workspace), "-m", "explicit-model", "-c", "model_context_window=4096",
        "-c", 'model_reasoning_effort="high"', "-c", "agents.max_threads=7",
        "-c", "agents.max_depth=0", "-"]


def test_shared_alias_precedence_and_explicit_empty_daemon_overrides(native_codex, monkeypatch):
    from ipfs_accelerate_py import llm_router
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as daemon
    workspace, binary = native_codex
    for i, alias in enumerate(ALIASES):
        monkeypatch.setenv(alias, f"choice-{i}")
    command = daemon._codex_implementation_command(codex=str(binary), workspace_path=workspace)
    assert command[command.index("-m") + 1] == "choice-0"
    assert llm_router._get_codex_cli_provider().generate("check alias order") == "done"
    args = json.loads((workspace / "calls.jsonl").read_text())["argv"]
    assert args[args.index("-m") + 1] == "choice-0"
    command = daemon._codex_implementation_command(codex=str(binary), workspace_path=workspace,
        model_override="", reasoning_effort_override="")
    assert "-m" not in command and not any(arg.startswith("model_reasoning_effort=") for arg in command)


@pytest.mark.parametrize("kwargs", [
    {"context_window": "123\nmalicious.setting=true"}, {"max_threads": True},
    {"max_threads": -1}, {"max_depth": "unknown"},
    {"reasoning_effort": "high; injected"},
    {"bypass_approvals_and_sandbox": "false"},
    {"sandbox": "workspace-write", "bypass_approvals_and_sandbox": True},
])
def test_shared_builder_rejects_invalid_configuration_before_launch(kwargs):
    from ipfs_accelerate_py import llm_router
    with pytest.raises(ValueError):
        llm_router.build_codex_cli_command(**kwargs)


def test_provider_cleans_temporary_message_when_command_policy_is_invalid(native_codex, monkeypatch):
    from ipfs_accelerate_py import llm_router
    workspace, _binary = native_codex
    files = []
    create = llm_router.tempfile.NamedTemporaryFile
    def observed(**kwargs):
        handle = create(**kwargs)
        files.append(Path(handle.name))
        return handle
    monkeypatch.setattr(llm_router.tempfile, "NamedTemporaryFile", observed)
    monkeypatch.setenv("ipfs_accelerate_py_CODEX_SANDBOX", "unsupported")
    with pytest.raises(ValueError, match="sandbox"):
        llm_router._get_codex_cli_provider().generate("must not launch")
    assert len(files) == 1 and not files[0].exists()
    assert not (workspace / "calls.jsonl").exists()
