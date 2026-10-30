"""Repair argv, physical parser and trusted quota handoff as one fake route.

Popen and readiness are explicit doubles; only private anonymous receipt pipes
and toy prompt files are real. No provider, service or supervisor DB is called.
"""
from __future__ import annotations

import contextlib
import importlib.util
import errno
import io
import json
import os
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor import provider_fallback_runner as fallback
from ipfs_accelerate_py.agent_supervisor.rescue import fleet_repair as repair


def _physical_runner():
    return Path(repair.__file__).resolve().parents[1] / "grok_cli_runner.py"


def _legacy_module():
    path = _physical_runner()
    spec = importlib.util.spec_from_file_location("_fleet_contract_physical_grok", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy(tmp_path):
    return dict(cwd=str(tmp_path), argv=["fixture-codex", "exec", "--model", "gpt-6.1-sol",
        "-c", 'model_reasoning_effort="high"', "--json"], implementation_route=dict(
        primary_provider_id="grok_cli", primary_model_id="grok-4.7",
        fallback_provider_id="codex", fallback_model_id="gpt-6.1-sol",
        fallback_trigger="primary_quota_exhausted", fallback_reasoning_effort="high"))


def _command(tmp_path):
    return repair.llm_router_repair_argv(_policy(tmp_path), tmp_path / "prompt", tmp_path / "last")


def test_repair_command_binds_physical_parser_and_trusted_pipe_protocol(tmp_path):
    from ipfs_accelerate_py.agent_supervisor import grok_cli_runner as alias

    command = _command(tmp_path)
    primary = json.loads(command[command.index("--primary-command-json") + 1])
    assert Path(alias.__file__).resolve() != _physical_runner()
    assert primary[:3] == [sys.executable, "-P", str(_physical_runner())]
    assert "--require-terminal-quota-frame" in primary
    assert primary[primary.index("--model") + 1] == "grok-4.7"
    assert fallback._uses_packaged_grok_adapter(primary)
    assert command[command.index("--fallback-policy") + 1] == "grok_quota_only"
    assert command[command.index("--probe-grok-model") + 1] == "grok-4.7"
    assert command[command.index("--probe-codex-model") + 1] == "gpt-6.1-sol"


@pytest.mark.parametrize("prefix", [(), ("-P",)])
def test_receipt_channel_accepts_only_existing_physical_script_shapes(prefix):
    assert fallback._uses_packaged_grok_adapter([sys.executable, *prefix, str(_physical_runner())])


@pytest.mark.parametrize("shape", ["runtime_alias", "wrong_python", "same_basename", "module",
    "code", "isolation", "additional_option", "missing_script", "empty"])
def test_receipt_channel_rejects_unrelated_or_ambiguous_commands(tmp_path, shape):
    physical = _physical_runner()
    other = tmp_path / physical.name
    other.write_text("raise SystemExit(0)\n")
    commands = {
        "runtime_alias": [sys.executable, "-P", str(physical.parent / "runtime" / physical.name)],
        "wrong_python": ["/bin/false", "-P", str(physical)],
        "same_basename": [sys.executable, "-P", str(other)],
        "module": [sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"],
        "code": [sys.executable, "-c", "pass", str(physical)],
        "isolation": [sys.executable, "-I", str(physical)],
        "additional_option": [sys.executable, "-P", "-S", str(physical)],
        "missing_script": [sys.executable, "-P"], "empty": [],
    }
    assert not fallback._uses_packaged_grok_adapter(commands[shape])


class _PromptSink:
    def __init__(self):
        self.value = ""
        self.closed = False
    def write(self, value):
        self.value += value
    def flush(self):
        pass
    def close(self):
        self.closed = True


class _Process:
    def __init__(self, code, stdout="", stderr=""):
        self.code = code
        self.stdin = _PromptSink()
        self.stdout, self.stderr = io.StringIO(stdout), io.StringIO(stderr)
    def wait(self):
        return self.code


_QUOTA = json.dumps(dict(type="error", code="usage_pool_exhausted")) + "\n"
_PRELUDE = json.dumps(dict(type="available_commands", tools=["read_file"], commands=["models"])) + "\n"
_EMBEDDED = json.dumps(dict(type="error", message='Internal error: {"message": "API error (status 402 Payment Required): Grok Build usage balance exhausted", "http_status": 402}')) + "\n"


@pytest.mark.parametrize("stream,stderr,code,eligible", [
    (_QUOTA, "", 19, True),
    (_PRELUDE + _QUOTA, "", 19, True),
    (_PRELUDE + _EMBEDDED, "", 19, True),
    (_EMBEDDED, "", 19, False),
    ("", "usage_pool_exhausted", 19, False),
    (json.dumps(dict(type="assistant", message="quota")) + "\n" + _QUOTA, "", 19, False),
    (_QUOTA + _QUOTA, "", 19, False),
    (json.dumps(dict(type="error", code="usage_pool_exhausted", error=dict(code="usage_limit_reached"))) + "\n", "", 19, False),
    ('{"type":"error","message":"' + "x" * (256 * 1024) + '"}\n' + _QUOTA, "", 19, False),
    ("not JSON\n" + _QUOTA, "", 19, False),
    ("\ufffd\n" + _QUOTA, "", 19, False),
    (_PRELUDE + json.dumps(dict(type="available_commands", tools=["different"], commands=["models"])) + "\n" + _QUOTA, "", 19, False),
    (json.dumps(dict(type="error", code="authentication_failed")) + "\n", "", 19, False),
    (_QUOTA, "", 0, False),
], ids=["exact_quota", "startup_then_quota", "authenticated_startup_embedded_balance", "unbound_embedded_balance",
        "stderr_only_quota", "assistant_before_quota", "duplicate_terminal", "conflicting_codes", "overlong",
        "malformed_stream", "replacement_character", "changed_startup", "authentication_not_quota", "primary_success"])
def test_generated_repair_route_runs_physical_parser_and_falls_back_only_on_trusted_quota(
    tmp_path, monkeypatch, stream, stderr, code, eligible,
):
    legacy = _legacy_module()
    command = _command(tmp_path)
    primary = json.loads(command[command.index("--primary-command-json") + 1])
    # Only the fake process executable is selected explicitly; all generated
    # physical parser, model, workspace and strict quota options stay intact.
    primary += ["--grok-bin", "/bin/true"]
    command[command.index("--primary-command-json") + 1] = json.dumps(primary)
    prompt = "repair only this fictional engine fixture\n"
    calls = {"primary": [], "grok": [], "codex": [], "readiness": []}

    def popen(argv, **kwargs):
        if list(argv) == primary:
            assert kwargs["cwd"] == tmp_path
            assert kwargs["close_fds"] is True
            fd = int(kwargs["env"][legacy.TRUSTED_FAILURE_RECEIPT_FD_ENV])
            assert fd in kwargs["pass_fds"]
            # A simulated fork owns a separate FD, just as a real child does.
            child_fd = os.dup(fd)
            output, diagnostic = io.StringIO(), io.StringIO()
            try:
                with monkeypatch.context() as child:
                    child.setenv(legacy.TRUSTED_FAILURE_RECEIPT_FD_ENV, str(child_fd))
                    child.setattr(sys, "stdin", io.StringIO(prompt))
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostic):
                        result = legacy.main(primary[3:])
            finally:
                # A real child closes inherited descriptors at exit, including
                # success paths which emit no failure receipt. Model that exit.
                try:
                    os.close(child_fd)
                except OSError as exc:
                    if exc.errno != errno.EBADF:
                        raise
            process = _Process(result, output.getvalue(), diagnostic.getvalue())
            calls["primary"].append(process)
            return process
        if argv[0] == "/bin/true":
            assert argv[argv.index("--model") + 1] == "grok-4.7"
            assert argv[argv.index("--output-format") + 1] == "streaming-json"
            assert Path(argv[argv.index("--prompt-file") + 1]).read_text() == prompt
            calls["grok"].append(list(argv))
            return _Process(code, stream, stderr)
        if argv[0] == "fixture-codex":
            process = _Process(0)
            calls["codex"].append(process)
            return process
        pytest.fail("unexpected real process attempt: " + repr(argv))

    def ready(**kwargs):
        calls["readiness"].append(kwargs)
        return llm_router.AgentCLIRouteReadiness(True, True, "grok", "fixture_ready", None)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(fallback, "_prepare_provider_boundary_runtime", lambda **_: fallback._ProviderBoundaryRuntime((), {}, None))
    monkeypatch.setattr(fallback, "snapshot_agent_cli_workspace", lambda _: llm_router.AgentCLIWorkspaceSnapshot("fixture-stable", True))
    monkeypatch.setattr(fallback, "probe_grok_codex_agent_route_readiness", ready)
    monkeypatch.setattr(fallback.subprocess, "Popen", popen)
    result = fallback.main(command[4:])
    assert result == (0 if eligible or code == 0 else code)
    assert len(calls["primary"]) == len(calls["grok"]) == len(calls["readiness"]) == 1
    assert calls["primary"][0].stdin.value == prompt
    assert len(calls["codex"]) == int(eligible)
    if eligible:
        assert calls["codex"][0].stdin.value == prompt
    assert calls["readiness"][0]["grok_model"] == "grok-4.7"
    assert calls["readiness"][0]["codex_model"] == "gpt-6.1-sol"
    assert "--require-terminal-quota-frame" in primary
