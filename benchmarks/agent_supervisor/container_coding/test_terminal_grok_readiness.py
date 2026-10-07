"""Isolated catalog fixtures only; no real Grok, credentials or network calls."""
import asyncio
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_grok_deployment as grok


@pytest.mark.parametrize("stdout,stderr,code,status,reason", [
    (b"Available models:\n grok-4.7\n grok-code-fast\n", b"", 0, "ready", "catalog_observed"),
    (b"\x1b[32mAvailable models:\x1b[0m\n - grok-4.7 (default)\n", b"", 0, "ready", "catalog_observed"),
    (b"You are not authenticated.\nDefault model: grok-4.7\n", b"", 0, "unauthenticated", "authentication_refused"),
    (b"Available models:\n grok-4.7\n", b"Not signed in. PRIVATE", 0, "unauthenticated", "authentication_refused"),
    (b"Available models:\n grok-4.7\n", b"network unavailable PRIVATE", 0, "unavailable", "unexpected_diagnostic"),
    (b"Available models:\n grok-4.7\n", b"", 1, "unavailable", "native_command_failed"),
    (b"Error loading model grok-4.7: PRIVATE", b"", 0, "unavailable", "catalog_shape_unknown"),
    (b"Available models:\n grok-4.7\n error PRIVATE\n", b"", 0, "unavailable", "catalog_shape_unknown"),
    (b"Available models:\n grok-4.7\n grok-4.7\n", b"", 0, "unavailable", "catalog_shape_unknown"),
    (b"Available models:\n grok-code-fast\n", b"", 0, "unavailable", "selected_model_unavailable"),
    (b"", b"", 0, "unavailable", "catalog_shape_unknown"),
])
def test_catalog_parser_requires_closed_positive_shape_and_prioritizes_auth(stdout, stderr, code, status, reason):
    receipt = grok._catalog_observation(stdout, stderr, code)
    assert (receipt["status"], receipt["reason"]) == (status, reason)
    assert receipt["native_exit_code"] == code
    assert receipt["stdout_bytes"] == len(stdout) and receipt["stderr_bytes"] == len(stderr)
    assert receipt["text_generation_calls"] == 0
    assert all(receipt[key] is False for key in (
        "authentication_authority", "proof_authority", "execution_authority", "completion_authority"))
    assert "PRIVATE" not in json.dumps(receipt)


def test_probe_refuses_host_identity_without_launch_or_credential_read(monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("no file read before worker identity"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("host CLI must not launch"))
    assert grok.probe_container_grok_readiness()["reason"] == "isolated_boundary_unavailable"


@pytest.fixture
def copied_boundary(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.runtime import container_worker_boundary as boundary
    root = tmp_path / "runtime"
    home = root / "worker-home"
    (home / ".grok").mkdir(parents=True)
    home.chmod(0o700)
    (home / ".grok").chmod(0o700)
    auth = home / ".grok/auth.json"
    auth.write_bytes(b"synthetic fixture, not a credential")
    auth.chmod(0o600)
    binary = root / "provider-bin/grok"
    binary.parent.mkdir()
    binary.parent.chmod(0o755)
    binary.write_bytes(b"synthetic native asset")
    binary.chmod(0o555)
    (root / "container-boundary.json").write_text(json.dumps({"provider": "grok_cli"}))
    (root / "container-boundary.json").chmod(0o444)
    monkeypatch.setattr(grok, "READINESS_ROOT", root)
    monkeypatch.setattr(grok, "GROK_BYTES", binary.stat().st_size)
    monkeypatch.setattr(grok, "GROK_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest())
    monkeypatch.setattr(os, "getuid", lambda: 1001)
    monkeypatch.setattr(os, "geteuid", lambda: 1001)
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **k: SimpleNamespace(prctl=lambda *a: 0))
    verified = []
    monkeypatch.setattr(boundary, "verify_container_worker_boundary", lambda **kw: verified.append(kw))
    original = Path.lstat
    def authored_ownership(path, **kwargs):
        info = original(path, **kwargs)
        uid = 1001 if path in (home, home / ".grok", auth) else 0
        return SimpleNamespace(**{name: getattr(info, name) for name in dir(info) if name.startswith("st_") and name != "st_uid"}, st_uid=uid)
    monkeypatch.setattr(Path, "lstat", authored_ownership)
    return root, home, auth, binary, verified


def test_probe_uses_private_copy_without_reading_credential_body(copied_boundary, monkeypatch):
    root, home, auth, binary, verified = copied_boundary
    original = Path.open
    def noncredential_open(path, *args, **kwargs):
        assert path != auth, "the helper must not read copied credentials itself"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", noncredential_open)
    assert grok._container_grok_inputs() == (binary, home)
    assert verified[0]["workspace"] == Path("/app")
    assert verified[0]["purpose"] == "validation"
    assert verified[0]["artifact"] == root / "container-boundary.json"


@pytest.mark.parametrize("change", ["auth_missing", "auth_mode", "auth_symlink", "home_mode", "binary_hash", "provider"])
def test_copy_or_native_identity_changes_refuse_before_models(copied_boundary, monkeypatch, change):
    root, home, auth, binary, _ = copied_boundary
    if change == "auth_missing":
        auth.unlink()
    elif change == "auth_mode":
        auth.chmod(0o644)
    elif change == "auth_symlink":
        target = home / "synthetic-other"
        auth.rename(target)
        auth.symlink_to(target)
    elif change == "home_mode":
        home.chmod(0o750)
    elif change == "binary_hash":
        binary.chmod(0o755)
        binary.write_bytes(b"X" * binary.stat().st_size)
    else:
        (root / "container-boundary.json").chmod(0o644)
        (root / "container-boundary.json").write_text(json.dumps({"provider": "codex_cli"}))
        (root / "container-boundary.json").chmod(0o444)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("changed boundary must not launch"))
    assert grok.probe_container_grok_readiness()["reason"] == "isolated_boundary_unavailable"


@pytest.mark.parametrize("mode", ["catalog", "timeout", "oversize"])
def test_bounded_native_fixture_uses_only_models_and_is_reaped(tmp_path, monkeypatch, mode):
    binary = tmp_path / "fixture-cli"
    home = tmp_path / "isolated-home"
    home.mkdir()
    body = ("import os,sys,time\n"
        "assert sys.argv[1:] == ['models']\n"
        f"assert os.environ['HOME'] == {str(home)!r}\n"
        f"assert os.environ['GROK_HOME'] == {str(home / '.grok')!r}\n"
        "assert 'XAI_API_KEY' not in os.environ\n")
    body += {"catalog": "print('Available models:\\n grok-4.7')\n",
             "timeout": "time.sleep(5)\n", "oversize": "os.write(1,b'X'*70000)\n"}[mode]
    binary.write_text("#!" + sys.executable + "\n" + body)
    binary.chmod(0o755)
    monkeypatch.setenv("XAI_API_KEY", "synthetic ambient value must not reach fixture")
    monkeypatch.setattr(grok, "READINESS_TIMEOUT", .1 if mode == "timeout" else 2)
    native = subprocess.Popen
    processes = []
    def launch(*args, **kwargs):
        process = native(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", launch)
    stdout, stderr, code, reason = grok._bounded_catalog_command(binary, home)
    assert len(stdout) + len(stderr) <= grok.READINESS_MAX_BYTES
    assert len(processes) == 1 and processes[0].poll() is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(processes[0].pid, os.WNOHANG)
    assert reason == {"catalog": None, "timeout": "native_timeout", "oversize": "output_limit"}[mode]
    if mode == "catalog":
        assert grok._catalog_observation(stdout, stderr, code)["status"] == "ready"


@pytest.mark.parametrize("change", ["transport", "raw_output", "extra_field", "forged_ready",
    "ready_with_stderr", "ready_without_stdout", "combined_output_limit"])
def test_async_consumer_persists_only_sanitized_closed_results(tmp_path, change):
    calls = []
    async def execute(**kwargs):
        calls.append(kwargs)
        if change == "transport":
            raise RuntimeError("PRIVATE transport payload")
        value = grok._catalog_observation(b"Available models:\n grok-4.7\n", b"", 0)
        if change == "extra_field":
            value["credential_fingerprint"] = "PRIVATE"
        elif change == "forged_ready":
            value["native_exit_code"] = 1
        elif change == "ready_with_stderr":
            value["stderr_bytes"] = 1
        elif change == "ready_without_stdout":
            value["stdout_bytes"] = 0
        elif change == "combined_output_limit":
            value.update(status="unavailable", reason="output_limit",
                stdout_bytes=grok.READINESS_MAX_BYTES, stderr_bytes=1,
                models_observed=0, selected_model_observed=False)
        return SimpleNamespace(return_code=0, stdout="PRIVATE" if change == "raw_output" else json.dumps(value), stderr="")
    with pytest.raises(grok.GrokCredentialReadinessError) as failed:
        asyncio.run(grok.require_grok_credential_readiness(SimpleNamespace(exec=execute), output=tmp_path / "check"))
    receipt = json.loads((tmp_path / "check/readiness.json").read_text())
    assert receipt == failed.value.receipt
    assert receipt["status"] == "unavailable"
    assert "PRIVATE" not in json.dumps(receipt) + str(failed.value)
    assert calls[0]["user"] == "benchmarkworker" and calls[0]["cwd"] == "/"
    assert calls[0]["env"]["HOME"] == "/opt/ipfs-supervisor/worker-home"
    assert calls[0]["timeout_sec"] == 30
    assert "subprocess.Popen([str(binary), \"models\"]" in calls[0]["command"]


def test_standalone_probe_does_not_depend_on_new_module_in_retained_archive(monkeypatch, capsys):
    script = grok.grok_credential_readiness_script()
    tree = ast.parse(script)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "benchmarks.agent_supervisor.container_coding.terminal_grok_deployment" not in imports
    assert "ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary" in imports
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("host probe must refuse"))
    before = list(sys.path)
    try:
        exec(compile(script, "retained-archive-readiness-fixture", "exec"), {})
    finally:
        sys.path[:] = before
    result = json.loads(capsys.readouterr().out)
    assert result["reason"] == "isolated_boundary_unavailable"
    assert result["native_catalog_commands"] == 0
