"""Trusted ordinary-agent effort reaches the real container argv, without models."""

from __future__ import annotations

from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor.runtime import grok_cli_runner as G


@pytest.fixture
def ordinary_launch(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "operator-home"
    home.mkdir()
    grok = tmp_path / "fixture-grok"
    grok.write_text("inert fixture; never executed\n")
    lease_root = tmp_path / "lease"
    lease_root.mkdir()
    docker_config = lease_root / "docker-config"
    docker_config.mkdir()
    for key in tuple(os.environ):
        if key.startswith(("IPFS_", "ipfs_", "GROK", "CODEX", "XAI_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GROK_HOME", str(home / ".grok"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "control-plane"))
    monkeypatch.setattr(G.sys, "stdin", io.StringIO("synthetic task instructions"))
    monkeypatch.chdir(workspace)
    observed = {"closed": [], "created": [], "started": []}
    command_environment = SimpleNamespace(
        required_commands=(), wrapper_path="/fixture/provider-command-env",
        contract_sha256="1" * 64, formal_toolchain_contract_sha256="2" * 64,
    )
    lease = SimpleNamespace(
        docker_bin="/fixture/docker", docker_config=docker_config,
        container_name="ipfs-accelerate-grok-1-" + "a" * 32,
        cidfile=lease_root / "container.cid", lease_root=lease_root,
        close=lambda **kw: observed["closed"].append(kw),
    )
    monkeypatch.setattr(G, "sealed_provider_command_environment", lambda *_a, **_kw: nullcontext(command_environment))
    monkeypatch.setattr(G, "_select_grok_isolation_backend", lambda **_kw: G.GROK_ISOLATION_DOCKER)
    monkeypatch.setattr(G, "_docker_isolation_binary", lambda: lease.docker_bin)
    monkeypatch.setattr(G._LegacyDockerContainerLease, "create", lambda *_a, **_kw: lease)
    monkeypatch.setattr(G, "_docker_ordinary_task_toolchain_image_id", lambda *_a, **_kw: G._ordinary_task_toolchain_image_pin())
    monkeypatch.setattr(G, "_grok_task_tex_toolchain", lambda _env: {})
    monkeypatch.setattr(G, "_research_task_toolchain", lambda _env: {})
    monkeypatch.setattr(G, "_git_metadata_roots", lambda _workspace: ())

    def create(command, **_kwargs):
        observed["created"].append(list(command))
        return ["/fixture/docker", "start", "inert-container"]

    def start(command, **kwargs):
        observed["started"].append((list(command), dict(kwargs)))
        return 0, b"", 0, False

    def forbidden(*_args, **_kwargs):
        pytest.fail("test attempted to dispatch an external process")

    # Keep the actual llm_router builder, isolated-home policy, task bootstrap,
    # and ordinary Docker command builder. Intercept only process admission.
    monkeypatch.setattr(G, "_create_grok_container_and_build_start_command", create)
    monkeypatch.setattr(G, "_run_grok_with_bounded_stderr", start)
    monkeypatch.setattr(G.subprocess, "Popen", forbidden)
    monkeypatch.setattr(G.subprocess, "run", forbidden)
    observed["argv"] = ["--workspace", str(workspace), "--grok-bin", str(grok), "--model", "grok-4.7"]
    try:
        yield observed
    finally:
        # This inert lease has no Docker cleanup owner. Restore only its own
        # synthetic deny masks so pytest can remove the bounded fixture tree.
        for parent, directories, files in os.walk(lease_root):
            for name in directories:
                (Path(parent) / name).chmod(0o700)
            for name in files:
                (Path(parent) / name).chmod(0o600)


@pytest.mark.parametrize("private", (False, True))
@pytest.mark.parametrize("effort", (None, "", "low", "medium", "high"))
def test_ordinary_effort_reaches_real_docker_command(ordinary_launch, monkeypatch, private, effort):
    if effort is not None:
        monkeypatch.setenv(G.GROK_REASONING_EFFORT_ENV, effort)
    # Router text-generation aliases are deliberately not trusted agent input.
    monkeypatch.setenv("IPFS_ACCELERATE_PY_GROK_CLI_REASONING_EFFORT", "high")
    monkeypatch.setenv("ipfs_accelerate_py_GROK_CLI_REASONING_EFFORT", "high")
    if private:
        monkeypatch.setenv(G.GROK_PRIVATE_PATHS_ENV, '["data"]')
    assert G.main(ordinary_launch["argv"]) == 0
    assert len(ordinary_launch["created"]) == len(ordinary_launch["started"]) == 1
    command = ordinary_launch["created"][0]
    image = G._ordinary_task_toolchain_image_pin()
    inner = command[command.index(image) + 1:]
    assert inner[inner.index("--model") + 1] == "grok-4.7"
    assert inner.count("--reasoning-effort") == bool(effort)
    if effort:
        assert inner[inner.index("--reasoning-effort") + 1] == effort
    assert "--effort" not in inner
    # Preference becomes argv on the host; it cannot reconfigure the isolated
    # worker or its terminal children through an inherited environment alias.
    environment = ordinary_launch["started"][0][1]["env"]
    assert G.GROK_REASONING_EFFORT_ENV not in environment
    assert G.GROK_REASONING_EFFORT_ENV not in command
    assert ordinary_launch["closed"] == [{"docker_run_finished": True}]


@pytest.mark.parametrize("effort", ("auto", "MEDIUM", "medium --model other", "0", "none"))
def test_invalid_effort_fails_before_discovery_or_dispatch(tmp_path, monkeypatch, capsys, effort):
    monkeypatch.setenv(G.GROK_REASONING_EFFORT_ENV, effort)

    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid effort reached provider discovery or dispatch")

    monkeypatch.setattr(llm_router, "find_grok_cli", forbidden)
    monkeypatch.setattr(G, "_parse_codex_fallback_command", forbidden)
    monkeypatch.setattr(G.subprocess, "Popen", forbidden)
    assert G.main(["--workspace", str(tmp_path)]) == 2
    assert "must be low, medium, or high" in capsys.readouterr().err


@pytest.mark.parametrize("restricted", (
    ["--codex-fallback-command-json", '["/fixture/codex"]'],
    ["--agent-implementation-route-json", '{"signed":"fixture"}'],
    ["--agent-implementation-recovery-json", '{"signed":"fixture"}'],
    ["--grok-failure-receipt-nonce", "a" * 64],
    [G.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG],
    [G.GROK_INVOCATION_ID_FLAG, "a" * 32],
    [G.GROK_INVOCATION_BINDING_FLAG, "b" * 64],
    ["--mode", "chat"],
))
def test_effort_cannot_override_protected_or_ordered_route(tmp_path, monkeypatch, capsys, restricted):
    monkeypatch.setenv(G.GROK_REASONING_EFFORT_ENV, "medium")

    def forbidden(*_args, **_kwargs):
        pytest.fail("effort override reached protected route or provider")

    monkeypatch.setattr(G, "_parse_codex_fallback_command", forbidden)
    monkeypatch.setattr(G, "_run_protected_effect_recovery", forbidden)
    monkeypatch.setattr(G.subprocess, "Popen", forbidden)
    assert G.main(["--workspace", str(tmp_path), *restricted]) == 2
    assert "cannot be overridden" in capsys.readouterr().err


@pytest.mark.parametrize("effort", (None, ""))
def test_absent_effort_leaves_ordered_route_admission_unchanged(tmp_path, monkeypatch, effort):
    if effort is None:
        monkeypatch.delenv(G.GROK_REASONING_EFFORT_ENV, raising=False)
    else:
        monkeypatch.setenv(G.GROK_REASONING_EFFORT_ENV, effort)
    raw = json.dumps(["/fixture/codex", "exec", "authored-route"])
    seen = []

    class ExistingAdmission(Exception):
        pass

    def existing_admission(value, **kwargs):
        seen.append((value, kwargs))
        raise ExistingAdmission

    monkeypatch.setattr(G, "_parse_codex_fallback_command", existing_admission)
    with pytest.raises(ExistingAdmission):
        G.main(["--workspace", str(tmp_path), "--codex-fallback-command-json", raw])
    assert seen == [(raw, {"expected_fallback_reasoning_effort": G.DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT})]
