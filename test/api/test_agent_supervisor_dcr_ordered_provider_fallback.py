"""Focused DCR-000 checks for the ordered Grok quota fallback contract."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import grok_cli_runner
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def test_builder_binds_high_for_dcr_and_keeps_legacy_medium(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/usr/local/bin/codex",
    )
    high = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable="/usr/bin/python3",
        grok_bin="/usr/bin/grok",
        codex_bin="/usr/local/bin/codex",
        fallback_reasoning_effort="high",
    )
    assert high[high.index("--codex-fallback-reasoning-effort") + 1] == "high"
    high_fallback = json.loads(high[high.index("--codex-fallback-command-json") + 1])
    assert 'model_reasoning_effort="high"' in high_fallback
    assert (
        grok_cli_runner._parse_codex_fallback_command(
            json.dumps(high_fallback),
            expected_fallback_reasoning_effort="high",
        )
        == high_fallback
    )

    medium = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        codex_bin="/usr/local/bin/codex",
    )
    assert medium[medium.index("--codex-fallback-reasoning-effort") + 1] == "medium"
    with pytest.raises(ValueError, match="must be one of"):
        grok_cli_runner.build_grok_quota_routed_agent_command(
            workspace=tmp_path,
            fallback_reasoning_effort="low",
        )


def test_daemon_passes_configured_high_effort_to_exact_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(implementation_daemon._CODEX_REASONING_EFFORT_ENV, "high")
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: True)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )

    command = implementation_daemon._grok_cli_command(workspace_path=tmp_path)
    assert command[command.index("--codex-fallback-reasoning-effort") + 1] == "high"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback


def test_ordered_grok_builder_uses_shared_configured_binary_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-user/configured Grok path must survive a PATH-minimal daemon."""

    workspace = tmp_path / "worktree"
    workspace.mkdir()
    providers = tmp_path / "providers"
    providers.mkdir()
    grok = providers / "grok"
    codex = providers / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_grok_cli_for_ordered_provider",
        lambda: str(grok),
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_readiness",
        lambda: ("ready", ""),
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: str(codex) if name == "codex" else None,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )

    command = implementation_daemon._grok_cli_command(workspace_path=workspace)

    assert command[command.index("--grok-bin") + 1] == str(grok)


def test_primary_unavailable_route_invokes_only_containerized_terra_high(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex = shutil.which("codex")
    assert codex is not None
    command = grok_cli_runner.build_grok_primary_unavailable_codex_fallback_command(
        workspace=tmp_path,
        python_executable=sys.executable,
        codex_bin=codex,
        primary_unavailability_reason="grok_auth_unavailable",
    )
    assert command[:3] == [
        sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    ]
    assert "--grok-bin" not in command
    assert "--grok-failure-receipt-nonce" not in command
    assert command[-1] == "--codex-fallback-on-primary-unavailable"
    assert command[command.index("--primary-unavailability-reason") + 1] == (
        "grok_auth_unavailable"
    )
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback == [
        codex,
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "-s",
        "workspace-write",
        "-C",
        str(tmp_path),
        "-m",
        "gpt-5.6-terra",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'web_search="disabled"',
        "-",
    ]

    calls: list[dict[str, object]] = []

    def fake_containerized_fallback(
        *,
        host_fallback_command,
        workspace,
        base_env,
        prompt,
    ):
        calls.append(
            {
                "command": list(host_fallback_command),
                "workspace": workspace,
                "prompt": prompt,
                "base_env": dict(base_env),
            }
        )
        return subprocess.CompletedProcess(host_fallback_command, 0)

    monkeypatch.setattr(
        grok_cli_runner,
        "grok_primary_readiness_for_ordered_provider",
        lambda: ("unavailable", "grok_auth_unavailable"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_preflight_containerized_codex_fallback_workspace",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_containerized_codex_quota_fallback",
        fake_containerized_fallback,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "build_grok_cli_command",
        lambda *_args, **_kwargs: pytest.fail(
            "direct unavailable route must not invoke Grok"
        ),
        raising=False,
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair it"))

    assert grok_cli_runner.main(command[3:]) == 0
    assert len(calls) == 1
    assert calls[0]["command"] == fallback
    assert calls[0]["workspace"] == tmp_path.resolve()
    assert calls[0]["prompt"] == "repair it"


@pytest.mark.parametrize(
    "readiness",
    (
        ("ready", ""),
        ("indeterminate", ""),
        ("unavailable", "grok_provider_unavailable"),
    ),
)
def test_primary_unavailable_route_rejects_readiness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readiness: tuple[str, str],
) -> None:
    codex = shutil.which("codex")
    assert codex is not None
    command = grok_cli_runner.build_grok_primary_unavailable_codex_fallback_command(
        workspace=tmp_path,
        python_executable=sys.executable,
        codex_bin=codex,
        primary_unavailability_reason="grok_auth_unavailable",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "grok_primary_readiness_for_ordered_provider",
        lambda: readiness,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_containerized_codex_quota_fallback",
        lambda **_kwargs: pytest.fail("readiness drift reached Codex"),
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair it"))

    assert grok_cli_runner.main(command[3:]) == 2


@pytest.mark.parametrize("shape", ("duplicate", "reordered"))
def test_primary_unavailable_route_rejects_noncanonical_raw_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> None:
    codex = shutil.which("codex")
    assert codex is not None
    command = grok_cli_runner.build_grok_primary_unavailable_codex_fallback_command(
        workspace=tmp_path,
        python_executable=sys.executable,
        codex_bin=codex,
        primary_unavailability_reason="grok_auth_unavailable",
    )
    raw = command[3:]
    if shape == "duplicate":
        raw = [*raw[:-1], "--model", "grok-4.5", raw[-1]]
    else:
        raw = [*raw[2:4], *raw[:2], *raw[4:]]
    calls: list[object] = []
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_containerized_codex_quota_fallback",
        lambda **_kwargs: calls.append(_kwargs),
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair it"))

    assert grok_cli_runner.main(raw) == 2
    assert calls == []


@pytest.mark.parametrize(
    ("boundary_probe", "finding"),
    (
        ("_workspace_regular_file_hardlinks", "multiply linked"),
        ("_workspace_descendant_mountpoints", "mountpoints"),
    ),
)
def test_primary_unavailable_route_rejects_workspace_boundary_drift_before_docker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary_probe: str,
    finding: str,
) -> None:
    codex = shutil.which("codex")
    assert codex is not None
    command = grok_cli_runner.build_grok_primary_unavailable_codex_fallback_command(
        workspace=tmp_path,
        python_executable=sys.executable,
        codex_bin=codex,
        primary_unavailability_reason="grok_auth_unavailable",
    )
    sensitive_root = tmp_path.parent / "sensitive"
    sensitive_root.mkdir(exist_ok=True)
    auth = sensitive_root / "auth.json"
    auth.write_text("{}", encoding="utf-8")
    package = sensitive_root / "package"
    package.mkdir(exist_ok=True)
    bwrap = sensitive_root / "bwrap"
    bwrap.write_text("#!/bin/sh\n", encoding="utf-8")
    checkpoint = sensitive_root / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    monkeypatch.setattr(
        grok_cli_runner,
        "grok_primary_readiness_for_ordered_provider",
        lambda: ("unavailable", "grok_auth_unavailable"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_containerized_codex_fallback_assets",
        lambda **_kwargs: (
            Path("/usr/bin/docker"),
            auth,
            package,
            bwrap,
            checkpoint,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        boundary_probe,
        lambda _workspace: (_workspace / "unsafe",),
    )
    calls: list[object] = []
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_containerized_codex_quota_fallback",
        lambda **_kwargs: calls.append(_kwargs),
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair it"))

    assert grok_cli_runner.main(command[3:]) == 127
    assert calls == []
    assert finding


def test_primary_unavailable_route_returns_container_failure_without_grok_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex = shutil.which("codex")
    assert codex is not None
    command = grok_cli_runner.build_grok_primary_unavailable_codex_fallback_command(
        workspace=tmp_path,
        python_executable=sys.executable,
        codex_bin=codex,
        primary_unavailability_reason="grok_auth_unavailable",
    )
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        grok_cli_runner,
        "grok_primary_readiness_for_ordered_provider",
        lambda: ("unavailable", "grok_auth_unavailable"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_preflight_containerized_codex_fallback_workspace",
        lambda **_kwargs: None,
    )

    def failed_container(**kwargs):
        calls.append(dict(kwargs))
        return subprocess.CompletedProcess(kwargs["host_fallback_command"], 47)

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_containerized_codex_quota_fallback",
        failed_container,
    )
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("repair it"))

    assert grok_cli_runner.main(command[3:]) == 47
    assert len(calls) == 1


def test_ordered_daemon_selects_direct_route_only_for_closed_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "worktree"
    workspace.mkdir()
    codex = shutil.which("codex")
    assert codex is not None
    daemon = object.__new__(TodoImplementationDaemon)
    daemon.repo_root = tmp_path
    daemon.worktree_root = tmp_path
    daemon.use_ephemeral_worktree = True
    daemon.manual_completion_authority_revalidation_only = False
    task = PortalTask(
        task_id="DCR-UNAVAILABLE",
        title="Use the sealed unavailable route",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"implementation mode": "ordered_provider"},
    )
    calls: list[dict[str, object]] = []
    real_builder = grok_cli_runner.build_grok_primary_unavailable_codex_fallback_command

    def capture_builder(**kwargs):
        calls.append(dict(kwargs))
        return real_builder(**kwargs)

    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_readiness",
        lambda: ("unavailable", "grok_auth_unavailable"),
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: codex if name == "codex" else None,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "build_grok_primary_unavailable_codex_fallback_command",
        capture_builder,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_command",
        lambda **_kwargs: pytest.fail("unavailable route invoked Grok builder"),
    )

    command = daemon._build_implementation_command(workspace, task=task)

    assert len(calls) == 1
    assert not hasattr(daemon, "_ordered_provider_proposal_authorities")
    assert calls[0]["primary_unavailability_reason"] == "grok_auth_unavailable"
    assert command[-1] == "--codex-fallback-on-primary-unavailable"
    assert command[command.index("--codex-fallback-reasoning-effort") + 1] == "high"

    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_readiness",
        lambda: ("indeterminate", ""),
    )
    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="readiness is indeterminate",
    ):
        daemon._build_implementation_command(workspace, task=task)
