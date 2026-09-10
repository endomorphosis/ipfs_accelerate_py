"""Contracts for typed Grok failure → Codex Terra/high fallback authority."""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.parse
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ipfs_accelerate_py.agent_supervisor.control import provider_attempt_store
from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.integrations import (
    llm_merge_resolver_fallback as merge_resolver_fallback,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    provider_executable_trust,
    provider_failure_policy,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    GROK_NOT_SIGNED_IN_GUIDANCE,
)
from ipfs_accelerate_py.agent_supervisor.runtime.worker_network import (
    PROVIDER_HOSTNAME_ALLOWLISTS,
    WORKER_NETWORK_AUTHORIZATION_SCHEMA,
    WorkerNetworkProfile,
    derived_worker_network_name,
    worker_network_approval_cid,
    worker_network_authorization_relative_path,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor import grok_cli_runner

_NATIVE_SESSION_ID = "00000000-0000-4000-8000-000000000001"
_SPENDING_LIMIT_MESSAGE = (
    "API error (status 403 Forbidden): personal-team-blocked:spending-limit: "
    "You have run out of credits or need a Grok subscription. Add credits at "
    "https://grok.com/?_s=usage or upgrade at https://grok.com/supergrok."
)
import signal
import socket
import stat
import tempfile
import threading
from ipfs_accelerate_py.agent_supervisor.runtime import provider_failure_policy
from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator

_REQUIRE_LIVE_DOCKER_CLEANUP_VALIDATION_ENV = (
    "IPFS_ACCELERATE_AGENT_REQUIRE_LIVE_DOCKER_CLEANUP_VALIDATION"
)
_TEST_PRELOAD_GROK_NATIVE_ENV = (
    "IPFS_ACCELERATE_AGENT_TEST_PRELOAD_GROK_NATIVE"
)
_TEST_CODEX_EXECUTABLE_ENV = (
    "IPFS_ACCELERATE_AGENT_TEST_CODEX_EXECUTABLE"
)
_EXACT_TEST_CODEX_EXECUTABLE = "/usr/local/bin/codex"

def _signed_network_fixture(
    tmp_path: Path,
    *,
    provider: str,
    workspace: Path,
    container_name: str,
    lease_root: Path,
    prompt: str = "implement",
) -> tuple[SimpleNamespace, WorkerNetworkProfile]:
    """Create a real, fresh, reviewer-signed worker-network fixture."""

    profile_dir = tmp_path / f"signed-network-profile-{provider}"
    profile_dir.mkdir(mode=0o700, exist_ok=True)
    reviewer_key = Ed25519PrivateKey.generate()
    reviewer_did = ed25519_did_key(reviewer_key.public_key())
    worker_did = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    provider_did = ed25519_did_key(Ed25519PrivateKey.generate().public_key())
    now_ms = int(time.time() * 1000)

    def cid(label: str) -> str:
        return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()

    invocation = SimpleNamespace(
        invocation_id=cid(f"invocation-{provider}-{container_name}"),
        content_id=cid(f"binding-{provider}-{container_name}"),
        logical_attempt_id=cid(f"attempt-{provider}-{container_name}"),
        task_id="EAAEF-NETWORK-BOUNDARY-TEST",
        worktree_id=cid(f"worktree-{workspace}"),
        route_id=cid(f"route-{provider}"),
        profile_dir=str(profile_dir),
        reviewer_identity=reviewer_did,
        profile_identity_did=reviewer_did,
        expected_worker_principal_did=worker_did,
        expected_provider_principal_did=provider_did,
        primary_provider_id="grok_cli",
        fallback_provider_id="codex",
        expires_at_ms=now_ms + 120_000,
        control_plane=SimpleNamespace(capsule_id=cid("capsule")),
        prompt_cid=grok_cli_runner._agent_prompt_cid(prompt),
        workspace_path=str(workspace),
    )
    unsigned = {
        "schema": WORKER_NETWORK_AUTHORIZATION_SCHEMA,
        "invocation_binding_id": invocation.content_id,
        "logical_attempt_id": invocation.logical_attempt_id,
        "task_id": invocation.task_id,
        "worktree_id": invocation.worktree_id,
        "control_plane_capsule_id": invocation.control_plane.capsule_id,
        "effect_cid": invocation.content_id,
        "provider": provider,
        "route_id": invocation.route_id,
        "workspace": str(workspace),
        "container_name": container_name,
        "lease_id": lease_root.name,
        "lease_root": str(lease_root),
        "docker_network": derived_worker_network_name(invocation.worktree_id),
        "docker_network_id": "b" * 64,
        "docker_network_internal": True,
        "proxy_endpoint": "http://172.28.0.2:3128",
        "proxy_container_id": "c" * 64,
        "proxy_image_id": "sha256:" + "d" * 64,
        "allowed_hostnames": list(PROVIDER_HOSTNAME_ALLOWLISTS[provider]),
        "issued_at_ms": now_ms - 1_000,
        "expires_at_ms": now_ms + 60_000,
        "one_use_nonce": "network-nonce:0123456789abcdef",
        "signer_did": reviewer_did,
        "worker_principal_did": worker_did,
        "provider_principal_did": provider_did,
    }
    authorization_id = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    signed = {**unsigned, "authorization_id": authorization_id}
    record = {
        **signed,
        "signature": base64.b64encode(
            reviewer_key.sign(
                json.dumps(
                    signed,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            )
        ).decode("ascii"),
    }
    authorization_path = (
        profile_dir
        / worker_network_authorization_relative_path(
            invocation.invocation_id,
            provider,
        )
    )
    authorization_path.parent.mkdir(parents=True)
    authorization_path.parent.parent.chmod(0o700)
    authorization_path.parent.chmod(0o700)
    authorization_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    authorization_path.chmod(0o600)
    profile = grok_cli_runner._signed_worker_network_profile(
        invocation_binding=invocation,
        provider=provider,
        workspace=workspace,
    )
    return invocation, profile

def _native_update(
    update: dict[str, object],
    *,
    session_id: str = _NATIVE_SESSION_ID,
) -> dict[str, object]:
    return {
        "method": "_x.ai/session/update",
        "params": {
            "sessionId": session_id,
            "update": update,
        },
    }

def _write_native_session_home(
    grok_home: Path,
    updates: list[dict[str, object]],
    *,
    session_id: str = _NATIVE_SESSION_ID,
    model: str = "grok-4.6",
    workspace: Path | None = None,
    encoded_workspace: bool = True,
) -> Path:
    session_root = grok_home / "sessions"
    if workspace is not None:
        workspace = workspace.resolve(strict=True)
    if workspace is not None and encoded_workspace:
        session_root /= quote(str(workspace), safe="")
    session = session_root / session_id
    session.mkdir(parents=True)
    (session / "updates.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in updates),
        encoding="utf-8",
    )
    (session / "summary.json").write_text(
        json.dumps(
            {
                "info": {
                    "id": session_id,
                    **(
                        {"cwd": str(workspace)}
                        if workspace is not None
                        else {}
                    ),
                },
                "current_model_id": model,
                "grok_home": str(grok_home),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return grok_home

def _write_native_session(
    root: Path,
    updates: list[dict[str, object]],
    *,
    model: str = "grok-4.6",
) -> Path:
    return _write_native_session_home(
        root / "grok-home",
        updates,
        model=model,
    )

def _spending_limit_retry(
    *, session_id: str = _NATIVE_SESSION_ID
) -> dict[str, object]:
    return _native_update(
        {
            "sessionUpdate": "retry_state",
            "type": "failed",
            "error_type": "api",
            "message": _SPENDING_LIMIT_MESSAGE,
        },
        session_id=session_id,
    )

def _spending_limit_terminal(
    *,
    message: str = _SPENDING_LIMIT_MESSAGE,
    session_id: str = _NATIVE_SESSION_ID,
) -> dict[str, object]:
    return _native_update(
        {
            "sessionUpdate": "turn_completed",
            "stop_reason": "error",
            "agent_result": message,
        },
        session_id=session_id,
    )

def _daemon(root: Path) -> TodoImplementationDaemon:
    board = root / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    return TodoImplementationDaemon(
        todo_path=board,
        state_path=root / "state" / "task-state.json",
        strategy_path=root / "state" / "strategy.json",
        events_path=root / "state" / "events.jsonl",
        repo_root=root,
        worktree_root=root,
    )

def _terra_fallback_command(
    codex: str,
    workspace: str | Path,
    *,
    reasoning_effort: str = "high",
) -> list[str]:
    return [
        str(codex),
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--json",
        "-s",
        "workspace-write",
        "-C",
        str(workspace),
        "-m",
        "gpt-5.6-terra",
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-",
    ]

def _codex_capacity_log_start(nonce: str) -> str:
    return json.dumps(
        {
            "schema": (
                grok_cli_runner.
                AGENT_IMPLEMENTATION_CODEX_CAPACITY_LOG_SENTINEL_SCHEMA
            ),
            "type": "runner.capacity.start",
            "log_nonce": nonce,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

def _seal_auth_or_quota_route(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_or_auth_unavailable"
        ),
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        implementation_daemon,
        "_configured_agent_implementation_route_plan",
        lambda _repo_root: (
            llm_router._EAAEF_AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
        ),
    )

def _seal_quota_high_route(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV: "grok_cli",
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV: "codex",
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV: (
            "primary_quota_exhausted"
        ),
        implementation_daemon._GROK_MODEL_ENV: "grok-4.6",
        implementation_daemon._CODEX_MODEL_ENV: "gpt-5.6-terra",
        implementation_daemon._CODEX_REASONING_EFFORT_ENV: "high",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

def _install_fake_grok_docker_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    create_returncode: int = 0,
    create_stdout: bytes | None = None,
    cidfile_container_id: str | None = None,
) -> dict[str, object]:
    workspace = tmp_path / "workspace"
    provider_home = tmp_path / "asref-grok-home-test"
    lease_root = tmp_path / "asref-grok-container-test"
    workspace.mkdir()
    provider_home.mkdir(mode=0o700)
    lease_root.mkdir(mode=0o700)
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(mode=0o700)
    policy_path = provider_home / "sandbox.toml"
    policy_path.write_text("[profiles.test]\n", encoding="utf-8")
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    container_id = "d" * 64
    close_calls: list[bool] = []
    create_calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeHome:
        name = str(provider_home)

        def cleanup(self) -> None:
            return None

    class FakeCommandEnvironment:
        wrapper_path = "/opt/provider-command-wrapper"
        contract_sha256 = "sha256:" + "1" * 64
        formal_toolchain_contract_sha256 = "sha256:" + "2" * 64

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        container_name = "ipfs-accelerate-grok-1-" + "c" * 32

        def close(self, *, docker_run_finished: bool) -> None:
            close_calls.append(docker_run_finished)

    FakeLease.docker_config = docker_config
    FakeLease.cidfile = lease_root / "container.cid"
    FakeLease.lease_root = lease_root
    invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=FakeLease.container_name,
        lease_root=lease_root,
    )

    def fake_create_run(command, **kwargs):
        create_calls.append((list(command), dict(kwargs)))
        if create_returncode == 0:
            FakeLease.cidfile.write_text(
                (cidfile_container_id or container_id) + "\n",
                encoding="ascii",
            )
        return subprocess.CompletedProcess(
            command,
            create_returncode,
            stdout=(
                (container_id + "\n").encode("ascii")
                if create_stdout is None
                else create_stdout
            ),
            stderr=(b"create failed" if create_returncode else b""),
        )

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "sealed_provider_command_environment",
        lambda *_args, **_kwargs: FakeCommandEnvironment(),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        lambda **kwargs: (
            FakeHome(),
            {
                **kwargs["child_env"],
                "GROK_HOME": str(provider_home),
                "HOME": str(provider_home),
            },
            policy_path,
            (provider_home,),
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_DOCKER,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerContainerLease,
        "create",
        lambda *_args, **_kwargs: FakeLease(),
    )
    original_resolve_route = llm_router.resolve_agent_implementation_route
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route",
        lambda **kwargs: replace(
            original_resolve_route(**kwargs),
            invocation_binding=invocation,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **kwargs: (
            None
            if kwargs["profile"].authorization is not None
            else pytest.fail("network inspection accepted unsigned profile")
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_image_id",
        lambda *_args, **_kwargs: "sha256:" + "e" * 64,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_create_run)
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.chdir(workspace)
    return {
        "workspace": workspace,
        "grok": grok,
        "codex": codex,
        "container_id": container_id,
        "close_calls": close_calls,
        "create_calls": create_calls,
        "invocation": invocation,
        "network_profile": network_profile,
    }

def test_daemon_auth_or_quota_route_embeds_strict_terra_high_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV, raising=False
    )
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV, raising=False
    )
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok")
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/opt/providers/grok")
    monkeypatch.setattr(
        implementation_daemon, "_goose_meta_spark_available", lambda: False
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/opt/providers/codex" if name == "codex" else None,
    )
    monkeypatch.setattr(
        provider_executable_trust,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )

    daemon = _daemon(tmp_path)
    daemon._require_primary_provider_readiness(None)
    command = daemon._build_implementation_command(tmp_path)
    assert "--codex-fallback-command-json" in command
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[0] == "/opt/providers/codex"
    assert fallback[1] == "exec"
    assert "--ignore-user-config" in fallback
    assert "--ignore-rules" in fallback
    assert "--ephemeral" in fallback
    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    nonce = command[command.index("--grok-failure-receipt-nonce") + 1]
    assert len(nonce) == 64
    assert set(nonce) <= set("0123456789abcdef")
    head = command[: command.index("--codex-fallback-command-json")]
    assert fallback[0] not in head
    assert json.dumps(fallback, separators=(",", ":")) not in head

def test_daemon_auth_or_quota_route_embeds_strict_terra_high_fallback_aseh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ROUTE_ENABLED_ENV, raising=False
    )
    monkeypatch.delenv(
        implementation_daemon.PRODUCTION_PROVIDER_ALLOW_RAW_COMMAND_ENV, raising=False
    )
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
    monkeypatch.setattr(implementation_daemon, "_grok_binary", lambda: "/opt/providers/grok")
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/opt/providers/grok")
    monkeypatch.setattr(
        implementation_daemon, "_goose_meta_spark_available", lambda: False
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

    daemon = _daemon(tmp_path)
    daemon._require_primary_provider_readiness(None)
    command = daemon._build_implementation_command(tmp_path)
    assert "--codex-fallback-command-json" in command
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[0] == "/opt/providers/codex"
    assert fallback[1] == "exec"
    assert "--ignore-user-config" in fallback
    assert "--ignore-rules" in fallback
    assert "--ephemeral" in fallback
    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback
    nonce = command[command.index("--grok-failure-receipt-nonce") + 1]
    assert len(nonce) == 64
    assert set(nonce) <= set("0123456789abcdef")
    head = command[: command.index("--codex-fallback-command-json")]
    assert fallback[0] not in head
    assert json.dumps(fallback, separators=(",", ":")) not in head

def test_daemon_quota_high_route_reaches_exact_typed_fallback_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
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
        provider_executable_trust,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )

    daemon = _daemon(tmp_path)
    daemon._require_primary_provider_readiness(None)
    command = daemon._build_implementation_command(tmp_path)

    assert "--canonical-legacy-preflight-route" not in command
    assert Path(command[1]).name == "provider_fallback_runner.py"
    assert command[command.index("--primary-provider") + 1] == "grok"
    assert command[command.index("--fallback-provider") + 1] == "codex"
    assert command[command.index("--fallback-policy") + 1] == "grok_quota_only"
    primary = json.loads(command[command.index("--primary-command-json") + 1])
    fallback = json.loads(command[command.index("--fallback-command-json") + 1])
    assert primary[primary.index("--model") + 1] == "grok-4.6"
    assert "--require-terminal-quota-frame" in primary
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="high"' in fallback

def test_daemon_quota_only_route_requires_authenticated_grok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(implementation_daemon, "_grok_cli_available", lambda: False)
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
        provider_executable_trust,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/opt/providers/codex",
    )
    daemon = _daemon(tmp_path)

    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="quota-only.*ready Grok primary",
    ):
        daemon._require_primary_provider_readiness(None)
    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="quota-only.*ready Grok primary",
    ):
        daemon._build_implementation_command(tmp_path)

def test_daemon_quota_route_requires_trusted_codex_at_both_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
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
        provider_executable_trust,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "",
    )
    daemon = _daemon(tmp_path)

    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="requires the trusted Codex CLI fallback",
    ):
        daemon._require_primary_provider_readiness(None)
    with pytest.raises(
        RuntimeError,
        match="sealed Grok/Codex route requires a trusted Codex CLI",
    ):
        daemon._build_implementation_command(tmp_path)

def test_daemon_quota_route_keeps_trusted_codex_outside_sealed_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_quota_high_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    # The live sealed daemon PATH is /usr/bin:/bin, while the reviewed Codex
    # entry may live in /usr/local/bin and is resolved by the trust boundary.
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda _name: None,
    )
    monkeypatch.setattr(
        provider_executable_trust,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/usr/local/bin/codex",
    )

    daemon = _daemon(tmp_path)
    daemon._require_primary_provider_readiness(None)
    command = daemon._build_implementation_command(tmp_path)

    fallback = json.loads(command[command.index("--fallback-command-json") + 1])
    assert fallback[0] == "/usr/local/bin/codex"

@pytest.mark.parametrize("override_source", ("constructor", "environment"))
def test_auth_or_quota_route_rejects_raw_command_override_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_source: str,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    command = "codex exec -"
    if override_source == "environment":
        monkeypatch.setenv("IMPLEMENTATION_DAEMON_COMMAND", command)
        constructor_command = ""
    else:
        monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
        constructor_command = command
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Tasks\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=board,
        state_path=tmp_path / "state" / "task-state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        worktree_root=tmp_path,
        implementation_command=constructor_command,
    )

    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="sealed Grok/Codex route rejects",
    ):
        daemon._require_primary_provider_readiness(None)
    with pytest.raises(
        implementation_daemon.ImplementationRetryDeferred,
        match="sealed Grok/Codex route rejects",
    ):
        daemon._build_implementation_command(tmp_path)

def test_auth_or_quota_route_keeps_explicit_grok_task_grok_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
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
    task = PortalTask(
        task_id="AUTH-ROUTE-001",
        title="Stay on Grok",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"Provider role": "grok-only"},
    )
    daemon = _daemon(tmp_path)

    daemon._require_primary_provider_readiness(task)
    command = daemon._build_implementation_command(tmp_path, task=task)

    assert command[command.index("--model") + 1] == "grok-4.6"
    assert "--codex-fallback-command-json" not in command
    assert "--grok-failure-receipt-nonce" not in command

def test_auth_or_quota_route_denies_independent_codex_review_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_auth_or_quota_route(monkeypatch)
    monkeypatch.delenv("IMPLEMENTATION_DAEMON_COMMAND", raising=False)
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/opt/providers/grok",
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: f"/opt/providers/{name}",
    )
    task = PortalTask(
        task_id="AUTH-ROUTE-REVIEW-001",
        title="Keep review independent",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
        outputs=["src/provider.py"],
        metadata={"Provider role": "codex-review"},
    )
    daemon = _daemon(tmp_path)

    for action in (
        lambda: daemon._require_primary_provider_readiness(task),
        lambda: daemon._build_implementation_command(tmp_path, task=task),
    ):
        with pytest.raises(
            implementation_daemon.ImplementationRetryDeferred,
            match="requires independent Codex review",
        ):
            action()

def test_typed_preflight_process_uses_its_isolated_workspace_as_os_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    probe_home = tmp_path / "isolated-grok-home"

    class FakeIsolatedHome:
        name = str(probe_home)

        def cleanup(self) -> None:
            return None

    def fake_isolated_home(**kwargs):
        probe_home.mkdir(mode=0o700)
        captured["isolated_workspace"] = kwargs["workspace"]
        return (
            FakeIsolatedHome(),
            {"GROK_HOME": str(probe_home)},
            probe_home / "settings.json",
            (),
        )

    def fake_probe(command, *, env, cwd):
        captured["command"] = list(command)
        captured["env"] = dict(env)
        captured["cwd"] = cwd
        prompt_path = Path(command[command.index("--prompt-file") + 1])
        captured["prompt"] = prompt_path.read_text(encoding="utf-8")
        return 41, "Error: Not signed in", len("Error: Not signed in"), False

    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        fake_isolated_home,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_isolated_grok_quota_probe",
        fake_probe,
    )

    returncode, receipt, overflow = grok_cli_runner._run_typed_grok_preflight(
        grok_bin="/usr/local/bin/grok",
        base_env={},
        nonce="a" * 64,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert returncode == 41
    assert overflow is False
    assert receipt["failure_class"] == "authentication_unavailable"
    assert captured["cwd"] == captured["isolated_workspace"]
    assert captured["env"]["PWD"] == str(captured["cwd"])
    assert "OLDPWD" not in captured["env"]
    assert command[command.index("--cwd") + 1] == str(captured["cwd"])
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--max-turns") + 1] == "1"
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert captured["prompt"] == grok_cli_runner.GROK_QUOTA_PROBE_PROMPT
    assert not Path(captured["cwd"]).exists()

def _typed_preflight_attempt(
    stderr_text: str,
    *,
    nonce: str = "a" * 64,
    returncode: int = 41,
) -> tuple[int, dict[str, object], bool, str]:
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=stderr_text,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=returncode,
        primary_dispatched=False,
    )
    return returncode, receipt, False, stderr_text

def test_typed_preflight_receipts_are_bound_to_the_exact_route_model() -> None:
    nonce = "9" * 64
    receipts = {
        model: grok_cli_runner.build_grok_failure_receipt(
            probe_stderr_text="Error: Not signed in",
            nonce=nonce,
            model=model,
            probe_returncode=41,
            primary_dispatched=False,
        )
        for model in ("grok-4.5", "grok-4.6")
    }

    assert receipts["grok-4.5"]["probe_contract_id"] != receipts["grok-4.6"][
        "probe_contract_id"
    ]
    for model, receipt in receipts.items():
        assert provider_failure_policy.valid_grok_failure_receipt(
            receipt,
            nonce=nonce,
            model=model,
            returncode=41,
        )
        other_model = "grok-4.6" if model == "grok-4.5" else "grok-4.5"
        assert not provider_failure_policy.valid_grok_failure_receipt(
            receipt,
            nonce=nonce,
            model=other_model,
            returncode=41,
        )

def test_legacy_route_rejects_an_eaaef_model_receipt() -> None:
    nonce = "8" * 64
    route = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Error: Not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )

    decision = llm_router.decide_agent_implementation_fallback(
        route,
        repo_root=Path.cwd(),
        failure_receipt=receipt,
        expected_nonce=nonce,
        expected_model="grok-4.6",
        expected_probe_returncode=41,
    )

    assert decision.authorized is False
    assert decision.reason_code == "route_primary_model_mismatch"

def test_typed_preflight_retries_exact_max_turns_artifact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    results = iter(
        (
            _typed_preflight_attempt("Error: max turns reached\n"),
            (0, {}, False, ""),
        )
    )

    def fake_attempt(**kwargs):
        calls.append(dict(kwargs))
        return next(results)

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )

    result = grok_cli_runner._run_typed_grok_preflight(
        grok_bin="/usr/local/bin/grok",
        base_env={},
        nonce="a" * 64,
    )

    assert result == (0, {}, False)
    assert len(calls) == 2
    assert calls[0] == calls[1]

@pytest.mark.parametrize(
    "stderr_text",
    (
        "Error: max turns reached",
        "error: max turns reached\n",
        "Error: max turns reached\n\n",
        "Error: Not signed in",
        "Grok Build usage balance exhausted",
    ),
)
def test_typed_preflight_does_not_retry_near_match_auth_or_quota(
    monkeypatch: pytest.MonkeyPatch,
    stderr_text: str,
) -> None:
    calls = 0
    attempt = _typed_preflight_attempt(stderr_text)

    def fake_attempt(**_kwargs):
        nonlocal calls
        calls += 1
        return attempt

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )

    result = grok_cli_runner._run_typed_grok_preflight(
        grok_bin="/usr/local/bin/grok",
        base_env={},
        nonce="a" * 64,
    )

    assert result == attempt[:3]
    assert calls == 1

def test_repeated_exact_max_turns_is_one_unknown_denial_without_terra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    nonce = "a" * 64
    attempt = _typed_preflight_attempt(
        "Error: max turns reached\n",
        nonce=nonce,
    )
    calls = 0

    def fake_attempt(**_kwargs):
        nonlocal calls
        calls += 1
        return attempt

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: "clean",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "b" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: pytest.fail("unknown evidence must not run verifier"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail("Terra must remain forbidden"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: pytest.fail("task Grok must not run after denial"),
    )
    route_plan = llm_router._EAAEF_AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router._agent_implementation_route,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(_terra_fallback_command(str(codex), workspace)),
            "--codex-fallback-reasoning-effort",
            "high",
            "--grok-failure-receipt-nonce",
            nonce,
            "--agent-implementation-route-json",
            json.dumps(route_plan.as_binding_dict()),
        ]
    )

    assert result == 41
    assert calls == 2
    rendered = capsys.readouterr().err
    assert rendered.count(grok_cli_runner.GROK_FAILURE_RECEIPT_PREFIX) == 1
    outcomes = provider_failure_policy.extract_grok_route_outcomes(rendered)
    assert len(outcomes) == 1
    assert outcomes[0]["decision"] == "denied"
    assert outcomes[0]["failure_class"] == "unknown"
    assert outcomes[0]["fallback_dispatched"] is False

def test_repeated_exact_max_turns_is_one_unknown_denial_without_terra_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    nonce = "a" * 64
    attempt = _typed_preflight_attempt(
        "Error: max turns reached\n",
        nonce=nonce,
    )
    calls = 0

    def fake_attempt(**_kwargs):
        nonlocal calls
        calls += 1
        return attempt

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("implement"))
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight_once",
        fake_attempt,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: "clean",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "b" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: pytest.fail("unknown evidence must not run verifier"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail("Terra must remain forbidden"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: pytest.fail("task Grok must not run after denial"),
    )
    route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router._agent_implementation_route,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(_terra_fallback_command(str(codex), workspace)),
            "--codex-fallback-reasoning-effort",
            "high",
            "--grok-failure-receipt-nonce",
            nonce,
            "--agent-implementation-route-json",
            json.dumps(route_plan.as_binding_dict()),
        ]
    )

    assert result == 41
    assert calls == 2
    rendered = capsys.readouterr().err
    assert rendered.count(grok_cli_runner.GROK_FAILURE_RECEIPT_PREFIX) == 1
    outcomes = provider_failure_policy.extract_grok_route_outcomes(rendered)
    assert len(outcomes) == 1
    assert outcomes[0]["decision"] == "denied"
    assert outcomes[0]["failure_class"] == "unknown"
    assert outcomes[0]["fallback_dispatched"] is False

def test_scoped_route_rejects_prompt_cid_before_grok_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    invocation = SimpleNamespace(
        prompt_cid=grok_cli_runner._agent_prompt_cid("signed prompt"),
        control_plane=object(),
        provider_attempt_store=str(tmp_path / "attempt-store"),
        provider_attempt_store_identity="sha256:" + "a" * 64,
        logical_attempt_id="sha256:" + "b" * 64,
    )
    route_plan = SimpleNamespace(
        invocation_binding=invocation,
        primary_model_id="grok-4.6",
        fallback_reasoning_effort="high",
    )

    class EmptyAttemptStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def read(self, _logical_attempt_id: str) -> None:
            return None

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("wrong prompt"))
    monkeypatch.setattr(grok_cli_runner.sys, "argv", ["/proc/self/fd/71"])
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin, _descriptor: "/proc/self/fd/71",
    )
    monkeypatch.setattr(
        provider_attempt_store,
        "DurableProviderAttemptCAS",
        EmptyAttemptStore,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "c" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        lambda **_kwargs: pytest.fail(
            "prompt CID mismatch must stop before Grok preflight"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(_terra_fallback_command(str(codex), workspace)),
            "--codex-fallback-reasoning-effort",
            "high",
            "--grok-failure-receipt-nonce",
            "a" * 64,
            "--agent-implementation-route-json",
            "{}",
        ]
    )

    assert result == 2
    assert "does not match the task prompt" in capsys.readouterr().err

def test_scoped_route_rejects_prompt_cid_before_grok_preflight_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    grok = tmp_path / "grok"
    codex = tmp_path / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    invocation = SimpleNamespace(
        prompt_cid=grok_cli_runner._agent_prompt_cid("signed prompt"),
        control_plane=object(),
        provider_attempt_store=str(tmp_path / "attempt-store"),
        provider_attempt_store_identity="sha256:" + "a" * 64,
        logical_attempt_id="sha256:" + "b" * 64,
    )
    route_plan = SimpleNamespace(
        invocation_binding=invocation,
        fallback_reasoning_effort="high",
    )

    class EmptyAttemptStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def read(self, _logical_attempt_id: str) -> None:
            return None

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("wrong prompt"))
    monkeypatch.setattr(grok_cli_runner.sys, "argv", ["/proc/self/fd/71"])
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router,
        "verify_agent_implementation_sealed_control_plane",
        lambda _pin, _descriptor: "/proc/self/fd/71",
    )
    monkeypatch.setattr(
        provider_attempt_store,
        "DurableProviderAttemptCAS",
        EmptyAttemptStore,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "c" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        lambda **_kwargs: pytest.fail(
            "prompt CID mismatch must stop before Grok preflight"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(_terra_fallback_command(str(codex), workspace)),
            "--codex-fallback-reasoning-effort",
            "high",
            "--grok-failure-receipt-nonce",
            "a" * 64,
            "--agent-implementation-route-json",
            "{}",
        ]
    )

    assert result == 2
    assert "does not match the task prompt" in capsys.readouterr().err

def test_typed_preflight_probe_overflow_is_measured_fail_closed(
    tmp_path: Path,
) -> None:
    guidance = GROK_NOT_SIGNED_IN_GUIDANCE.encode("utf-8")
    code = (
        "import os\n"
        "payload = b'HTTP 429\\n' + "
        f"(b'x' * {grok_cli_runner.MAX_GROK_FAILURE_EVIDENCE_BYTES + 4096}) + "
        f"b'\\n' + {guidance!r}\n"
        "os.write(2, payload)\n"
        "raise SystemExit(41)\n"
    )

    returncode, retained, evidence_size, overflow = (
        grok_cli_runner._run_isolated_grok_quota_probe(
            [grok_cli_runner.sys.executable, "-c", code],
            env=os.environ.copy(),
            cwd=tmp_path,
        )
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=retained,
        nonce="f" * 64,
        model="grok-4.6",
        probe_returncode=returncode,
        primary_dispatched=False,
        evidence_size=evidence_size,
        evidence_overflow=overflow,
    )

    assert returncode == 41
    assert evidence_size > grok_cli_runner.MAX_GROK_FAILURE_EVIDENCE_BYTES
    assert overflow is True
    assert receipt["evidence_overflow"] is True
    assert len(retained.encode("utf-8")) <= (
        grok_cli_runner.MAX_GROK_FAILURE_EVIDENCE_BYTES
    )

def test_independent_quota_verifier_uses_isolated_os_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    verifier_home = tmp_path / "verifier-home"
    inherited_temp = tmp_path.joinpath(*(["long-inherited-temp-root"] * 12))
    inherited_temp.mkdir(parents=True)
    monkeypatch.setattr(grok_cli_runner.tempfile, "tempdir", str(inherited_temp))

    class FakeIsolatedHome:
        name = str(verifier_home)

        def cleanup(self) -> None:
            return None

    def fake_isolated_home(**_kwargs):
        verifier_home.mkdir(mode=0o700)
        return (
            FakeIsolatedHome(),
            {"GROK_HOME": str(verifier_home), "OLDPWD": "/task/worktree"},
            verifier_home / "settings.json",
            (),
        )

    def fake_run(command, **kwargs):
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = dict(kwargs["env"])
        captured["umask"] = kwargs["umask"]
        session_id = command[command.index("--session-id") + 1]
        _write_native_session_home(
            verifier_home,
            [
                _spending_limit_retry(session_id=session_id),
                _spending_limit_terminal(session_id=session_id),
            ],
            session_id=session_id,
            workspace=Path(kwargs["cwd"]),
        )
        return subprocess.CompletedProcess(command, 23)

    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="1" * 64,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        fake_isolated_home,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    evidence = grok_cli_runner._independently_verify_grok_quota(
        grok_bin="/opt/providers/grok",
        base_env={},
        failure_receipt=receipt,
    )

    assert evidence is not None
    assert captured["env"]["PWD"] == str(captured["cwd"])
    assert "OLDPWD" not in captured["env"]
    assert captured["umask"] == 0o077
    verifier_workspace = Path(captured["cwd"])
    assert verifier_workspace.parent.parent == Path("/tmp")
    assert len(
        urllib.parse.quote(str(verifier_workspace), safe="").encode("utf-8")
    ) <= 255

def test_independent_quota_verifier_uses_isolated_os_cwd_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    verifier_home = tmp_path / "verifier-home"

    class FakeIsolatedHome:
        name = str(verifier_home)

        def cleanup(self) -> None:
            return None

    def fake_isolated_home(**_kwargs):
        verifier_home.mkdir(mode=0o700)
        return (
            FakeIsolatedHome(),
            {"GROK_HOME": str(verifier_home), "OLDPWD": "/task/worktree"},
            verifier_home / "settings.json",
            (),
        )

    def fake_run(command, **kwargs):
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = dict(kwargs["env"])
        session_id = command[command.index("--session-id") + 1]
        _write_native_session_home(
            verifier_home,
            [
                _spending_limit_retry(session_id=session_id),
                _spending_limit_terminal(session_id=session_id),
            ],
            session_id=session_id,
            workspace=Path(kwargs["cwd"]),
        )
        return subprocess.CompletedProcess(command, 23)

    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="1" * 64,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_grok_home",
        fake_isolated_home,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    evidence = grok_cli_runner._independently_verify_grok_quota(
        grok_bin="/opt/providers/grok",
        base_env={},
        failure_receipt=receipt,
    )

    assert isinstance(evidence, llm_router.AgentImplementationQuotaEvidence)
    assert evidence.verifier_result == "spending_limit_exhausted"
    assert captured["env"]["PWD"] == str(captured["cwd"])
    assert "OLDPWD" not in captured["env"]

@pytest.mark.parametrize("verifier_returncode", (1, 41))
def test_native_quota_verifier_rejects_ambiguous_session_layouts(
    tmp_path: Path,
    verifier_returncode: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verifier_home = tmp_path / "verifier-home"
    updates = [_spending_limit_retry(), _spending_limit_terminal()]
    _write_native_session_home(verifier_home, updates)
    _write_native_session_home(
        verifier_home,
        updates,
        workspace=workspace,
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="1" * 64,
        model="grok-4.6",
        probe_returncode=verifier_returncode,
        primary_dispatched=False,
    )

    assert (
        llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=verifier_returncode,
            failure_receipt=receipt,
            verifier_workspace=workspace,
        )
        is None
    )

def test_native_quota_verifier_rejects_foreign_workspace_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    foreign_workspace = tmp_path / "foreign-workspace"
    workspace.mkdir()
    foreign_workspace.mkdir()
    verifier_home = tmp_path / "grok-home"
    _write_native_session_home(
        verifier_home,
        [_spending_limit_retry(), _spending_limit_terminal()],
        workspace=foreign_workspace,
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="2" * 64,
        model="grok-4.6",
        probe_returncode=1,
        primary_dispatched=False,
    )

    evidence = llm_router.validate_agent_implementation_quota_evidence(
        grok_home=verifier_home,
        expected_session_id=_NATIVE_SESSION_ID,
        verifier_returncode=1,
        failure_receipt=receipt,
        verifier_workspace=workspace,
    )

    assert evidence is None

def test_native_quota_verifier_rejects_direct_session_wrong_cwd(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    foreign_workspace = tmp_path / "foreign-workspace"
    workspace.mkdir()
    foreign_workspace.mkdir()
    verifier_home = tmp_path / "grok-home"
    _write_native_session_home(
        verifier_home,
        [_spending_limit_retry(), _spending_limit_terminal()],
        workspace=foreign_workspace,
        encoded_workspace=False,
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="5" * 64,
        model="grok-4.6",
        probe_returncode=1,
        primary_dispatched=False,
    )

    evidence = llm_router.validate_agent_implementation_quota_evidence(
        grok_home=verifier_home,
        expected_session_id=_NATIVE_SESSION_ID,
        verifier_returncode=1,
        failure_receipt=receipt,
        verifier_workspace=workspace,
    )

    assert evidence is None

def test_native_quota_verifier_rejects_workspace_traversal_alias(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    verifier_home = tmp_path / "grok-home"
    _write_native_session_home(
        verifier_home,
        [_spending_limit_retry(), _spending_limit_terminal()],
        workspace=workspace,
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce="4" * 64,
        model="grok-4.6",
        probe_returncode=1,
        primary_dispatched=False,
    )
    traversal_alias = tmp_path / "missing" / ".." / workspace.name

    evidence = llm_router.validate_agent_implementation_quota_evidence(
        grok_home=verifier_home,
        expected_session_id=_NATIVE_SESSION_ID,
        verifier_returncode=1,
        failure_receipt=receipt,
        verifier_workspace=traversal_alias,
    )

    assert evidence is None

def test_quota_fallback_command_rejects_model_or_effort_drift() -> None:
    valid = _terra_fallback_command("/usr/local/bin/codex", "/repo")
    assert (
        grok_cli_runner._parse_codex_fallback_command(
            json.dumps(valid),
            expected_fallback_reasoning_effort="high",
        )
        == valid
    )
    # Parsing without a route-bound expectation validates the closed argv
    # shape only.  Dispatch callers always supply their sealed route effort.
    assert grok_cli_runner._parse_codex_fallback_command(json.dumps(valid)) == valid
    model_drift = list(valid)
    model_drift[model_drift.index("-m") + 1] = "gpt-5.6-sol"
    effort_drift = list(valid)
    effort_idx = next(
        i for i, item in enumerate(effort_drift) if "model_reasoning_effort=" in item
    )
    effort_drift[effort_idx] = 'model_reasoning_effort="low"'
    for drifted in (model_drift, effort_drift):
        with pytest.raises(ValueError):
            grok_cli_runner._parse_codex_fallback_command(
                json.dumps(drifted),
                expected_fallback_reasoning_effort="high",
            )

def test_quota_fallback_command_rejects_model_or_effort_drift_aseh() -> None:
    valid = _terra_fallback_command("/usr/local/bin/codex", "/repo")
    assert (
        grok_cli_runner._parse_codex_fallback_command(
            json.dumps(valid),
            expected_fallback_reasoning_effort="high",
        )
        == valid
    )
    with pytest.raises(
        ValueError,
        match="reasoning does not match the sealed provider route",
    ):
        grok_cli_runner._parse_codex_fallback_command(json.dumps(valid))
    model_drift = list(valid)
    model_drift[model_drift.index("-m") + 1] = "gpt-5.6-sol"
    effort_drift = list(valid)
    effort_idx = next(
        i for i, item in enumerate(effort_drift) if "model_reasoning_effort=" in item
    )
    effort_drift[effort_idx] = 'model_reasoning_effort="low"'
    for drifted in (model_drift, effort_drift):
        with pytest.raises(ValueError):
            grok_cli_runner._parse_codex_fallback_command(
                json.dumps(drifted),
                expected_fallback_reasoning_effort="high",
            )

def test_quota_fallback_rejects_workspace_codex_executable(tmp_path: Path) -> None:
    attacker = tmp_path / "codex"
    attacker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    attacker.chmod(0o700)
    fallback = _terra_fallback_command(str(attacker), tmp_path)
    with pytest.raises(ValueError):
        grok_cli_runner._validate_codex_quota_fallback_command(
            fallback, workspace=tmp_path
        )

def test_quota_classifier_is_fail_closed_for_incomplete_diagnostics() -> None:
    assert (
        grok_cli_runner._grok_quota_exhausted(
            "Grok implementation failed\n"
            "usage balance exhausted\n"
            "402 Payment Required"
        )
        is False
    )
    assert (
        grok_cli_runner._grok_quota_exhausted(
            '{"provider":"xAI","error":{"type":"insufficient_quota"}}'
        )
        is False
    )

def test_quota_classifier_accepts_exact_balance_exhausted_envelope() -> None:
    transcript = (
        'Internal error: {"message":"API error (status 402 Payment Required): '
        'Grok Build usage balance exhausted","http_status":402}'
    )
    assert grok_cli_runner._grok_quota_exhausted(transcript) is True
    parsed = grok_cli_runner.parse_grok_quota_error(transcript)
    assert parsed["kind"] == "usage_balance_exhausted"
    assert parsed["http_status"] == 402

def test_direct_no_nonce_native_quota_cannot_cross_providers(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    fallback = _terra_fallback_command(
        str(codex),
        workspace,
        reasoning_effort="medium",
    )
    prompt = "repair the failed implementation"

    def fake_primary(command, *, env) -> int:
        session_id = command[command.index("--session-id") + 1]
        _write_native_session_home(
            Path(env["GROK_HOME"]),
            [
                _spending_limit_retry(session_id=session_id),
                _native_update(
                    {
                        "sessionUpdate": "user_message_chunk",
                        "content": prompt,
                    },
                    session_id=session_id,
                ),
                _spending_limit_terminal(session_id=session_id),
            ],
            session_id=session_id,
        )
        return 23

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_GROK_SANDBOX,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_primary,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: pytest.fail(
            "direct no-nonce route must not run a fallback verifier"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "direct no-nonce route must not cross providers"
        ),
    )
    monkeypatch.chdir(workspace)

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == 23
    assert "Direct no-nonce Grok failure cannot authorize" in capsys.readouterr().err

def test_direct_no_nonce_native_quota_cannot_cross_providers_aseh(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    fallback = _terra_fallback_command(
        str(codex),
        workspace,
        reasoning_effort="medium",
    )
    prompt = "repair the failed implementation"

    def fake_primary(command, *, env, provider_stdin=None) -> int:
        assert provider_stdin is None
        session_id = command[command.index("--session-id") + 1]
        _write_native_session_home(
            Path(env["GROK_HOME"]),
            [
                _spending_limit_retry(session_id=session_id),
                _native_update(
                    {
                        "sessionUpdate": "user_message_chunk",
                        "content": prompt,
                    },
                    session_id=session_id,
                ),
                _spending_limit_terminal(session_id=session_id),
            ],
            session_id=session_id,
        )
        return 23

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: grok_cli_runner.GROK_ISOLATION_GROK_SANDBOX,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_primary,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        lambda **_kwargs: pytest.fail(
            "direct no-nonce route must not run a fallback verifier"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "direct no-nonce route must not cross providers"
        ),
    )
    monkeypatch.chdir(workspace)

    returncode = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
        ]
    )

    assert returncode == 23
    assert "Direct no-nonce Grok failure cannot authorize" in capsys.readouterr().err

@pytest.mark.parametrize(
    ("nonce", "route_binding"),
    (
        ("ab" * 32, ""),
        ("", "{}"),
        ("ab" * 32, "{}"),
    ),
)
def test_typed_route_metadata_without_codex_fallback_is_rejected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    nonce: str,
    route_binding: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    argv = ["--workspace", str(workspace)]
    if nonce:
        argv.extend(["--grok-failure-receipt-nonce", nonce])
    if route_binding:
        argv.extend(["--agent-implementation-route-json", route_binding])

    assert grok_cli_runner.main(argv) == 2
    assert (
        "typed Grok route requires a Codex fallback command"
        in capsys.readouterr().err
    )

def test_merge_resolver_marker_mints_fresh_legacy_preflight_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: str(grok))
    wrapper_command = shlex.split(
        merge_resolver_fallback.llm_merge_resolver_fallback_command(
            python_executable="python-test"
        )
    )
    assert wrapper_command == [
        "python-test",
        "-m",
        (
            "ipfs_accelerate_py.agent_supervisor.integrations."
            "llm_merge_resolver_fallback"
        ),
    ]
    prompt = "resolve the merge conflict"
    preflight_nonces: list[str] = []
    fallback_calls: list[list[str]] = []

    def fake_preflight(**kwargs):
        nonce = str(kwargs["nonce"])
        model = str(kwargs["model"])
        preflight_nonces.append(nonce)
        receipt = grok_cli_runner.build_grok_failure_receipt(
            probe_stderr_text="Grok Build usage balance exhausted",
            nonce=nonce,
            model=model,
            probe_returncode=41,
            primary_dispatched=False,
        )
        return 41, receipt, False

    def fake_verifier(**kwargs) -> object:
        verifier_home = _write_native_session(
            tmp_path / "legacy-independent-verifier",
            [_spending_limit_retry(), _spending_limit_terminal()],
            model="grok-4.5",
        )
        return llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=41,
            failure_receipt=kwargs["failure_receipt"],
        )

    def fake_fallback(command, **kwargs) -> int:
        fallback_calls.append(list(command))
        assert kwargs["effect_claim"] is None
        assert kwargs["effect_terminal"] is None
        assert kwargs["capacity_evidence"] is None
        kwargs["pre_effect_validator"]()
        return 0

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setenv("AGENT_RESOLVER_LOCK_BYPASS", "1")
    monkeypatch.setenv(merge_resolver_fallback._INVOCATION_DEPTH_ENV, "0")
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        fake_preflight,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        fake_verifier,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: "clean",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "a" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy fallback must run before the task Grok process"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )

    assert merge_resolver_fallback.main([str(workspace)]) == 0
    assert len(preflight_nonces) == 1
    assert re.fullmatch(r"[0-9a-f]{64}", preflight_nonces[0])
    assert len(fallback_calls) == 1
    assert 'model_reasoning_effort="medium"' in fallback_calls[0]
    assert "quota is exhausted" in capsys.readouterr().err

def test_merge_resolver_marker_mints_fresh_legacy_preflight_route_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: str(grok))
    wrapper_command = shlex.split(
        merge_resolver_fallback.llm_merge_resolver_fallback_command(
            python_executable="python-test"
        )
    )
    assert wrapper_command == [
        "python-test",
        "-m",
        (
            "ipfs_accelerate_py.agent_supervisor.integrations."
            "llm_merge_resolver_fallback"
        ),
    ]
    prompt = "resolve the merge conflict"
    preflight_nonces: list[str] = []
    fallback_calls: list[list[str]] = []

    def fake_preflight(**kwargs):
        nonce = str(kwargs["nonce"])
        preflight_nonces.append(nonce)
        receipt = grok_cli_runner.build_grok_failure_receipt(
            probe_stderr_text="Grok Build usage balance exhausted",
            nonce=nonce,
            model="grok-4.6",
            probe_returncode=41,
            primary_dispatched=False,
        )
        return 41, receipt, False

    def fake_verifier(**kwargs) -> object:
        verifier_home = _write_native_session(
            tmp_path / "legacy-independent-verifier",
            [_spending_limit_retry(), _spending_limit_terminal()],
        )
        return llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=41,
            failure_receipt=kwargs["failure_receipt"],
        )

    def fake_fallback(command, **kwargs) -> int:
        fallback_calls.append(list(command))
        kwargs["pre_effect_validator"]()
        return 0

    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO(prompt))
    monkeypatch.setenv("AGENT_RESOLVER_LOCK_BYPASS", "1")
    monkeypatch.setenv(merge_resolver_fallback._INVOCATION_DEPTH_ENV, "0")
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        fake_preflight,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        fake_verifier,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: "clean",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "a" * 40,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy fallback must run before the task Grok process"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )

    assert merge_resolver_fallback.main([str(workspace)]) == 0
    assert len(preflight_nonces) == 1
    assert re.fullmatch(r"[0-9a-f]{64}", preflight_nonces[0])
    assert len(fallback_calls) == 1
    assert 'model_reasoning_effort="medium"' in fallback_calls[0]
    assert "quota is exhausted" in capsys.readouterr().err

@pytest.mark.parametrize(
    "external_route_arguments",
    (
        ("--grok-failure-receipt-nonce", "a" * 64),
        ("--agent-implementation-route-json", "{}"),
    ),
)
def test_legacy_preflight_marker_rejects_external_route_authority(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    external_route_arguments: tuple[str, str],
) -> None:
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    assert (
        grok_cli_runner.main(
            [
                "--workspace",
                str(tmp_path),
                "--codex-fallback-command-json",
                json.dumps(
                    _terra_fallback_command(
                        str(codex),
                        tmp_path,
                        reasoning_effort="medium",
                    )
                ),
                grok_cli_runner.CANONICAL_LEGACY_PREFLIGHT_ROUTE_FLAG,
                *external_route_arguments,
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    if external_route_arguments[0] == "--grok-failure-receipt-nonce":
        assert "cannot be combined with an external nonce" in error
    else:
        assert "legacy quota route forbids an auth/high route binding" in error

@pytest.mark.parametrize(
    (
        "probe_stderr",
        "fingerprints",
        "verifier_failure_type",
        "expected_returncode",
        "expected_fallback_count",
        "expected_verifier_count",
    ),
    (
        (
            "Error: Not signed in",
            ("clean", "clean", "clean", "clean"),
            "",
            0,
            1,
            0,
        ),
        (
            GROK_NOT_SIGNED_IN_GUIDANCE,
            ("clean", "clean", "clean", "clean"),
            "",
            0,
            1,
            0,
        ),
        ("Error: Not signed in", ("clean", "mutated"), "", 41, 0, 0),
        ("HTTP 429 Too Many Requests", ("clean",), "", 41, 0, 0),
        ("HTTP 403", ("clean",), "", 41, 0, 1),
        ("Forbidden", ("clean",), "", 41, 0, 1),
        ("Not signed in\nHTTP 429", ("clean",), "", 41, 0, 1),
        ("Not signed in\nHTTP 403", ("clean",), "", 41, 0, 1),
        ("Forbidden\nNot signed in", ("clean",), "", 41, 0, 1),
        (
            GROK_NOT_SIGNED_IN_GUIDANCE + "\nHTTP 429",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            "HTTP 429\n" + GROK_NOT_SIGNED_IN_GUIDANCE,
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            "Not signed in\nGrok Build usage balance exhausted",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            _SPENDING_LIMIT_MESSAGE,
            ("clean", "clean", "clean", "clean"),
            "spending_limit_exhausted",
            0,
            1,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean", "clean", "clean", "clean"),
            "spending_limit_exhausted",
            0,
            1,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean", "clean", "clean", "mutated"),
            "spending_limit_exhausted",
            41,
            0,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
    ),
)
def test_typed_preflight_requires_independent_quota_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    probe_stderr: str,
    fingerprints: tuple[str, ...],
    verifier_failure_type: str,
    expected_returncode: int,
    expected_fallback_count: int,
    expected_verifier_count: int,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    nonce = "a" * 64
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=probe_stderr,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    fallback = _terra_fallback_command(str(codex), workspace)
    fallback_calls: list[list[str]] = []
    verifier_calls: list[dict[str, object]] = []
    preflight_calls: list[dict[str, object]] = []
    fingerprint_values = iter(fingerprints)
    route_plan = llm_router._EAAEF_AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE

    class PreflightOrderedStdin(io.StringIO):
        def read(self, *args, **kwargs) -> str:
            assert preflight_calls, "task prompt was read before Grok preflight"
            return super().read(*args, **kwargs)

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        PreflightOrderedStdin("implement"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: pytest.fail(
            "authorized typed preflight must skip task-Grok isolation"
        ),
    )
    def fake_preflight(**kwargs):
        preflight_calls.append(dict(kwargs))
        return 41, receipt, False

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        fake_preflight,
    )

    def fake_verifier(**kwargs) -> object:
        verifier_calls.append(dict(kwargs))
        if verifier_failure_type != "spending_limit_exhausted":
            return None
        verifier_home = _write_native_session(
            tmp_path / "independent-verifier",
            [_spending_limit_retry(), _spending_limit_terminal()],
        )
        return llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=41,
            failure_receipt=kwargs["failure_receipt"],
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        fake_verifier,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: next(fingerprint_values),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "task Grok must not run after a failed typed preflight"
        ),
    )

    def fake_fallback(command, **kwargs) -> int:
        kwargs["pre_effect_validator"]()
        fallback_calls.append(list(command))
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router._agent_implementation_route,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "b" * 40,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
            "--codex-fallback-reasoning-effort",
            "high",
            "--grok-failure-receipt-nonce",
            nonce,
            "--agent-implementation-route-json",
            json.dumps(route_plan.as_binding_dict()),
        ]
    )

    assert result == expected_returncode
    assert len(fallback_calls) == expected_fallback_count
    assert len(verifier_calls) == expected_verifier_count
    rendered = capsys.readouterr().err
    assert grok_cli_runner.GROK_FAILURE_RECEIPT_PREFIX in rendered
    if expected_fallback_count:
        assert 'model_reasoning_effort="high"' in fallback_calls[0]
        expected_reason = (
            "authentication is unavailable"
            if receipt["failure_class"] == "authentication_unavailable"
            else "quota is exhausted"
        )
        assert expected_reason in rendered
    else:
        assert "Codex fallback is forbidden" in rendered

@pytest.mark.parametrize(
    (
        "probe_stderr",
        "fingerprints",
        "verifier_failure_type",
        "expected_returncode",
        "expected_fallback_count",
        "expected_verifier_count",
    ),
    (
        (
            "Error: Not signed in",
            ("clean", "clean", "clean", "clean"),
            "",
            0,
            1,
            0,
        ),
        (
            GROK_NOT_SIGNED_IN_GUIDANCE,
            ("clean", "clean", "clean", "clean"),
            "",
            0,
            1,
            0,
        ),
        ("Error: Not signed in", ("clean", "mutated"), "", 41, 0, 0),
        ("HTTP 429 Too Many Requests", ("clean",), "", 41, 0, 0),
        ("HTTP 403", ("clean",), "", 41, 0, 1),
        ("Forbidden", ("clean",), "", 41, 0, 1),
        ("Not signed in\nHTTP 429", ("clean",), "", 41, 0, 1),
        ("Not signed in\nHTTP 403", ("clean",), "", 41, 0, 1),
        ("Forbidden\nNot signed in", ("clean",), "", 41, 0, 1),
        (
            GROK_NOT_SIGNED_IN_GUIDANCE + "\nHTTP 429",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            "HTTP 429\n" + GROK_NOT_SIGNED_IN_GUIDANCE,
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            "Not signed in\nGrok Build usage balance exhausted",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
        (
            _SPENDING_LIMIT_MESSAGE,
            ("clean", "clean", "clean", "clean"),
            "spending_limit_exhausted",
            0,
            1,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean", "clean", "clean", "clean"),
            "spending_limit_exhausted",
            0,
            1,
            1,
        ),
        (
            "Grok Build usage balance exhausted",
            ("clean",),
            "",
            41,
            0,
            1,
        ),
    ),
)
def test_typed_preflight_requires_independent_quota_confirmation_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    probe_stderr: str,
    fingerprints: tuple[str, ...],
    verifier_failure_type: str,
    expected_returncode: int,
    expected_fallback_count: int,
    expected_verifier_count: int,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    workspace.mkdir()
    provider_bin.mkdir()
    grok = provider_bin / "grok"
    codex = provider_bin / "codex"
    for executable in (grok, codex):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)

    nonce = "a" * 64
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=probe_stderr,
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    fallback = _terra_fallback_command(str(codex), workspace)
    fallback_calls: list[list[str]] = []
    fallback_authorities: list[dict[str, object]] = []
    verifier_calls: list[dict[str, object]] = []
    preflight_calls: list[dict[str, object]] = []
    fingerprint_values = iter(fingerprints)
    route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE

    class PreflightOrderedStdin(io.StringIO):
        def read(self, *args, **kwargs) -> str:
            assert preflight_calls, "task prompt was read before Grok preflight"
            return super().read(*args, **kwargs)

    monkeypatch.setattr(
        grok_cli_runner.sys,
        "stdin",
        PreflightOrderedStdin("implement"),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_select_grok_isolation_backend",
        lambda **_kwargs: pytest.fail(
            "authorized typed preflight must skip task-Grok isolation"
        ),
    )
    def fake_preflight(**kwargs):
        preflight_calls.append(dict(kwargs))
        return 41, receipt, False

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_typed_grok_preflight",
        fake_preflight,
    )

    def fake_verifier(**kwargs) -> object:
        verifier_calls.append(dict(kwargs))
        if verifier_failure_type != "spending_limit_exhausted":
            return None
        verifier_home = _write_native_session(
            tmp_path / "independent-verifier",
            [_spending_limit_retry(), _spending_limit_terminal()],
        )
        return llm_router.validate_agent_implementation_quota_evidence(
            grok_home=verifier_home,
            expected_session_id=_NATIVE_SESSION_ID,
            verifier_returncode=41,
            failure_receipt=kwargs["failure_receipt"],
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_independently_verify_grok_quota",
        fake_verifier,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_workspace_content_fingerprint",
        lambda _workspace: next(fingerprint_values),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "task Grok must not run after a failed typed preflight"
        ),
    )

    def fake_fallback(command, **kwargs) -> int:
        fallback_calls.append(list(command))
        fallback_authorities.append(
            {
                "effect_claim": kwargs.get("effect_claim"),
                "effect_terminal": kwargs.get("effect_terminal"),
            }
        )
        kwargs["pre_effect_validator"]()
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        fake_fallback,
    )
    monkeypatch.setattr(
        llm_router,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        llm_router._agent_implementation_route,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_repository_head",
        lambda _workspace: "b" * 40,
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(workspace),
            "--grok-bin",
            str(grok),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(fallback),
            "--codex-fallback-reasoning-effort",
            "high",
            "--grok-failure-receipt-nonce",
            nonce,
            "--agent-implementation-route-json",
            json.dumps(route_plan.as_binding_dict()),
        ]
    )

    assert result == expected_returncode
    assert len(fallback_calls) == expected_fallback_count
    assert len(verifier_calls) == expected_verifier_count
    rendered = capsys.readouterr().err
    assert grok_cli_runner.GROK_FAILURE_RECEIPT_PREFIX in rendered
    if expected_fallback_count:
        assert 'model_reasoning_effort="high"' in fallback_calls[0]
        assert fallback_authorities == [
            {"effect_claim": None, "effect_terminal": None}
        ]
        expected_reason = (
            "authentication is unavailable"
            if receipt["failure_class"] == "authentication_unavailable"
            else "quota is exhausted"
        )
        assert expected_reason in rendered
    else:
        assert "Codex fallback is forbidden" in rendered

def test_terminal_route_outcome_is_bound_to_receipt_route_and_runner_exit() -> None:
    nonce = "c" * 64
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce=nonce,
        model="grok-4.5",
        probe_returncode=41,
        primary_dispatched=False,
    )
    route = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    ).as_binding_dict()
    evidence_id = "sha256:" + "d" * 64
    cases = (
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                quota_evidence_id=evidence_id,
                decision="fallback_succeeded",
                verifier_status="confirmed_quota",
                fallback_dispatched=True,
                fallback_returncode=0,
            ),
            0,
        ),
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                quota_evidence_id=evidence_id,
                decision="fallback_failed",
                verifier_status="confirmed_quota",
                fallback_dispatched=True,
                fallback_returncode=17,
            ),
            17,
        ),
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                decision="denied",
                verifier_status="not_confirmed",
                fallback_dispatched=False,
                fallback_returncode=None,
            ),
            41,
        ),
    )

    for outcome, runner_returncode in cases:
        assert provider_failure_policy.valid_grok_route_outcome(
            outcome,
            receipt=receipt,
            route_plan=route,
            runner_returncode=runner_returncode,
        )
        rendered = provider_failure_policy.render_grok_route_outcome(outcome)
        assert provider_failure_policy.extract_grok_route_outcomes(
            rendered + "\n"
        ) == (outcome,)

    tampered = dict(cases[0][0])
    tampered["fallback_returncode"] = 9
    assert not provider_failure_policy.valid_grok_route_outcome(
        tampered,
        receipt=receipt,
        route_plan=route,
        runner_returncode=0,
    )
    duplicate_log = "\n".join(
        provider_failure_policy.render_grok_route_outcome(cases[1][0])
        for _ in range(2)
    )
    assert len(
        provider_failure_policy.extract_grok_route_outcomes(duplicate_log)
    ) == 2
    assert provider_failure_policy.extract_grok_route_outcomes(
        provider_failure_policy.GROK_ROUTE_OUTCOME_PREFIX
        + '{"schema":"first","schema":"last"}\n'
    ) == ()
    assert provider_failure_policy.extract_grok_failure_receipts(
        provider_failure_policy.GROK_FAILURE_RECEIPT_PREFIX
        + '{"schema":"first","schema":"last"}\n'
    ) == ()

def test_terminal_route_outcome_is_bound_to_receipt_route_and_runner_exit_aseh() -> None:
    nonce = "c" * 64
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    route = llm_router.resolve_agent_implementation_route(
        default_route="legacy"
    ).as_binding_dict()
    evidence_id = "sha256:" + "d" * 64
    cases = (
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                quota_evidence_id=evidence_id,
                decision="fallback_succeeded",
                verifier_status="confirmed_quota",
                fallback_dispatched=True,
                fallback_returncode=0,
            ),
            0,
        ),
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                quota_evidence_id=evidence_id,
                decision="fallback_failed",
                verifier_status="confirmed_quota",
                fallback_dispatched=True,
                fallback_returncode=17,
            ),
            17,
        ),
        (
            provider_failure_policy.build_grok_route_outcome(
                receipt=receipt,
                route_plan=route,
                decision="denied",
                verifier_status="not_confirmed",
                fallback_dispatched=False,
                fallback_returncode=None,
            ),
            41,
        ),
    )

    for outcome, runner_returncode in cases:
        assert provider_failure_policy.valid_grok_route_outcome(
            outcome,
            receipt=receipt,
            route_plan=route,
            runner_returncode=runner_returncode,
        )
        rendered = provider_failure_policy.render_grok_route_outcome(outcome)
        assert provider_failure_policy.extract_grok_route_outcomes(
            rendered + "\n"
        ) == (outcome,)

    tampered = dict(cases[0][0])
    tampered["fallback_returncode"] = 9
    assert not provider_failure_policy.valid_grok_route_outcome(
        tampered,
        receipt=receipt,
        route_plan=route,
        runner_returncode=0,
    )
    duplicate_log = "\n".join(
        provider_failure_policy.render_grok_route_outcome(cases[1][0])
        for _ in range(2)
    )
    assert len(
        provider_failure_policy.extract_grok_route_outcomes(duplicate_log)
    ) == 2
    assert provider_failure_policy.extract_grok_route_outcomes(
        provider_failure_policy.GROK_ROUTE_OUTCOME_PREFIX
        + '{"schema":"first","schema":"last"}\n'
    ) == ()
    assert provider_failure_policy.extract_grok_failure_receipts(
        provider_failure_policy.GROK_FAILURE_RECEIPT_PREFIX
        + '{"schema":"first","schema":"last"}\n'
    ) == ()

@pytest.mark.parametrize("outcome_case", ("valid", "missing", "forged", "duplicate"))
def test_nonce_route_nonzero_never_restores_provider_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome_case: str,
) -> None:
    classifier_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        implementation_daemon,
        "classify_provider_capacity_failure",
        lambda *args, **_kwargs: classifier_calls.append(args),
    )
    nonce = "e" * 64
    route_plan = llm_router._EAAEF_AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Error: Not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route_plan.as_binding_dict(),
        decision="fallback_failed",
        verifier_status="not_required_exact_auth",
        fallback_dispatched=True,
        fallback_returncode=17,
    )
    if outcome_case == "forged":
        outcome = {**outcome, "fallback_returncode": 16}
    records = [provider_failure_policy.render_grok_failure_receipt(receipt)]
    if outcome_case != "missing":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    if outcome_case == "duplicate":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    log_path = tmp_path / "route.log"
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        implementation_daemon,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    command = [
        "/usr/bin/python3",
        "grok_cli_runner.py",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(route_plan.as_binding_dict()),
    ]

    capacity = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )

    assert capacity["exhausted"] is False
    assert capacity["providers"] == []
    assert classifier_calls == []
    assert route_plan.invocation_binding is None
    assert "route_outcome_id" not in capacity

@pytest.mark.parametrize("outcome_case", ("valid", "missing", "forged", "duplicate"))
def test_nonce_route_nonzero_never_restores_provider_attempt_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome_case: str,
) -> None:
    classifier_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        implementation_daemon,
        "classify_provider_capacity_failure",
        lambda *args, **_kwargs: classifier_calls.append(args),
    )
    nonce = "e" * 64
    route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Error: Not signed in",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route_plan.as_binding_dict(),
        decision="fallback_failed",
        verifier_status="not_required_exact_auth",
        fallback_dispatched=True,
        fallback_returncode=17,
    )
    if outcome_case == "forged":
        outcome = {**outcome, "fallback_returncode": 16}
    records = [provider_failure_policy.render_grok_failure_receipt(receipt)]
    if outcome_case != "missing":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    if outcome_case == "duplicate":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    log_path = tmp_path / "route.log"
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        implementation_daemon,
        "resolve_agent_implementation_route_binding",
        lambda *_args, **_kwargs: route_plan,
    )
    command = [
        "/usr/bin/python3",
        "grok_cli_runner.py",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(route_plan.as_binding_dict()),
    ]

    capacity = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=17,
    )

    assert capacity["exhausted"] is False
    assert capacity["providers"] == []
    assert classifier_calls == []
    assert route_plan.invocation_binding is None
    assert "route_outcome_id" not in capacity

@pytest.mark.parametrize(
    ("record_case", "expected_exhausted"),
    (
        ("valid_hard_quota", True),
        ("live_canonical_medium", True),
        ("authentication", False),
        ("malformed", False),
        ("missing", False),
        ("forged", False),
        ("duplicate", False),
        ("quota_high_canonical", False),
        ("foreign_implementer", False),
        ("other_unscoped_route", False),
    ),
)
def test_invocation_null_route_restores_only_exact_denied_hard_quota_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_case: str,
    expected_exhausted: bool,
) -> None:
    classifier_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        implementation_daemon,
        "classify_provider_capacity_failure",
        lambda *args, **_kwargs: classifier_calls.append(args),
    )
    nonce = "9" * 64
    if record_case == "valid_hard_quota":
        route_plan = implementation_daemon._HISTORICAL_INVOCATION_NULL_GROK_ROUTE
    elif record_case == "live_canonical_medium":
        route_plan = implementation_daemon._CANONICAL_INVOCATION_NULL_GROK_ROUTE
    elif record_case == "quota_high_canonical":
        route_plan = llm_router.resolve_agent_implementation_route(
            primary_provider_id="grok_cli",
            primary_model_id="grok-4.6",
            fallback_provider_id="codex",
            fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_exhausted",
            fallback_reasoning_effort="high",
        )
    elif record_case == "foreign_implementer":
        route_plan = llm_router.AgentImplementationRoutePlan(
            primary_provider_id="grok_cli",
            primary_model_id="grok-4.6",
            fallback_provider_id="codex",
            fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_exhausted",
            fallback_reasoning_effort="medium",
            route_id=(
                "agent-supervisor-grok45-terra56-medium-hard-quota-v1"
            ),
            authorization=None,
            fallback_implementer_identity="not-codex",
            invocation_binding=None,
        )
    elif record_case == "other_unscoped_route":
        route_plan = llm_router.AgentImplementationRoutePlan(
            primary_provider_id="grok_cli",
            primary_model_id="grok-4.6",
            fallback_provider_id="codex",
            fallback_model_id="gpt-5.6-terra",
            fallback_trigger="primary_quota_exhausted",
            fallback_reasoning_effort="medium",
            route_id="unscoped-noncanonical-route",
            authorization=None,
            fallback_implementer_identity="codex",
            invocation_binding=None,
        )
    else:
        route_plan = llm_router._AUTH_OR_QUOTA_AGENT_IMPLEMENTATION_ROUTE
    authentication = record_case == "authentication"
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text=(
            "Error: Not signed in"
            if authentication
            else "Grok Build usage balance exhausted"
        ),
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route_plan.as_binding_dict(),
        decision="denied",
        verifier_status=(
            "not_required_exact_auth" if authentication else "not_confirmed"
        ),
        fallback_dispatched=False,
        fallback_returncode=None,
    )
    if record_case == "forged":
        outcome = {**outcome, "outcome_id": "sha256:" + "0" * 64}

    records = [provider_failure_policy.render_grok_failure_receipt(receipt)]
    if record_case == "malformed":
        records.append(
            provider_failure_policy.GROK_ROUTE_OUTCOME_PREFIX
            + '{"schema":"first","schema":"last"}'
        )
    elif record_case != "missing":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    if record_case == "duplicate":
        records.append(provider_failure_policy.render_grok_route_outcome(outcome))
    log_path = tmp_path / f"invocation-null-{record_case}.log"
    log_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    log_path.chmod(0o600)
    command = [
        "/usr/bin/python3",
        "grok_cli_runner.py",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(route_plan.as_binding_dict()),
    ]

    capacity = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=41,
    )

    assert capacity["exhausted"] is expected_exhausted
    assert classifier_calls == []
    if expected_exhausted:
        assert capacity["providers"] == ["grok"]
        assert capacity["reason"] == "provider_capacity_exhausted"
        assert capacity["failure_class"] == "hard_quota_exhausted"
        assert capacity["hard_quota_exhausted_providers"] == ["grok"]
        assert capacity["hard_quota_evidence_sha256"] == receipt[
            "evidence_sha256"
        ]
        assert capacity["evidence"] == [
            "runner_receipt:" + str(receipt["receipt_id"])
        ]
        assert capacity["quota_probe_receipt"] == receipt
        assert capacity["route_outcome"] == outcome
    else:
        assert capacity["providers"] == []
        assert capacity["reason"] == ""

@pytest.mark.parametrize(
    ("command_case", "expected_match"),
    (
        ("script", True),
        ("module", True),
        ("module_suffix", False),
        ("lookalike_script_path", False),
        ("lookalike_script_interpreter", False),
        ("lookalike_module_interpreter", False),
        ("runner_value_token", False),
        ("duplicate_nonce", False),
        ("duplicate_model", False),
        ("equals_nonce_override", False),
        ("equals_model_override", False),
    ),
)
def test_quota_start_event_requires_exact_runner_and_unique_flags(
    tmp_path: Path,
    command_case: str,
    expected_match: bool,
) -> None:
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="PCPC-025",
        title="Bind exact Grok runner start",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
    )
    nonce = "7" * 64
    trusted_python = implementation_daemon.sys.executable
    trusted_script = str(
        (
            Path(implementation_daemon.__file__).resolve(strict=True).parents[1]
            / "grok_cli_runner.py"
        ).resolve(strict=True)
    )
    prefix = (
        [trusted_python, trusted_script]
        if command_case == "script"
        else [
            trusted_python,
            "-m",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
    )
    if command_case == "module_suffix":
        prefix = [
            trusted_python,
            "-m",
            "evil.ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
    elif command_case == "lookalike_script_path":
        lookalike_script = tmp_path / "grok_cli_runner.py"
        lookalike_script.write_text("raise SystemExit(0)\n", encoding="utf-8")
        prefix = [trusted_python, str(lookalike_script)]
    elif command_case in {
        "lookalike_script_interpreter",
        "lookalike_module_interpreter",
    }:
        lookalike_python = tmp_path / "python"
        lookalike_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        lookalike_python.chmod(0o755)
        prefix = (
            [str(lookalike_python), trusted_script]
            if command_case == "lookalike_script_interpreter"
            else [
                str(lookalike_python),
                "-m",
                "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
            ]
        )
    elif command_case == "runner_value_token":
        prefix = [
            trusted_python,
            "not-the-runner.py",
            "--workspace",
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        ]
    command = [
        *prefix,
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
    ]
    if command_case == "duplicate_nonce":
        command.extend(["--grok-failure-receipt-nonce", "6" * 64])
    elif command_case == "duplicate_model":
        command.extend(["--model", "grok-4.5"])
    elif command_case == "equals_nonce_override":
        command.append("--grok-failure-receipt-nonce=" + "6" * 64)
    elif command_case == "equals_model_override":
        command.append("--model=grok-4.5")
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task.task_id,
            "canonical_task_cid": daemon._canonical_ref(task),
            "attempt": 1,
            "command": command,
        },
    )

    matched = daemon._matching_quota_fallback_start_event(
        task=task,
        attempt=1,
        receipt={"nonce": nonce, "primary_model": "grok-4.6"},
    )

    assert (matched is not None) is expected_match

def test_module_form_grok_start_event_builds_durable_auto_codex_latch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        implementation_daemon,
        "_provider_capacity_now",
        lambda: fixed_now,
    )
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="PCPC-025",
        title="Resume after sealed Grok quota",
        status="ready",
        completion="manual",
        priority="P0",
        track="provider",
    )
    canonical_task_cid = daemon._canonical_ref(task)
    nonce = "8" * 64
    route_plan = implementation_daemon._CANONICAL_INVOCATION_NULL_GROK_ROUTE
    command = [
        implementation_daemon.sys.executable,
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--model",
        "grok-4.6",
        "--grok-failure-receipt-nonce",
        nonce,
        "--agent-implementation-route-json",
        json.dumps(route_plan.as_binding_dict()),
    ]
    log_path = tmp_path / "module-form-grok-quota.log"
    state = implementation_daemon.PortalTaskState()
    daemon._mark_implementation_started(
        state,
        task=task,
        attempt=1,
        started_at=fixed_now.isoformat(),
        log_path=log_path,
    )
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task.task_id,
            "canonical_task_cid": canonical_task_cid,
            "attempt": 1,
            "command": command,
        },
    )
    receipt = grok_cli_runner.build_grok_failure_receipt(
        probe_stderr_text="Grok Build usage balance exhausted",
        nonce=nonce,
        model="grok-4.6",
        probe_returncode=41,
        primary_dispatched=False,
    )
    outcome = provider_failure_policy.build_grok_route_outcome(
        receipt=receipt,
        route_plan=route_plan.as_binding_dict(),
        decision="denied",
        verifier_status="not_confirmed",
        fallback_dispatched=False,
        fallback_returncode=None,
    )
    log_path.write_text(
        "\n".join(
            (
                provider_failure_policy.render_grok_failure_receipt(receipt),
                provider_failure_policy.render_grok_route_outcome(outcome),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    log_path.chmod(0o600)
    failure = daemon._provider_capacity_failure_from_log(
        log_path,
        command=command,
        returncode=41,
    )
    assert failure["exhausted"] is True
    start = daemon._matching_quota_fallback_start_event(
        task=task,
        attempt=1,
        receipt=receipt,
    )
    assert start is not None
    result = daemon._record_provider_capacity_deferral(
        task=task,
        state=state,
        attempt=1,
        started_at=fixed_now.isoformat(),
        returncode=41,
        log_path=log_path,
        failure=failure,
    )
    authority = result["quota_fallback_authority"]
    assert authority["start_event_id"] == start["event_id"]
    assert authority["command_sha256"] == (
        daemon._implementation_command_identity(command)
    )
    assert result["hard_quota_exhausted_providers"] == ["grok"]

    states = daemon._provider_capacity_latch_states()
    assert states["grok"]["active"] is True
    assert states["grok"]["hard_quota_exhausted"] is True

    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        "auto",
    )
    monkeypatch.delenv(
        implementation_daemon.PROVIDER_EXTERNAL_ISOLATION_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/usr/local/bin/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_codex_ready_for_automatic_routing",
        lambda: True,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_goose_meta_spark_available",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_host_cli_binary",
        lambda _name: "",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_copilot_has_auth",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_daemon.shutil,
        "which",
        lambda name: "/usr/local/bin/codex" if name == "codex" else None,
    )
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: True)
    monkeypatch.setattr(llm_router, "get_llm_provider", lambda _name: object())
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        cli_provider_balance,
    )

    monkeypatch.setattr(
        cli_provider_balance,
        "probe_all_cli_provider_readiness",
        lambda: {},
    )

    selected = daemon._build_implementation_command(tmp_path)
    assert selected[:2] == ["/usr/local/bin/codex", "exec"]
    assert selected[selected.index("-m") + 1] == "gpt-5.6-terra"

def test_legacy_non_route_capacity_classification_remains_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_classifier = (
        implementation_daemon.classify_provider_capacity_failure
    )
    classifier_calls: list[tuple[object, ...]] = []

    def classify(*args: object, **kwargs: object) -> dict[str, object]:
        classifier_calls.append(args)
        return original_classifier(*args, **kwargs)

    monkeypatch.setattr(
        implementation_daemon,
        "classify_provider_capacity_failure",
        classify,
    )
    log_path = tmp_path / "legacy-capacity.log"
    log_path.write_text(
        "Grok quota exhausted before implementation\n",
        encoding="utf-8",
    )

    capacity = _daemon(tmp_path)._provider_capacity_failure_from_log(
        log_path,
        command=["/usr/local/bin/grok"],
        returncode=23,
    )

    assert capacity["exhausted"] is True
    assert capacity["providers"] == ["grok"]
    assert len(classifier_calls) == 1

def test_docker_codex_boundary_transforms_only_validated_sandbox(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    lease_root = tmp_path / "asref-codex-container-test"
    docker_config = lease_root / "docker-config"
    workspace.mkdir()
    provider_bin.mkdir()
    docker_config.mkdir(parents=True)
    codex = provider_bin / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    fallback = _terra_fallback_command(str(codex), workspace)
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    child_env = grok_cli_runner._codex_task_container_environment()
    network_values = {
        "provider": "codex",
        "docker_network": "eaaef-worker-test",
        "proxy_endpoint": "http://172.30.0.2:3128",
        "approval_identity": "eaaef-network-approval:test",
        "effect_cid": "sha256:" + hashlib.sha256(b"test-effect").hexdigest(),
        "workspace": workspace,
        "container_name": container_name,
        "lease_id": lease_root.name,
        "lease_root": lease_root,
    }
    network_profile = WorkerNetworkProfile(
        **network_values,
        allowed_hostnames=PROVIDER_HOSTNAME_ALLOWLISTS["codex"],
        approval_cid=worker_network_approval_cid(**network_values),
    )

    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=fallback,
        workspace=workspace,
        source_auth=source_auth,
        child_env=child_env,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=image,
        network_profile=network_profile,
    )

    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert command[0] == "/usr/bin/docker"
    assert f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}" in command
    assert "--pull=never" in command
    assert "--read-only" in command
    assert "--network=eaaef-worker-test" in command
    assert "--network=bridge" not in command
    assert "--runtime=runc" in command
    assert "--entrypoint=/usr/bin/env" in command
    assert "--log-driver=json-file" in command
    assert command.count("--log-opt") == 2
    assert "max-size=16m" in command
    assert "max-file=2" in command
    unbounded_log_command = list(command)
    unbounded_log_command[unbounded_log_command.index("max-size=16m")] = (
        "max-size=1g"
    )
    with pytest.raises(ValueError, match="bounded Docker logging"):
        grok_cli_runner.validate_provider_worker_command(
            unbounded_log_command,
            profile=network_profile,
            expected_image=image,
        )
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--device" not in command
    assert "ipfs_accelerate.codex_fallback_isolation=true" in command
    assert image in command
    docker_env = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    assert docker_env == [
        *grok_cli_runner._CODEX_DOCKER_IMAGE_ENV_OVERRIDES,
        *(
            f"{name}={value}"
            for name, value in sorted(network_profile.proxy_environment().items())
        ),
    ]
    assert "NVIDIA_VISIBLE_DEVICES=void" in docker_env
    assert "BASH_ENV=" in docker_env
    assert "ENV=" in docker_env

    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    writable_mounts = [mount for mount in mounts if "readonly" not in mount]
    assert f"type=bind,src={workspace},dst={workspace}" in writable_mounts
    auth_mounts = [
        mount
        for mount in mounts
        if f"dst={grok_cli_runner._CODEX_CONTAINER_AUTH_PATH}" in mount
    ]
    assert len(auth_mounts) == 1
    assert f"src={source_auth}" not in auth_mounts[0]
    assert "readonly" not in auth_mounts[0]
    assert not any(str(source_auth) in mount for mount in mounts)
    assert "type=bind,src=/usr,dst=/usr,readonly" in mounts
    assert (
        "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,readonly" in mounts
    )
    assert (
        f"type=bind,src={grok_cli_runner._HOST_CODEX_TASK_TOOLCHAIN_PYTHON},"
        f"dst={grok_cli_runner._CODEX_TASK_TOOLCHAIN_PYTHON},readonly"
        in mounts
    )
    assert not any("/var/run/docker.sock" in mount for mount in mounts)
    assert not any("/home/" in mount for mount in mounts)

    inner = command[command.index(image) + 1 :]
    expected_provider = list(fallback)
    expected_provider[expected_provider.index("-s") + 1] = (
        "danger-full-access"
    )
    try:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            _host_codex_vendor_binaries,
        )
    except Exception:
        vendor = None
    else:
        vendor = _host_codex_vendor_binaries()
    if vendor is not None:
        host_codex, _host_companion = vendor
        expected_provider[0] = str(host_codex)
        assert not any(
            "dst=/usr/local/bin/codex" in mount for mount in mounts
        )
    expected_environment = [
        f"{name}={value}" for name, value in sorted(child_env.items())
    ]
    expected_environment.extend(
        f"{name}={value}"
        for name, value in sorted(network_profile.proxy_environment().items())
    )
    assert inner[: 2 + len(expected_environment)] == [
        "-i",
        *expected_environment,
        str(grok_cli_runner._CODEX_TASK_TOOLCHAIN_PYTHON),
    ]
    assert inner[2 + len(expected_environment) : 5 + len(expected_environment)] == [
        "-I",
        "-c",
        grok_cli_runner.AGENT_IMPLEMENTATION_CODEX_CAPACITY_LOG_WRAPPER,
    ]
    assert inner[5 + len(expected_environment)] == (
        grok_cli_runner.AGENT_IMPLEMENTATION_CODEX_CAPACITY_LOG_SENTINEL_SCHEMA
    )
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        inner[6 + len(expected_environment)],
    )
    capacity_log_nonce = inner[6 + len(expected_environment)]
    assert grok_cli_runner._recorded_codex_capacity_log_nonce(
        {"command_receipt": {"create_argv": command}}
    ) == capacity_log_nonce
    assert inner[7 + len(expected_environment) :] == expected_provider
    provider_receipt = grok_cli_runner._codex_provider_argv_receipt(
        fallback,
        command,
    )
    assert provider_receipt == [expected_provider[0], *fallback[1:]]
    assert not any("/home/barberb" in item for item in command)
    assert "--dangerously-bypass-approvals-and-sandbox" not in inner

def test_docker_codex_boundary_transforms_only_validated_sandbox_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    lease_root = tmp_path / "asref-codex-container-boundary"
    docker_config = lease_root / "docker-config"
    provider_home = tmp_path / "asref-codex-home-test"
    workspace.mkdir()
    provider_bin.mkdir()
    lease_root.mkdir(mode=0o700)
    docker_config.mkdir()
    provider_home.mkdir(mode=0o700)
    monkeypatch.setattr(
        grok_cli_runner.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    codex = provider_bin / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    vendor_bin = tmp_path / "vendor-bin"
    vendor_bin.mkdir()
    host_codex = vendor_bin / "codex"
    host_companion = vendor_bin / "codex-code-mode-host"
    host_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    host_companion.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    host_codex.chmod(0o755)
    host_companion.chmod(0o755)
    monkeypatch.setattr(
        grok_cli_runner,
        "find_codex_vendor_binaries",
        lambda: (host_codex.resolve(), host_companion.resolve()),
    )
    fallback = _terra_fallback_command(str(codex), workspace)
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    child_env = grok_cli_runner._codex_task_container_environment()

    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=fallback,
        workspace=workspace,
        source_auth=source_auth,
        provider_home=provider_home,
        child_env=child_env,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=lease_root / "container.cid",
        docker_bin="/usr/bin/docker",
        isolation_image=image,
    )

    assert fallback[fallback.index("-s") + 1] == "workspace-write"
    assert command[0] == "/usr/bin/docker"
    assert f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}" in command
    assert "--pull=never" in command
    assert "--read-only" in command
    assert "--network=bridge" in command
    assert "--runtime=runc" in command
    assert "--entrypoint=/usr/bin/env" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--device" not in command
    assert "ipfs_accelerate.codex_fallback_isolation=true" in command
    assert image in command
    docker_env = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--env"
    ]
    assert docker_env == list(grok_cli_runner._CODEX_DOCKER_IMAGE_ENV_OVERRIDES)
    assert "NVIDIA_VISIBLE_DEVICES=void" in docker_env
    assert "BASH_ENV=" in docker_env
    assert "ENV=" in docker_env

    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    writable_mounts = [mount for mount in mounts if "readonly" not in mount]
    assert f"type=bind,src={workspace},dst={workspace}" in writable_mounts
    auth_mounts = [
        mount
        for mount in mounts
        if f"dst={grok_cli_runner._CODEX_CONTAINER_AUTH_PATH}" in mount
    ]
    assert len(auth_mounts) == 1
    auth_mount_fields = {
        name: value
        for item in auth_mounts[0].split(",")
        for name, separator, value in (item.partition("="),)
        if separator
    }
    auth_mount_source = Path(auth_mount_fields["src"])
    assert auth_mount_source == provider_home / "auth.json"
    assert auth_mount_source.is_relative_to(provider_home)
    assert auth_mount_source.resolve(strict=True).is_relative_to(
        provider_home.resolve(strict=True)
    )
    assert f"src={source_auth}" not in auth_mounts[0]
    assert "readonly" not in auth_mounts[0]
    assert not any(str(source_auth) in mount for mount in mounts)
    assert "type=bind,src=/usr,dst=/usr,readonly" in mounts
    assert (
        "type=bind,src=/etc/ssl/certs,dst=/etc/ssl/certs,readonly" in mounts
    )
    assert (
        f"type=bind,src={vendor_bin.resolve()},"
        "dst=/usr/local/bin,readonly"
        in mounts
    )
    assert not any("dst=/usr/local/bin/codex," in mount for mount in mounts)
    assert not any(
        "dst=/usr/local/bin/codex-code-mode-host," in mount
        for mount in mounts
    )
    assert (
        f"type=bind,src={grok_cli_runner._HOST_CODEX_TASK_TOOLCHAIN_PYTHON},"
        f"dst={grok_cli_runner._CODEX_TASK_TOOLCHAIN_PYTHON},readonly"
        in mounts
    )
    assert not any("/var/run/docker.sock" in mount for mount in mounts)
    assert not any("/home/" in mount for mount in mounts)
    assert not any("provider-start" in mount for mount in mounts)

    inner = command[command.index(image) + 1 :]
    expected_inner = list(fallback)
    expected_inner[expected_inner.index("-s") + 1] = "danger-full-access"
    if grok_cli_runner.find_codex_vendor_binaries() is not None:
        expected_inner[0] = "/usr/local/bin/codex"
    expected_environment = [
        f"{name}={value}" for name, value in sorted(child_env.items())
    ]
    assert inner == [
        "-i",
        *expected_environment,
        "/bin/sh",
        "-c",
        grok_cli_runner._DOCKER_PROVIDER_START_SCRIPT,
        "aseh-provider-start",
        *expected_inner,
    ]
    assert not any("/home/barberb" in item for item in command)
    assert "--dangerously-bypass-approvals-and-sandbox" not in inner

def test_docker_codex_boundary_rejects_vendor_pair_outside_projected_usr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    lease_root = tmp_path / "asref-codex-container-vendor-outside"
    docker_config = lease_root / "docker-config"
    workspace.mkdir()
    provider_bin.mkdir()
    docker_config.mkdir(parents=True)
    codex = provider_bin / "codex"
    companion = provider_bin / "codex-code-mode-host"
    for executable in (codex, companion):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    monkeypatch.setattr(
        implementation_daemon,
        "_host_codex_vendor_binaries",
        lambda: (codex.resolve(), companion.resolve()),
    )
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    _invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )

    with pytest.raises(
        ValueError,
        match="vendor pair is not safely projected by pinned host /usr",
    ):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            source_auth=source_auth,
            child_env=grok_cli_runner._codex_task_container_environment(),
            docker_config=docker_config,
            container_name=container_name,
            cidfile=lease_root / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
            network_profile=network_profile,
        )

def test_docker_codex_boundary_requires_native_code_mode_vendor_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-codex-container-vendor-missing"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    monkeypatch.setattr(
        implementation_daemon,
        "_host_codex_vendor_binaries",
        lambda: None,
    )
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    _invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )

    with pytest.raises(
        ValueError,
        match="requires a matching native code-mode vendor pair",
    ):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            source_auth=source_auth,
            child_env=grok_cli_runner._codex_task_container_environment(),
            docker_config=docker_config,
            container_name=container_name,
            cidfile=lease_root / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
            network_profile=network_profile,
        )

def test_codex_capacity_capture_accepts_complete_no_candidate_jsonl() -> None:
    nonce = "sha256:" + "8" * 64
    terminal = json.dumps(
        {
            "type": "error",
            "message": (
                "You've hit your usage limit. Try again at "
                "Aug 31st, 2026 12:42 AM."
            ),
        },
        separators=(",", ":"),
    )
    stream = "\n".join(
        (
            _codex_capacity_log_start(nonce),
            '{"type":"thread.started","thread_id":"thread:1"}',
            '{"type":"turn.started"}',
            terminal,
            '{"type":"turn.failed","error":{"message":"usage limit"}}',
            "",
        )
    )
    capture = grok_cli_runner._CodexTerminalCapacityCapture(
        expected_log_nonce=nonce
    )
    for chunk in (stream[:17], stream[17:103], stream[103:]):
        capture.feed(chunk)

    assert capture.finish() == terminal

@pytest.mark.parametrize(
    "records",
    (
        (lambda nonce: ['{"type":"error","message":"usage limit"}']),
        (
            lambda nonce: [
                _codex_capacity_log_start("sha256:" + "9" * 64),
                '{"type":"error","message":"usage limit"}',
            ]
        ),
        (
            lambda nonce: [
                _codex_capacity_log_start(nonce),
                '{"type":"item.completed","item":{"type":"command_execution"}}',
                '{"type":"error","message":"usage limit"}',
            ]
        ),
        (
            lambda nonce: [
                _codex_capacity_log_start(nonce),
                '{"type":"error","message":"usage limit"}',
                '{"type":"error","message":"usage limit"}',
            ]
        ),
        (
            lambda nonce: [
                _codex_capacity_log_start(nonce),
                '{"type":"error","type":"error","message":"usage limit"}',
            ]
        ),
    ),
    ids=(
        "missing-stream-start",
        "foreign-stream-start",
        "candidate-activity",
        "duplicate-terminal",
        "duplicate-json-key",
    ),
)
def test_codex_capacity_capture_fails_closed_for_incomplete_or_active_stream(
    records,
) -> None:
    nonce = "sha256:" + "8" * 64
    capture = grok_cli_runner._CodexTerminalCapacityCapture(
        expected_log_nonce=nonce
    )
    capture.feed("\n".join(records(nonce)) + "\n")

    assert capture.finish() == ""

def test_docker_grok_create_is_followed_by_attached_exact_container_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-direct"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    cidfile = lease_root / "container.cid"
    container_name = "ipfs-accelerate-grok-1-" + "c" * 32
    invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=container_name,
        lease_root=lease_root,
    )
    container_id = "d" * 64
    create_command = ["/usr/bin/docker", "create", "fixture-image"]
    observed: dict[str, object] = {}

    def fake_create(command, **kwargs):
        observed["create_command"] = list(command)
        observed["create_kwargs"] = dict(kwargs)
        cidfile.write_text(container_id + "\n", encoding="ascii")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    def fake_start(command, *, env):
        observed["start_command"] = list(command)
        observed["start_env"] = dict(env)
        return 19

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_create)
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_start,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **_kwargs: None,
    )

    returncode = (
        grok_cli_runner._run_created_grok_container_with_typed_failure_capture(
            create_command,
            docker_bin="/usr/bin/docker",
            docker_config=docker_config,
            cidfile=cidfile,
            workspace=workspace,
            env={"PATH": "/usr/bin"},
            network_profile=network_profile,
            invocation_binding=invocation,
        )
    )

    assert returncode == 19
    assert observed["create_command"] == create_command
    assert observed["create_kwargs"]["cwd"] == workspace
    assert observed["create_kwargs"]["stdin"] is subprocess.DEVNULL
    assert observed["start_command"] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    assert observed["start_env"] == {"PATH": "/usr/bin"}

def test_docker_grok_create_is_followed_by_attached_exact_container_start_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    cidfile = tmp_path / "container.cid"
    container_id = "d" * 64
    create_command = ["/usr/bin/docker", "create", "fixture-image"]
    observed: dict[str, object] = {}
    events: list[str] = []
    provider_sender, provider_stdin = (
        grok_cli_runner._provider_start_socketpair()
    )

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        preserve_for_recovery = False

        def __init__(self) -> None:
            self.docker_config = docker_config
            self.cidfile = cidfile

        def create_inert_container(self, command, **kwargs):
            observed["create_command"] = list(command)
            observed["create_kwargs"] = dict(kwargs)
            cidfile.write_text(container_id + "\n", encoding="ascii")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(container_id + "\n").encode("ascii"),
                stderr=b"",
            )

        def take_provider_start_stdin(self) -> socket.socket:
            return provider_stdin

        def capture_running_termination_fence(self) -> None:
            events.append("fence")
            provider_sender.sendall(
                grok_cli_runner._DOCKER_PROVIDER_START_MARKER
            )
            provider_sender.close()

    def fake_start(command, *, env, provider_stdin):
        observed["start_command"] = list(command)
        observed["start_env"] = dict(env)
        assert provider_stdin.recv(
            len(grok_cli_runner._DOCKER_PROVIDER_START_MARKER)
        ) == grok_cli_runner._DOCKER_PROVIDER_START_MARKER
        events.append("provider")
        provider_stdin.close()
        return 19

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_start,
    )

    returncode = (
        grok_cli_runner._run_created_grok_container_with_typed_failure_capture(
            create_command,
            workspace=workspace,
            env={"PATH": "/usr/bin"},
            docker_lease=FakeLease(),
        )
    )

    assert returncode == 19
    assert observed["create_command"] == create_command
    assert observed["create_kwargs"]["cwd"] == workspace
    assert observed["create_kwargs"]["env"] == {"PATH": "/usr/bin"}
    assert observed["start_command"] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    assert observed["start_env"] == {"PATH": "/usr/bin"}
    assert events == ["fence", "provider"]

@pytest.mark.parametrize(
    "failure",
    ("nonzero", "malformed", "mismatch", "missing_cidfile", "oversized"),
)
def test_docker_grok_create_rejects_untrusted_container_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-direct-invalid"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    cidfile = lease_root / "container.cid"
    container_id = "d" * 64
    invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name="ipfs-accelerate-grok-1-" + "c" * 32,
        lease_root=lease_root,
    )
    start_called = False

    def fake_create(command, **_kwargs):
        if failure not in {"missing_cidfile", "nonzero"}:
            recorded = "e" * 64 if failure == "mismatch" else container_id
            cidfile.write_text(recorded + "\n", encoding="ascii")
        stdout = (container_id + "\n").encode("ascii")
        if failure == "malformed":
            stdout = b"not-a-container-id\n"
        elif failure == "oversized":
            stdout = b"x" * (grok_cli_runner._DOCKER_INSPECTION_MAX_BYTES + 1)
        return subprocess.CompletedProcess(
            command,
            17 if failure == "nonzero" else 0,
            stdout=stdout,
            stderr=b"create failed" if failure == "nonzero" else b"",
        )

    def fail_if_started(*_args, **_kwargs):
        nonlocal start_called
        start_called = True
        return 0

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_create)
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fail_if_started,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError):
        grok_cli_runner._run_created_grok_container_with_typed_failure_capture(
            ["/usr/bin/docker", "create", "fixture-image"],
            docker_bin="/usr/bin/docker",
            docker_config=docker_config,
            cidfile=cidfile,
                workspace=workspace,
                env={"PATH": "/usr/bin"},
                network_profile=network_profile,
                invocation_binding=invocation,
            )

    assert start_called is False

@pytest.mark.parametrize(
    "failure",
    ("nonzero", "malformed", "mismatch", "missing_cidfile", "oversized"),
)
def test_docker_grok_create_rejects_untrusted_container_identity_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    cidfile = tmp_path / "container.cid"
    container_id = "d" * 64
    start_called = False

    class FakeLease:
        docker_bin = "/usr/bin/docker"

        def __init__(self) -> None:
            self.docker_config = docker_config
            self.cidfile = cidfile

        def create_inert_container(self, command, **_kwargs):
            if failure not in {"missing_cidfile", "nonzero"}:
                recorded = (
                    "e" * 64 if failure == "mismatch" else container_id
                )
                cidfile.write_text(recorded + "\n", encoding="ascii")
            stdout = (container_id + "\n").encode("ascii")
            if failure == "malformed":
                stdout = b"not-a-container-id\n"
            elif failure == "oversized":
                stdout = b"x" * (
                    grok_cli_runner._DOCKER_INSPECTION_MAX_BYTES + 1
                )
            return subprocess.CompletedProcess(
                command,
                17 if failure == "nonzero" else 0,
                stdout=stdout,
                stderr=b"create failed" if failure == "nonzero" else b"",
            )

    def fail_if_started(*_args, **_kwargs):
        nonlocal start_called
        start_called = True
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fail_if_started,
    )

    with pytest.raises(ValueError):
        grok_cli_runner._run_created_grok_container_with_typed_failure_capture(
            ["/usr/bin/docker", "create", "fixture-image"],
            workspace=workspace,
            env={"PATH": "/usr/bin"},
            docker_lease=FakeLease(),
        )

    assert start_called is False

@pytest.mark.parametrize("valid_label", (True, False))
def test_codex_task_toolchain_image_requires_exact_identity_and_label(
    tmp_path: Path,
    monkeypatch,
    valid_label: bool,
) -> None:
    expected_line = (
        f"{grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID}|linux|arm64|"
        f"{grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_LABEL}"
    )
    observed_commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        observed_commands.append(list(command))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                expected_line
                if valid_label
                else expected_line.removesuffix(
                    grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_LABEL
                )
                + "untrusted"
            ),
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    resolved = grok_cli_runner._docker_codex_task_toolchain_image_id(
        "/usr/bin/docker",
        docker_config=tmp_path,
    )

    assert resolved == (
        grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID if valid_label else ""
    )
    assert observed_commands[0][-1] == (
        grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    )

@pytest.mark.parametrize(
    "invalid_case",
    ("symlink", "group_readable", "hardlink", "wrong_owner"),
)
def test_codex_auth_boundary_rejects_ambient_or_mutable_authority(
    tmp_path: Path,
    monkeypatch,
    invalid_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    auth_path.chmod(0o600)

    if invalid_case == "symlink":
        target = tmp_path / "real-auth.json"
        target.write_text("{}\n", encoding="utf-8")
        target.chmod(0o600)
        auth_path.unlink()
        auth_path.symlink_to(target)
    elif invalid_case == "group_readable":
        auth_path.chmod(0o640)
    elif invalid_case == "hardlink":
        os.link(auth_path, tmp_path / "auth-alias.json")
    else:
        current_uid = os.getuid()
        monkeypatch.setattr(
            grok_cli_runner.os,
            "getuid",
            lambda: current_uid + 1,
        )

    with pytest.raises(ValueError, match="private, owned, regular"):
        grok_cli_runner._codex_quota_fallback_env(
            workspace=workspace,
            base_env={
                "HOME": str(tmp_path),
                "CODEX_HOME": str(codex_home),
            },
        )

def test_codex_auth_defaults_to_account_home_when_runtime_home_is_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    account_home = tmp_path / "account-home"
    runtime_home = tmp_path / "runtime-home"
    codex_home = account_home / ".codex"
    workspace.mkdir()
    runtime_home.mkdir()
    codex_home.mkdir(parents=True)
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    auth_path.chmod(0o600)
    monkeypatch.setattr(
        grok_cli_runner,
        "_operating_system_account_home",
        lambda: account_home.resolve(strict=True),
    )

    environment = grok_cli_runner._codex_quota_fallback_env(
        workspace=workspace,
        base_env={"HOME": str(runtime_home)},
    )

    assert environment["HOME"] == str(codex_home.resolve(strict=True))
    assert environment["CODEX_HOME"] == str(codex_home.resolve(strict=True))

def test_runner_owned_lazy_import_cannot_drift_workspace_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "lgcvf_lazy_provider_dependency"
    (tmp_path / f"{module_name}.py").write_text(
        "VALUE = 'loaded'\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    (tmp_path / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(grok_cli_runner.sys, "dont_write_bytecode", False)
    baseline = grok_cli_runner._workspace_content_fingerprint(tmp_path)

    def lazy_import(_args, _receipt_fd: int) -> int:
        assert grok_cli_runner.sys.dont_write_bytecode is True
        imported = importlib.import_module(module_name)
        assert imported.VALUE == "loaded"
        assert (
            grok_cli_runner._workspace_content_fingerprint(tmp_path)
            == baseline
        )
        return 0

    monkeypatch.setattr(grok_cli_runner, "_run", lazy_import)
    try:
        assert grok_cli_runner.main(["--workspace", str(tmp_path)]) == 0
    finally:
        sys.modules.pop(module_name, None)

    assert grok_cli_runner.sys.dont_write_bytecode is False
    assert not (tmp_path / "__pycache__").exists()

    # The fix suppresses runner-owned cache writes; it must not weaken the
    # fence by exempting a cache-shaped mutation from the fingerprint.
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    external_cache = cache / "external.cpython-312.pyc"
    external_cache.write_bytes(b"external mutation")
    ignored = subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "check-ignore",
            "-q",
            "--",
            external_cache,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert ignored.returncode == 0
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) != baseline

def test_codex_isolated_home_seals_environment_without_host_toolchain(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    codex_home.mkdir()
    auth_path = codex_home / "auth.json"
    auth_path.write_text("{}\n", encoding="utf-8")
    auth_path.chmod(0o600)

    temporary_home, environment, source_auth = (
        grok_cli_runner._isolated_codex_quota_fallback_home(
            workspace=workspace,
            base_env={
                "CODEX_HOME": str(codex_home),
                "BASH_ENV": "/workspace/untrusted-hook",
                "PYTHONPATH": "/home/barberb/.local/lib/python3.12/site-packages",
            },
        )
    )
    try:
        assert environment == (
            grok_cli_runner._codex_task_container_environment()
        )
        assert source_auth == auth_path
        assert not any("/home/barberb" in value for value in environment.values())
        assert not any(Path(temporary_home.name).iterdir())
    finally:
        grok_cli_runner._robust_remove_runner_temp_tree(
            Path(temporary_home.name)
        )
        temporary_home.cleanup()

def test_grok_docker_create_binds_exact_id_to_attached_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    container_id = "d" * 64
    docker_environment = {"PATH": "/usr/bin"}
    create_command = ["/usr/bin/docker", "create", "sealed-grok"]
    calls: list[tuple[list[str], dict[str, object]]] = []
    lease_root = tmp_path / "asref-grok-container-identity"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        container_name = "ipfs-accelerate-grok-1-" + "a" * 32

    FakeLease.docker_config = docker_config
    FakeLease.cidfile = lease_root / "container.cid"
    FakeLease.lease_root = lease_root

    invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=FakeLease.container_name,
        lease_root=FakeLease.lease_root,
    )

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        FakeLease.cidfile.write_text(container_id + "\n", encoding="ascii")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **_kwargs: None,
    )

    start_command = (
        grok_cli_runner._create_grok_container_and_build_start_command(
            create_command,
            workspace=workspace,
            docker_environment=docker_environment,
            docker_lease=FakeLease(),
            network_profile=network_profile,
            invocation_binding=invocation,
        )
    )

    assert start_command == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(FakeLease.docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    assert calls[0][0] == create_command
    assert calls[0][1]["cwd"] == workspace
    assert calls[0][1]["env"] is docker_environment
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["stdout"] is subprocess.PIPE
    assert calls[0][1]["stderr"] is subprocess.PIPE

def test_grok_docker_create_binds_exact_id_to_attached_start_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "d" * 64
    docker_environment = {"PATH": "/usr/bin"}
    create_command = ["/usr/bin/docker", "create", "sealed-grok"]
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        docker_config = tmp_path / "docker-config"
        cidfile = tmp_path / "container.cid"

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        FakeLease.cidfile.write_text(container_id + "\n", encoding="ascii")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)

    start_command = (
        grok_cli_runner._create_grok_container_and_build_start_command(
            create_command,
            workspace=tmp_path,
            docker_environment=docker_environment,
            docker_lease=FakeLease(),
        )
    )

    assert start_command == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(FakeLease.docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    assert calls[0][0] == create_command
    assert calls[0][1]["env"] is docker_environment
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["stdout"] is subprocess.PIPE
    assert calls[0][1]["stderr"] is subprocess.PIPE

@pytest.mark.parametrize(
    ("create_stdout", "cidfile_container_id", "error"),
    (
        (b"container-name\n", "d" * 64, "identity is invalid"),
        (
            ("a" * 64 + "\n" + "b" * 64 + "\n").encode("ascii"),
            "a" * 64,
            "identity is invalid",
        ),
        (("A" * 64 + "\n").encode("ascii"), "A" * 64, "identity is invalid"),
        (("a" * 64 + "\n").encode("ascii"), "b" * 64, "identity is invalid"),
        (("a" * 64 + "\n").encode("ascii"), None, "identity is unavailable"),
    ),
)
def test_grok_docker_create_rejects_untrusted_container_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_stdout: bytes,
    cidfile_container_id: str | None,
    error: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease_root = tmp_path / "asref-grok-container-invalid-identity"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)

    class FakeLease:
        docker_bin = "/usr/bin/docker"
        container_name = "ipfs-accelerate-grok-1-" + "a" * 32

    FakeLease.docker_config = docker_config
    FakeLease.cidfile = lease_root / "container.cid"
    FakeLease.lease_root = lease_root

    invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="grok",
        workspace=workspace,
        container_name=FakeLease.container_name,
        lease_root=FakeLease.lease_root,
    )

    if cidfile_container_id is not None:
        FakeLease.cidfile.write_text(
            cidfile_container_id + "\n",
            encoding="ascii",
        )

    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=create_stdout,
            stderr=b"",
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **_kwargs: None,
    )

    with pytest.raises(ValueError, match=error):
        grok_cli_runner._create_grok_container_and_build_start_command(
            ["/usr/bin/docker", "create", "sealed-grok"],
            workspace=workspace,
            docker_environment={},
            docker_lease=FakeLease(),
            network_profile=network_profile,
            invocation_binding=invocation,
        )

@pytest.mark.parametrize(
    ("create_stdout", "cidfile_container_id", "error"),
    (
        (b"container-name\n", "d" * 64, "identity is invalid"),
        (
            ("a" * 64 + "\n" + "b" * 64 + "\n").encode("ascii"),
            "a" * 64,
            "identity is invalid",
        ),
        (("A" * 64 + "\n").encode("ascii"), "A" * 64, "identity is invalid"),
        (("a" * 64 + "\n").encode("ascii"), "b" * 64, "identity is invalid"),
        (("a" * 64 + "\n").encode("ascii"), None, "identity is unavailable"),
    ),
)
def test_grok_docker_create_rejects_untrusted_container_identity_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_stdout: bytes,
    cidfile_container_id: str | None,
    error: str,
) -> None:
    class FakeLease:
        docker_bin = "/usr/bin/docker"
        docker_config = tmp_path / "docker-config"
        cidfile = tmp_path / "container.cid"

    if cidfile_container_id is not None:
        FakeLease.cidfile.write_text(
            cidfile_container_id + "\n",
            encoding="ascii",
        )

    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=create_stdout,
            stderr=b"",
        ),
    )

    with pytest.raises(ValueError, match=error):
        grok_cli_runner._create_grok_container_and_build_start_command(
            ["/usr/bin/docker", "create", "sealed-grok"],
            workspace=tmp_path,
            docker_environment={},
            docker_lease=FakeLease(),
        )

def test_grok_docker_primary_rejects_forged_stdout_identity_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_grok_docker_primary(
        tmp_path,
        monkeypatch,
        cidfile_container_id="e" * 64,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_bounded_stderr",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched container identity must never be attached"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
        ]
    )

    assert result == 2
    assert harness["close_calls"] == [False]

@pytest.mark.parametrize("typed_route", (False, True))
def test_grok_docker_primary_parses_attached_start_not_create_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    typed_route: bool,
) -> None:
    harness = _install_fake_grok_docker_primary(tmp_path, monkeypatch)
    provider_calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_bounded(command, *, env):
        provider_calls.append((list(command), dict(env)))
        return 0, b"", 0, False

    def fake_typed(command, *, env):
        provider_calls.append((list(command), dict(env)))
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_bounded_stderr",
        fake_bounded,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_typed,
    )
    argv = [
        "--workspace",
        str(harness["workspace"]),
        "--grok-bin",
        str(harness["grok"]),
        "--model",
        "grok-4.6",
    ]
    if typed_route:
        argv.extend(
            [
                "--codex-fallback-command-json",
                json.dumps(
                    _terra_fallback_command(
                        str(harness["codex"]),
                        harness["workspace"],
                        reasoning_effort="medium",
                    )
                ),
            ]
        )

    assert grok_cli_runner.main(argv) == 0

    create_calls = harness["create_calls"]
    assert isinstance(create_calls, list)
    assert len(create_calls) == 1
    assert "create" in create_calls[0][0]
    assert len(provider_calls) == 1
    assert provider_calls[0][0] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(tmp_path / "asref-grok-container-test" / "docker-config"),
        "start",
        "--attach",
        "--interactive",
        harness["container_id"],
    ]
    assert harness["close_calls"] == [True]

@pytest.mark.parametrize("typed_route", (False, True))
def test_grok_docker_primary_parses_attached_start_not_create_output_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    typed_route: bool,
) -> None:
    harness = _install_fake_grok_docker_primary(tmp_path, monkeypatch)
    provider_calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_bounded(command, *, env, provider_stdin):
        assert provider_stdin.family == socket.AF_UNIX
        assert provider_stdin.recv(
            len(grok_cli_runner._DOCKER_PROVIDER_START_MARKER)
        ) == grok_cli_runner._DOCKER_PROVIDER_START_MARKER
        provider_stdin.close()
        provider_calls.append((list(command), dict(env)))
        return 0, b"", 0, False

    def fake_typed(command, *, env, provider_stdin):
        assert provider_stdin.family == socket.AF_UNIX
        assert provider_stdin.recv(
            len(grok_cli_runner._DOCKER_PROVIDER_START_MARKER)
        ) == grok_cli_runner._DOCKER_PROVIDER_START_MARKER
        provider_stdin.close()
        provider_calls.append((list(command), dict(env)))
        return 0

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_bounded_stderr",
        fake_bounded,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fake_typed,
    )
    argv = [
        "--workspace",
        str(harness["workspace"]),
        "--grok-bin",
        str(harness["grok"]),
        "--model",
        "grok-4.6",
    ]
    if typed_route:
        argv.extend(
            [
                "--codex-fallback-command-json",
                json.dumps(
                    _terra_fallback_command(
                        str(harness["codex"]),
                        harness["workspace"],
                        reasoning_effort="medium",
                    )
                ),
            ]
        )

    assert grok_cli_runner.main(argv) == 0

    create_calls = harness["create_calls"]
    assert isinstance(create_calls, list)
    assert len(create_calls) == 1
    assert "create" in create_calls[0][0]
    assert len(provider_calls) == 1
    assert provider_calls[0][0] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(tmp_path / "asref-grok-container-test" / "docker-config"),
        "start",
        "--attach",
        "--interactive",
        harness["container_id"],
    ]
    assert harness["close_calls"] == [True]

@pytest.mark.parametrize(
    ("create_returncode", "create_stdout"),
    (
        (125, b""),
        (0, b"container-name\n"),
    ),
)
def test_grok_docker_create_failure_cleans_without_provider_or_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_returncode: int,
    create_stdout: bytes,
) -> None:
    harness = _install_fake_grok_docker_primary(
        tmp_path,
        monkeypatch,
        create_returncode=create_returncode,
        create_stdout=create_stdout,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "failed container creation must never start Grok"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "failed container creation must not fall through to Terra"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(
                _terra_fallback_command(
                    str(harness["codex"]),
                    harness["workspace"],
                    reasoning_effort="medium",
                )
            ),
        ]
    )

    assert result == 127
    assert harness["close_calls"] == [False]

@pytest.mark.parametrize(
    ("create_returncode", "create_stdout"),
    (
        (125, b""),
        (0, b"container-name\n"),
    ),
)
def test_grok_docker_create_failure_cleans_without_provider_or_fallback_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_returncode: int,
    create_stdout: bytes,
) -> None:
    harness = _install_fake_grok_docker_primary(
        tmp_path,
        monkeypatch,
        create_returncode=create_returncode,
        create_stdout=create_stdout,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        lambda *_args, **_kwargs: pytest.fail(
            "failed container creation must never start Grok"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "failed container creation must not fall through to Terra"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(
                _terra_fallback_command(
                    str(harness["codex"]),
                    harness["workspace"],
                    reasoning_effort="medium",
                )
            ),
        ]
    )

    assert result == 2
    assert harness["close_calls"] == [False]

def test_grok_docker_start_failure_cleans_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_grok_docker_primary(tmp_path, monkeypatch)
    start_calls: list[list[str]] = []

    def fail_start(command, *, env):
        del env
        start_calls.append(list(command))
        raise OSError("docker start failed")

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fail_start,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "failed Docker start must not fall through to Terra"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(
                _terra_fallback_command(
                    str(harness["codex"]),
                    harness["workspace"],
                    reasoning_effort="medium",
                )
            ),
        ]
    )

    assert result == 127
    assert len(start_calls) == 1
    assert "start" in start_calls[0]
    assert "create" not in start_calls[0]
    assert harness["close_calls"] == [False]

def test_grok_docker_start_failure_cleans_without_fallback_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_fake_grok_docker_primary(tmp_path, monkeypatch)
    start_calls: list[list[str]] = []

    def fail_start(command, *, env, provider_stdin):
        del env
        assert provider_stdin.family == socket.AF_UNIX
        # Simulate failure before Docker inherited the capability: closing the
        # attached endpoint makes the owner-side release observe EPIPE.
        provider_stdin.close()
        start_calls.append(list(command))
        raise OSError("docker start failed")

    monkeypatch.setattr(
        grok_cli_runner,
        "_run_grok_with_typed_failure_capture",
        fail_start,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_run_codex_quota_fallback_in_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "failed Docker start must not fall through to Terra"
        ),
    )

    result = grok_cli_runner.main(
        [
            "--workspace",
            str(harness["workspace"]),
            "--grok-bin",
            str(harness["grok"]),
            "--model",
            "grok-4.6",
            "--codex-fallback-command-json",
            json.dumps(
                _terra_fallback_command(
                    str(harness["codex"]),
                    harness["workspace"],
                    reasoning_effort="medium",
                )
            ),
        ]
    )

    assert result == 127
    assert len(start_calls) == 1
    assert "start" in start_calls[0]
    assert "create" not in start_calls[0]
    assert harness["close_calls"] == [False]

@pytest.mark.parametrize(
    "invalid_case",
    (
        "mutable_image",
        "unapproved_image",
        "wrong_provider_name",
        "workspace_mismatch",
        "ambient_environment",
    ),
)
def test_docker_codex_boundary_rejects_unpinned_or_mismatched_authority(
    tmp_path: Path,
    invalid_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other-workspace"
    workspace.mkdir()
    other_workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    fallback_workspace = (
        other_workspace if invalid_case == "workspace_mismatch" else workspace
    )
    image = {
        "mutable_image": "ubuntu:24.04",
        "unapproved_image": "sha256:" + "a" * 64,
    }.get(invalid_case, grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID)
    container_name = (
        "ipfs-accelerate-grok-1-" + "b" * 32
        if invalid_case == "wrong_provider_name"
        else "ipfs-accelerate-codex-1-" + "b" * 32
    )
    child_env = grok_cli_runner._codex_task_container_environment()
    if invalid_case == "ambient_environment":
        child_env["BASH_ENV"] = "/workspace/untrusted-hook"

    with pytest.raises(ValueError):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(
                str(codex),
                fallback_workspace,
            ),
            workspace=workspace,
            source_auth=source_auth,
            child_env=child_env,
            docker_config=tmp_path,
            container_name=container_name,
            cidfile=tmp_path / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=image,
        )

@pytest.mark.parametrize(
    "invalid_case",
    (
        "mutable_image",
        "unapproved_image",
        "wrong_provider_name",
        "workspace_mismatch",
        "ambient_environment",
    ),
)
def test_docker_codex_boundary_rejects_unpinned_or_mismatched_authority_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_case: str,
) -> None:
    workspace = tmp_path / "workspace"
    other_workspace = tmp_path / "other-workspace"
    workspace.mkdir()
    other_workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    provider_home = tmp_path / "asref-codex-home-test"
    provider_home.mkdir(mode=0o700)
    monkeypatch.setattr(
        grok_cli_runner.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    fallback_workspace = (
        other_workspace if invalid_case == "workspace_mismatch" else workspace
    )
    image = {
        "mutable_image": "ubuntu:24.04",
        "unapproved_image": "sha256:" + "a" * 64,
    }.get(invalid_case, grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID)
    container_name = (
        "ipfs-accelerate-grok-1-" + "b" * 32
        if invalid_case == "wrong_provider_name"
        else "ipfs-accelerate-codex-1-" + "b" * 32
    )
    child_env = grok_cli_runner._codex_task_container_environment()
    if invalid_case == "ambient_environment":
        child_env["BASH_ENV"] = "/workspace/untrusted-hook"

    with pytest.raises(ValueError):
        grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(
                str(codex),
                fallback_workspace,
            ),
            workspace=workspace,
            source_auth=source_auth,
            provider_home=provider_home,
            child_env=child_env,
            docker_config=tmp_path,
            container_name=container_name,
            cidfile=tmp_path / "container.cid",
            docker_bin="/usr/bin/docker",
            isolation_image=image,
        )

@pytest.mark.parametrize(
    ("effect_claim", "capacity_evidence", "effect_terminal"),
    (
        (lambda _context: None, None, None),
        (None, None, lambda _returncode: None),
        (None, lambda _returncode, _record: None, None),
    ),
)
def test_docker_codex_fallback_rejects_partial_effect_lifecycle(
    tmp_path: Path,
    effect_claim,
    capacity_evidence,
    effect_terminal,
) -> None:
    with pytest.raises(ValueError):
        grok_cli_runner._run_codex_quota_fallback_in_docker(
            ["codex"],
            workspace=tmp_path,
            prompt="repair",
            prompt_path=tmp_path / "prompt.txt",
            base_env={},
            effect_claim=effect_claim,
            capacity_evidence=capacity_evidence,
            effect_terminal=effect_terminal,
        )

@pytest.mark.parametrize(
    ("outcome", "expected_finished"),
    (
        ("success", True),
        ("signal", True),
        ("error", False),
        ("auth_swap", False),
    ),
)
def test_docker_codex_fallback_always_closes_its_separate_lease(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
    expected_finished: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    provider_home = tmp_path / "asref-codex-home-test"
    provider_home.mkdir()
    prompt_path = tmp_path / "asref-grok-prompt-test.txt"
    prompt_path.write_text("repair", encoding="utf-8")
    close_calls: list[bool] = []
    create_kwargs: list[dict[str, object]] = []
    create_commands: list[list[str]] = []
    popen_calls: list[bool] = []
    boundary_events: list[str] = []
    created_container_id = "d" * 64

    class FakeHome:
        name = str(provider_home)

        def cleanup(self) -> None:
            return None

    class FakeLease:
        lease_root = tmp_path / "asref-codex-container-test"
        docker_config = lease_root / "docker-config"
        container_name = "ipfs-accelerate-codex-1-" + "c" * 32
        cidfile = tmp_path / "container.cid"

        def close(self, *, docker_run_finished: bool) -> None:
            close_calls.append(docker_run_finished)

    FakeLease.docker_config.mkdir(parents=True)
    _invocation, network_profile = _signed_network_fixture(
        tmp_path,
        provider="codex",
        workspace=workspace,
        container_name=FakeLease.container_name,
        lease_root=FakeLease.lease_root,
        prompt="repair",
    )

    def fake_create(*_args, **kwargs):
        create_kwargs.append(dict(kwargs))
        return FakeLease()

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            return 0 if outcome == "success" else -15

    def fake_run(command, **_kwargs):
        create_command = list(command)
        create_commands.append(create_command)
        boundary_events.append("create")
        assert create_command == ["docker", "create"]
        return subprocess.CompletedProcess(
            create_command,
            0,
            stdout=(created_container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    def fake_popen(command, **_kwargs):
        boundary_events.append("popen")
        assert list(command) == [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "--config",
            str(FakeLease.docker_config),
            "start",
            "--attach",
            "--interactive",
            created_container_id,
        ]
        if outcome == "error":
            raise OSError("docker launch failed")
        popen_calls.append(True)
        return FakeProcess()

    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_codex_quota_fallback_home",
        lambda **_kwargs: (
            FakeHome(),
            grok_cli_runner._codex_task_container_environment(),
            source_auth,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerContainerLease,
        "create",
        fake_create,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_task_toolchain_image_id",
        lambda *_args, **_kwargs: (
            grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_fallback_command",
        lambda **_kwargs: ["docker", "create"],
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_inspect_signed_worker_network",
        lambda **kwargs: (
            None
            if kwargs["profile"].authorization is not None
            else pytest.fail("Codex cleanup fixture lost signed authority")
        ),
    )
    validate_auth = grok_cli_runner._validated_codex_auth_path

    def record_and_validate_auth(**kwargs):
        try:
            validated = validate_auth(**kwargs)
        except (OSError, ValueError):
            boundary_events.append("auth-fail")
            raise
        boundary_events.append("auth")
        return validated

    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_codex_auth_path",
        record_and_validate_auth,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)

    def invocation() -> int:
        def swap_auth() -> None:
            boundary_events.append("route")
            replacement = tmp_path / "replacement-auth.json"
            replacement.write_text("{}\n", encoding="utf-8")
            replacement.chmod(0o600)
            source_auth.unlink()
            source_auth.symlink_to(replacement)

        def validate_route() -> None:
            boundary_events.append("route")

        return grok_cli_runner._run_codex_quota_fallback_in_docker(
            _terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            prompt="repair",
            prompt_path=prompt_path,
            base_env={},
            pre_effect_validator=(
                swap_auth if outcome == "auth_swap" else validate_route
            ),
            network_profile=network_profile,
        )
    if outcome in {"error", "auth_swap"}:
        with pytest.raises((OSError, ValueError)):
            invocation()
    else:
        expected_returncode = 0 if outcome == "success" else -15
        assert invocation() == expected_returncode

    assert close_calls == [expected_finished]
    if outcome == "auth_swap":
        assert popen_calls == []
        assert create_commands == []
        assert boundary_events == ["route", "auth-fail"]
    else:
        assert create_commands == [["docker", "create"]]
        assert popen_calls == ([] if outcome == "error" else [True])
        assert boundary_events == [
            "route",
            "auth",
            "route",
            "create",
            "popen",
        ]
    assert create_kwargs == [
        {
            "provider": "codex",
            "provider_home": provider_home,
            "prompt_path": prompt_path,
            "authorized_container_name": FakeLease.container_name,
            "authorized_lease_root": FakeLease.lease_root,
        }
    ]

@pytest.mark.parametrize(
    ("outcome", "expected_finished"),
    (
        ("success", True),
        ("signal", True),
        ("error", False),
        ("auth_swap", False),
    ),
)
def test_docker_codex_fallback_always_closes_its_separate_lease_aseh(
    tmp_path: Path,
    monkeypatch,
    outcome: str,
    expected_finished: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    provider_home = tmp_path / "asref-codex-home-test"
    provider_home.mkdir()
    prompt_path = tmp_path / "asref-grok-prompt-test.txt"
    prompt_path.write_text("repair", encoding="utf-8")
    close_calls: list[bool] = []
    create_kwargs: list[dict[str, object]] = []
    create_commands: list[list[str]] = []
    popen_calls: list[bool] = []
    provider_payloads: list[bytes] = []
    boundary_events: list[str] = []
    created_container_id = "d" * 64

    class FakeHome:
        name = str(provider_home)

        def cleanup(self) -> None:
            return None

    lease_provider_home = provider_home

    class FakeLease:
        docker_config = tmp_path / "docker-config"
        container_name = "ipfs-accelerate-codex-1-" + "c" * 32
        cidfile = tmp_path / "container.cid"
        provider_home = lease_provider_home

        def __init__(self) -> None:
            self._sender, self._docker_stdin = (
                grok_cli_runner._provider_start_socketpair()
            )
            self._fence_published = False

        def take_provider_start_stdin(self) -> socket.socket:
            return self._docker_stdin

        def capture_running_termination_fence(self) -> dict[str, object]:
            self._fence_published = True
            self._sender.sendall(
                grok_cli_runner._DOCKER_PROVIDER_START_MARKER
            )
            return {"docker_state": "running", "init_pid": 123}

        def finish_provider_input(self, payload: str = "") -> None:
            assert self._fence_published
            self._sender.sendall(payload.encode("utf-8"))
            self._sender.shutdown(socket.SHUT_WR)
            self._sender.close()

        def _abort_provider_start(self) -> None:
            self._sender.close()
            self._docker_stdin.close()

        def bind_isolation_image(self, image_id: str) -> None:
            assert image_id == grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID

        def mark_cas_owned(self) -> None:
            pytest.fail("unscoped fallback must remain lease-owned")

        def mark_cas_terminal(self) -> None:
            pytest.fail("unscoped fallback cannot claim CAS terminality")

        def close(self, *, docker_run_finished: bool) -> None:
            close_calls.append(docker_run_finished)
            self._sender.close()
            self._docker_stdin.close()

    FakeLease.docker_config.mkdir()

    def fake_create(*_args, **kwargs):
        create_kwargs.append(dict(kwargs))
        return FakeLease()

    class FakeProcess:
        def __init__(self, provider_input: socket.socket) -> None:
            self._provider_input = provider_input
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self) -> int:
            payload = bytearray()
            while True:
                chunk = self._provider_input.recv(4096)
                if not chunk:
                    break
                payload.extend(chunk)
            self._provider_input.close()
            provider_payloads.append(bytes(payload))
            return 0 if outcome == "success" else -15

    def fake_run(command, **_kwargs):
        create_command = list(command)
        create_commands.append(create_command)
        boundary_events.append("create")
        assert create_command == ["docker", "create"]
        return subprocess.CompletedProcess(
            create_command,
            0,
            stdout=(created_container_id + "\n").encode("ascii"),
            stderr=b"",
        )

    def fake_popen(command, **kwargs):
        boundary_events.append("popen")
        assert list(command) == [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "--config",
            str(FakeLease.docker_config),
            "start",
            "--attach",
            "--interactive",
            created_container_id,
        ]
        if outcome == "error":
            raise OSError("docker launch failed")
        provider_input = kwargs["stdin"]
        assert isinstance(provider_input, socket.socket)
        duplicate = socket.fromfd(
            provider_input.fileno(),
            socket.AF_UNIX,
            socket.SOCK_STREAM,
        )
        popen_calls.append(True)
        return FakeProcess(duplicate)

    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_codex_quota_fallback_home",
        lambda **_kwargs: (
            FakeHome(),
            grok_cli_runner._codex_task_container_environment(),
            source_auth,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerContainerLease,
        "create",
        fake_create,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_task_toolchain_image_id",
        lambda *_args, **_kwargs: (
            grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
        ),
    )
    def fake_docker_codex_fallback_command(**kwargs: object) -> list[str]:
        assert kwargs["provider_home"] == provider_home
        return ["docker", "create"]

    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_codex_fallback_command",
        fake_docker_codex_fallback_command,
    )
    validate_auth = grok_cli_runner._validated_codex_auth_path

    def record_and_validate_auth(**kwargs):
        try:
            validated = validate_auth(**kwargs)
        except (OSError, ValueError):
            boundary_events.append("auth-fail")
            raise
        boundary_events.append("auth")
        return validated

    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_codex_auth_path",
        record_and_validate_auth,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)

    def invocation() -> int:
        def swap_auth() -> None:
            boundary_events.append("route")
            replacement = tmp_path / "replacement-auth.json"
            replacement.write_text("{}\n", encoding="utf-8")
            replacement.chmod(0o600)
            source_auth.unlink()
            source_auth.symlink_to(replacement)

        def validate_route() -> None:
            boundary_events.append("route")

        return grok_cli_runner._run_codex_quota_fallback_in_docker(
            _terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            prompt="repair",
            prompt_path=prompt_path,
            base_env={},
            pre_effect_validator=(
                swap_auth if outcome == "auth_swap" else validate_route
            ),
        )
    if outcome in {"error", "auth_swap"}:
        with pytest.raises((OSError, ValueError)):
            invocation()
    else:
        expected_returncode = 0 if outcome == "success" else -15
        assert invocation() == expected_returncode

    assert close_calls == [expected_finished]
    if outcome == "auth_swap":
        assert popen_calls == []
        assert create_commands == []
        assert boundary_events == ["route", "auth-fail"]
    else:
        assert create_commands == [["docker", "create"]]
        assert popen_calls == ([] if outcome == "error" else [True])
        assert boundary_events == [
            "route",
            "auth",
            "route",
            "create",
            "popen",
        ]
        if outcome != "error":
            assert provider_payloads == [
                grok_cli_runner._DOCKER_PROVIDER_START_MARKER + b"repair"
            ]
    assert create_kwargs == [
        {
            "provider": "codex",
            "provider_home": provider_home,
            "prompt_path": prompt_path,
        }
    ]

def test_codex_fallback_without_signed_network_authority_is_pre_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    prompt_path = tmp_path / "asref-grok-prompt-test.txt"
    prompt_path.write_text("repair", encoding="utf-8")
    isolated = tmp_path / "asref-codex-home-test"
    isolated.mkdir()

    class FakeHome:
        name = str(isolated)

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: str(codex),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_isolated_codex_quota_fallback_home",
        lambda **_kwargs: (
            FakeHome(),
            grok_cli_runner._codex_task_container_environment(),
            source_auth,
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner._DockerContainerLease,
        "create",
        lambda *_args, **_kwargs: pytest.fail(
            "unsigned authority must not allocate a Docker lease"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "unsigned authority must not create a container"
        ),
    )
    monkeypatch.setattr(
        grok_cli_runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail(
            "unsigned authority must not start a container"
        ),
    )

    with pytest.raises(ValueError, match="lacks signed network authority"):
        grok_cli_runner._run_codex_quota_fallback_in_docker(
            _terra_fallback_command(str(codex), workspace),
            workspace=workspace,
            prompt="repair",
            prompt_path=prompt_path,
            base_env={},
        )

def test_real_disposable_codex_container_and_board_toolchain_probe(
    tmp_path: Path,
) -> None:
    workspace = Path(__file__).resolve().parents[2]
    docker_bin = grok_cli_runner._docker_isolation_binary()
    codex = grok_cli_runner.resolve_codex_quota_fallback_executable(
        workspace=workspace,
    )
    if not docker_bin or not codex:
        pytest.skip("trusted local Docker/Codex boundary is unavailable")
    vendor = implementation_daemon._host_codex_vendor_binaries()
    if vendor is None:
        pytest.skip("matching native Codex/code-mode vendor pair is unavailable")
    host_codex = str(vendor[0].resolve(strict=True))
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    inspect_docker_config = tmp_path / "inspect-docker-config"
    inspect_docker_config.mkdir()
    image = grok_cli_runner._docker_codex_task_toolchain_image_id(
        docker_bin,
        docker_config=inspect_docker_config,
    )
    if not image:
        pytest.skip("pinned local Docker image is unavailable")
    assert image == grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    child_env = grok_cli_runner._codex_task_container_environment()
    probe_network = f"eaaef-probe-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    created_network = subprocess.run(
        [
            docker_bin,
            f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}",
            "--config",
            str(inspect_docker_config),
            "network",
            "create",
            "--internal",
            probe_network,
        ],
        env=grok_cli_runner._docker_control_env(child_env),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if created_network.returncode != 0:
        pytest.skip(
            "could not create diagnostic worker network: "
            + (created_network.stderr or created_network.stdout)[-400:]
        )

    def create_start_wait_and_cleanup(
        create_command: list[str],
        *,
        container_name: str,
        cidfile: Path,
        docker_config: Path,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert create_command[4] == "create"
        assert "run" not in create_command
        assert "--rm" not in create_command
        assert "--pull=never" in create_command
        labels = [
            create_command[index + 1]
            for index, item in enumerate(create_command[:-1])
            if item == "--label"
        ]
        assert "ipfs_accelerate.codex_fallback_isolation=true" in labels
        assert create_command.index(image) > create_command.index("--label")
        container_id = ""
        try:
            created = subprocess.run(
                create_command,
                cwd=workspace,
                env=grok_cli_runner._docker_control_env(child_env),
                input="",
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert created.returncode == 0, created.stderr
            created_fields = created.stdout.split()
            assert len(created_fields) == 1
            container_id = created_fields[0]
            assert re.fullmatch(r"[0-9a-f]{64}", container_id)
            assert cidfile.read_text(encoding="ascii").strip() == container_id

            inspected = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "container",
                    "inspect",
                    container_id,
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert inspected.returncode == 0, inspected.stderr
            inspection = json.loads(inspected.stdout)
            assert isinstance(inspection, list) and len(inspection) == 1
            record = inspection[0]
            assert record["Id"] == container_id
            assert record["Name"] == "/" + container_name
            assert record["Image"] == image
            assert record["State"]["Status"] == "created"
            assert record["Config"]["Labels"].get(
                "ipfs_accelerate.codex_fallback_isolation"
            ) == "true"

            started = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "start",
                    "--attach",
                    "--interactive",
                    container_id,
                ],
                cwd=workspace,
                env=grok_cli_runner._docker_control_env(child_env),
                input="",
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            waited = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "container",
                    "wait",
                    container_id,
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert waited.returncode == 0, waited.stderr
            assert waited.stdout.split() == [str(started.returncode)]
            return started
        finally:
            grok_cli_runner._remove_exact_docker_container(
                docker_bin=docker_bin,
                docker_config=docker_config,
                container_name=container_name,
                settle_for_creation=False,
            )
            absent = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(docker_config),
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{container_name}$",
                    "--format",
                    "{{.Names}}",
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert absent.returncode == 0, absent.stderr
            assert absent.stdout.strip() == ""

    def network_bound_command(
        *,
        container_name: str,
        label: str,
        docker_config: Path,
        cidfile: Path,
        lease_root: Path,
    ) -> list[str]:
        network_values = {
            "provider": "codex",
            "docker_network": probe_network,
            "proxy_endpoint": "http://172.30.0.2:3128",
            "approval_identity": f"eaaef-network-approval:{label}",
            "effect_cid": "sha256:"
            + hashlib.sha256(container_name.encode("utf-8")).hexdigest(),
            "workspace": workspace,
            "container_name": container_name,
            "lease_id": lease_root.name,
            "lease_root": lease_root,
        }
        return grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(codex, workspace),
            workspace=workspace,
            source_auth=source_auth,
            child_env=child_env,
            docker_config=docker_config,
            container_name=container_name,
            cidfile=cidfile,
            docker_bin=docker_bin,
            isolation_image=image,
            network_profile=WorkerNetworkProfile(
                **network_values,
                allowed_hostnames=PROVIDER_HOSTNAME_ALLOWLISTS["codex"],
                approval_cid=worker_network_approval_cid(**network_values),
            ),
        )

    version_lease_root = tmp_path / "asref-codex-container-version"
    version_docker_config = version_lease_root / "docker-config"
    version_docker_config.mkdir(parents=True)
    version_container_name = (
        f"ipfs-accelerate-codex-{os.getpid()}-{uuid.uuid4().hex}"
    )
    version_cidfile = version_lease_root / "container.cid"
    command = network_bound_command(
        container_name=version_container_name,
        label="version",
        docker_config=version_docker_config,
        cidfile=version_cidfile,
        lease_root=version_lease_root,
    )
    image_index = command.index(image)
    codex_index = command.index(host_codex, image_index + 1)
    probe_command = [*command[:codex_index], host_codex, "--version"]

    completed = create_start_wait_and_cleanup(
        probe_command,
        container_name=version_container_name,
        cidfile=version_cidfile,
        docker_config=version_docker_config,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    version_lines = completed.stdout.splitlines()
    assert len(version_lines) == 2
    capacity_start = json.loads(version_lines[0])
    assert capacity_start["schema"] == (
        grok_cli_runner.AGENT_IMPLEMENTATION_CODEX_CAPACITY_LOG_SENTINEL_SCHEMA
    )
    assert capacity_start["type"] == "runner.capacity.start"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", capacity_start["log_nonce"])
    assert version_lines[1].startswith("codex-cli ")
    assert "bwrap:" not in completed.stderr

    validation_lease_root = tmp_path / "asref-codex-container-validation"
    validation_docker_config = validation_lease_root / "docker-config"
    validation_docker_config.mkdir(parents=True)
    validation_container_name = (
        f"ipfs-accelerate-codex-{os.getpid()}-{uuid.uuid4().hex}"
    )
    validation_cidfile = validation_lease_root / "container.cid"
    validation_command = network_bound_command(
        container_name=validation_container_name,
        label="validation",
        docker_config=validation_docker_config,
        cidfile=validation_cidfile,
        lease_root=validation_lease_root,
    )
    validation_image_index = validation_command.index(image)
    validation_codex_index = validation_command.index(
        host_codex,
        validation_image_index + 1,
    )
    validation_command[validation_codex_index:] = [
        str(grok_cli_runner._CODEX_TASK_TOOLCHAIN_PYTHON),
        "-c",
        (
            "import pytest,sys; "
            "print('toolchain', sys.version.split()[0], pytest.__version__)"
        ),
    ]
    validation = create_start_wait_and_cleanup(
        validation_command,
        container_name=validation_container_name,
        cidfile=validation_cidfile,
        docker_config=validation_docker_config,
        timeout=30,
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert "toolchain" in validation.stdout
    subprocess.run(
        [
            docker_bin,
            f"--host={grok_cli_runner._DOCKER_LOCAL_HOST}",
            "--config",
            str(inspect_docker_config),
            "network",
            "rm",
            probe_network,
        ],
        env=grok_cli_runner._docker_control_env(child_env),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

def test_real_disposable_codex_container_and_board_toolchain_probe_aseh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )

    workspace = Path(__file__).resolve().parents[2]
    docker_bin = grok_cli_runner._docker_isolation_binary()
    if os.environ.get(_REQUIRE_LIVE_DOCKER_CLEANUP_VALIDATION_ENV) == "1":
        configured_codex = os.environ.get(_TEST_CODEX_EXECUTABLE_ENV)
        if configured_codex != _EXACT_TEST_CODEX_EXECUTABLE:
            pytest.fail(
                "live Docker cleanup validation requires the exact reviewed "
                f"{_TEST_CODEX_EXECUTABLE_ENV}="
                f"{_EXACT_TEST_CODEX_EXECUTABLE} opt-in"
            )
        codex = grok_cli_runner.resolve_codex_quota_fallback_executable(
            workspace=workspace,
            configured=configured_codex,
        )
    else:
        codex = grok_cli_runner.resolve_codex_quota_fallback_executable(
            workspace=workspace,
        )
    if not docker_bin or not codex:
        _skip_or_fail_live_cleanup_validation(
            "trusted local Docker/Codex boundary is unavailable"
        )
    state_root = tmp_path / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True)
    profile = LifecycleProfile(
        target_id="live-codex-toolchain-probe",
        run_id="run-live-codex-toolchain-probe",
        configuration_root="sha256:" + "a" * 64,
        repository_root=str(workspace),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(workspace),
    )
    for name, value in profile.launch_environment(11).items():
        monkeypatch.setenv(name, value)
    observation_docker_config = tmp_path / "docker-config"
    observation_docker_config.mkdir()
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    image = grok_cli_runner._docker_codex_task_toolchain_image_id(
        docker_bin,
        docker_config=observation_docker_config,
    )
    if not image:
        _skip_or_fail_live_cleanup_validation(
            "pinned local Docker image is unavailable"
        )
    assert image == grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    child_env = grok_cli_runner._codex_task_container_environment()
    version_provider_home = Path(
        tempfile.mkdtemp(prefix="asref-codex-home-version-")
    )
    validation_provider_home = Path(
        tempfile.mkdtemp(prefix="asref-codex-home-validation-")
    )

    def create_start_wait_and_cleanup(
        lease: grok_cli_runner._DockerContainerLease,
        create_command: list[str],
        *,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        assert create_command[4] == "create"
        assert "run" not in create_command
        assert "--rm" not in create_command
        assert "--pull=never" in create_command
        label_index = create_command.index("--label")
        assert create_command[label_index + 1] == (
            "ipfs_accelerate.codex_fallback_isolation=true"
        )
        assert create_command.index(image) > label_index
        container_id = ""
        started_process: subprocess.Popen[str] | None = None
        try:
            lease.bind_isolation_image(image)
            created = lease.create_inert_container(
                create_command,
                cwd=workspace,
                env=grok_cli_runner._docker_control_env(child_env),
            )
            assert created.returncode == 0, created.stderr.decode(
                errors="replace"
            )
            created_fields = created.stdout.decode("ascii").split()
            assert len(created_fields) == 1
            container_id = created_fields[0]
            assert re.fullmatch(r"[0-9a-f]{64}", container_id)
            assert lease.cidfile.read_text(
                encoding="ascii"
            ).strip() == container_id

            inspected = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(lease.docker_config),
                    "container",
                    "inspect",
                    container_id,
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert inspected.returncode == 0, inspected.stderr
            inspection = json.loads(inspected.stdout)
            assert isinstance(inspection, list) and len(inspection) == 1
            record = inspection[0]
            assert record["Id"] == container_id
            assert record["Name"] == "/" + lease.container_name
            assert record["Image"] == image
            assert record["State"]["Status"] == "created"
            assert record["Config"]["Labels"].get(
                "ipfs_accelerate.codex_fallback_isolation"
            ) == "true"

            provider_stdin = lease.take_provider_start_stdin()
            try:
                started_process = subprocess.Popen(
                    [
                        docker_bin,
                        "--host=unix:///var/run/docker.sock",
                        "--config",
                        str(lease.docker_config),
                        "start",
                        "--attach",
                        "--interactive",
                        container_id,
                    ],
                    cwd=workspace,
                    env=grok_cli_runner._docker_control_env(child_env),
                    stdin=provider_stdin,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            finally:
                provider_stdin.close()
            termination_fence = lease.capture_running_termination_fence()
            assert termination_fence["container_id"] == container_id
            assert termination_fence["container_name"] == lease.container_name
            assert termination_fence["docker_state"] == "running"
            assert int(termination_fence["init_pid"]) > 0
            assert termination_fence["kernel_scope"]
            lease.finish_provider_input()
            stdout, stderr = started_process.communicate(timeout=timeout)
            started = subprocess.CompletedProcess(
                started_process.args,
                started_process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
            waited = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(lease.docker_config),
                    "container",
                    "wait",
                    container_id,
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert waited.returncode == 0, waited.stderr
            assert waited.stdout.split() == [str(started.returncode)]
            return started
        finally:
            docker_run_finished = bool(
                started_process is not None
                and started_process.poll() is not None
            )
            lease.close(docker_run_finished=docker_run_finished)
            if started_process is not None and started_process.poll() is None:
                try:
                    started_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    started_process.kill()
                    started_process.wait(timeout=2.0)
            absent = subprocess.run(
                [
                    docker_bin,
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(observation_docker_config),
                    "container",
                    "ls",
                    "--all",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{lease.container_name}$",
                    "--format",
                    "{{.Names}}",
                ],
                env=grok_cli_runner._docker_control_env(child_env),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            assert absent.returncode == 0, absent.stderr
            assert absent.stdout.strip() == ""
            assert not lease.lease_root.exists()

    version_prompt_fd, version_prompt_name = tempfile.mkstemp(
        prefix="asref-grok-prompt-version-"
    )
    os.close(version_prompt_fd)
    version_prompt_path = Path(version_prompt_name)
    version_prompt_path.chmod(0o600)
    version_lease = grok_cli_runner._DockerContainerLease.create(
        docker_bin,
        provider="codex",
        provider_home=version_provider_home,
        prompt_path=version_prompt_path,
    )
    try:
        command = grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(codex, workspace),
            workspace=workspace,
            source_auth=source_auth,
            provider_home=version_provider_home,
            child_env=child_env,
            docker_config=version_lease.docker_config,
            container_name=version_lease.container_name,
            cidfile=version_lease.cidfile,
            docker_bin=docker_bin,
            isolation_image=image,
        )
        image_index = command.index(image)
        provider_argv0 = command.index(
            "aseh-provider-start",
            image_index + 1,
        ) + 1
        probe_command = [
            *command[:provider_argv0],
            command[provider_argv0],
            "--version",
        ]
        completed = create_start_wait_and_cleanup(
            version_lease,
            probe_command,
            timeout=30,
        )
    finally:
        version_lease.close(docker_run_finished=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("codex-cli ")
    assert "bwrap:" not in completed.stderr

    validation_prompt_fd, validation_prompt_name = tempfile.mkstemp(
        prefix="asref-grok-prompt-validation-"
    )
    os.close(validation_prompt_fd)
    validation_prompt_path = Path(validation_prompt_name)
    validation_prompt_path.chmod(0o600)
    validation_lease = grok_cli_runner._DockerContainerLease.create(
        docker_bin,
        provider="codex",
        provider_home=validation_provider_home,
        prompt_path=validation_prompt_path,
    )
    try:
        validation_command = grok_cli_runner._docker_codex_fallback_command(
            codex_command=_terra_fallback_command(codex, workspace),
            workspace=workspace,
            source_auth=source_auth,
            provider_home=validation_provider_home,
            child_env=child_env,
            docker_config=validation_lease.docker_config,
            container_name=validation_lease.container_name,
            cidfile=validation_lease.cidfile,
            docker_bin=docker_bin,
            isolation_image=image,
        )
        validation_image_index = validation_command.index(image)
        validation_argv0 = validation_command.index(
            "aseh-provider-start",
            validation_image_index + 1,
        ) + 1
        validation_command[validation_argv0:] = [
            "python",
            "-m",
            "pytest",
            "-q",
            "--color=no",
            "-p",
            "no:cacheprovider",
            "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
            "test/api/test_agent_supervisor_managed_daemon_kernel_fence.py",
            "test/api/test_agent_supervisor_grok_quota_terra_gate.py",
            "-k",
            (
                "state_authority_handoff_rejects_unqualified_ptrace_scope or "
                "state_authority_handoff_rejects_cap_sys_ptrace_peer or "
                "aseh_repair_provider_cleanup_fence_transition_is_closed_and_chained or "
                "reparented_nondumpable_daemon_session_prevents_false_death or "
                "forged_managed_daemon_sidecar_is_unknown_and_never_signalled or "
                "exact_kernel_bound_daemon_is_fenced_by_birth_and_group or "
                "docker_create_positive_grammar_admits_canonical_vendor_command_only or "
                "provider_start_socket_cannot_be_reopened_through_proc or "
                "fenced_docker_issuer_uses_three_nonreopenable_socketpairs or "
                "concurrent_docker_create_calls_dispatch_at_most_once"
            ),
        ]
        validation = create_start_wait_and_cleanup(
            validation_lease,
            validation_command,
            # Qualify the current R11 security boundary inside the exact
            # pinned toolchain. Prompt-v3 convergence remains recorded as its
            # own R10 known-baseline rather than coupling unrelated history to
            # this isolation probe.
            timeout=30,
        )
    finally:
        validation_lease.close(docker_run_finished=False)

    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert re.search(r"\b10 passed\b", validation.stdout)

def test_daemon_liveness_accepts_exact_codex_fallback_container_label(
    tmp_path: Path,
    monkeypatch,
) -> None:
    daemon = _daemon(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        if command[:2] == ["docker", "ps"]:
            label = command[command.index("--filter") + 1]
            stdout = "codex-container\n" if "codex_fallback" in label else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                [{"Source": str(tmp_path), "Destination": str(tmp_path)}]
            ),
        )

    monkeypatch.setattr(implementation_daemon.subprocess, "run", fake_run)

    assert daemon._docker_isolation_active_for_worktree(str(tmp_path)) is True
    filters = [
        call[call.index("--filter") + 1]
        for call in calls
        if call[:2] == ["docker", "ps"]
    ]
    assert filters == [
        "label=ipfs_accelerate.grok_isolation=true",
        "label=ipfs_accelerate.codex_fallback_isolation=true",
    ]

def test_build_grok_quota_routed_agent_command_embeds_terra_shape(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_k: "/usr/local/bin/codex",
    )
    command = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable="/usr/bin/python3",
        grok_bin="/usr/bin/grok",
        codex_bin="/usr/local/bin/codex",
    )
    assert command[:3] == [
        "/usr/bin/python3",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    ]
    assert command[command.index("--model") + 1] == "grok-4.6"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback
    assert "--ephemeral" in fallback

def test_build_grok_quota_routed_agent_command_embeds_terra_shape_aseh(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_k: "/usr/local/bin/codex",
    )
    command = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=tmp_path,
        python_executable="/usr/bin/python3",
        grok_bin="/usr/bin/grok",
        codex_bin="/usr/local/bin/codex",
    )
    assert command[:4] == [
        "/usr/bin/python3",
        "-B",
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    ]
    assert command[command.index("--model") + 1] == "grok-4.6"
    fallback = json.loads(command[command.index("--codex-fallback-command-json") + 1])
    assert fallback[fallback.index("-m") + 1] == "gpt-5.6-terra"
    assert 'model_reasoning_effort="medium"' in fallback
    assert "--ephemeral" in fallback

def test_quota_grok_command_authorizes_canonical_legacy_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    def unexpected_secret_store_access(*_args, **_kwargs):
        pytest.fail("Grok route construction requested an unrelated secret fingerprint")

    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/usr/bin/grok",
    )
    monkeypatch.setattr(llm_router, "_grok_cli_auth_available", lambda: True)
    monkeypatch.setattr(llm_router, "_get_grok_cli_provider", lambda: object())
    monkeypatch.setattr(
        llm_router,
        "meta_model_api_key_fingerprint",
        unexpected_secret_store_access,
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        provider_executable_trust,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/usr/local/bin/codex",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/usr/local/bin/codex",
    )
    command = implementation_daemon._grok_cli_command(workspace_path=tmp_path)
    assert "--codex-fallback-command-json" in command
    assert "--canonical-legacy-preflight-route" in command
    assert command[command.index("--model") + 1] == "grok-4.5"

    nonce_bound = implementation_daemon._grok_cli_command(
        workspace_path=tmp_path,
        failure_receipt_nonce="ab" * 32,
    )
    assert "--canonical-legacy-preflight-route" not in nonce_bound

def test_quota_grok_command_authorizes_canonical_legacy_preflight_aseh(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_binary",
        lambda: "/usr/bin/grok",
    )
    monkeypatch.setattr(
        implementation_daemon,
        "_grok_cli_available",
        lambda: True,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "resolve_codex_quota_fallback_executable",
        lambda **_kwargs: "/usr/local/bin/codex",
    )
    command = implementation_daemon._grok_cli_command(workspace_path=tmp_path)
    assert "--codex-fallback-command-json" in command
    assert "--canonical-legacy-preflight-route" in command

    nonce_bound = implementation_daemon._grok_cli_command(
        workspace_path=tmp_path,
        failure_receipt_nonce="ab" * 32,
    )
    assert "--canonical-legacy-preflight-route" not in nonce_bound

def test_incomplete_quota_route_defaults_medium_reasoning_effort(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_PROVIDER_ENV,
        "grok_cli",
    )
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_FALLBACK_PROVIDER_ENV,
        "codex",
    )
    monkeypatch.setenv(
        implementation_daemon.IMPLEMENTATION_FALLBACK_TRIGGER_ENV,
        "primary_quota_exhausted",
    )
    monkeypatch.setenv(implementation_daemon._GROK_MODEL_ENV, "grok-4.6")
    monkeypatch.setenv(implementation_daemon._CODEX_MODEL_ENV, "gpt-5.6-terra")
    monkeypatch.delenv(
        implementation_daemon._CODEX_REASONING_EFFORT_ENV,
        raising=False,
    )

    plan = implementation_daemon._configured_agent_implementation_route_plan(
        tmp_path
    )

    assert plan is not None
    assert plan.fallback_trigger == "primary_quota_exhausted"
    assert plan.fallback_reasoning_effort == "medium"
    assert plan.permits_authentication_unavailable is False

@pytest.mark.parametrize(
    ("bound", "value", "reason"),
    [
        ("_WORKSPACE_FINGERPRINT_MAX_ENTRIES", 1, "entry"),
        ("_WORKSPACE_FINGERPRINT_MAX_BYTES", 3, "byte"),
    ],
)
def test_workspace_fingerprint_rejects_archives_without_partial_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    bound: str, value: int, reason: str,
) -> None:
    archive = tmp_path / "data" / "campaign.pre-reboot" / "worktrees"
    archive.mkdir(parents=True)
    (archive / "candidate.py").write_bytes(b"payload")
    monkeypatch.setattr(grok_cli_runner, bound, value)
    with pytest.raises(ValueError, match=f"{reason} budget exceeded"):
        grok_cli_runner._workspace_content_fingerprint(tmp_path)

def test_workspace_fingerprint_checks_deadline_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "large.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    # Initialization, directory, file, first chunk, second chunk.
    ticks = iter([0.0, 0.0, 0.0, 0.0, 61.0])
    monkeypatch.setattr(grok_cli_runner.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="time budget exceeded"):
        grok_cli_runner._workspace_content_fingerprint(tmp_path)

def test_workspace_fingerprint_does_not_silently_skip_walk_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable_walk(*args, **kwargs):
        kwargs["onerror"](PermissionError("unreadable archive"))
        yield  # pragma: no cover

    monkeypatch.setattr(grok_cli_runner.os, "walk", unreadable_walk)
    with pytest.raises(ValueError, match="unable to fingerprint"):
        grok_cli_runner._workspace_content_fingerprint(tmp_path)

def test_workspace_fingerprint_keeps_ignored_archive_bytes_in_fence(
    tmp_path: Path,
) -> None:
    (tmp_path / ".gitignore").write_text("data/\n")
    archive = tmp_path / "data" / "campaign.pre-reboot" / "worktrees"
    archive.mkdir(parents=True)
    source = archive / "candidate.py"
    source.write_text("before")
    baseline = grok_cli_runner._workspace_content_fingerprint(tmp_path)
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) == baseline
    source.write_text("after")
    assert grok_cli_runner._workspace_content_fingerprint(tmp_path) != baseline

def test_merge_preflight_fingerprint_budget_denies_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "archive.bin").write_bytes(b"oversized archive")
    grok = tmp_path / "grok"
    grok.write_text("#!/bin/sh\nexit 0\n")
    grok.chmod(0o700)
    monkeypatch.setattr(grok_cli_runner.sys, "stdin", io.StringIO("resolve merge"))
    monkeypatch.setattr(grok_cli_runner, "_WORKSPACE_FINGERPRINT_MAX_BYTES", 1)
    monkeypatch.setattr(grok_cli_runner, "_repository_head", lambda _workspace: "a" * 40)
    monkeypatch.setattr(
        grok_cli_runner, "_resolve_trusted_grok_bin", lambda **_kwargs: str(grok),
    )
    for name in (
        "_run_typed_grok_preflight", "_independently_verify_grok_quota",
        "_run_grok_with_typed_failure_capture", "_run_codex_quota_fallback_in_docker",
    ):
        monkeypatch.setattr(
            grok_cli_runner, name,
            lambda *args, **kwargs: pytest.fail("over-budget scan must deny provider dispatch"),
        )
    command = grok_cli_runner.build_grok_quota_routed_agent_command(
        workspace=workspace, python_executable=sys.executable,
        fallback_reasoning_effort="medium", enable_internal_legacy_preflight=True,
    )
    assert grok_cli_runner.main(command[3:]) == 2
    assert "fingerprint byte budget exceeded" in capsys.readouterr().err

@pytest.fixture(scope="session", autouse=True)
def _preload_exact_sealed_grok_native_dependency() -> object:
    """Preload the reviewed R11 native payload only for its live test command."""

    if os.environ.get(_TEST_PRELOAD_GROK_NATIVE_ENV) != "1":
        yield None
        return
    assert (
        os.environ.get(
            "IPFS_ACCELERATE_AGENT_REQUIRE_LIVE_NATIVE_DEPENDENCY_VALIDATION"
        )
        == "1"
    )
    source_text = str(
        os.environ.get(
            "IPFS_ACCELERATE_AGENT_LIVE_NATIVE_DEPENDENCY_SOURCE"
        )
        or ""
    )
    assert source_text
    source = Path(source_text)
    source_stat = os.lstat(source)
    assert (
        source.is_absolute()
        and not source.is_symlink()
        and source.resolve(strict=True) == source
        and source.name == "_duckdb.cpython-312-aarch64-linux-gnu.so"
        and stat.S_ISREG(source_stat.st_mode)
        and source_stat.st_nlink == 1
    )
    pin = llm_router.inspect_agent_supervisor_native_dependency_source(
        source,
        distribution_version="1.5.5",
        engine_version="v1.5.5",
    )
    assert pin.as_dict() == aseh_operator.ASEH_R11_NATIVE_DEPENDENCY_PIN
    native_launch = llm_router.seal_agent_supervisor_native_dependency(
        source,
        expected_pin=pin,
        accepted_authorization_id=aseh_operator._identity(
            b"aseh-r11-grok-live-validation-native-fixture"
        ),
    )
    sealed_fd = native_launch.descriptor.descriptor
    try:
        native_module = llm_router.preload_agent_supervisor_native_dependency(
            native_launch
        )
        assert native_module is sys.modules["_duckdb"]
        assert native_module is sys.modules["duckdb"]
        os.fstat(sealed_fd)
        yield native_launch
    finally:
        try:
            os.close(sealed_fd)
        except OSError:
            pass

def _skip_or_fail_live_cleanup_validation(reason: str) -> None:
    if os.environ.get(_REQUIRE_LIVE_DOCKER_CLEANUP_VALIDATION_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)

def _canonical_codex_docker_create_argv(
    *,
    cwd: Path,
    docker_config: Path,
    cidfile: Path,
    container_name: str,
    image: str,
    container_command: tuple[str, ...] = ("/bin/true",),
    projected_environment: tuple[str, ...] = (),
) -> list[str]:
    """Build the minimal positive-grammar Codex create fixture."""

    environment_assignments = [
        f"{name}={value}"
        for name, value in sorted(
            grok_cli_runner._codex_task_container_environment().items()
        )
    ]
    docker_environment = [
        argument
        for name in projected_environment
        for argument in ("--env", name)
    ]
    return [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(docker_config),
        "create",
        "--pull=never",
        "--interactive",
        "--name",
        container_name,
        "--cidfile",
        str(cidfile),
        "--read-only",
        "--network=bridge",
        "--runtime=runc",
        "--entrypoint=/usr/bin/env",
        "--tmpfs",
        (
            "/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        "--tmpfs",
        (
            "/var/tmp:rw,nosuid,nodev,noexec,mode=0700,"
            f"uid={os.getuid()},gid={os.getgid()}"
        ),
        "--tmpfs",
        (
            f"{grok_cli_runner._CODEX_CONTAINER_HOME}:rw,nosuid,nodev,"
            f"noexec,mode=0700,uid={os.getuid()},gid={os.getgid()}"
        ),
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=1024",
        "--label",
        "ipfs_accelerate.codex_fallback_isolation=true",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--workdir",
        str(cwd),
        *docker_environment,
        image,
        "-i",
        *environment_assignments,
        "/bin/sh",
        "-c",
        grok_cli_runner._DOCKER_PROVIDER_START_SCRIPT,
        "aseh-provider-start",
        *container_command,
    ]

def _same_boot_materialization_fixture(tmp_path: Path):
    """Create one canonical observed-create journal without calling Docker."""

    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    lease_root = tmp_path / "lease"
    docker_config = lease_root / "docker-config"
    lease_root.mkdir(mode=0o700)
    docker_config.mkdir(mode=0o700)
    provider_home = tmp_path / "provider-home"
    provider_home.mkdir(mode=0o700)
    prompt_path = tmp_path / "prompt"
    prompt_path.write_text("prompt", encoding="utf-8")
    cidfile = lease_root / "container.cid"
    container_id = "a" * 64
    cidfile.write_text(container_id + "\n", encoding="ascii")
    container_name = "ipfs-accelerate-codex-123-" + ("b" * 32)
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    environment_id = "sha256:" + ("c" * 64)
    command = _canonical_codex_docker_create_argv(
        cwd=tmp_path,
        docker_config=docker_config,
        cidfile=cidfile,
        container_name=container_name,
        image=image,
    )
    command_id, command_body = grok_cli_runner._docker_create_command_identity(
        provider="codex",
        docker_bin="/usr/bin/docker",
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        cwd=tmp_path,
        environment_id=environment_id,
        expected_image=image,
        argv=command,
    )
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    issuer = {
        "pid": 999_999_991,
        "start_time_ticks": 123,
        "boot_id": boot_id,
        "parent_pid": 999_999_990,
    }
    journal = grok_cli_runner._docker_create_journal_value(
        command_body=command_body,
        command_id=command_id,
        state="create_observed",
        issuer_process_birth=issuer,
        returncode=0,
    )
    grok_cli_runner._write_private_control_record(
        lease_root,
        grok_cli_runner._DOCKER_CREATE_JOURNAL_NAME,
        journal,
        replace_existing=False,
    )
    path_identities = {
        "docker_config": grok_cli_runner._cleanup_path_identity(
            docker_config,
            directory=True,
        ),
        "lease_root": grok_cli_runner._cleanup_path_identity(
            lease_root,
            directory=True,
        ),
        "prompt_path": grok_cli_runner._cleanup_path_identity(
            prompt_path,
            directory=False,
        ),
        "provider_home": grok_cli_runner._cleanup_path_identity(
            provider_home,
            directory=True,
        ),
    }
    record = multi_supervisor_runner._DurableDockerCleanupBinding(
        docker_bin="/usr/bin/docker",
        provider="codex",
        container_name=container_name,
        cleanup_root=tmp_path,
        cleanup_root_identity=(
            grok_cli_runner._docker_cleanup_root_identity(tmp_path)
        ),
        lease_root=lease_root,
        docker_config=docker_config,
        cidfile=cidfile,
        provider_home=provider_home,
        prompt_path=prompt_path,
        effect_observation={},
        create_command_id=command_id,
        create_cwd=tmp_path,
        create_environment_id=environment_id,
        termination_fence={},
        binding_state="command_bound",
        path_identities=path_identities,
        runner_pid=999_999_989,
        runner_start_ticks=456,
        watchdog_pid=issuer["parent_pid"],
        watchdog_start_ticks=789,
        boot_id=boot_id,
        record_path=tmp_path / "binding.json",
        record_device=1,
        record_inode=2,
        record_id="sha256:" + ("d" * 64),
    )
    return multi_supervisor_runner, record, journal, container_id, image

def _created_docker_termination_fence(
    *,
    container_name: str,
    container_id: str,
    image_id: str,
) -> dict[str, object]:
    """Build one validated no-init fence for an exact created container."""

    body: dict[str, object] = {
        "schema": grok_cli_runner._DOCKER_TERMINATION_FENCE_SCHEMA,
        "provider": "codex",
        "container_id": container_id,
        "container_name": container_name,
        "image_id": image_id,
        "isolation_label": "ipfs_accelerate.codex_fallback_isolation",
        "docker_state": "created",
        "init_pid": 0,
        "kernel_scope": {},
    }
    body["fence_id"] = grok_cli_runner._effect_receipt_identity(body)
    return grok_cli_runner._validated_docker_termination_fence(
        body,
        provider="codex",
        container_name=container_name,
        expected_container_id=container_id,
        expected_image_id=image_id,
    )

def _docker_removal_dispatch_fixture(
    root: Path,
    *,
    token: str,
) -> tuple[
    Path,
    dict[str, int],
    dict[str, object],
    dict[str, object],
    Path,
]:
    """Create one private binding and a harmless observable Docker stub."""

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    binding_directory = root / f"bindings-{token}"
    binding_directory.mkdir(mode=0o700)
    binding_path = binding_directory / (token * 64 + ".json")
    docker_config = root / f"docker-config-{token}"
    docker_config.mkdir(mode=0o700)
    invocation_log = root / f"docker-rm-{token}.log"
    docker_stub = root / f"docker-{token}"
    docker_stub.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(invocation_log))}\n",
        encoding="utf-8",
    )
    docker_stub.chmod(0o700)
    docker_metadata = docker_stub.stat()
    container_name = f"ipfs-accelerate-codex-1-{token * 32}"
    container_id = token * 64
    image_id = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    termination_fence = _created_docker_termination_fence(
        container_name=container_name,
        container_id=container_id,
        image_id=image_id,
    )
    binding_record: dict[str, object] = {
        "binding_path": str(binding_path),
        "record_id": "sha256:" + token * 64,
        "provider": "codex",
        "container_name": container_name,
        "docker_bin": str(docker_stub),
        "docker_device": docker_metadata.st_dev,
        "docker_inode": docker_metadata.st_ino,
        "docker_mode": docker_metadata.st_mode,
        "docker_uid": docker_metadata.st_uid,
        "docker_config": str(docker_config),
        "path_identities": {
            "docker_config": grok_cli_runner._cleanup_path_identity(
                docker_config,
                directory=True,
            ),
        },
        "termination_fence": termination_fence,
    }
    grok_cli_runner._write_private_control_record(
        binding_path.parent,
        binding_path.name,
        binding_record,
        replace_existing=False,
    )
    binding_identity = grok_cli_runner._cleanup_path_identity(
        binding_path,
        directory=False,
    )
    return (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    )

def _wait_for_docker_removal_dispatch(
    binding_path: Path,
    *,
    states: set[str],
    timeout: float = 8.0,
) -> dict[str, object]:
    """Wait for one admitted removal state without treating time as evidence."""

    dispatch_path = grok_cli_runner._docker_removal_dispatch_path(binding_path)
    deadline = time.monotonic() + timeout
    observed: dict[str, object] | None = None
    while time.monotonic() < deadline:
        try:
            candidate = grok_cli_runner._read_private_control_record(
                dispatch_path.parent,
                dispatch_path.name,
            )
        except ValueError:
            # Atomic replacement can leave an already-open descriptor pointing
            # at the just-unlinked predecessor.  The secure reader rejects that
            # stale inode; retrying observes no state and admits no evidence.
            time.sleep(0.01)
            continue
        if candidate is not None:
            observed = candidate
            if candidate.get("state") in states:
                return candidate
        time.sleep(0.01)
    pytest.fail(
        "Docker removal dispatch did not reach "
        f"{sorted(states)!r}; last observed record was {observed!r}"
    )

def _dead_same_boot_process_birth() -> dict[str, object]:
    """Return an exact-shaped birth identity that is absent on this boot."""

    return {
        "pid": 2_000_000_000,
        "start_time_ticks": 1,
        "boot_id": Path("/proc/sys/kernel/random/boot_id")
        .read_text(encoding="ascii")
        .strip(),
        "parent_pid": 1_999_999_999,
    }

def _configure_docker_removal_stub(
    binding_record: dict[str, object],
    invocation_log: Path,
    *,
    dispatch_snapshot: Path | None = None,
    returncode: int = 0,
    timeout: bool = False,
) -> None:
    """Configure the exact harmless executable used by a removal fixture."""

    docker_stub = Path(str(binding_record["docker_bin"]))
    lines = ["#!/bin/sh"]
    if dispatch_snapshot is not None:
        binding_path = Path(str(binding_record["binding_path"]))
        dispatch_path = grok_cli_runner._docker_removal_dispatch_path(
            binding_path
        )
        lines.append(
            "/bin/cp "
            f"{shlex.quote(str(dispatch_path))} "
            f"{shlex.quote(str(dispatch_snapshot))}"
        )
    lines.append(
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(invocation_log))}"
    )
    if timeout:
        # ``exec`` ensures the bounded issuer kills the actual sleeping process,
        # rather than leaving a shell child behind after timeout reconciliation.
        lines.append("exec /bin/sleep 30")
    else:
        lines.append(f"exit {returncode}")
    docker_stub.write_text("\n".join(lines) + "\n", encoding="utf-8")
    docker_stub.chmod(0o700)

def test_quota_classifier_accepts_live_prompt_usage_envelope() -> None:
    transcript = (
        'num_turns":45,"total_cost_usd":0.580093Error: Internal error: '
        '{"message":"API error (status 402 Payment Required): '
        'Grok Build usage balance exhausted","http_status":402,'
        '"promptUsage":{"inputTokens":1,"outputTokens":1}}'
    )
    assert grok_cli_runner._grok_quota_exhausted(transcript) is True
    parsed = grok_cli_runner.parse_grok_quota_error(transcript)
    assert parsed == {
        "kind": "usage_balance_exhausted",
        "http_status": 402,
    }

def test_docker_cleanup_watchdog_cannot_write_candidate_bytecode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeLauncher:
        returncode = 0

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            return 0

        def kill(self) -> None:
            pytest.fail("admitted cleanup launcher was killed")

    class FakeWatchdog:
        pid = 424242

        def poll(self) -> None:
            return None

        def wait(self, *, timeout: float) -> int:
            assert timeout > 0
            return 0

        def terminate(self) -> None:
            pytest.fail("admitted cleanup watchdog was terminated")

        def kill(self) -> None:
            pytest.fail("admitted cleanup watchdog was killed")

    def fake_popen(command: list[str], **kwargs: object) -> FakeLauncher:
        captured["command"] = list(command)
        captured.update(kwargs)
        return FakeLauncher()

    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        grok_cli_runner,
        "_remove_exact_docker_container",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_read_detached_docker_cleanup_watchdog",
        lambda *_args, **_kwargs: FakeWatchdog(),
    )
    provider_home = tmp_path / "asref-codex-home-test"
    provider_home.mkdir(mode=0o700)
    prompt_path = tmp_path / "asref-grok-prompt-test"
    prompt_path.write_text("test", encoding="utf-8")
    prompt_path.chmod(0o600)
    lease = grok_cli_runner._DockerContainerLease.create(
        "/usr/bin/docker",
        provider="codex",
        provider_home=provider_home,
        prompt_path=prompt_path,
    )
    try:
        command = captured["command"]
        assert isinstance(command, list)
        assert command[:3] == [grok_cli_runner.sys.executable, "-I", "-B"]
        assert command[3] == str(Path(grok_cli_runner.__file__).resolve())
        assert command[4] == (
            grok_cli_runner._DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG
        )
        assert grok_cli_runner._DOCKER_CLEANUP_WATCHDOG_ARG in command
        assert captured["cwd"] == "/"
    finally:
        lease.close(docker_run_finished=False)

def test_docker_cleanup_binding_ignores_repository_root_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in grok_cli_runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        grok_cli_runner.REPOSITORY_ROOT_ENV,
        "/tmp/aseh-repo-root",
    )
    assert (
        grok_cli_runner._docker_cleanup_binding_path(
            "ipfs-accelerate-grok-1-" + ("a" * 32),
            create_directory=False,
        )
        is None
    )
    environment = grok_cli_runner._docker_cleanup_watchdog_env()
    assert grok_cli_runner.REPOSITORY_ROOT_ENV not in environment

def test_docker_cleanup_binding_ignores_provider_lifecycle_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in grok_cli_runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    state_root = tmp_path / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True)
    monkeypatch.setenv(grok_cli_runner.RUN_ID_ENV, "run-provider-leak")
    monkeypatch.setenv(grok_cli_runner.PROFILE_ID_ENV, "profile-provider-leak")
    monkeypatch.setenv(grok_cli_runner.TARGET_ID_ENV, "target-provider-leak")
    monkeypatch.setenv(grok_cli_runner.STATE_ROOT_ENV, str(state_root))
    monkeypatch.setenv(grok_cli_runner.RUN_ROOT_ENV, str(run_root))
    monkeypatch.setenv(grok_cli_runner.FENCING_EPOCH_ENV, "0")
    monkeypatch.setenv(
        grok_cli_runner.CONFIGURATION_ROOT_ENV,
        "sha256:" + "a" * 64,
    )
    assert (
        grok_cli_runner._docker_cleanup_binding_path(
            "ipfs-accelerate-grok-1-" + ("b" * 32),
            create_directory=False,
        )
        is None
    )
    environment = grok_cli_runner._docker_cleanup_watchdog_env()
    for name in grok_cli_runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        assert name not in environment

def test_docker_cleanup_binding_rejects_asymmetric_docker_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in grok_cli_runner._DOCKER_WATCHDOG_LIFECYCLE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        grok_cli_runner.STATE_ROOT_ENV,
        str(tmp_path / "state"),
    )
    with pytest.raises(
        ValueError,
        match="Docker cleanup lifecycle binding is partial",
    ):
        grok_cli_runner._docker_cleanup_binding_path(
            "ipfs-accelerate-grok-1-" + ("c" * 32),
            create_directory=False,
        )
    with pytest.raises(
        ValueError,
        match="Docker cleanup watchdog lifecycle identity is partial",
    ):
        grok_cli_runner._docker_cleanup_watchdog_env()

def test_docker_cleanup_root_survives_ambient_tempdir_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator_root = (tmp_path / "private-allocator").resolve()
    allocator_root.mkdir(mode=0o700)
    lease_root = allocator_root / "asref-codex-container-test"
    provider_home = allocator_root / "asref-codex-home-test"
    prompt_path = allocator_root / "asref-grok-prompt-test"
    lease_root.mkdir(mode=0o700)
    provider_home.mkdir(mode=0o700)
    prompt_path.write_bytes(b"")
    prompt_path.chmod(0o600)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN",
        "must-not-reach-cleanup-watchdog",
    )

    environment = grok_cli_runner._docker_cleanup_watchdog_env()
    cleanup_root, cleanup_root_identity = (
        grok_cli_runner._validated_docker_cleanup_root(
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
        )
    )
    monkeypatch.setattr(tempfile, "tempdir", None)
    readmitted_root, readmitted_identity = (
        grok_cli_runner._validated_docker_cleanup_root(
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
            expected_root=cleanup_root,
            expected_identity=cleanup_root_identity,
        )
    )

    assert "TMPDIR" not in environment
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment
    assert readmitted_root == cleanup_root == allocator_root
    assert readmitted_identity == cleanup_root_identity

    drifted_identity = dict(cleanup_root_identity)
    drifted_identity["inode"] += 1
    with pytest.raises(ValueError, match="cleanup root identity drifted"):
        grok_cli_runner._validated_docker_cleanup_root(
            lease_root=lease_root,
            provider_home=provider_home,
            prompt_path=prompt_path,
            expected_root=cleanup_root,
            expected_identity=drifted_identity,
        )

    allocator_root.chmod(0o755)
    try:
        with pytest.raises(ValueError, match="cleanup root identity is unsafe"):
            grok_cli_runner._validated_docker_cleanup_root(
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
            )
    finally:
        allocator_root.chmod(0o700)

    relocated_root = tmp_path / "relocated-private-allocator"
    allocator_root.rename(relocated_root)
    allocator_root.symlink_to(relocated_root, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="cleanup root identity is unavailable"):
            grok_cli_runner._validated_docker_cleanup_root(
                lease_root=lease_root,
                provider_home=provider_home,
                prompt_path=prompt_path,
            )
    finally:
        allocator_root.unlink()
        relocated_root.rename(allocator_root)

def test_durable_cleanup_binding_uses_recorded_root_after_ambient_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
        ProcessIdentity,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    cleanup_root = (tmp_path / "private-cleanup-root").resolve()
    cleanup_root.mkdir(mode=0o700)
    state_root = cleanup_root / "state"
    run_root = state_root / "run"
    binding_directory = run_root / "provider-cleanup-bindings"
    binding_directory.mkdir(parents=True, mode=0o700)
    binding_directory.chmod(0o700)
    profile = LifecycleProfile(
        target_id="recorded-cleanup-root-test",
        run_id="run-recorded-cleanup-root-test",
        configuration_root="sha256:" + ("7" * 64),
        repository_root=str(tmp_path),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=(sys.executable, "supervisor.py"),
        cwd=str(tmp_path),
    )
    for name, value in profile.launch_environment(5).items():
        monkeypatch.setenv(name, value)

    lease_root = cleanup_root / "asref-codex-container-recorded"
    docker_config = lease_root / "docker-config"
    provider_home = cleanup_root / "asref-codex-home-recorded"
    prompt_path = cleanup_root / "asref-grok-prompt-recorded"
    lease_root.mkdir(mode=0o700)
    docker_config.mkdir(mode=0o700)
    provider_home.mkdir(mode=0o700)
    prompt_path.write_bytes(b"")
    prompt_path.chmod(0o600)
    cidfile = lease_root / "container.cid"
    container_name = "ipfs-accelerate-codex-123-" + ("9" * 32)
    binding_path = binding_directory / (
        hashlib.sha256(container_name.encode("ascii")).hexdigest() + ".json"
    )
    birth = grok_cli_runner.read_process_birth(os.getpid())
    assert birth is not None
    path_identities = {
        "docker_config": grok_cli_runner._cleanup_path_identity(
            docker_config,
            directory=True,
        ),
        "lease_root": grok_cli_runner._cleanup_path_identity(
            lease_root,
            directory=True,
        ),
        "prompt_path": grok_cli_runner._cleanup_path_identity(
            prompt_path,
            directory=False,
        ),
        "provider_home": grok_cli_runner._cleanup_path_identity(
            provider_home,
            directory=True,
        ),
    }
    binding = grok_cli_runner._docker_cleanup_binding_value(
        binding_state="prepared_no_dispatch",
        provider="codex",
        docker_bin="/usr/bin/docker",
        container_name=container_name,
        lease_root=lease_root,
        docker_config=docker_config,
        cidfile=cidfile,
        provider_home=provider_home,
        prompt_path=prompt_path,
        effect_observation={},
        path_identities=path_identities,
        binding_path=binding_path,
        runner_pid=os.getpid(),
        runner_start_ticks=birth.start_time_ticks,
        watchdog_pid=os.getpid(),
        watchdog_start_ticks=birth.start_time_ticks,
    )
    grok_cli_runner._write_private_control_record(
        binding_directory,
        binding_path.name,
        binding,
        replace_existing=False,
    )

    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.delenv("TMPDIR", raising=False)
    records = multi_supervisor_runner._durable_docker_cleanup_bindings(
        profile,
        fencing_epoch=5,
    )
    assert len(records) == 1
    assert records[0].cleanup_root == cleanup_root
    assert dict(records[0].cleanup_root_identity) == binding[
        "cleanup_root_identity"
    ]

    watchdog_argv = (
        sys.executable,
        str(Path(grok_cli_runner.__file__).resolve()),
        grok_cli_runner._DOCKER_CLEANUP_WATCHDOG_LAUNCHER_ARG,
        "--control-fd",
        "9",
        grok_cli_runner._DOCKER_CLEANUP_WATCHDOG_ARG,
        "--provider",
        "codex",
        "--docker-bin",
        "/usr/bin/docker",
        "--container-name",
        container_name,
        "--cidfile",
        str(cidfile),
        "--lease-root",
        str(lease_root),
        "--provider-home",
        str(provider_home),
        "--prompt-path",
        str(prompt_path),
        "--cleanup-binding-record",
        str(binding_path),
        "--runner-pid",
        str(os.getpid()),
        "--runner-start-ticks",
        str(birth.start_time_ticks),
    )
    identity = ProcessIdentity(
        pid=os.getpid(),
        start_time_ticks=birth.start_time_ticks,
        parent_pid=1,
        process_group_id=os.getpgrp(),
        session_id=os.getsid(0),
        boot_id=birth.boot_id,
        argv=watchdog_argv,
        cwd="/",
        executable=str(Path(sys.executable).resolve()),
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=5,
        configuration_root=profile.configuration_root,
    )
    assert multi_supervisor_runner._detached_docker_cleanup_binding(
        identity,
        runner_pid=os.getpid(),
        runner_start_ticks=birth.start_time_ticks,
        runner_boot_id=birth.boot_id,
        records=records,
    ) == ("/usr/bin/docker", container_name, str(lease_root))

def test_docker_cleanup_watchdog_detaches_beyond_strict_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        CONFIGURATION_ROOT_ENV,
        FENCING_EPOCH_ENV,
        PROFILE_ID_ENV,
        REPOSITORY_ROOT_ENV,
        RUN_ID_ENV,
        RUN_ROOT_ENV,
        STATE_ROOT_ENV,
        TARGET_ID_ENV,
        LifecycleProfile,
        LinuxProcessAdapter,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    docker = Path("/usr/bin/docker")
    if not docker.exists():
        _skip_or_fail_live_cleanup_validation(
            "trusted Docker CLI is unavailable"
        )
    probe = subprocess.run(
        [
            str(docker),
            "--host=unix:///var/run/docker.sock",
            "info",
            "--format",
            "{{.ServerVersion}}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    if probe.returncode != 0:
        _skip_or_fail_live_cleanup_validation(
            "local Docker control plane is unavailable"
        )
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    image_probe = subprocess.run(
        [
            str(docker),
            "--host=unix:///var/run/docker.sock",
            "image",
            "inspect",
            image,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    if image_probe.returncode != 0:
        _skip_or_fail_live_cleanup_validation(
            "sealed task image is unavailable"
        )

    repository_root = tmp_path.resolve()
    state_root = repository_root / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True)
    profile = LifecycleProfile(
        target_id="detached-cleanup-test",
        run_id="run-detached-cleanup-test",
        configuration_root="sha256:" + "a" * 64,
        repository_root=str(repository_root),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=("python3", "supervisor.py"),
        cwd=str(repository_root),
    )
    lifecycle_environment = profile.launch_environment(7)
    for name in (
        RUN_ID_ENV,
        PROFILE_ID_ENV,
        TARGET_ID_ENV,
        REPOSITORY_ROOT_ENV,
        STATE_ROOT_ENV,
        RUN_ROOT_ENV,
        FENCING_EPOCH_ENV,
        CONFIGURATION_ROOT_ENV,
    ):
        monkeypatch.setenv(name, lifecycle_environment[name])
    authority_environment = {
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "quack-secret",
        "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT": "quack:127.0.0.1:1",
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR": "/tmp/forbidden-inbox",
        "IPFS_ACCELERATE_AGENT_STATE_OWNER_SOCKET": "/tmp/forbidden.sock",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SOCKET": "/tmp/broker.sock",
        "IPFS_ACCELERATE_AGENT_STATE_GRANT_BROKER_SECRET_FD": "99",
    }
    for name, value in authority_environment.items():
        monkeypatch.setenv(name, value)

    provider_home = Path(tempfile.mkdtemp(prefix="asref-codex-home-"))
    prompt_fd, prompt_name = tempfile.mkstemp(prefix="asref-grok-prompt-")
    os.close(prompt_fd)
    prompt_path = Path(prompt_name)
    lease = grok_cli_runner._DockerContainerLease.create(
        str(docker),
        provider="codex",
        provider_home=provider_home,
        prompt_path=prompt_path,
    )
    lease_root = lease.lease_root
    started_process: subprocess.Popen[bytes] | None = None
    try:
        lease.bind_isolation_image(image)
        sentinel = "ASEH_DURABLE_ENV_SENTINEL=watchdog-pipe-preserved"
        docker_environment = grok_cli_runner._docker_control_env(
            {
                "PATH": "/usr/bin:/bin",
                "HOME": "/nonexistent",
                "ASEH_DURABLE_ENV_SENTINEL": "watchdog-pipe-preserved",
            }
        )
        created = lease.create_inert_container(
            _canonical_codex_docker_create_argv(
                cwd=repository_root,
                docker_config=lease.docker_config,
                cidfile=lease.cidfile,
                container_name=lease.container_name,
                image=image,
                container_command=("/bin/sleep", "300"),
                projected_environment=("ASEH_DURABLE_ENV_SENTINEL",),
            ),
            cwd=repository_root,
            env=docker_environment,
        )
        assert created.returncode == 0, created.stderr.decode(errors="replace")
        journal = grok_cli_runner._validated_docker_create_journal(
            lease_root=lease.lease_root,
            provider="codex",
            docker_bin=str(docker),
            docker_config=lease.docker_config,
            container_name=lease.container_name,
            cidfile=lease.cidfile,
        )
        assert journal is not None
        assert journal["state"] == "create_observed"
        provider_stdin = lease.take_provider_start_stdin()
        try:
            started_process = subprocess.Popen(
                [
                    str(docker),
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(lease.docker_config),
                    "start",
                    "--attach",
                    "--interactive",
                    lease.container_name,
                ],
                env=docker_environment,
                stdin=provider_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            provider_stdin.close()
        termination_fence = lease.capture_running_termination_fence()
        lease.finish_provider_input()
        assert termination_fence["container_name"] == lease.container_name
        assert termination_fence["docker_state"] == "running"
        assert int(termination_fence["init_pid"]) > 0
        assert termination_fence["kernel_scope"]
        assert lease.cleanup_binding_record is not None
        assert lease.cleanup_binding_record.exists()
        durable_bindings = (
            multi_supervisor_runner._durable_docker_cleanup_bindings(
                profile,
                fencing_epoch=7,
            )
        )
        assert len(durable_bindings) == 1
        assert durable_bindings[0].container_name == lease.container_name
        assert durable_bindings[0].create_cwd == repository_root
        assert durable_bindings[0].create_environment_id == journal[
            "environment_id"
        ]
        assert durable_bindings[0].termination_fence == termination_fence
        environment_inspection = subprocess.run(
            [
                str(docker),
                "--host=unix:///var/run/docker.sock",
                "--config",
                str(lease.docker_config),
                "container",
                "inspect",
                "--format",
                "{{range .Config.Env}}{{println .}}{{end}}",
                lease.container_name,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )
        assert environment_inspection.returncode == 0
        assert sentinel.encode("ascii") in environment_inspection.stdout.splitlines()
        raw = Path(f"/proc/{lease._watchdog.pid}/stat").read_text(
            encoding="ascii"
        )
        closing_parenthesis = raw.rfind(")")
        fields = raw[closing_parenthesis + 2 :].split()
        assert int(fields[1]) == 1
        assert lease._watchdog.poll() is None
        watchdog_environment = Path(
            f"/proc/{lease._watchdog.pid}/environ"
        ).read_bytes()
        watchdog_entries = {
            item.partition(b"=")[0]: item.partition(b"=")[2]
            for item in watchdog_environment.split(b"\0")
            if b"=" in item
        }
        for name, value in authority_environment.items():
            assert name.encode() not in watchdog_entries
            assert value.encode() not in watchdog_entries.values()
        tree = LinuxProcessAdapter().snapshot(profile)
        assert lease._watchdog.pid in {member.pid for member in tree.members}
    finally:
        lease.close(docker_run_finished=False)
        if started_process is not None:
            started_process.communicate(timeout=10.0)
    assert lease._watchdog.poll() == 0
    assert not lease_root.exists()
    assert lease.cleanup_binding_record is not None
    assert not lease.cleanup_binding_record.exists()
    assert lease._watchdog.pid not in {
        member.pid for member in LinuxProcessAdapter().snapshot(profile).members
    }

def test_docker_cleanup_waits_for_late_create_before_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"remove": 0, "list": 0}
    clock = [0.0]
    container_name = "ipfs-accelerate-codex-123-" + ("a" * 32)

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if "rm" in command:
            calls["remove"] += 1
            return SimpleNamespace(
                returncode=0 if calls["remove"] > 1 else 1,
                stdout=b"",
            )
        calls["list"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout=(
                container_name.encode("ascii")
                if calls["list"] == 2
                else b""
            ),
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        grok_cli_runner.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        grok_cli_runner.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_DOCKER_CLEANUP_TIMEOUT_SECONDS",
        0.25,
    )
    grok_cli_runner._remove_exact_docker_container(
        docker_bin="/usr/bin/docker",
        docker_config=tmp_path,
        container_name=container_name,
        settle_for_creation=True,
    )

    # An unfenced reserved name is observation-only: cleanup samples the
    # exact-name view through the full settle window and never dispatches a
    # destructive rm.
    assert calls["remove"] == 0
    assert calls["list"] == 4
    assert clock[0] == pytest.approx(0.3)

def test_observed_create_cleanup_reconciles_timed_out_rm_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "ipfs-accelerate-codex-1-" + "d" * 32
    clock = [0.0]
    calls = {"remove": 0, "list": 0}

    def fake_run(command: list[str], **_kwargs: object):
        if "rm" in command:
            calls["remove"] += 1
            raise subprocess.TimeoutExpired(command, 2.0)
        assert "ls" in command
        calls["list"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout=(
                container_name.encode("ascii")
                if calls["list"] < 3
                else b""
            ),
        )

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        grok_cli_runner.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        grok_cli_runner.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_DOCKER_CLEANUP_TIMEOUT_SECONDS",
        0.25,
    )
    container_id = "d" * 64
    termination_fence = _created_docker_termination_fence(
        container_name=container_name,
        container_id=container_id,
        image_id=grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
    )

    grok_cli_runner._remove_exact_docker_container(
        docker_bin="/usr/bin/docker",
        docker_config=tmp_path,
        container_name=container_name,
        settle_for_creation=False,
        termination_fence=termination_fence,
        issue_removal=True,
    )

    # One fenced rm may have an unknown CLI outcome.  Reconciliation observes
    # both exact CID and exact name twice and never replays the destructive call.
    assert calls == {"remove": 1, "list": 6}
    assert clock[0] == pytest.approx(0.2)

def test_docker_removal_arm_uses_clean_launcher_and_durable_state_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token="a")
    dispatch_snapshot = tmp_path / "request-started.json"
    _configure_docker_removal_stub(
        binding_record,
        invocation_log,
        dispatch_snapshot=dispatch_snapshot,
    )
    release_thread = threading.Event()
    other_thread = threading.Thread(target=release_thread.wait, daemon=True)
    other_thread.start()
    real_write = grok_cli_runner._write_private_control_record
    prepared_records: list[dict[str, object]] = []

    def capture_prepared(
        directory: Path,
        name: str,
        value: object,
        *,
        replace_existing: bool,
        directory_fd: int | None = None,
    ) -> None:
        if (
            name.endswith(".remove-dispatched")
            and isinstance(value, dict)
            and value.get("state") == "prepared"
        ):
            prepared_records.append(dict(value))
        real_write(
            directory,
            name,
            value,
            replace_existing=replace_existing,
            directory_fd=directory_fd,
        )

    def forbidden_parent_fork() -> int:
        raise AssertionError(
            "the multithreaded supervisor parent called Python os.fork"
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_write_private_control_record",
        capture_prepared,
    )
    monkeypatch.setattr(grok_cli_runner.os, "fork", forbidden_parent_fork)

    try:
        assert grok_cli_runner._arm_docker_removal_once(
            binding_path=binding_path,
            expected_binding_identity=binding_identity,
            binding_record=binding_record,
            termination_fence=termination_fence,
        ) is False
    finally:
        release_thread.set()
        other_thread.join(timeout=2.0)

    completed = _wait_for_docker_removal_dispatch(
        binding_path,
        states={"request_completed"},
    )
    started = json.loads(dispatch_snapshot.read_text(encoding="utf-8"))
    assert len(prepared_records) == 1
    prepared = prepared_records[0]
    assert prepared["state"] == "prepared"
    assert prepared["generation"] == 1
    assert prepared["previous_dispatch_id"] == ""
    assert started["schema"] == grok_cli_runner._DOCKER_REMOVAL_DISPATCH_SCHEMA
    assert started["state"] == "request_started"
    assert started["generation"] == 1
    assert started["previous_dispatch_id"] == prepared["dispatch_id"]
    assert started["docker_returncode"] is None
    assert started["failure_kind"] == ""
    assert completed["state"] == "request_completed"
    assert completed["generation"] == 1
    assert completed["previous_dispatch_id"] == started["dispatch_id"]
    assert completed["docker_returncode"] == 0
    assert completed["failure_kind"] == ""
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    invocation = shlex.split(invocations[0])
    assert invocation[:2] == [
        "--host=unix:///var/run/docker.sock",
        "--config",
    ]
    assert invocation[2].startswith("/proc/self/fd/")
    assert invocation[3:] == [
        "rm",
        "--force",
        str(termination_fence["container_id"]),
    ]

    # A terminal dispatch is an observation-only idempotency barrier.  It
    # never authorizes a second destructive request.
    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    assert invocation_log.read_text(encoding="utf-8").splitlines() == invocations

def test_docker_binding_lock_name_replacement_cannot_split_exclusion(
    tmp_path: Path,
) -> None:
    binding_directory = tmp_path / "bindings"
    binding_directory.mkdir(mode=0o700)
    binding_path = binding_directory / ("a" * 64 + ".json")
    first = grok_cli_runner._docker_binding_lock_descriptor(binding_path)
    lock_path = binding_path.with_suffix(".lock")
    displaced = binding_path.with_suffix(".lock.displaced")
    try:
        lock_path.rename(displaced)
        lock_path.touch(mode=0o600)
        lock_path.chmod(0o600)
        with pytest.raises(ValueError, match="lock is contended"):
            grok_cli_runner._docker_binding_lock_descriptor(
                binding_path,
                deadline=time.monotonic() + 0.05,
            )
    finally:
        first.close()

    successor = grok_cli_runner._docker_binding_lock_descriptor(binding_path)
    successor.close()

def test_docker_binding_lock_rejects_canonical_directory_replacement(
    tmp_path: Path,
) -> None:
    binding_directory = tmp_path / "bindings"
    binding_directory.mkdir(mode=0o700)
    binding_path = binding_directory / ("b" * 64 + ".json")
    first = grok_cli_runner._docker_binding_lock_descriptor(binding_path)
    displaced_directory = tmp_path / "bindings.displaced"
    try:
        binding_directory.rename(displaced_directory)
        binding_directory.mkdir(mode=0o700)
        with pytest.raises(ValueError, match="directory is not owned"):
            first.write(
                binding_path,
                {"state": "must-not-publish"},
                replace_existing=False,
            )
        assert not binding_path.exists()
        with pytest.raises(ValueError, match="lock is contended"):
            grok_cli_runner._docker_binding_lock_descriptor(
                binding_path,
                deadline=time.monotonic() + 0.05,
            )
    finally:
        first.close()

    successor = grok_cli_runner._docker_binding_lock_descriptor(binding_path)
    successor.close()

def test_docker_removal_arm_failure_before_prepared_has_no_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token="b")

    def fail_before_write(
        directory: Path,
        name: str,
        value: object,
        *,
        replace_existing: bool,
        directory_fd: int | None = None,
    ) -> None:
        del directory, value, replace_existing, directory_fd
        if name.endswith(".remove-dispatched"):
            raise OSError("synthetic pre-publication failure")
        raise AssertionError("unexpected private-record write")

    monkeypatch.setattr(
        grok_cli_runner,
        "_write_private_control_record",
        fail_before_write,
    )

    with pytest.raises(OSError, match="pre-publication"):
        grok_cli_runner._arm_docker_removal_once(
            binding_path=binding_path,
            expected_binding_identity=binding_identity,
            binding_record=binding_record,
            termination_fence=termination_fence,
        )
    # The clean launcher sees EOF but cannot cross the effect boundary without
    # the exact durable prepared record.  Give it ample time to fail closed.
    time.sleep(0.5)
    assert not invocation_log.exists()
    assert not grok_cli_runner._docker_removal_dispatch_path(
        binding_path
    ).exists()

def test_docker_removal_arm_failure_after_prepared_still_converges_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token="8")
    real_write = grok_cli_runner._write_private_control_record
    injected = False

    def write_then_raise(
        directory: Path,
        name: str,
        value: object,
        *,
        replace_existing: bool,
        directory_fd: int | None = None,
    ) -> None:
        nonlocal injected
        real_write(
            directory,
            name,
            value,
            replace_existing=replace_existing,
            directory_fd=directory_fd,
        )
        if (
            not injected
            and name.endswith(".remove-dispatched")
            and isinstance(value, dict)
            and value.get("state") == "prepared"
        ):
            injected = True
            raise OSError("synthetic post-publication failure")

    monkeypatch.setattr(
        grok_cli_runner,
        "_write_private_control_record",
        write_then_raise,
    )
    with pytest.raises(OSError, match="post-publication"):
        grok_cli_runner._arm_docker_removal_once(
            binding_path=binding_path,
            expected_binding_identity=binding_identity,
            binding_record=binding_record,
            termination_fence=termination_fence,
        )
    terminal = _wait_for_docker_removal_dispatch(
        binding_path,
        states={"request_completed"},
    )
    assert terminal["generation"] == 1
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1

    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    assert invocation_log.read_text(encoding="utf-8").splitlines() == invocations

def test_dead_prepared_removal_rearms_one_chained_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token="c")
    dead_prepared = grok_cli_runner._docker_removal_dispatch_value(
        binding_path=binding_path,
        binding_record=binding_record,
        termination_fence=termination_fence,
        state="prepared",
        generation=1,
        previous_dispatch_id="",
        issuer_process_birth=_dead_same_boot_process_birth(),
        docker_returncode=None,
        failure_kind="",
    )
    dispatch_path = grok_cli_runner._docker_removal_dispatch_path(binding_path)
    grok_cli_runner._write_private_control_record(
        dispatch_path.parent,
        dispatch_path.name,
        dead_prepared,
        replace_existing=False,
    )
    real_write = grok_cli_runner._write_private_control_record
    replacement_prepared: list[dict[str, object]] = []

    def capture_replacement(
        directory: Path,
        name: str,
        value: object,
        *,
        replace_existing: bool,
        directory_fd: int | None = None,
    ) -> None:
        if (
            name.endswith(".remove-dispatched")
            and isinstance(value, dict)
            and value.get("state") == "prepared"
            and value.get("generation") == 2
        ):
            replacement_prepared.append(dict(value))
        real_write(
            directory,
            name,
            value,
            replace_existing=replace_existing,
            directory_fd=directory_fd,
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_write_private_control_record",
        capture_replacement,
    )
    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    completed = _wait_for_docker_removal_dispatch(
        binding_path,
        states={"request_completed"},
    )
    assert len(replacement_prepared) == 1
    assert replacement_prepared[0]["generation"] == 2
    assert replacement_prepared[0]["previous_dispatch_id"] == dead_prepared[
        "dispatch_id"
    ]
    assert completed["generation"] == 2
    assert completed["docker_returncode"] == 0
    assert len(invocation_log.read_text(encoding="utf-8").splitlines()) == 1

def test_dead_request_started_becomes_unknown_without_removal_replay(
    tmp_path: Path,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token="d")
    prepared = grok_cli_runner._docker_removal_dispatch_value(
        binding_path=binding_path,
        binding_record=binding_record,
        termination_fence=termination_fence,
        state="prepared",
        generation=1,
        previous_dispatch_id="",
        issuer_process_birth=_dead_same_boot_process_birth(),
        docker_returncode=None,
        failure_kind="",
    )
    request_started = grok_cli_runner._docker_removal_dispatch_value(
        binding_path=binding_path,
        binding_record=binding_record,
        termination_fence=termination_fence,
        state="request_started",
        generation=1,
        previous_dispatch_id=str(prepared["dispatch_id"]),
        issuer_process_birth=prepared["issuer_process_birth"],
        docker_returncode=None,
        failure_kind="",
    )
    dispatch_path = grok_cli_runner._docker_removal_dispatch_path(binding_path)
    grok_cli_runner._write_private_control_record(
        dispatch_path.parent,
        dispatch_path.name,
        request_started,
        replace_existing=False,
    )

    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    unknown = _wait_for_docker_removal_dispatch(
        binding_path,
        states={"request_outcome_unknown"},
    )
    assert unknown["generation"] == 1
    assert unknown["previous_dispatch_id"] == request_started["dispatch_id"]
    assert unknown["docker_returncode"] is None
    assert unknown["failure_kind"] == "issuer_failure"
    assert not invocation_log.exists()

    # Reconciliation is terminal for automatic issuing: an unknown external
    # outcome is never treated as if no request occurred.
    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    assert not invocation_log.exists()

@pytest.mark.parametrize(
    ("token", "stub_returncode", "times_out", "state", "returncode", "failure"),
    [
        ("e", 17, False, "request_completed", 17, ""),
        ("f", 0, True, "request_outcome_unknown", None, "timeout"),
    ],
)
def test_removal_outcome_is_durable_and_never_silently_replayed(
    tmp_path: Path,
    token: str,
    stub_returncode: int,
    times_out: bool,
    state: str,
    returncode: int | None,
    failure: str,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token=token)
    _configure_docker_removal_stub(
        binding_record,
        invocation_log,
        returncode=stub_returncode,
        timeout=times_out,
    )
    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    terminal = _wait_for_docker_removal_dispatch(
        binding_path,
        states={state},
    )
    assert terminal["docker_returncode"] == returncode
    assert terminal["failure_kind"] == failure
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1

    assert grok_cli_runner._arm_docker_removal_once(
        binding_path=binding_path,
        expected_binding_identity=binding_identity,
        binding_record=binding_record,
        termination_fence=termination_fence,
    ) is False
    assert invocation_log.read_text(encoding="utf-8").splitlines() == invocations

def test_docker_removal_contenders_issue_one_request(
    tmp_path: Path,
) -> None:
    (
        binding_path,
        binding_identity,
        binding_record,
        termination_fence,
        invocation_log,
    ) = _docker_removal_dispatch_fixture(tmp_path, token="9")
    barrier = threading.Barrier(8)
    results: list[bool] = []
    failures: list[BaseException] = []
    result_lock = threading.Lock()

    def contender() -> None:
        try:
            barrier.wait(timeout=2.0)
            result = grok_cli_runner._arm_docker_removal_once(
                binding_path=binding_path,
                expected_binding_identity=binding_identity,
                binding_record=binding_record,
                termination_fence=termination_fence,
            )
            with result_lock:
                results.append(result)
        except BaseException as exc:
            with result_lock:
                failures.append(exc)

    threads = [threading.Thread(target=contender) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert results == [False] * 8
    terminal = _wait_for_docker_removal_dispatch(
        binding_path,
        states={"request_completed"},
    )
    assert terminal["generation"] == 1
    assert len(invocation_log.read_text(encoding="utf-8").splitlines()) == 1

def test_durable_create_nonzero_is_unknown_and_preserved_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    docker = Path("/usr/bin/docker")
    if not docker.exists():
        _skip_or_fail_live_cleanup_validation("trusted Docker CLI is unavailable")
    probe = subprocess.run(
        [str(docker), "--host=unix:///var/run/docker.sock", "info"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    if probe.returncode != 0:
        _skip_or_fail_live_cleanup_validation(
            "local Docker control plane is unavailable"
        )
    state_root = tmp_path / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True)
    profile = LifecycleProfile(
        target_id="failed-create-cleanup-test",
        run_id="run-failed-create-cleanup-test",
        configuration_root="sha256:" + ("8" * 64),
        repository_root=str(tmp_path),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    for name, value in profile.launch_environment(8).items():
        monkeypatch.setenv(name, value)
    provider_home = Path(tempfile.mkdtemp(prefix="asref-grok-home-"))
    prompt_fd, prompt_name = tempfile.mkstemp(prefix="asref-grok-prompt-")
    os.close(prompt_fd)
    lease = grok_cli_runner._DockerContainerLease.create(
        str(docker),
        provider="grok",
        provider_home=provider_home,
        prompt_path=Path(prompt_name),
    )
    lease_root = lease.lease_root
    binding = lease.cleanup_binding_record
    try:
        missing_image = "sha256:" + ("0" * 64)
        lease.bind_isolation_image(missing_image)
        result = lease.create_inert_container(
            [
                str(docker),
                "--host=unix:///var/run/docker.sock",
                "--config",
                str(lease.docker_config),
                "create",
                "--pull=never",
                "--interactive",
                "--name",
                lease.container_name,
                "--cidfile",
                str(lease.cidfile),
                "--read-only",
                "--entrypoint=/bin/sh",
                "--tmpfs",
                (
                    "/tmp:rw,nosuid,nodev,noexec,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
                "--tmpfs",
                (
                    "/var/tmp:rw,nosuid,nodev,noexec,mode=0700,"
                    f"uid={os.getuid()},gid={os.getgid()}"
                ),
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=1024",
                "--label",
                "ipfs_accelerate.grok_isolation=true",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--workdir",
                str(tmp_path),
                missing_image,
                "-c",
                grok_cli_runner._DOCKER_PROVIDER_START_SCRIPT,
                "aseh-provider-start",
                "/bin/true",
            ],
            cwd=tmp_path,
            env=grok_cli_runner._docker_control_env(),
        )
        assert result.returncode != 0
        journal = grok_cli_runner._validated_docker_create_journal(
            lease_root=lease_root,
            provider="grok",
            docker_bin=str(docker),
            docker_config=lease.docker_config,
            container_name=lease.container_name,
            cidfile=lease.cidfile,
        )
        assert journal is not None
        assert journal["state"] == "create_outcome_unknown"
        durable = multi_supervisor_runner._durable_docker_cleanup_bindings(
            profile,
            fencing_epoch=8,
        )
        assert len(durable) == 1
        assert multi_supervisor_runner._durable_docker_create_state(
            durable[0]
        ) == "create_outcome_unknown"
    finally:
        lease.close(docker_run_finished=False)
    assert lease._watchdog.wait(timeout=5.0) == 0
    durable = multi_supervisor_runner._durable_docker_cleanup_bindings(
        profile,
        fencing_epoch=8,
    )
    assert len(durable) == 1
    # The runner that issued this request is deliberately still alive, and a
    # nonzero CLI result is not proof that the daemon observed no effect.  The
    # durable fence therefore remains in place on this boot; recovery never
    # converts the outcome to a retryable failure merely to clean test state.
    assert not multi_supervisor_runner._reconcile_durable_docker_cleanup(
        durable[0],
        deadline=time.monotonic() + 12.0,
    )
    observed = subprocess.run(
        [
            str(docker),
            "--host=unix:///var/run/docker.sock",
            "--config",
            str(durable[0].docker_config),
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--filter",
            f"name=^/{durable[0].container_name}$",
            "--format",
            "{{.Names}}",
        ],
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=2.0,
        check=False,
    )
    assert observed.returncode == 0 and not observed.stdout.strip()
    assert binding is not None
    completion = grok_cli_runner._cleanup_completion_path(binding)
    # The canonical finalizer owns resource quarantine and binding retirement
    # under the stable binding lock; callers must not pre-remove its inodes.
    assert multi_supervisor_runner._remove_durable_cleanup_record(durable[0])
    assert not lease_root.exists()
    assert not binding.exists()
    assert grok_cli_runner._cleanup_authority_path(binding).exists()
    assert completion.exists()

def test_watchdog_preserves_durable_claim_without_local_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_root = tmp_path / "asref-codex-container-crashgap"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    container_name = "ipfs-accelerate-codex-123-" + ("b" * 32)
    cleanup = {
        "lease_root": str(lease_root),
        "docker_config": str(docker_config),
        "watchdog_pid": 444,
        "watchdog_start_ticks": 555,
    }
    reservation = SimpleNamespace(
        state="effect_started",
        effect_launch_receipt={"cleanup_receipt": cleanup},
    )
    observer = SimpleNamespace(
        observe=lambda logical_attempt_id: (
            reservation if logical_attempt_id == "attempt-1" else None
        )
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_cleanup_identity",
        lambda _receipt: (lease_root, docker_config, container_name),
    )

    assert not (lease_root / "cas-owned").exists()
    assert grok_cli_runner._observed_provider_attempt_cleanup_state(
        observer,
        logical_attempt_id="attempt-1",
        lease_root=lease_root,
        docker_config=docker_config,
        container_name=container_name,
        watchdog_pid=444,
        watchdog_start_ticks=555,
    ) == "owned"

    reservation.state = "terminal"
    assert grok_cli_runner._observed_provider_attempt_cleanup_state(
        observer,
        logical_attempt_id="attempt-1",
        lease_root=lease_root,
        docker_config=docker_config,
        container_name=container_name,
        watchdog_pid=444,
        watchdog_start_ticks=555,
    ) == "terminal"

    foreign_root = tmp_path / "asref-codex-container-foreign"
    foreign_config = foreign_root / "docker-config"
    foreign_config.mkdir(parents=True)
    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_cleanup_identity",
        lambda _receipt: (
            foreign_root,
            foreign_config,
            "ipfs-accelerate-codex-123-" + ("c" * 32),
        ),
    )
    reservation.state = "effect_started"
    assert grok_cli_runner._observed_provider_attempt_cleanup_state(
        observer,
        logical_attempt_id="attempt-1",
        lease_root=lease_root,
        docker_config=docker_config,
        container_name=container_name,
        watchdog_pid=444,
        watchdog_start_ticks=555,
    ) == "foreign"

def test_terminal_recovery_removes_dead_watchdog_journal_and_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )

    docker = Path("/usr/bin/docker")
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    if not docker.exists():
        _skip_or_fail_live_cleanup_validation("trusted Docker CLI is unavailable")
    probe = subprocess.run(
        [
            str(docker),
            "--host=unix:///var/run/docker.sock",
            "image",
            "inspect",
            image,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    if probe.returncode != 0:
        _skip_or_fail_live_cleanup_validation("sealed task image is unavailable")
    state_root = tmp_path / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True)
    profile = LifecycleProfile(
        target_id="terminal-cleanup-recovery-test",
        run_id="run-terminal-cleanup-recovery-test",
        configuration_root="sha256:" + ("9" * 64),
        repository_root=str(tmp_path),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    for name, value in profile.launch_environment(9).items():
        monkeypatch.setenv(name, value)
    provider_home = Path(tempfile.mkdtemp(prefix="asref-codex-home-"))
    prompt_fd, prompt_name = tempfile.mkstemp(prefix="asref-grok-prompt-")
    os.close(prompt_fd)
    prompt_path = Path(prompt_name)
    lease = grok_cli_runner._DockerContainerLease.create(
        str(docker),
        provider="codex",
        provider_home=provider_home,
        prompt_path=prompt_path,
    )
    command = _canonical_codex_docker_create_argv(
        cwd=tmp_path,
        docker_config=lease.docker_config,
        cidfile=lease.cidfile,
        container_name=lease.container_name,
        image=image,
        container_command=("/bin/sleep", "300"),
    )
    binding = lease.cleanup_binding_record
    lease_root = lease.lease_root
    started_process: subprocess.Popen[bytes] | None = None
    try:
        lease.bind_isolation_image(image)
        created = lease.create_inert_container(
            command,
            cwd=tmp_path,
            env=grok_cli_runner._docker_control_env(),
        )
        assert created.returncode == 0
        provider_stdin = lease.take_provider_start_stdin()
        try:
            started_process = subprocess.Popen(
                [
                    str(docker),
                    "--host=unix:///var/run/docker.sock",
                    "--config",
                    str(lease.docker_config),
                    "start",
                    "--attach",
                    "--interactive",
                    lease.container_name,
                ],
                env=grok_cli_runner._docker_control_env(),
                stdin=provider_stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            provider_stdin.close()
        termination_fence = lease.capture_running_termination_fence()
        lease.finish_provider_input()
        assert termination_fence["docker_state"] == "running"
        assert int(termination_fence["init_pid"]) > 0
        cleanup_receipt: dict[str, object] = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "provider-effect-cleanup@1"
            ),
            "lease_root": str(lease_root),
            "docker_config": str(lease.docker_config),
            "cidfile": str(lease.cidfile),
            "provider_home": str(provider_home),
            "prompt_path": str(prompt_path),
            "watchdog_pid": lease._watchdog.pid,
            "watchdog_start_ticks": lease._watchdog.start_ticks,
        }
        cleanup_receipt["receipt_id"] = (
            grok_cli_runner._effect_receipt_identity(cleanup_receipt)
        )
        launch_receipt = {
            "container_name": lease.container_name,
            "container_id": "sha256:" + lease.cidfile.read_text(
                encoding="ascii"
            ).strip(),
            "image_id": image,
            "cleanup_id": cleanup_receipt["receipt_id"],
            "cleanup_receipt": cleanup_receipt,
            "runtime_receipt": {"path": str(docker)},
            "command_receipt": {"create_argv": command},
        }
        terminal_reservation = SimpleNamespace(
            logical_attempt_id="terminal-cleanup-attempt",
            reservation_id="terminal-cleanup-reservation",
            content_id="sha256:" + ("e" * 64),
            effect_launch_receipt=dict(launch_receipt),
        )
        terminal_outcome = {
            "reservation_id": terminal_reservation.reservation_id,
            "effect_launch_receipt": dict(launch_receipt),
            "fallback_dispatched": True,
            "fallback_returncode": 0,
        }
        cleanup_evidence = (
            grok_cli_runner._recorded_codex_terminal_cleanup_evidence(
                launch_receipt
            )
        )
        terminal_cleanup_authority = (
            provider_attempt_store._terminal_cleanup_authority_value(
                terminal_reservation,
                terminal_outcome=terminal_outcome,
                returncode=0,
                evidence=cleanup_evidence,
            )
        )
        observed_terminal = SimpleNamespace(
            schema=provider_attempt_store.CAS_SCHEMA,
            logical_attempt_id=terminal_reservation.logical_attempt_id,
            reservation_id=terminal_reservation.reservation_id,
            content_id=terminal_reservation.content_id,
            state="terminal",
            terminal=True,
            effect_launch_receipt=dict(launch_receipt),
            terminal_returncode=0,
            terminal_outcome=terminal_outcome,
            terminal_cleanup_authority=terminal_cleanup_authority,
            terminal_cleanup_progress={},
        )

        class TerminalObserver:
            def __init__(self) -> None:
                self.current = observed_terminal

            def observe(self, logical_attempt_id: str):
                return (
                    self.current
                    if logical_attempt_id
                    == terminal_reservation.logical_attempt_id
                    else None
                )

            def commit_terminal_cleanup_intent(
                self,
                reservation,
                *,
                intent: dict[str, object],
            ):
                if self.current.terminal_cleanup_progress:
                    assert self.current.terminal_cleanup_progress[
                        "intent"
                    ] == intent
                    return self.current
                progress = (
                    provider_attempt_store._terminal_cleanup_progress_value(
                        self.current,
                        intent=intent,
                        phase="intent_committed",
                    )
                )
                self.current = SimpleNamespace(
                    **{
                        **vars(self.current),
                        "terminal_cleanup_progress": progress,
                    }
                )
                return self.current

            def commit_terminal_cleanup_completion(
                self,
                reservation,
                *,
                intent_id: str,
                completion_id: str,
            ):
                progress = self.current.terminal_cleanup_progress
                if progress["phase"] == "completion_committed":
                    assert progress["completion_id"] == completion_id
                    return self.current
                assert progress["intent_id"] == intent_id
                committed = (
                    provider_attempt_store._terminal_cleanup_progress_value(
                        self.current,
                        intent=progress["intent"],
                        phase="completion_committed",
                        completion_id=completion_id,
                        previous_progress_id=progress["progress_id"],
                    )
                )
                self.current = SimpleNamespace(
                    **{
                        **vars(self.current),
                        "terminal_cleanup_progress": committed,
                    }
                )
                return self.current

        terminal_observer = TerminalObserver()
        lease._watchdog.kill()
        lease._watchdog.wait(timeout=2.0)

        assert binding is not None
        completion = grok_cli_runner._cleanup_completion_path(binding)
        grok_cli_runner._release_recorded_codex_effect_cleanup(
            launch_receipt,
            terminal_observer=terminal_observer,
            terminal_reservation=terminal_reservation,
        )
        assert not lease_root.exists()
        assert not provider_home.exists()
        assert not prompt_path.exists()
        assert not binding.exists()
        assert grok_cli_runner._cleanup_authority_path(binding).exists()
        assert completion.exists()

        grok_cli_runner._release_recorded_codex_effect_cleanup(
            launch_receipt,
            terminal_observer=terminal_observer,
            terminal_reservation=terminal_reservation,
        )
        assert completion.exists()
    finally:
        lease.close(docker_run_finished=False)
        if started_process is not None:
            started_process.communicate(timeout=10.0)

def test_watchdog_attempt_observer_never_creates_missing_store(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
        DurableProviderAttemptCAS,
        ProviderAttemptStoreError,
    )

    missing = tmp_path / "missing-attempt-store"
    with pytest.raises(
        ProviderAttemptStoreError,
        match="directory is absent",
    ):
        DurableProviderAttemptCAS(missing, create_if_missing=False)
    assert not missing.exists()

    existing = tmp_path / "existing-attempt-store"
    existing.mkdir(mode=0o700)
    observer = DurableProviderAttemptCAS(existing, create_if_missing=False)
    assert observer.observe("absent-attempt") is None
    assert list(existing.iterdir()) == []

def test_strict_fence_rejects_missing_detached_cleanup_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
        ProcessIdentity,
        ProcessTreeSnapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    profile = LifecycleProfile(
        target_id="cleanup-failure-test",
        run_id="run-cleanup-failure-test",
        configuration_root="sha256:" + ("c" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path / "state"),
        run_root=str(tmp_path / "state" / "run"),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    root = ProcessIdentity(
        pid=101,
        start_time_ticks=202,
        parent_pid=1,
        process_group_id=101,
        session_id=101,
        boot_id="boot",
        argv=profile.argv,
        cwd=profile.cwd,
        executable=sys.executable,
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=3,
        configuration_root=profile.configuration_root,
    )
    populated = ProcessTreeSnapshot(
        profile_id=profile.profile_id,
        run_id=profile.run_id,
        members=(root,),
    )
    empty = ProcessTreeSnapshot(
        profile_id=profile.profile_id,
        run_id=profile.run_id,
    )

    class FakeProcess:
        pid = root.pid

        def __init__(self) -> None:
            self.done = False
            self._agent_supervisor_lifecycle_profile = profile
            self._agent_supervisor_process_identity = root

        def poll(self) -> int | None:
            return 0 if self.done else None

        def wait(self, *, timeout: float) -> int:
            assert timeout >= 0
            return 0

    process = FakeProcess()

    class FakeAdapter:
        def snapshot(self, _profile: LifecycleProfile) -> ProcessTreeSnapshot:
            return empty if process.done else populated

        def identity_alive(self, _identity: ProcessIdentity) -> bool:
            return not process.done

        def terminate(
            self,
            _tree: ProcessTreeSnapshot,
            *,
            grace_seconds: float,
            deadline_ms: int,
        ) -> None:
            assert grace_seconds >= 0
            assert deadline_ms > 0
            process.done = True

    clock = [0.0]
    observed: list[tuple[tuple[str, str, str], ...]] = []
    binding = (
        "/usr/bin/docker",
        "ipfs-accelerate-codex-101-" + ("d" * 32),
        str(tmp_path / "missing-cleanup-lease"),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "LinuxProcessAdapter",
        lambda: FakeAdapter(),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_detached_docker_cleanup_bindings",
        lambda *_args, **_kwargs: (binding,),
    )

    def missing(
        bindings: tuple[tuple[str, str, str], ...],
        *,
        deadline: float,
    ) -> bool:
        assert deadline > clock[0]
        observed.append(bindings)
        return False

    monkeypatch.setattr(
        multi_supervisor_runner,
        "_detached_docker_cleanup_absence_verified",
        missing,
    )
    monkeypatch.setattr(
        multi_supervisor_runner.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        multi_supervisor_runner.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + max(0.02, seconds)),
    )

    fenced, members = multi_supervisor_runner._terminate_managed_process(
        process,  # type: ignore[arg-type]
        grace_seconds=0.0,
    )

    assert fenced is False
    assert members == (root.pid,)
    # Missing durable evidence is itself a failed fence.  Do not consult the
    # legacy Docker-name absence observer after the authoritative record scan
    # has failed closed.
    assert observed == []

def test_strict_fence_waits_for_detached_container_cleanup(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
        LinuxProcessAdapter,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    docker = Path("/usr/bin/docker")
    if not docker.exists():
        _skip_or_fail_live_cleanup_validation(
            "trusted Docker CLI is unavailable"
        )
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    probe = subprocess.run(
        [
            str(docker),
            "--host=unix:///var/run/docker.sock",
            "image",
            "inspect",
            image,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
    )
    if probe.returncode != 0:
        _skip_or_fail_live_cleanup_validation(
            "sealed task image is unavailable"
        )

    repository_root = Path.cwd().resolve()
    state_root = tmp_path / "state"
    run_root = state_root / "run"
    run_root.mkdir(parents=True)
    ready_path = tmp_path / "provider-ready.json"
    profile = LifecycleProfile(
        target_id="strict-cleanup-test",
        run_id="run-strict-cleanup-test",
        configuration_root="sha256:" + "b" * 64,
        repository_root=str(repository_root),
        state_root=str(state_root),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "strict-cleanup-test"),
        cwd=str(repository_root),
    )
    codex_environment_assignments = [
        f"{name}={value}"
        for name, value in sorted(
            grok_cli_runner._codex_task_container_environment().items()
        )
    ]
    child_code = "\n".join(
        (
            "import json, os, subprocess, tempfile, time",
            "from pathlib import Path",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime "
                "import grok_cli_runner as runner"
            ),
            "home = Path(tempfile.mkdtemp(prefix='asref-codex-home-'))",
            "prompt_fd, prompt_name = tempfile.mkstemp(prefix='asref-grok-prompt-')",
            "os.close(prompt_fd)",
            (
                "lease = runner._DockerContainerLease.create("
                "'/usr/bin/docker', provider='codex', provider_home=home, "
                "prompt_path=Path(prompt_name))"
            ),
            f"lease.bind_isolation_image({image!r})",
            "command = [",
            "    '/usr/bin/docker', '--host=unix:///var/run/docker.sock',",
            "    '--config', str(lease.docker_config), 'create', '--pull=never',",
            "    '--cidfile', str(lease.cidfile), '--name', lease.container_name,",
            "    '--interactive', '--read-only', '--network=bridge',",
            "    '--runtime=runc', '--entrypoint=/usr/bin/env',",
            f"    '--tmpfs', {'/tmp:rw,nosuid,nodev,noexec,mode=0700,uid=' + str(os.getuid()) + ',gid=' + str(os.getgid())!r},",
            f"    '--tmpfs', {'/var/tmp:rw,nosuid,nodev,noexec,mode=0700,uid=' + str(os.getuid()) + ',gid=' + str(os.getgid())!r},",
            f"    '--tmpfs', {str(grok_cli_runner._CODEX_CONTAINER_HOME) + ':rw,nosuid,nodev,noexec,mode=0700,uid=' + str(os.getuid()) + ',gid=' + str(os.getgid())!r},",
            "    '--cap-drop=ALL',",
            "    '--security-opt=no-new-privileges', '--pids-limit=1024',",
            "    '--label', 'ipfs_accelerate.codex_fallback_isolation=true',",
            f"    '--user', {f'{os.getuid()}:{os.getgid()}'!r},",
            f"    '--workdir', {str(repository_root)!r},",
            (
                f"    {image!r}, '-i', *{codex_environment_assignments!r}, "
                "'/bin/sh', '-c',"
            ),
            (
                f"    {grok_cli_runner._DOCKER_PROVIDER_START_SCRIPT!r}, "
                "'aseh-provider-start', '/bin/sleep', '300',"
            ),
            "]",
            (
                "created = lease.create_inert_container("
                "command, cwd=Path.cwd(), env=runner._docker_control_env())"
            ),
            "if created.returncode != 0: raise SystemExit(91)",
            "provider_stdin = lease.take_provider_start_stdin()",
            (
                "try:\n"
                "    started = subprocess.Popen(["
                "'/usr/bin/docker', '--host=unix:///var/run/docker.sock', "
                "'--config', str(lease.docker_config), 'start', '--attach', "
                "'--interactive', lease.container_name], "
                "env=runner._docker_control_env(), stdin=provider_stdin, "
                "stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n"
                "finally:\n"
                "    provider_stdin.close()"
            ),
            "fence = lease.capture_running_termination_fence()",
            "if int(fence.get('init_pid') or 0) <= 0: raise SystemExit(93)",
            (
                f"Path({str(ready_path)!r}).write_text(json.dumps({{"
                "'lease_root': str(lease.lease_root), "
                "'container_name': lease.container_name, "
                "'watchdog_pid': lease._watchdog.pid, "
                "'fence_id': fence['fence_id']}), encoding='utf-8')"
            ),
            "while True: time.sleep(1)",
        )
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code],
        cwd=repository_root,
        env=profile.launch_environment(11),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    container_name = ""
    lease_root = Path("/nonexistent")
    try:
        adapter = LinuxProcessAdapter()
        identity = adapter._identity(child.pid, profile)
        child._agent_supervisor_lifecycle_profile = profile  # type: ignore[attr-defined]
        child._agent_supervisor_process_identity = identity  # type: ignore[attr-defined]
        # This is a live pre-dispatch efficiency qualification, not the
        # production Docker-create effect timeout. The fixture must reach its
        # running fence without heavyweight supervisor imports consuming the
        # bounded readiness budget.
        ready_timeout = 30.0
        deadline = time.monotonic() + ready_timeout
        while not ready_path.exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                _stdout, stderr = child.communicate(timeout=1.0)
                pytest.fail(
                    "provider fixture exited early: "
                    f"{child.returncode}: {stderr.decode(errors='replace')}"
                )
            time.sleep(0.05)
        if not ready_path.is_file():
            pytest.fail(
                "provider fixture did not publish readiness within its "
                f"{ready_timeout:.0f}-second efficiency bound; "
                f"child_state={'running' if child.poll() is None else child.returncode}"
            )
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
        container_name = str(payload["container_name"])
        lease_root = Path(str(payload["lease_root"]))
        tree = adapter.snapshot(profile)
        assert int(payload["watchdog_pid"]) in {
            member.pid for member in tree.members
        }
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", payload["fence_id"])
        cleanup_bindings = (
            multi_supervisor_runner._detached_docker_cleanup_bindings(
                tree,
                process_pid=child.pid,
                process_start_ticks=identity.start_time_ticks,
                process_boot_id=identity.boot_id,
                records=(
                    multi_supervisor_runner._durable_docker_cleanup_bindings(
                        profile,
                        fencing_epoch=11,
                    )
                ),
            )
        )
        assert cleanup_bindings == (
            (str(docker), container_name, str(lease_root)),
        )

        fenced, stopped_members = (
            multi_supervisor_runner._terminate_managed_process(
                child,
                grace_seconds=10.0,
            )
        )
        assert fenced is True
        assert child.pid in stopped_members
        child.wait(timeout=2.0)

        assert not adapter.snapshot(profile).members
        assert not lease_root.exists()
        inspected = subprocess.run(
            [
                str(docker),
                "--host=unix:///var/run/docker.sock",
                "container",
                "inspect",
                container_name,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        assert inspected.returncode != 0
        assert multi_supervisor_runner._detached_docker_cleanup_absence_verified(
            cleanup_bindings,
            deadline=time.monotonic() + 5.0,
        )
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2.0)
        if container_name:
            subprocess.run(
                [
                    str(docker),
                    "--host=unix:///var/run/docker.sock",
                    "rm",
                    "--force",
                    container_name,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )

@pytest.mark.parametrize(
    (
        "primary_survives",
        "record_runner_liveness",
        "watchdog_liveness",
    ),
    (
        (True, True, False),
        (False, True, False),
        (False, None, False),
        (False, False, True),
    ),
)
def test_strict_fence_never_reconciles_before_both_exact_owners_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary_survives: bool,
    record_runner_liveness: bool | None,
    watchdog_liveness: bool,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
        ProcessIdentity,
        ProcessTreeSnapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    profile = LifecycleProfile(
        target_id="cleanup-owner-gate-test",
        run_id="run-cleanup-owner-gate-test",
        configuration_root="sha256:" + ("d" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path / "state"),
        run_root=str(tmp_path / "state" / "run"),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    root = ProcessIdentity(
        pid=101,
        start_time_ticks=202,
        parent_pid=1,
        process_group_id=101,
        session_id=101,
        boot_id="boot",
        argv=profile.argv,
        cwd=profile.cwd,
        executable=sys.executable,
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=3,
        configuration_root=profile.configuration_root,
    )
    populated = ProcessTreeSnapshot(
        profile_id=profile.profile_id,
        run_id=profile.run_id,
        members=(root,),
    )
    empty = ProcessTreeSnapshot(
        profile_id=profile.profile_id,
        run_id=profile.run_id,
    )
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    record = multi_supervisor_runner._DurableDockerCleanupBinding(
        docker_bin="/usr/bin/docker",
        provider="codex",
        container_name="ipfs-accelerate-codex-101-" + ("e" * 32),
        cleanup_root=tmp_path,
        cleanup_root_identity=(
            grok_cli_runner._docker_cleanup_root_identity(tmp_path)
        ),
        lease_root=tmp_path / "lease",
        docker_config=tmp_path / "lease" / "docker-config",
        cidfile=tmp_path / "lease" / "container.cid",
        provider_home=tmp_path / "provider-home",
        prompt_path=tmp_path / "prompt",
        effect_observation={},
        create_command_id="sha256:" + ("1" * 64),
        create_cwd=tmp_path,
        create_environment_id="sha256:" + ("2" * 64),
        termination_fence={},
        binding_state="command_bound",
        path_identities={},
        runner_pid=root.pid,
        runner_start_ticks=root.start_time_ticks,
        watchdog_pid=303,
        watchdog_start_ticks=404,
        boot_id=current_boot,
        record_path=tmp_path / "record.json",
        record_device=1,
        record_inode=2,
        record_id="sha256:" + ("3" * 64),
    )

    class FakeProcess:
        pid = root.pid

        def __init__(self) -> None:
            self.done = False
            self._agent_supervisor_lifecycle_profile = profile
            self._agent_supervisor_process_identity = root

        def poll(self) -> int | None:
            return 0 if self.done else None

        def wait(self, *, timeout: float) -> int:
            assert timeout >= 0
            return 0

    process = FakeProcess()

    class FakeAdapter:
        def snapshot(self, _profile: LifecycleProfile) -> ProcessTreeSnapshot:
            return empty if process.done else populated

        def identity_alive(self, identity: ProcessIdentity) -> bool:
            return identity.pid == root.pid and not process.done

        def terminate(
            self,
            _tree: ProcessTreeSnapshot,
            *,
            grace_seconds: float,
            deadline_ms: int,
        ) -> None:
            assert grace_seconds >= 0 and deadline_ms > 0
            process.done = not primary_survives

    reconciled: list[str] = []
    clock = [0.0]
    monkeypatch.setattr(
        multi_supervisor_runner,
        "LinuxProcessAdapter",
        lambda: FakeAdapter(),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_durable_docker_cleanup_bindings",
        lambda *_args, **_kwargs: (record,),
    )

    def exact_record_liveness(
        *,
        pid: int,
        start_ticks: int,
        boot_id: str,
    ) -> bool | None:
        assert boot_id == current_boot
        if (pid, start_ticks) == (
            record.runner_pid,
            record.runner_start_ticks,
        ):
            return record_runner_liveness
        if (pid, start_ticks) == (
            record.watchdog_pid,
            record.watchdog_start_ticks,
        ):
            return watchdog_liveness
        pytest.fail("unexpected durable cleanup process birth")

    monkeypatch.setattr(
        multi_supervisor_runner,
        "_exact_process_birth_alive",
        exact_record_liveness,
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_durable_docker_create_state",
        lambda *_args, **_kwargs: "create_observed",
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_reconcile_durable_docker_cleanup",
        lambda *_args, **_kwargs: reconciled.append("called") or True,
    )
    monkeypatch.setattr(
        multi_supervisor_runner.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        multi_supervisor_runner.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + max(0.02, seconds)),
    )

    fenced, _members = multi_supervisor_runner._terminate_managed_process(
        process,  # type: ignore[arg-type]
        grace_seconds=0.0,
    )

    assert fenced is False
    assert reconciled == []

def test_strict_fence_reconciles_record_created_during_shutdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
        ProcessIdentity,
        ProcessTreeSnapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    profile = LifecycleProfile(
        target_id="late-cleanup-record-test",
        run_id="run-late-cleanup-record-test",
        configuration_root="sha256:" + ("f" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path / "state"),
        run_root=str(tmp_path / "state" / "run"),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    root = ProcessIdentity(
        pid=111,
        start_time_ticks=222,
        parent_pid=1,
        process_group_id=111,
        session_id=111,
        boot_id="boot",
        argv=profile.argv,
        cwd=profile.cwd,
        executable=sys.executable,
        run_id=profile.run_id,
        profile_id=profile.profile_id,
        target_id=profile.target_id,
        repository_root=profile.repository_root,
        state_root=profile.state_root,
        run_root=profile.run_root,
        fencing_epoch=4,
        configuration_root=profile.configuration_root,
    )
    populated = ProcessTreeSnapshot(
        profile_id=profile.profile_id,
        run_id=profile.run_id,
        members=(root,),
    )
    empty = ProcessTreeSnapshot(
        profile_id=profile.profile_id,
        run_id=profile.run_id,
    )
    record = multi_supervisor_runner._DurableDockerCleanupBinding(
        docker_bin="/usr/bin/docker",
        provider="codex",
        container_name="ipfs-accelerate-codex-111-" + ("a" * 32),
        cleanup_root=tmp_path,
        cleanup_root_identity=(
            grok_cli_runner._docker_cleanup_root_identity(tmp_path)
        ),
        lease_root=tmp_path / "late-lease",
        docker_config=tmp_path / "late-lease" / "docker-config",
        cidfile=tmp_path / "late-lease" / "container.cid",
        provider_home=tmp_path / "late-home",
        prompt_path=tmp_path / "late-prompt",
        effect_observation={},
        create_command_id="sha256:" + ("4" * 64),
        create_cwd=tmp_path,
        create_environment_id="sha256:" + ("5" * 64),
        termination_fence={},
        binding_state="command_bound",
        path_identities={},
        runner_pid=root.pid,
        runner_start_ticks=root.start_time_ticks,
        watchdog_pid=999_999_998,
        watchdog_start_ticks=1,
        boot_id="not-current",
        record_path=tmp_path / "late-record.json",
        record_device=1,
        record_inode=2,
        record_id="sha256:" + ("6" * 64),
    )

    class FakeProcess:
        pid = root.pid

        def __init__(self) -> None:
            self.done = False
            self._agent_supervisor_lifecycle_profile = profile
            self._agent_supervisor_process_identity = root

        def poll(self) -> int | None:
            return 0 if self.done else None

        def wait(self, *, timeout: float) -> int:
            return 0

    process = FakeProcess()

    class FakeAdapter:
        def snapshot(self, _profile: LifecycleProfile) -> ProcessTreeSnapshot:
            return empty if process.done else populated

        def identity_alive(self, identity: ProcessIdentity) -> bool:
            return identity.pid == root.pid and not process.done

        def terminate(self, *_args: object, **_kwargs: object) -> None:
            process.done = True

    state = {"initial_scan": True, "reconciled": False}
    reconciled: list[str] = []

    def records(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        if state["initial_scan"]:
            state["initial_scan"] = False
            return ()
        return () if state["reconciled"] else (record,)

    def reconcile(*_args: object, **_kwargs: object) -> bool:
        reconciled.append("called")
        state["reconciled"] = True
        return True

    clock = [0.0]
    monkeypatch.setattr(
        multi_supervisor_runner,
        "LinuxProcessAdapter",
        lambda: FakeAdapter(),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_durable_docker_cleanup_bindings",
        records,
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_reconcile_durable_docker_cleanup",
        reconcile,
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_durable_docker_create_state",
        lambda *_args, **_kwargs: "create_observed",
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_detached_docker_cleanup_absence_verified",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        multi_supervisor_runner.time,
        "monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        multi_supervisor_runner.time,
        "sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + max(0.02, seconds)),
    )

    fenced, members = multi_supervisor_runner._terminate_managed_process(
        process,  # type: ignore[arg-type]
        grace_seconds=0.0,
    )

    assert fenced is True
    assert members == (root.pid,)
    assert reconciled == ["called"]

def test_durable_cleanup_record_reader_rejects_symlink(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    run_root = tmp_path / "run"
    directory = run_root / "provider-cleanup-bindings"
    directory.mkdir(parents=True, mode=0o700)
    target = tmp_path / "foreign.json"
    target.write_text("{}\n", encoding="utf-8")
    (directory / (("a" * 64) + ".json")).symlink_to(target)
    profile = LifecycleProfile(
        target_id="symlink-cleanup-record-test",
        run_id="run-symlink-cleanup-record-test",
        configuration_root="sha256:" + ("7" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )

    with pytest.raises(ValueError, match="record"):
        multi_supervisor_runner._durable_docker_cleanup_bindings(
            profile,
            fencing_epoch=1,
        )

def test_durable_cleanup_scan_tolerates_only_a_concurrently_unlinked_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    run_root = tmp_path / "run"
    directory = run_root / "provider-cleanup-bindings"
    directory.mkdir(parents=True, mode=0o700)
    record_path = directory / (("a" * 64) + ".json")
    record_path.write_text("{}\n", encoding="utf-8")
    record_path.chmod(0o600)
    profile = LifecycleProfile(
        target_id="concurrent-cleanup-record-test",
        run_id="run-concurrent-cleanup-record-test",
        configuration_root="sha256:" + ("9" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )

    def unlink_during_read(path: Path, **_kwargs: object) -> dict[str, object]:
        path.unlink()
        raise ValueError("injected admitted unlink race")

    monkeypatch.setattr(
        multi_supervisor_runner,
        "_read_durable_docker_cleanup_record",
        unlink_during_read,
    )

    assert multi_supervisor_runner._durable_docker_cleanup_bindings(
        profile,
        fencing_epoch=1,
    ) == ()

def test_durable_cleanup_scan_rejects_an_extant_failed_record_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    run_root = tmp_path / "run"
    directory = run_root / "provider-cleanup-bindings"
    directory.mkdir(parents=True, mode=0o700)
    record_path = directory / (("b" * 64) + ".json")
    record_path.write_text("{}\n", encoding="utf-8")
    record_path.chmod(0o600)
    profile = LifecycleProfile(
        target_id="extant-cleanup-record-test",
        run_id="run-extant-cleanup-record-test",
        configuration_root="sha256:" + ("a" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    monkeypatch.setattr(
        multi_supervisor_runner,
        "_read_durable_docker_cleanup_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("injected extant read failure")
        ),
    )

    with pytest.raises(ValueError, match="injected extant read failure"):
        multi_supervisor_runner._durable_docker_cleanup_bindings(
            profile,
            fencing_epoch=1,
        )

def test_parent_held_cleanup_directory_anchor_rejects_inode_replacement(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    run_root = tmp_path / "run"
    anchor = multi_supervisor_runner._open_durable_cleanup_directory_anchor(
        run_root
    )
    directory = run_root / "provider-cleanup-bindings"
    displaced = run_root / "provider-cleanup-bindings-displaced"
    profile = LifecycleProfile(
        target_id="cleanup-directory-anchor-test",
        run_id="run-cleanup-directory-anchor-test",
        configuration_root="sha256:" + ("8" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    try:
        directory.rename(displaced)
        directory.mkdir(mode=0o700)

        with pytest.raises(ValueError, match="anchor identity drifted"):
            multi_supervisor_runner._durable_docker_cleanup_bindings(
                profile,
                fencing_epoch=1,
                directory_anchor=anchor,
            )
    finally:
        os.close(anchor.descriptor)

def test_self_hashed_cleanup_completion_cannot_remove_live_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding_directory = tmp_path / "provider-cleanup-bindings"
    binding_directory.mkdir(mode=0o700)
    container_name = "ipfs-accelerate-codex-123-" + ("c" * 32)
    binding_path = binding_directory / (
        hashlib.sha256(container_name.encode("ascii")).hexdigest()
        + ".json"
    )
    completion_path = grok_cli_runner._cleanup_completion_path(binding_path)
    with tempfile.TemporaryDirectory(
        prefix="asref-codex-container-"
    ) as lease_name, tempfile.TemporaryDirectory(
        prefix="asref-codex-home-"
    ) as home_name:
        lease_root = Path(lease_name)
        docker_config = lease_root / "docker-config"
        docker_config.mkdir(mode=0o700)
        provider_home = Path(home_name)
        prompt_descriptor, prompt_name = tempfile.mkstemp(
            prefix="asref-grok-prompt-"
        )
        os.close(prompt_descriptor)
        prompt_path = Path(prompt_name)
        try:
            body: dict[str, object] = {
                "schema": grok_cli_runner._DOCKER_CLEANUP_BINDING_SCHEMA,
                "provider": "codex",
                "docker_bin": "/usr/bin/docker",
                "container_name": container_name,
                "run_id": "run-cleanup-completion-test",
                "profile_id": "profile-cleanup-completion-test",
                "target_id": "target-cleanup-completion-test",
                "repository_root": str(tmp_path),
                "state_root": str(tmp_path),
                "run_root": str(tmp_path),
                "configuration_root": "sha256:" + ("d" * 64),
                "fencing_epoch": 1,
                "cleanup_root": str(lease_root.parent),
                "cleanup_root_identity": (
                    grok_cli_runner._docker_cleanup_root_identity(
                        lease_root.parent
                    )
                ),
                "lease_root": str(lease_root),
                "docker_config": str(docker_config),
                "cidfile": str(lease_root / "container.cid"),
                "provider_home": str(provider_home),
                "prompt_path": str(prompt_path),
                "path_identities": {
                    "docker_config": grok_cli_runner._cleanup_path_identity(
                        docker_config,
                        directory=True,
                    ),
                    "lease_root": grok_cli_runner._cleanup_path_identity(
                        lease_root,
                        directory=True,
                    ),
                    "prompt_path": grok_cli_runner._cleanup_path_identity(
                        prompt_path,
                        directory=False,
                    ),
                    "provider_home": grok_cli_runner._cleanup_path_identity(
                        provider_home,
                        directory=True,
                    ),
                },
                "binding_path": str(binding_path),
            }
            binding_record = dict(body)
            binding_record["record_id"] = (
                grok_cli_runner._effect_receipt_identity(body)
            )
            grok_cli_runner._write_private_control_record(
                binding_directory,
                binding_path.name,
                binding_record,
                replace_existing=False,
            )
            completion_value = grok_cli_runner._cleanup_completion_value(
                binding_path=binding_path,
                binding_identity=grok_cli_runner._cleanup_path_identity(
                    binding_path,
                    directory=False,
                ),
                binding_record=binding_record,
            )
            grok_cli_runner._write_private_control_record(
                binding_directory,
                completion_path.name,
                completion_value,
                replace_existing=False,
            )

            assert not grok_cli_runner._recover_cleanup_completion(
                completion_path,
                docker_absence_verified=True,
            )
            assert binding_path.exists()
            assert lease_root.exists()
            assert docker_config.exists()
            assert provider_home.exists()
            assert prompt_path.exists()

            reobserved: list[dict[str, object]] = []

            def admit_absence(record: dict[str, object]) -> bool:
                reobserved.append(record)
                return record == binding_record

            monkeypatch.setattr(
                grok_cli_runner,
                "_reobserve_cleanup_binding_docker_absence",
                admit_absence,
            )

            # Rename-away plus a forged public marker cannot become cleanup
            # authority. The precommit must observe the exact original inode.
            escaped_prompt = prompt_path.with_name(prompt_path.name + ".escaped")
            prompt_identity = binding_record["path_identities"]["prompt_path"]  # type: ignore[index]
            quarantine, _owned, marker, tombstone = (
                grok_cli_runner._cleanup_path_quarantine(
                    prompt_path,
                    directory=False,
                    identity=prompt_identity,  # type: ignore[arg-type]
                )
            )
            prompt_path.rename(escaped_prompt)
            quarantine.mkdir(mode=0o700)
            grok_cli_runner._write_private_control_record(
                quarantine,
                marker.name,
                tombstone,
                replace_existing=False,
            )
            assert not grok_cli_runner._finalize_verified_cleanup_completion(
                binding_path=binding_path,
                binding_identity=completion_value["binding_identity"],  # type: ignore[arg-type]
                binding_record=binding_record,
            )
            marker.unlink()
            quarantine.rmdir()
            escaped_prompt.rename(prompt_path)
            completion_path.unlink()

            # Retirement is create-only: a raced foreign authority name is
            # never overwritten and the live binding remains untouched.
            authority_path = grok_cli_runner._cleanup_authority_path(
                binding_path
            )
            foreign_authority = {"kind": "foreign-same-uid-record"}
            grok_cli_runner._write_private_control_record(
                binding_directory,
                authority_path.name,
                foreign_authority,
                replace_existing=False,
            )
            assert not grok_cli_runner._retire_cleanup_binding_authority(
                binding_path,
                binding_identity=completion_value["binding_identity"],  # type: ignore[arg-type]
                binding_record=binding_record,
            )
            assert binding_path.exists()
            assert (
                grok_cli_runner._read_private_control_record(
                    binding_directory,
                    authority_path.name,
                )
                == foreign_authority
            )
            authority_path.unlink()

            # The existing terminal attempt CAS precommits all exact inode
            # tombstones. Simulate a crash after their removal but before the
            # completion bytes are published; replay must converge without a
            # second Docker effect.
            authority_body: dict[str, object] = {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "terminal-cleanup-authority@1"
                ),
                "logical_attempt_id": "attempt-cleanup-crash-gap",
                "reservation_id": "reservation-cleanup-crash-gap",
                "cleanup_id": "sha256:" + ("a" * 64),
                "binding_path": str(binding_path),
                "binding_record_id": binding_record["record_id"],
                "termination_fence_id": "",
            }
            authority_body["authority_id"] = (
                grok_cli_runner._effect_receipt_identity(authority_body)
            )
            terminal = SimpleNamespace(
                logical_attempt_id=authority_body["logical_attempt_id"],
                reservation_id=authority_body["reservation_id"],
                state="terminal",
                terminal_cleanup_authority=authority_body,
                terminal_cleanup_progress={},
            )

            class CleanupCAS:
                def __init__(self) -> None:
                    self.current = terminal

                def observe(self, logical_attempt_id: str):
                    return (
                        self.current
                        if logical_attempt_id
                        == self.current.logical_attempt_id
                        else None
                    )

                def commit_terminal_cleanup_intent(
                    self,
                    reservation,
                    *,
                    intent: dict[str, object],
                ):
                    assert reservation.reservation_id == (
                        self.current.reservation_id
                    )
                    if self.current.terminal_cleanup_progress:
                        assert self.current.terminal_cleanup_progress[
                            "intent"
                        ] == intent
                        return self.current
                    progress: dict[str, object] = {
                        "schema": (
                            "ipfs_accelerate_py/agent-supervisor/"
                            "terminal-cleanup-progress@1"
                        ),
                        "logical_attempt_id": self.current.logical_attempt_id,
                        "reservation_id": self.current.reservation_id,
                        "authority_id": authority_body["authority_id"],
                        "phase": "intent_committed",
                        "intent_id": intent["intent_id"],
                        "intent": dict(intent),
                        "completion_id": "",
                        "previous_progress_id": "",
                    }
                    progress["progress_id"] = (
                        grok_cli_runner._effect_receipt_identity(progress)
                    )
                    self.current = SimpleNamespace(
                        **{
                            **vars(self.current),
                            "terminal_cleanup_progress": progress,
                        }
                    )
                    return self.current

                def commit_terminal_cleanup_completion(
                    self,
                    reservation,
                    *,
                    intent_id: str,
                    completion_id: str,
                ):
                    progress = self.current.terminal_cleanup_progress
                    if progress["phase"] == "completion_committed":
                        assert progress["completion_id"] == completion_id
                        return self.current
                    assert progress["intent_id"] == intent_id
                    committed: dict[str, object] = {
                        **progress,
                        "phase": "completion_committed",
                        "completion_id": completion_id,
                        "previous_progress_id": progress["progress_id"],
                    }
                    committed.pop("progress_id")
                    committed["progress_id"] = (
                        grok_cli_runner._effect_receipt_identity(committed)
                    )
                    self.current = SimpleNamespace(
                        **{
                            **vars(self.current),
                            "terminal_cleanup_progress": committed,
                        }
                    )
                    return self.current

            cleanup_cas = CleanupCAS()
            scoped_completion = grok_cli_runner._cleanup_completion_value(
                binding_path=binding_path,
                binding_identity=completion_value["binding_identity"],  # type: ignore[arg-type]
                binding_record=binding_record,
                terminal_cleanup_authority=authority_body,
            )
            cleanup_intent = scoped_completion["cleanup_intent"]
            assert isinstance(cleanup_intent, dict)
            cleanup_cas.commit_terminal_cleanup_intent(
                terminal,
                intent=cleanup_intent,
            )
            for resource in cleanup_intent["resources"]:
                assert grok_cli_runner._remove_owned_cleanup_path(
                    Path(resource["path"]),
                    directory=resource["directory"],
                    identity=resource["identity"],
                    admitted_tombstone_id=resource["tombstone_id"],
                )
            assert not completion_path.exists()
            canonical_retirement = (
                grok_cli_runner._retire_cleanup_binding_authority
            )

            def crash_after_retirement_link(
                binding: Path,
                *,
                binding_identity: dict[str, int],
                binding_record: dict[str, object],
                binding_lock: object | None = None,
            ) -> bool:
                del binding_identity, binding_record, binding_lock
                authority = grok_cli_runner._cleanup_authority_path(binding)
                directory_fd = grok_cli_runner._private_control_directory(
                    binding.parent
                )
                try:
                    os.link(
                        binding.name,
                        authority.name,
                        src_dir_fd=directory_fd,
                        dst_dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                return False

            monkeypatch.setattr(
                grok_cli_runner,
                "_retire_cleanup_binding_authority",
                crash_after_retirement_link,
            )
            assert not grok_cli_runner._finalize_verified_cleanup_completion(
                binding_path=binding_path,
                binding_identity=completion_value["binding_identity"],  # type: ignore[arg-type]
                binding_record=binding_record,
                terminal_cleanup_store=cleanup_cas,
                terminal_cleanup_reservation=terminal,
            )
            assert binding_path.stat().st_ino == authority_path.stat().st_ino
            assert binding_path.stat().st_nlink == 2
            monkeypatch.setattr(
                grok_cli_runner,
                "_retire_cleanup_binding_authority",
                canonical_retirement,
            )
            assert grok_cli_runner._finalize_verified_cleanup_completion(
                binding_path=binding_path,
                binding_identity=completion_value["binding_identity"],  # type: ignore[arg-type]
                binding_record=binding_record,
                terminal_cleanup_store=cleanup_cas,
                terminal_cleanup_reservation=terminal,
            )
            assert not binding_path.exists()
            assert authority_path.exists()
            assert not lease_root.exists()
            assert not provider_home.exists()
            assert not prompt_path.exists()
            assert completion_path.exists()
            assert grok_cli_runner._finalize_verified_cleanup_completion(
                binding_path=binding_path,
                binding_identity=completion_value["binding_identity"],  # type: ignore[arg-type]
                binding_record=binding_record,
                terminal_cleanup_store=cleanup_cas,
                terminal_cleanup_reservation=terminal,
            )
            assert completion_path.exists()
            assert cleanup_cas.current.terminal_cleanup_progress["phase"] == (
                "completion_committed"
            )
            assert reobserved == [binding_record] * 4
        finally:
            completion_path.unlink(missing_ok=True)
            binding_path.unlink(missing_ok=True)
            grok_cli_runner._cleanup_authority_path(binding_path).unlink(
                missing_ok=True
            )
            prompt_path.unlink(missing_ok=True)

@pytest.mark.parametrize(
    "replacement_stage",
    [None, "pair_validation", "retirement"],
)
def test_dual_retirement_scan_closes_both_composite_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_stage: str | None,
) -> None:
    """A link-before-unlink crash binds one directory and leaks no lock."""

    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    docker = Path("/usr/bin/docker")
    if not docker.is_file():
        pytest.skip("trusted Docker CLI is unavailable")
    run_root = tmp_path / "run"
    binding_directory = run_root / "provider-cleanup-bindings"
    binding_directory.mkdir(parents=True, mode=0o700)
    profile = LifecycleProfile(
        target_id="dual-retirement-lock-test",
        run_id="run-dual-retirement-lock-test",
        configuration_root="sha256:" + ("e" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )
    for name, value in profile.launch_environment(11).items():
        monkeypatch.setenv(name, value)

    birth = grok_cli_runner.read_process_birth(os.getpid())
    assert birth is not None
    container_name = "ipfs-accelerate-codex-123-" + ("e" * 32)
    binding_path = binding_directory / (
        hashlib.sha256(container_name.encode("ascii")).hexdigest() + ".json"
    )
    with tempfile.TemporaryDirectory(
        prefix="asref-codex-container-"
    ) as lease_name, tempfile.TemporaryDirectory(
        prefix="asref-codex-home-"
    ) as home_name:
        lease_root = Path(lease_name)
        docker_config = lease_root / "docker-config"
        docker_config.mkdir(mode=0o700)
        provider_home = Path(home_name)
        prompt_descriptor, prompt_name = tempfile.mkstemp(
            prefix="asref-grok-prompt-"
        )
        os.close(prompt_descriptor)
        prompt_path = Path(prompt_name)
        try:
            path_identities = {
                "docker_config": grok_cli_runner._cleanup_path_identity(
                    docker_config,
                    directory=True,
                ),
                "lease_root": grok_cli_runner._cleanup_path_identity(
                    lease_root,
                    directory=True,
                ),
                "prompt_path": grok_cli_runner._cleanup_path_identity(
                    prompt_path,
                    directory=False,
                ),
                "provider_home": grok_cli_runner._cleanup_path_identity(
                    provider_home,
                    directory=True,
                ),
            }
            binding_record = grok_cli_runner._docker_cleanup_binding_value(
                binding_state="prepared_no_dispatch",
                provider="codex",
                docker_bin=str(docker),
                container_name=container_name,
                lease_root=lease_root,
                docker_config=docker_config,
                cidfile=lease_root / "container.cid",
                provider_home=provider_home,
                prompt_path=prompt_path,
                effect_observation={},
                path_identities=path_identities,
                binding_path=binding_path,
                runner_pid=os.getpid(),
                runner_start_ticks=birth.start_time_ticks,
                watchdog_pid=os.getpid(),
                watchdog_start_ticks=birth.start_time_ticks,
            )
            grok_cli_runner._write_private_control_record(
                binding_directory,
                binding_path.name,
                binding_record,
                replace_existing=False,
            )
            binding_identity = grok_cli_runner._cleanup_path_identity(
                binding_path,
                directory=False,
            )
            completion = grok_cli_runner._cleanup_completion_value(
                binding_path=binding_path,
                binding_identity=binding_identity,
                binding_record=binding_record,
            )
            cleanup_intent = completion["cleanup_intent"]
            assert isinstance(cleanup_intent, dict)
            resources = cleanup_intent["resources"]
            assert isinstance(resources, list)
            for resource in resources:
                assert isinstance(resource, dict)
                assert grok_cli_runner._remove_owned_cleanup_path(
                    Path(str(resource["path"])),
                    directory=bool(resource["directory"]),
                    identity=resource["identity"],
                )
            completion_path = grok_cli_runner._cleanup_completion_path(
                binding_path
            )
            grok_cli_runner._write_private_control_record(
                binding_directory,
                completion_path.name,
                completion,
                replace_existing=False,
            )
            assert grok_cli_runner._recover_cleanup_completion(
                completion_path,
                docker_absence_verified=True,
            )

            authority_path = grok_cli_runner._cleanup_authority_path(
                binding_path
            )
            os.link(binding_path, authority_path, follow_symlinks=False)
            directory_descriptor = grok_cli_runner._private_control_directory(
                binding_directory
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            assert binding_path.stat().st_nlink == 2

            canonical_lock = grok_cli_runner._docker_binding_lock_descriptor
            scanner_locks: list[object] = []

            def tracked_lock(path: Path, **kwargs: object):
                handle = canonical_lock(path, **kwargs)
                scanner_locks.append(handle)
                return handle

            monkeypatch.setattr(
                grok_cli_runner,
                "_docker_binding_lock_descriptor",
                tracked_lock,
            )
            displaced_directory = run_root / "provider-cleanup-bindings-displaced"

            def replace_binding_directory() -> None:
                binding_directory.rename(displaced_directory)
                binding_directory.mkdir(mode=0o700)

            if replacement_stage == "pair_validation":
                canonical_pair_validation = (
                    grok_cli_runner._cleanup_binding_retirement_pair_matches
                )

                def replacing_pair_validation(*args: object, **kwargs: object):
                    replace_binding_directory()
                    return canonical_pair_validation(*args, **kwargs)

                monkeypatch.setattr(
                    grok_cli_runner,
                    "_cleanup_binding_retirement_pair_matches",
                    replacing_pair_validation,
                )
            elif replacement_stage == "retirement":
                canonical_retirement = (
                    grok_cli_runner._retire_cleanup_binding_authority
                )

                def replacing_retirement(*args: object, **kwargs: object):
                    replace_binding_directory()
                    return canonical_retirement(*args, **kwargs)

                monkeypatch.setattr(
                    grok_cli_runner,
                    "_retire_cleanup_binding_authority",
                    replacing_retirement,
                )

            if replacement_stage is None:
                assert multi_supervisor_runner._durable_docker_cleanup_bindings(
                    profile,
                    fencing_epoch=11,
                ) == ()
            else:
                with pytest.raises(ValueError):
                    multi_supervisor_runner._durable_docker_cleanup_bindings(
                        profile,
                        fencing_epoch=11,
                    )
                binding_directory.rmdir()
                displaced_directory.rename(binding_directory)
            assert len(scanner_locks) == (
                1 if replacement_stage == "pair_validation" else 2
            )
            assert all(getattr(handle, "_closed", False) for handle in scanner_locks)
            if replacement_stage is None:
                assert not binding_path.exists()
                assert authority_path.exists()
            else:
                assert binding_path.exists()
                assert authority_path.exists()

            # Reacquisition proves both the inode flock and abstract Unix
            # uniqueness lease were released by both scanner branches.
            reacquired = canonical_lock(binding_path)
            reacquired.close()
        finally:
            prompt_path.unlink(missing_ok=True)

def test_old_cleanup_completions_do_not_consume_active_record_capacity(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleProfile,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    run_root = tmp_path / "run"
    directory = run_root / "provider-cleanup-bindings"
    directory.mkdir(parents=True, mode=0o700)
    for index in range(129):
        grok_cli_runner._write_private_control_record(
            directory,
            f"{index:064x}.complete",
            {"inert_transition_journal": index},
            replace_existing=False,
        )
    profile = LifecycleProfile(
        target_id="completion-capacity-test",
        run_id="run-completion-capacity-test",
        configuration_root="sha256:" + ("6" * 64),
        repository_root=str(tmp_path),
        state_root=str(tmp_path),
        run_root=str(run_root),
        argv=(sys.executable, "-c", "pass"),
        cwd=str(tmp_path),
    )

    assert multi_supervisor_runner._durable_docker_cleanup_bindings(
        profile,
        fencing_epoch=1,
    ) == ()

def test_durable_cleanup_rejects_path_replacement_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    target = tmp_path / "provider-home"
    target.mkdir(mode=0o700)
    (target / "owned.txt").write_text("owned", encoding="utf-8")
    identity = grok_cli_runner._cleanup_path_identity(
        target,
        directory=True,
    )
    original = tmp_path / "original-moved-away"
    real_rename = multi_supervisor_runner.os.rename
    swapped = False

    def replace_before_quarantine(source: Path, destination: Path) -> None:
        nonlocal swapped
        if not swapped and Path(source) == target:
            swapped = True
            real_rename(target, original)
            target.mkdir(mode=0o700)
            (target / "replacement.txt").write_text(
                "must-survive",
                encoding="utf-8",
            )
        real_rename(source, destination)

    monkeypatch.setattr(
        multi_supervisor_runner.os,
        "rename",
        replace_before_quarantine,
    )

    assert not multi_supervisor_runner._remove_owned_cleanup_path(
        target,
        directory=True,
        identity=identity,
    )
    assert swapped is True
    assert (target / "replacement.txt").read_text(encoding="utf-8") == (
        "must-survive"
    )
    assert (original / "owned.txt").read_text(encoding="utf-8") == "owned"

def test_cleanup_marker_only_cannot_authorize_renamed_credential_tree(
    tmp_path: Path,
) -> None:
    target = tmp_path / "provider-home"
    target.mkdir(mode=0o700)
    (target / "credential.json").write_text("secret", encoding="utf-8")
    identity = grok_cli_runner._cleanup_path_identity(target, directory=True)
    quarantine, _owned, marker, tombstone = (
        grok_cli_runner._cleanup_path_quarantine(
            target,
            directory=True,
            identity=identity,
        )
    )
    renamed = tmp_path / "credential-tree-renamed-away"
    target.rename(renamed)
    quarantine.mkdir(mode=0o700)
    grok_cli_runner._write_private_control_record(
        quarantine,
        marker.name,
        tombstone,
        replace_existing=False,
    )

    assert not grok_cli_runner._remove_owned_cleanup_path(
        target,
        directory=True,
        identity=identity,
    )
    assert (renamed / "credential.json").read_text(encoding="utf-8") == (
        "secret"
    )

def test_cleanup_fd_walk_rejects_owned_inode_swap_after_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "provider-home"
    target.mkdir(mode=0o700)
    (target / "credential.json").write_text("secret", encoding="utf-8")
    identity = grok_cli_runner._cleanup_path_identity(target, directory=True)
    _quarantine, owned, marker, _tombstone = (
        grok_cli_runner._cleanup_path_quarantine(
            target,
            directory=True,
            identity=identity,
        )
    )
    renamed = tmp_path / "exact-owned-inode-renamed-away"
    real_write = grok_cli_runner._write_private_control_record
    swapped = False

    def swap_after_marker(*args: object, **kwargs: object) -> None:
        nonlocal swapped
        real_write(*args, **kwargs)  # type: ignore[arg-type]
        if not swapped and Path(args[1]) == Path(marker.name):
            swapped = True
            owned.rename(renamed)
            owned.mkdir(mode=0o700)
            (owned / "replacement.txt").write_text(
                "must-survive",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        grok_cli_runner,
        "_write_private_control_record",
        swap_after_marker,
    )

    assert not grok_cli_runner._remove_owned_cleanup_path(
        target,
        directory=True,
        identity=identity,
    )
    assert swapped is True
    assert (renamed / "credential.json").read_text(encoding="utf-8") == (
        "secret"
    )
    assert (owned / "replacement.txt").read_text(encoding="utf-8") == (
        "must-survive"
    )

def test_durable_cleanup_removes_same_inode_after_permission_drift(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    target = tmp_path / "provider-home"
    target.mkdir(mode=0o700)
    (target / "mutable-provider-state").write_text(
        "owned",
        encoding="utf-8",
    )
    identity = grok_cli_runner._cleanup_path_identity(
        target,
        directory=True,
    )
    target.chmod(0o000)
    changed = target.lstat()

    assert changed.st_dev == identity["device"]
    assert changed.st_ino == identity["inode"]
    assert changed.st_mode & 0o777 == 0
    assert multi_supervisor_runner._remove_owned_cleanup_path(
        target,
        directory=True,
        identity=identity,
    )
    assert not os.path.lexists(target)
    assert multi_supervisor_runner._discard_owned_cleanup_tombstone(
        target,
        directory=True,
        identity=identity,
    )

def test_durable_cleanup_rejects_rename_away_before_initial_lstat(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )

    target = tmp_path / "provider-home"
    target.mkdir(mode=0o700)
    (target / "owned.txt").write_text("owned", encoding="utf-8")
    identity = grok_cli_runner._cleanup_path_identity(
        target,
        directory=True,
    )
    moved_away = tmp_path / "provider-home-moved-away"
    target.rename(moved_away)

    assert not multi_supervisor_runner._remove_owned_cleanup_path(
        target,
        directory=True,
        identity=identity,
    )
    assert not os.path.lexists(target)
    assert (moved_away / "owned.txt").read_text(encoding="utf-8") == "owned"
    moved_metadata = moved_away.lstat()
    assert moved_metadata.st_dev == identity["device"]
    assert moved_metadata.st_ino == identity["inode"]

def test_docker_create_positive_grammar_admits_canonical_vendor_command_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_source = Path(grok_cli_runner.__file__).read_text(encoding="utf-8")
    assert "todo_daemon.implementation_daemon" not in runner_source

    workspace = tmp_path / "workspace"
    provider_bin = tmp_path / "provider-bin"
    lease_root = tmp_path / "asref-codex-container-grammar"
    docker_config = lease_root / "docker-config"
    provider_home = tmp_path / "asref-codex-home-grammar"
    for directory in (workspace, provider_bin, lease_root, provider_home):
        directory.mkdir(mode=0o700)
    docker_config.mkdir(mode=0o700)
    monkeypatch.setattr(
        grok_cli_runner.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    codex = provider_bin / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o700)
    source_auth = tmp_path / "auth.json"
    source_auth.write_text("{}\n", encoding="utf-8")
    source_auth.chmod(0o600)
    vendor_bin = tmp_path / "vendor-bin"
    vendor_bin.mkdir(mode=0o700)
    host_codex = vendor_bin / "codex"
    host_companion = vendor_bin / "codex-code-mode-host"
    for executable in (host_codex, host_companion):
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setattr(
        grok_cli_runner,
        "find_codex_vendor_binaries",
        lambda: (host_codex.resolve(), host_companion.resolve()),
    )
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    cidfile = lease_root / "container.cid"
    container_name = "ipfs-accelerate-codex-1-" + "a" * 32
    command = grok_cli_runner._docker_codex_fallback_command(
        codex_command=_terra_fallback_command(str(codex), workspace),
        workspace=workspace,
        source_auth=source_auth,
        provider_home=provider_home,
        child_env=grok_cli_runner._codex_task_container_environment(),
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        docker_bin="/usr/bin/docker",
        isolation_image=image,
    )
    identity_arguments = {
        "provider": "codex",
        "docker_bin": "/usr/bin/docker",
        "docker_config": docker_config,
        "container_name": container_name,
        "cidfile": cidfile,
        "cwd": workspace,
        "environment_id": "sha256:" + "1" * 64,
        "expected_image": image,
    }

    command_id, command_body = grok_cli_runner._docker_create_command_identity(
        **identity_arguments,
        argv=command,
    )

    assert re.fullmatch(r"sha256:[0-9a-f]{64}", command_id)
    assert command_body["image_id"] == image
    canonical_vendor_mount = (
        f"type=bind,src={vendor_bin.resolve()},"
        "dst=/usr/local/bin,readonly"
    )
    assert canonical_vendor_mount in command_body["argv"]

    image_index = command.index(image)
    mismatched_image = "sha256:" + ("0" if image[-1] != "0" else "1") * 64
    foreign_writable = tmp_path / "foreign-writable"
    foreign_writable.mkdir()
    foreign_git = tmp_path / "foreign-repository" / ".git"
    foreign_git.mkdir(parents=True)
    foreign_grok = tmp_path / "foreign-grok"
    foreign_grok.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    foreign_grok.chmod(0o700)

    def before_image(*arguments: str) -> list[str]:
        return [*command[:image_index], *arguments, *command[image_index:]]

    mutations = {
        "mutable image with later digest": [
            *command[:image_index],
            "ubuntu:latest",
            image,
            *command[image_index + 1 :],
        ],
        "mismatched immutable image": [
            *command[:image_index],
            mismatched_image,
            *command[image_index + 1 :],
        ],
        "privileged option": before_image("--privileged"),
        "unknown option": before_image("--restart=no"),
        "foreign writable mount": before_image(
            "--mount",
            (
                f"type=bind,src={foreign_writable.resolve()},"
                "dst=/mnt/foreign"
            ),
        ),
        "Docker authority environment": before_image(
            "--env",
            "DOCKER_HOST=unix:///tmp/foreign.sock",
        ),
        "foreign git metadata": before_image(
            "--mount",
            (
                f"type=bind,src={foreign_git.resolve()},"
                f"dst={foreign_git.resolve()},readonly"
            ),
        ),
        "Grok executable mount under Codex": before_image(
            "--mount",
            (
                f"type=bind,src={foreign_grok.resolve()},"
                "dst=/opt/ipfs-accelerate/grok,readonly"
            ),
        ),
    }
    for _case, mutated in mutations.items():
        with pytest.raises(ValueError, match="Docker create"):
            grok_cli_runner._docker_create_command_identity(
                **identity_arguments,
                argv=mutated,
            )

def test_docker_mount_emits_resolved_source_not_symlink(tmp_path: Path) -> None:
    real = tmp_path / "grok-1.0.13-linux-aarch64"
    real.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real.chmod(0o700)
    link = tmp_path / "grok"
    link.symlink_to(real)
    destination = Path("/opt/ipfs-accelerate/grok")

    mount = grok_cli_runner._docker_mount(
        link,
        destination=destination,
        read_only=True,
    )

    assert mount == [
        "--mount",
        f"type=bind,src={real.resolve()},dst={destination},readonly",
    ]

def _canonical_grok_docker_create_argv(tmp_path: Path) -> dict[str, object]:
    workspace = tmp_path / "workspace"
    lease_root = tmp_path / "asref-grok-container-socket-mask"
    docker_config = lease_root / "docker-config"
    grok_home = tmp_path / "asref-grok-home-socket-mask"
    prompt_path = tmp_path / "asref-grok-prompt-socket-mask"
    grok_bin = tmp_path / "grok-fixture"
    mask_root = lease_root / "provider-masks"
    for directory in (workspace, lease_root, grok_home):
        directory.mkdir(mode=0o700)
    docker_config.mkdir(mode=0o700)
    grok_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    grok_bin.chmod(0o700)
    prompt_path.write_text("prompt\n", encoding="utf-8")
    image = "sha256:" + "a" * 64
    container_name = "ipfs-accelerate-grok-1-" + "c" * 32
    cidfile = lease_root / "container.cid"
    command = grok_cli_runner._docker_grok_command(
        grok_command=[str(grok_bin), "-p", "hi"],
        grok_bin=grok_bin,
        workspace=workspace,
        prompt_path=prompt_path,
        grok_home=grok_home,
        base_env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        child_env={"PATH": "/usr/bin:/bin", "HOME": str(grok_home)},
        denied_paths=(
            Path("/var/run/docker.sock"),
            Path("/run/docker.sock"),
        ),
        mask_root=mask_root,
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
        docker_bin="/usr/bin/docker",
        isolation_image=image,
    )
    return {
        "command": command,
        "identity_arguments": {
            "provider": "grok",
            "docker_bin": "/usr/bin/docker",
            "docker_config": docker_config,
            "container_name": container_name,
            "cidfile": cidfile,
            "cwd": workspace,
            "environment_id": "sha256:" + "1" * 64,
            "expected_image": image,
        },
        "grok_bin": grok_bin,
        "lease_root": lease_root,
    }

def test_docker_create_identity_admits_provider_mask_over_docker_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.chmod(0o700)
    monkeypatch.setattr(
        grok_cli_runner.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    fixture = _canonical_grok_docker_create_argv(tmp_path)
    grok_bin = fixture["grok_bin"]
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok_bin.resolve()),
    )
    command = fixture["command"]
    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]

    assert any(
        f"dst={Path('/var/run/docker.sock')}" in mount
        or f"dst={Path('/run/docker.sock')}" in mount
        for mount in mounts
    )
    grok_cli_runner._docker_create_command_identity(
        **fixture["identity_arguments"],
        argv=command,
    )

def test_docker_create_identity_rejects_real_docker_socket_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/run/docker.sock")
    if not socket_path.exists():
        pytest.skip("host docker socket is required for this identity check")
    tmp_path.chmod(0o700)
    monkeypatch.setattr(
        grok_cli_runner.tempfile,
        "gettempdir",
        lambda: str(tmp_path),
    )
    fixture = _canonical_grok_docker_create_argv(tmp_path)
    grok_bin = fixture["grok_bin"]
    monkeypatch.setattr(
        grok_cli_runner,
        "_resolve_trusted_grok_bin",
        lambda **_kwargs: str(grok_bin.resolve()),
    )
    command = list(fixture["command"])
    image = fixture["identity_arguments"]["expected_image"]
    image_index = command.index(image)
    mutated = [
        *command[:image_index],
        "--mount",
        f"type=bind,src={socket_path},dst={socket_path},readonly",
        *command[image_index:],
    ]

    with pytest.raises(ValueError, match="mount path is unsafe"):
        grok_cli_runner._docker_create_command_identity(
            **fixture["identity_arguments"],
            argv=mutated,
        )

def test_router_codex_vendor_pair_requires_two_common_executables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vendor_bin = tmp_path / "vendor-bin"
    foreign_bin = tmp_path / "foreign-bin"
    vendor_bin.mkdir()
    foreign_bin.mkdir()
    codex = vendor_bin / "codex"
    companion = vendor_bin / "codex-code-mode-host"
    codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    companion.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    codex.chmod(0o755)
    companion.chmod(0o755)

    assert llm_router._codex_vendor_pair_from_bin_dir(vendor_bin) == (
        codex.resolve(),
        companion.resolve(),
    )
    admitted_pair = (codex.resolve(), companion.resolve())
    monkeypatch.setattr(
        implementation_daemon,
        "find_codex_vendor_binaries",
        lambda: admitted_pair,
    )
    assert implementation_daemon._host_codex_vendor_binaries() == admitted_pair

    companion.chmod(0o644)
    assert llm_router._codex_vendor_pair_from_bin_dir(vendor_bin) is None
    companion.unlink()
    foreign_companion = foreign_bin / "codex-code-mode-host"
    foreign_companion.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    foreign_companion.chmod(0o755)
    companion.symlink_to(foreign_companion)
    assert llm_router._codex_vendor_pair_from_bin_dir(vendor_bin) is None

def test_codex_vendor_lookup_does_not_import_the_implementation_daemon(
    tmp_path: Path,
) -> None:
    isolated_tmp = tmp_path / "isolated-tmp"
    isolated_tmp.mkdir()
    repository_root = Path(__file__).resolve().parents[2]
    daemon_name = (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    code = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            f"sys.path.insert(0, {str(repository_root)!r})",
            (
                "from ipfs_accelerate_py.agent_supervisor.runtime "
                "import grok_cli_runner"
            ),
            f"assert {daemon_name!r} not in sys.modules",
            "grok_cli_runner._docker_codex_host_vendor_mounts()",
            f"assert {daemon_name!r} not in sys.modules",
            (
                f"assert not list(Path({str(isolated_tmp)!r}).glob("
                "'asref-imported-control-plane-*'))"
            ),
        )
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("LD_", "PYTHON", "PYTEST"))
    }
    environment["TMPDIR"] = str(isolated_tmp)
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-B", "-c", code],
        cwd=repository_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        errors="replace"
    )

@pytest.mark.parametrize("payload", [None, b"forged-provider-start\n"])
def test_provider_start_wrapper_rejects_eof_or_wrong_marker(
    tmp_path: Path,
    payload: bytes | None,
) -> None:
    sentinel = tmp_path / "provider-executed"
    sender, docker_stdin = grok_cli_runner._provider_start_socketpair()
    try:
        process = subprocess.Popen(
            [
                "/bin/sh",
                "-c",
                grok_cli_runner._DOCKER_PROVIDER_START_SCRIPT,
                "aseh-provider-start",
                "/bin/sh",
                "-c",
                f"printf executed > {shlex.quote(str(sentinel))}",
            ],
            stdin=docker_stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    finally:
        docker_stdin.close()
    try:
        if payload is not None:
            sender.sendall(payload)
        sender.close()
        stdout, stderr = process.communicate(timeout=5.0)
        assert process.returncode == 125, (stdout, stderr)
        assert not sentinel.exists()
    finally:
        sender.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)

def test_running_fence_is_published_before_provider_start_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_root = tmp_path / "lease"
    lease_root.mkdir(mode=0o700)
    cidfile = lease_root / "container.cid"
    container_id = "e" * 64
    cidfile.write_text(container_id + "\n", encoding="ascii")
    provider_home = tmp_path / "provider-home"
    provider_home.mkdir(mode=0o700)
    prompt_path = tmp_path / "prompt"
    prompt_path.write_text("prompt", encoding="utf-8")
    binding_path = tmp_path / "binding.json"
    image_id = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    container_name = "ipfs-accelerate-codex-1-" + "e" * 32
    journal = {
        "state": "create_observed",
        "command_id": "sha256:" + "1" * 64,
        "cwd": str(tmp_path),
        "environment_id": "sha256:" + "2" * 64,
    }
    running_fence: dict[str, object] = {
        "docker_state": "running",
        "init_pid": 123,
        "container_id": container_id,
        "container_name": container_name,
        "image_id": image_id,
    }
    events: list[str] = []

    lease = object.__new__(grok_cli_runner._DockerContainerLease)
    lease._create_started = True
    lease._authorized_image_id = image_id
    lease._create_outcome_unknown = False
    lease.cidfile = cidfile
    lease.lease_root = lease_root
    lease.provider = "codex"
    lease.docker_bin = "/usr/bin/docker"
    lease.docker_config = lease_root / "docker-config"
    lease.docker_config.mkdir(mode=0o700)
    lease.container_name = container_name
    lease.cleanup_binding_record = binding_path
    lease._cleanup_binding_value = {
        "record_id": "sha256:" + "3" * 64,
        "termination_fence": {},
    }
    lease._cleanup_binding_identity = {
        "device": 1,
        "inode": 2,
        "mode": stat.S_IFREG,
        "uid": os.geteuid(),
    }
    lease._termination_fence = {}
    lease.effect_observation = {}
    lease.provider_home = provider_home
    lease.prompt_path = prompt_path
    lease._watchdog = SimpleNamespace(pid=321, start_ticks=654)
    provider_sender, provider_stdin = (
        grok_cli_runner._provider_start_socketpair()
    )
    lease._provider_start_sender = provider_sender
    lease._provider_start_stdin = provider_stdin
    lease._provider_start_stdin_taken = False
    lease._provider_start_released = False
    lease.preserve_for_recovery = False
    lease._admit_cleanup_authority = lambda _journal: True

    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_docker_create_journal",
        lambda **_kwargs: journal,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_attest_exact_docker_execution",
        lambda **_kwargs: running_fence,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_docker_termination_fence",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_runner_process_start_ticks",
        lambda _pid: 777,
    )

    observer = lease.take_provider_start_stdin()

    def publish_fence(**_kwargs: object):
        with pytest.raises(BlockingIOError):
            observer.recv(1, socket.MSG_DONTWAIT)
        events.append("fence-published")
        return (
            {
                "record_id": "sha256:" + "4" * 64,
                "termination_fence": running_fence,
            },
            {
                "device": 5,
                "inode": 6,
                "mode": stat.S_IFREG,
                "uid": os.geteuid(),
            },
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_publish_docker_termination_fence_binding",
        publish_fence,
    )
    try:
        assert lease.capture_running_termination_fence() == running_fence
        assert events == ["fence-published"]
        assert observer.recv(
            len(grok_cli_runner._DOCKER_PROVIDER_START_MARKER),
        ) == grok_cli_runner._DOCKER_PROVIDER_START_MARKER
        assert lease._provider_start_released is True
        lease.finish_provider_input()
        assert observer.recv(1) == b""
    finally:
        lease._abort_provider_start()
        observer.close()

def test_provider_start_socket_cannot_be_reopened_through_proc() -> None:
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        pytest.skip("Linux procfs descriptor semantics are unavailable")
    runner_socket, watchdog_socket = (
        grok_cli_runner._provider_start_socketpair()
    )
    try:
        assert not os.get_inheritable(runner_socket.fileno())
        assert not os.get_inheritable(watchdog_socket.fileno())
        os.set_inheritable(watchdog_socket.fileno(), True)
        descriptor_path = f"/proc/self/fd/{watchdog_socket.fileno()}"
        for flags in (os.O_RDONLY, os.O_WRONLY, os.O_RDWR):
            try:
                reopened = os.open(descriptor_path, flags)
            except OSError:
                continue
            os.close(reopened)
            pytest.fail(
                "an inherited AF_UNIX control capability was reopenable "
                f"through procfs with flags={flags}"
            )
    finally:
        runner_socket.close()
        watchdog_socket.close()

def test_running_adoption_rejects_missing_persisted_fence_without_attesting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_root = tmp_path / "asref-codex-container-fixture"
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(parents=True)
    binding_path = tmp_path / "binding.json"
    binding_path.write_text("{}", encoding="utf-8")
    container_id = "a" * 64
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    provider_home = tmp_path / "asref-codex-home-fixture"
    prompt_path = tmp_path / "asref-grok-prompt-fixture"
    cidfile = lease_root / "container.cid"
    candidate = {
        "docker_bin": "/usr/bin/docker",
        "binding_state": "command_bound",
        "effect_observation": {},
        "termination_fence": {},
        "record_id": "sha256:" + "c" * 64,
    }
    launch_receipt = {
        "container_id": "sha256:" + container_id,
        "image_id": grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
        "cleanup_receipt": {
            "provider_home": str(provider_home),
            "prompt_path": str(prompt_path),
            "cidfile": str(cidfile),
        },
    }
    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_lease_root",
        lambda _receipt: (lease_root, docker_config, container_name),
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_cleanup_binding_path",
        lambda *_args, **_kwargs: binding_path,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_read_private_control_record",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_docker_create_journal",
        lambda **_kwargs: {
            "state": "create_observed",
            "command_id": "sha256:" + "d" * 64,
            "cwd": str(tmp_path),
            "environment_id": "sha256:" + "e" * 64,
        },
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_validated_cleanup_binding_record",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_attest_exact_docker_execution",
        lambda **_kwargs: pytest.fail(
            "running adoption must not synthesize a lost fence"
        ),
    )

    with pytest.raises(ValueError, match="lacks its persisted termination fence"):
        grok_cli_runner._recorded_codex_running_fence(
            launch_receipt,
            capture_if_absent=False,
        )

def test_created_adoption_uses_fresh_socket_and_fence_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    container_id = "f" * 64
    container_name = "ipfs-accelerate-codex-1-" + "a" * 32
    runtime_id = "sha256:" + "1" * 64
    start_command = [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "--config",
        str(docker_config),
        "start",
        "--attach",
        "--interactive",
        container_id,
    ]
    launch_receipt = {
        "runtime_id": runtime_id,
        "container_name": container_name,
        "container_id": "sha256:" + container_id,
        "command_receipt": {"start_argv": start_command},
    }
    events: list[str] = []
    provider_payloads: list[bytes] = []

    class FakeProcess:
        def __init__(self, provider_input: socket.socket) -> None:
            self._provider_input = provider_input
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")

        def wait(self, timeout=None) -> int:
            del timeout
            payload = bytearray()
            while True:
                chunk = self._provider_input.recv(4096)
                if not chunk:
                    break
                payload.extend(chunk)
            self._provider_input.close()
            provider_payloads.append(bytes(payload))
            events.append("provider-observed-input")
            return 0

    def fake_popen(command, **kwargs):
        assert list(command) == start_command
        provider_input = kwargs["stdin"]
        assert isinstance(provider_input, socket.socket)
        assert provider_input.family == socket.AF_UNIX
        return FakeProcess(
            socket.fromfd(
                provider_input.fileno(),
                socket.AF_UNIX,
                socket.SOCK_STREAM,
            )
        )

    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_isolation_binary",
        lambda: "/usr/bin/docker",
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_docker_runtime_receipt_identity",
        lambda _docker_bin: runtime_id,
    )
    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_lease_root",
        lambda _receipt: (tmp_path, docker_config, container_name),
    )

    def admit_fence(_receipt, *, capture_if_absent: bool):
        assert capture_if_absent is True
        events.append("fence-published")
        return {"docker_state": "running", "init_pid": 123}

    monkeypatch.setattr(
        grok_cli_runner,
        "_recorded_codex_running_fence",
        admit_fence,
    )
    monkeypatch.setattr(grok_cli_runner.subprocess, "Popen", fake_popen)

    assert grok_cli_runner._start_recorded_codex_effect(
        launch_receipt,
        prompt="repair",
    ) == 0
    assert events == ["fence-published", "provider-observed-input"]
    assert provider_payloads == [
        grok_cli_runner._DOCKER_PROVIDER_START_MARKER + b"repair"
    ]

def test_fenced_docker_issuer_uses_three_nonreopenable_socketpairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        pytest.skip("Linux procfs descriptor semantics are unavailable")
    lease_root = tmp_path / "lease"
    lease_root.mkdir(mode=0o700)
    command_body: dict[str, object] = {
        "provider": "codex",
        "docker_bin": "/usr/bin/docker",
        "docker_config": str(lease_root / "docker-config"),
        "container_name": "ipfs-accelerate-codex-1-" + "c" * 32,
        "cidfile": str(lease_root / "container.cid"),
        "cwd": str(tmp_path),
        "environment_id": "sha256:" + "2" * 64,
        "image_id": grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID,
        "argv": ["/bin/true"],
    }
    command_id = grok_cli_runner._effect_receipt_identity(command_body)
    armed = grok_cli_runner._docker_create_journal_value(
        command_body=command_body,
        command_id=command_id,
        state="create_armed",
    )
    grok_cli_runner._write_private_control_record(
        lease_root,
        grok_cli_runner._DOCKER_CREATE_JOURNAL_NAME,
        armed,
        replace_existing=False,
    )
    real_socketpair = socket.socketpair
    observed_pairs: list[tuple[int, int]] = []

    def nonreopenable_socketpair(*args: object, **kwargs: object):
        left, right = real_socketpair(*args, **kwargs)
        assert left.family == socket.AF_UNIX
        assert right.family == socket.AF_UNIX
        assert left.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
        assert right.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
        for endpoint in (left, right):
            descriptor_path = f"/proc/self/fd/{endpoint.fileno()}"
            for flags in (os.O_RDONLY, os.O_WRONLY, os.O_RDWR):
                try:
                    reopened = os.open(descriptor_path, flags)
                except OSError:
                    continue
                os.close(reopened)
                left.close()
                right.close()
                pytest.fail(
                    "a fenced issuer socket endpoint was reopenable through "
                    f"procfs with flags={flags}"
                )
        observed_pairs.append((left.fileno(), right.fileno()))
        return left, right

    monkeypatch.setattr(
        grok_cli_runner.socket,
        "socketpair",
        nonreopenable_socketpair,
    )

    inflight, returncode, stdout, stderr, dispatched, forced_kill = (
        grok_cli_runner._run_fenced_docker_create_issuer(
            armed,
            lease_root=lease_root,
            cwd=tmp_path,
            environment={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
        )
    )

    assert len(observed_pairs) == 3
    assert inflight["state"] == "create_inflight"
    assert returncode == 0
    assert stdout == b""
    assert stderr == b""
    assert dispatched is True
    assert forced_kill is False

def test_same_boot_materialization_uses_bound_config_and_returns_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, record, _journal, container_id, image = (
        _same_boot_materialization_fixture(tmp_path)
    )
    observed: dict[str, object] = {}
    termination_fence = _created_docker_termination_fence(
        container_name=record.container_name,
        container_id=container_id,
        image_id=image,
    )

    def attest(**kwargs: object):
        observed.update(kwargs)
        config_argument = str(kwargs["docker_config"])
        assert config_argument.startswith("/proc/self/fd/")
        config_descriptor = int(config_argument.rpartition("/")[2])
        assert kwargs.get("pass_fds") == (config_descriptor,)
        assert kwargs["container_id"] == container_id
        assert kwargs["container_name"] == record.container_name
        assert kwargs["image_id"] == image
        assert kwargs["provider"] == "codex"
        return termination_fence

    monkeypatch.setattr(grok_cli_runner, "_attest_exact_docker_execution", attest)

    materialized = runner._exact_docker_container_materialized(
        record,
        deadline=time.monotonic() + 5.0,
    )
    assert materialized == termination_fence
    assert observed["pass_fds"]

def test_same_boot_materialization_rejects_replaced_config_before_inspect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, record, _journal, _container_id, _image = (
        _same_boot_materialization_fixture(tmp_path)
    )
    replacement = tmp_path / "replacement-config"
    replacement.mkdir(mode=0o700)
    record.docker_config.rmdir()
    replacement.rename(record.docker_config)

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "Docker inspect ran with a replaced config inode"
        ),
    )

    assert runner._exact_docker_container_materialized(
        record,
        deadline=time.monotonic() + 5.0,
    ) is None

def test_same_boot_materialization_revalidates_journal_after_scheduler_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, record, journal, _container_id, _image = (
        _same_boot_materialization_fixture(tmp_path)
    )
    assert runner._durable_docker_create_state(record) == "create_observed"
    command_body = {
        name: journal[name]
        for name in (
            "provider",
            "docker_bin",
            "docker_config",
            "container_name",
            "cidfile",
            "cwd",
            "environment_id",
            "image_id",
            "argv",
        )
    }
    swapped = grok_cli_runner._docker_create_journal_value(
        command_body=command_body,
        command_id=str(journal["command_id"]),
        state="create_outcome_unknown",
        issuer_process_birth=journal["issuer_process_birth"],
        returncode=125,
    )
    grok_cli_runner._write_private_control_record(
        record.lease_root,
        grok_cli_runner._DOCKER_CREATE_JOURNAL_NAME,
        swapped,
        replace_existing=True,
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "Docker inspect ran after the observed journal was replaced"
        ),
    )

    assert runner._exact_docker_container_materialized(
        record,
        deadline=time.monotonic() + 5.0,
    ) is None

def test_same_boot_recovery_removes_immutable_id_not_replaced_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, record, _journal, container_id, image = (
        _same_boot_materialization_fixture(tmp_path)
    )
    removals: list[tuple[str, bool]] = []
    termination_fence = _created_docker_termination_fence(
        container_name=record.container_name,
        container_id=container_id,
        image_id=image,
    )
    monkeypatch.setattr(runner, "_durable_cleanup_cas_allows_reap", lambda _r: True)
    monkeypatch.setattr(runner, "_durable_cleanup_runner_alive", lambda _r: False)
    monkeypatch.setattr(runner, "_durable_cleanup_watchdog_alive", lambda _r: False)
    monkeypatch.setattr(
        runner,
        "_exact_docker_container_materialized",
        lambda *_args, **_kwargs: termination_fence,
    )

    def publish_fence(observed_record, observed_fence):
        assert observed_record == record
        assert observed_fence == termination_fence
        return runner.replace(
            observed_record,
            termination_fence=termination_fence,
            record_id="sha256:" + ("e" * 64),
        )

    monkeypatch.setattr(
        runner,
        "_publish_durable_docker_termination_fence",
        publish_fence,
    )
    monkeypatch.setattr(
        runner,
        "_arm_fenced_durable_docker_removal",
        lambda observed_record: (
            observed_record.termination_fence == termination_fence
        ),
    )
    monkeypatch.setattr(
        runner,
        "_exact_docker_name_absent",
        # A different container acquired the same name after inspection.
        lambda *_args, **_kwargs: pytest.fail(
            "a replaced name cannot override the immutable termination fence"
        ),
    )

    def remove(observed_record, *, deadline: float, issue_removal: bool) -> bool:
        assert deadline > time.monotonic()
        removals.append(
            (
                str(observed_record.termination_fence["container_id"]),
                issue_removal,
            )
        )
        return False

    monkeypatch.setattr(runner, "_remove_fenced_durable_docker_effect", remove)

    assert not runner._reconcile_durable_docker_cleanup(
        record,
        deadline=time.monotonic() + 5.0,
    )
    assert len(removals) == 1
    assert removals == [(container_id, True)]
    assert removals[0][0] != record.container_name
    assert record.lease_root.exists()

def test_public_terminal_journal_cannot_replace_private_create_result(
    tmp_path: Path,
) -> None:
    workspace = tmp_path.resolve()
    lease_root = workspace / "lease"
    lease_root.mkdir(mode=0o700)
    docker_config = lease_root / "docker-config"
    docker_config.mkdir(mode=0o700)
    cidfile = lease_root / "container.cid"
    image = grok_cli_runner._CODEX_TASK_TOOLCHAIN_IMAGE_ID
    container_name = "ipfs-accelerate-codex-1-" + "b" * 32
    control_socket, watchdog_socket = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
    )
    lease = object.__new__(grok_cli_runner._DockerContainerLease)
    lease.provider = "codex"
    lease.docker_bin = "/usr/bin/docker"
    lease.docker_config = docker_config
    lease.container_name = container_name
    lease.cidfile = cidfile
    lease.lease_root = lease_root
    lease.cleanup_binding_record = workspace / "binding.json"
    lease._control_socket = control_socket
    lease._create_lock = threading.Lock()
    lease._create_started = False
    lease._authorized_image_id = image
    lease._create_command_id = ""
    lease._create_command_body = None
    lease.preserve_for_recovery = False
    peer_errors: list[BaseException] = []

    def read_exact(size: int) -> bytes:
        value = bytearray()
        while len(value) < size:
            chunk = watchdog_socket.recv(size - len(value))
            if not chunk:
                raise AssertionError("runner closed the private command handoff")
            value.extend(chunk)
        return bytes(value)

    def publish_only_public_terminal() -> None:
        try:
            assert read_exact(1) == b"Q"
            size = int.from_bytes(read_exact(8), "big")
            assert 0 < size <= grok_cli_runner._DOCKER_CREATE_HANDOFF_MAX_BYTES
            read_exact(size)
            assert read_exact(1) == b"D"
            armed = grok_cli_runner._validated_docker_create_journal(
                lease_root=lease_root,
                provider="codex",
                docker_bin="/usr/bin/docker",
                docker_config=docker_config,
                container_name=container_name,
                cidfile=cidfile,
            )
            assert armed is not None and armed["state"] == "create_armed"
            grok_cli_runner._transition_docker_create_journal(
                armed,
                lease_root=lease_root,
                state="create_observed",
                issuer_process_birth={
                    "pid": os.getpid(),
                    "start_time_ticks": 1,
                    "boot_id": "00000000-0000-4000-8000-000000000001",
                    "parent_pid": os.getppid(),
                },
                returncode=0,
            )
        except BaseException as exc:  # surface thread failures in the test
            peer_errors.append(exc)
        finally:
            watchdog_socket.close()

    peer = threading.Thread(target=publish_only_public_terminal)
    peer.start()
    try:
        with pytest.raises(ValueError, match="worker result is unavailable"):
            lease.create_inert_container(
                _canonical_codex_docker_create_argv(
                    cwd=workspace,
                    docker_config=docker_config,
                    cidfile=cidfile,
                    container_name=container_name,
                    image=image,
                ),
                cwd=workspace,
                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            )
    finally:
        control_socket.close()
        peer.join(timeout=5.0)
    assert not peer.is_alive()
    assert peer_errors == []
    assert lease.preserve_for_recovery is True
    terminal = grok_cli_runner._validated_docker_create_journal(
        lease_root=lease_root,
        provider="codex",
        docker_bin="/usr/bin/docker",
        docker_config=docker_config,
        container_name=container_name,
        cidfile=cidfile,
    )
    assert terminal is not None and terminal["state"] == "create_observed"

def test_concurrent_docker_create_calls_dispatch_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = object.__new__(grok_cli_runner._DockerContainerLease)
    lease._create_lock = threading.Lock()
    lease._create_started = False
    lease._authorized_image_id = "sha256:" + "f" * 64
    lease.cleanup_binding_record = None
    dispatch_started = threading.Event()
    release_dispatch = threading.Event()
    dispatches: list[list[str]] = []
    first_results: list[subprocess.CompletedProcess[bytes]] = []
    first_errors: list[BaseException] = []

    def fake_run(command: list[str], **_kwargs: object):
        dispatches.append(list(command))
        dispatch_started.set()
        assert release_dispatch.wait(timeout=5.0)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    def first_create() -> None:
        try:
            first_results.append(
                lease.create_inert_container(
                    ["/usr/bin/docker", "create", "sealed"],
                    cwd=Path.cwd(),
                    env={},
                )
            )
        except BaseException as exc:  # surface thread failures in the test
            first_errors.append(exc)

    monkeypatch.setattr(grok_cli_runner.subprocess, "run", fake_run)
    first = threading.Thread(target=first_create)
    first.start()
    assert dispatch_started.wait(timeout=5.0)
    try:
        with pytest.raises(ValueError, match="exactly one dispatch"):
            lease.create_inert_container(
                ["/usr/bin/docker", "create", "sealed"],
                cwd=Path.cwd(),
                env={},
            )
    finally:
        release_dispatch.set()
        first.join(timeout=5.0)

    assert not first.is_alive()
    assert first_errors == []
    assert len(first_results) == 1
    assert len(dispatches) == 1
