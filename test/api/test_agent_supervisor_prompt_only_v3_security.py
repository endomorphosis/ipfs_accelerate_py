"""ASE3-012 security adversary matrix for prompt-only facades."""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    configure_prompt_lifecycle_supervisor,
    prompt_entrypoints as pe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeSupervisor:
    composition_cid = "cid:security"

    def preview(self, prompt: str):
        from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
            SupervisorObservation,
        )

        return SupervisorObservation(
            run_id="",
            state="preview",
            health="unknown",
            event_cursor="",
            composition_cid=self.composition_cid,
            summary="preview-only",
            values={"effect_applied": False, "prompt_cid": "cid:p"},
        )

    def run(self, prompt: str):
        raise RuntimeError("unavailable backend")


def test_mcp_path_injection_denied_without_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(pe.REPOSITORY_ALLOWLIST_ENV, raising=False)
    configure_prompt_lifecycle_supervisor(None)
    result = asyncio.run(
        pe.agent_supervisor_run(
            prompt="x",
            repository="/etc/passwd",
        )
    )
    assert result["ok"] is False
    assert result["error_code"] == "path_denied"


def test_mcp_symlink_escape_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    allow = tmp_path / "allow"
    allow.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = allow / "escape"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink not permitted")
    monkeypatch.setenv(pe.REPOSITORY_ALLOWLIST_ENV, str(allow.resolve()))
    configure_prompt_lifecycle_supervisor(_FakeSupervisor())
    try:
        # Resolved path is outside allowlist root after symlink resolution.
        result = asyncio.run(
            pe.agent_supervisor_preview(
                prompt="x",
                repository=str(link),
            )
        )
    finally:
        configure_prompt_lifecycle_supervisor(None)
        monkeypatch.delenv(pe.REPOSITORY_ALLOWLIST_ENV, raising=False)
    # Symlink that resolves outside the allowlist must fail closed.
    assert Path(link).resolve() == outside.resolve()
    assert result["ok"] is False
    assert result["error_code"] == "path_denied"


def test_cli_preview_does_not_echo_prompt_secret() -> None:
    secret = "TOP-SECRET-PROMPT-BODY-ASE3-012"
    args = SimpleNamespace(
        supervisor_command="preview",
        prompt=secret,
        prompt_file=None,
        prompt_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, supervisor=_FakeSupervisor()
    )
    assert code == supervisor_cli.EXIT_SUCCESS
    assert secret not in out.getvalue()


def test_python_composition_manifest_has_no_secret_material() -> None:
    supervisor = Supervisor.open(repository=REPO_ROOT)
    blob = json.dumps(supervisor.composition_manifest.to_dict())
    for needle in ("password", "api_key", "BEGIN PRIVATE", "secret_token"):
        assert needle.lower() not in blob.lower()


def test_mcp_preview_rejects_prompt_body_leak() -> None:
    secret = "mcp-secret-prompt-body"
    configure_prompt_lifecycle_supervisor(_FakeSupervisor())
    try:
        result = asyncio.run(pe.agent_supervisor_preview(prompt=secret))
    finally:
        configure_prompt_lifecycle_supervisor(None)
    assert result["ok"] is True
    assert secret not in json.dumps(result)


def test_unauthorized_config_fails_typed(tmp_path: Path) -> None:
    with pytest.raises(Exception) as info:
        Supervisor.open(repository=tmp_path, require_activation=True)
    # Typed configuration failure — not a silent success.
    assert info.value.__class__.__name__ in {
        "SupervisorConfigurationError",
        "ConfigurationUnavailableError",
        "ActivationNotReadyError",
    } or "config" in str(info.value).lower() or "activation" in str(info.value).lower()
