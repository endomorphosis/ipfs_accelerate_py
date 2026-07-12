from __future__ import annotations

import subprocess

from ipfs_accelerate_py.utils.mistral_vibe import (
    ensure_mistral_vibe,
    mistral_vibe_auth_available,
)


def test_ensure_mistral_vibe_reuses_existing_executable():
    calls = []

    result = ensure_mistral_vibe(
        which=lambda name: "/opt/bin/vibe" if name == "vibe" else None,
        run=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result.available
    assert not result.installed
    assert result.executable == "/opt/bin/vibe"
    assert result.method == "existing"
    assert calls == []


def test_ensure_mistral_vibe_respects_auto_install_disable(tmp_path):
    result = ensure_mistral_vibe(
        auto_install=False,
        environ={"HOME": str(tmp_path)},
        which=lambda _name: None,
    )

    assert not result.available
    assert result.reason == "auto_install_disabled"


def test_ensure_mistral_vibe_prefers_uv_tool_install(tmp_path):
    installed = False
    commands = []

    def fake_which(name):
        if name == "vibe" and installed:
            return "/home/test/.local/bin/vibe"
        if name == "uv":
            return "/usr/bin/uv"
        return None

    def fake_run(command, **kwargs):
        nonlocal installed
        commands.append((command, kwargs))
        installed = True
        return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")

    result = ensure_mistral_vibe(
        auto_install=True,
        environ={"HOME": str(tmp_path), "PATH": "/usr/bin"},
        which=fake_which,
        run=fake_run,
    )

    assert result.available
    assert result.installed
    assert result.method == "uv_tool"
    assert result.executable == "/home/test/.local/bin/vibe"
    assert commands[0][0] == ["/usr/bin/uv", "tool", "install", "mistral-vibe"]
    assert commands[0][1]["check"] is False
    assert commands[0][1]["capture_output"] is True


def test_mistral_vibe_auth_available_ignores_empty_keys(tmp_path):
    vibe_dir = tmp_path / ".vibe"
    vibe_dir.mkdir()
    (vibe_dir / ".env").write_text("MISTRAL_API_KEY=\n", encoding="utf-8")

    assert not mistral_vibe_auth_available(environ={}, home=tmp_path)
    assert mistral_vibe_auth_available(
        environ={"IPFS_ACCELERATE_MISTRAL_API_KEY": "secret"},
        home=tmp_path,
    )

