"""Offline tests for Muse Code discovery and explicit official installer."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.cli_runtime.installers.muse import (
    OFFICIAL_INSTALL_URL,
    discover_muse,
    ensure_muse,
    muse_auto_install_enabled,
    muse_auth_available,
)


def test_discover_never_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_MUSE_PATH", raising=False)
    monkeypatch.delenv("MUSE_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def boom(*_a, **_k):
        raise AssertionError("discover must not download")

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse._default_download",
        boom,
    )
    result = discover_muse(probe_version=False)
    assert result.available is False
    assert result.reason == "not_installed"


def test_discover_explicit_path(tmp_path: Path) -> None:
    binary = tmp_path / "muse"
    binary.write_text("#!/bin/sh\necho Muse Code 0.1.0\n", encoding="utf-8")
    binary.chmod(0o755)
    result = discover_muse(explicit_path=str(binary), probe_version=False)
    assert result.available is True
    assert result.executable == str(binary)
    assert result.method == "explicit_path"


def test_ensure_muse_detect_only_does_not_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    def boom(*_a, **_k):
        raise AssertionError("detect-only ensure must not download")

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse._default_download",
        boom,
    )
    result = ensure_muse(auto_install=False)
    assert result.available is False


def test_ensure_muse_auto_install_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("IPFS_ACCELERATE_MUSE_AUTO_INSTALL", "0")
    assert muse_auto_install_enabled() is False

    def boom(*_a, **_k):
        raise AssertionError("disabled auto-install must not download")

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse._default_download",
        boom,
    )
    result = ensure_muse(auto_install=True)
    assert result.available is False
    assert result.reason == "auto_install_disabled"


def test_ensure_muse_runs_downloaded_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("MUSE_INSTALL_DIR", str(tmp_path / "bin"))
    monkeypatch.delenv("IPFS_ACCELERATE_MUSE_AUTO_INSTALL", raising=False)

    dest_bin = tmp_path / "bin"
    dest_bin.mkdir(parents=True)

    def fake_download(url: str, dest: Path) -> None:
        assert url == OFFICIAL_INSTALL_URL
        dest.write_text("#!/bin/bash\necho installer\n", encoding="utf-8")

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "bash":
            launcher = dest_bin / "muse"
            launcher.write_text("#!/bin/sh\necho Muse Code 0.1.0\n", encoding="utf-8")
            launcher.chmod(0o755)
            return type("P", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
        if argv and Path(argv[0]).name == "muse":
            return type(
                "P", (), {"returncode": 0, "stdout": "Muse Code 0.1.0\n", "stderr": ""}
            )()
        raise AssertionError(f"unexpected command: {argv!r}")

    result = ensure_muse(
        auto_install=True,
        download_fn=fake_download,
        run_fn=fake_run,
        probe_version=False,
    )
    assert result.available is True
    assert result.installed is True
    assert result.method == "official_installer"
    assert Path(result.executable).name == "muse"


def test_ensure_muse_succeeds_when_script_exits_nonzero_but_binary_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("MUSE_INSTALL_DIR", str(tmp_path / "bin"))
    monkeypatch.delenv("IPFS_ACCELERATE_MUSE_AUTO_INSTALL", raising=False)

    dest_bin = tmp_path / "bin"
    dest_bin.mkdir(parents=True)

    def fake_download(url: str, dest: Path) -> None:
        dest.write_text("#!/usr/bin/env bash\necho installer\n", encoding="utf-8")

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "bash":
            launcher = dest_bin / "muse"
            launcher.write_text("#!/bin/sh\necho Muse Code 0.1.0\n", encoding="utf-8")
            launcher.chmod(0o755)
            return type("P", (), {"returncode": 1, "stdout": "path skipped", "stderr": ""})()
        if argv and Path(argv[0]).name == "muse":
            return type(
                "P", (), {"returncode": 0, "stdout": "Muse Code 0.1.0\n", "stderr": ""}
            )()
        raise AssertionError(f"unexpected command: {argv!r}")

    result = ensure_muse(
        auto_install=True,
        download_fn=fake_download,
        run_fn=fake_run,
        probe_version=False,
    )
    assert result.available is True
    assert result.installed is True
    assert Path(result.executable).name == "muse"


def test_auth_available_is_presence_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "secret-value")
    assert muse_auth_available() is True
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("META_AI_API_KEY", raising=False)
    monkeypatch.delenv("ipfs_accelerate_py_META_AI_API_KEY", raising=False)
    monkeypatch.setattr(
        "ipfs_accelerate_py.common.meta_model_api.resolve_meta_model_api_key",
        lambda *a, **k: None,
    )
    assert muse_auth_available() is False
