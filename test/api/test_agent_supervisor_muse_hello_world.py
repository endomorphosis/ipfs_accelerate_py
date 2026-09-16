"""Supervisor Muse Code install + hello-world smoke check."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.muse_hello_world import (
    HELLO_WORLD_MARKER,
    live_muse_hello_world_enabled,
    run_muse_hello_world_check,
)
from ipfs_accelerate_py.cli_runtime.installers.muse import MuseInstallResult


_FAKE_MUSE = textwrap.dedent(
    """\
    #!/bin/sh
    echo '{"type":"text","role":"assistant","text":"hello world"}'
    echo '{"type":"completed","status":"ok"}'
    exit 0
    """
)


def _clear_muse_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "IPFS_ACCELERATE_MUSE_PATH",
        "IPFS_ACCELERATE_PY_MUSE_PATH",
        "ipfs_accelerate_py_MUSE_BIN",
        "MUSE_BIN",
        "MUSE_CLI_PATH",
        "MUSE_INSTALL_DIR",
        "IPFS_ACCELERATE_AGENT_MUSE_BIN",
    ):
        monkeypatch.delenv(name, raising=False)


def test_supervisor_installs_muse_code_and_hello_world_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_dir = tmp_path / "bin"
    home = tmp_path / "home"
    home.mkdir()
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    _clear_muse_path_env(monkeypatch)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", str(empty_path))
    monkeypatch.setenv("META_API_KEY", "test-key")

    def fake_ensure_muse(**_kwargs):
        install_dir.mkdir(parents=True, exist_ok=True)
        binary = install_dir / "muse"
        binary.write_text(_FAKE_MUSE, encoding="utf-8")
        binary.chmod(0o755)
        return MuseInstallResult(
            available=True,
            installed=True,
            executable=str(binary),
            method="official_installer",
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.cli_runtime.installers.muse.ensure_muse",
        fake_ensure_muse,
    )
    workspace = tmp_path / "work"
    receipt = run_muse_hello_world_check(
        workspace=workspace,
        auto_install=True,
        timeout_seconds=15,
    )
    assert receipt["executable"], receipt
    assert Path(str(receipt["executable"])).is_file(), receipt
    assert receipt["installed"] is True, receipt
    assert receipt["ok"] is True, receipt
    assert HELLO_WORLD_MARKER in str(receipt["text"]).lower(), receipt
    assert int(receipt["exit_code"]) == 0


@pytest.mark.skipif(
    not live_muse_hello_world_enabled(),
    reason="live smoke requires IPFS_ACCELERATE_MUSE_LIVE=1 and Meta API credentials",
)
def test_live_supervisor_muse_hello_world(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance import (
        muse_code_auth_available,
    )

    if not muse_code_auth_available():
        pytest.skip("no Meta API credential configured for live Muse hello-world")
    receipt = run_muse_hello_world_check(
        workspace=tmp_path / "live-work",
        auto_install=True,
        timeout_seconds=180,
        max_model_steps=8,
    )
    if not receipt.get("executable"):
        pytest.skip(f"muse install unavailable: {receipt.get('reason')}")
    # Must be the real muse binary talking to Meta, not --provider echo.
    assert Path(str(receipt["executable"])).name == "muse"
    assert str(receipt.get("provider") or "") == "meta"
    assert "muse-spark" in str(receipt.get("model_id") or "").lower(), receipt
    if receipt.get("reason") == "muse_api_error":
        pytest.skip(
            "muse exec reached Meta but generation failed: "
            + str(receipt.get("error") or receipt.get("text") or "")[:240]
        )
    assert receipt["ok"] is True, receipt
    assert HELLO_WORLD_MARKER in str(receipt["text"]).lower()
    secret = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
    if secret:
        assert secret not in str(receipt["text"])
