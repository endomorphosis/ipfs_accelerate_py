from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.agent_supervisor import (
    provider_fallback_runner as runner,
)


def _protected_provider_paths(tmp_path: Path) -> tuple[Path, Path]:
    state_root = tmp_path / "proof-backed-test-reuse-v9"
    workspace = state_root / "worktrees" / "ptr_lane_0" / "workspace"
    workspace.mkdir(mode=0o700, parents=True)
    return state_root, workspace


def _source_identity(path: Path) -> tuple[int, ...]:
    entry = path.stat()
    return (
        entry.st_dev,
        entry.st_ino,
        entry.st_mode,
        entry.st_nlink,
        entry.st_size,
        entry.st_mtime_ns,
        entry.st_ctime_ns,
    )


def test_sealed_home_projects_only_private_grok_auth_before_landlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, workspace = _protected_provider_paths(tmp_path)
    qualification_home = tmp_path / "trusted-duckdb-home"
    qualification_home.mkdir(mode=0o700)
    empty_qualification_grok_home = qualification_home / ".grok"
    empty_qualification_grok_home.mkdir(mode=0o700)
    operator_grok_home = tmp_path / "operator-home" / ".grok"
    operator_grok_home.mkdir(mode=0o700, parents=True)
    source_auth = operator_grok_home / "auth.json"
    payload = b'{"oauth":"private-grok-authority"}\n'
    source_auth.write_bytes(payload)
    source_auth.chmod(0o600)
    source_before = _source_identity(source_auth)

    monkeypatch.setenv("HOME", str(qualification_home))
    monkeypatch.delenv("GROK_HOME", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv(
        runner.PROVIDER_PROTECTED_STATE_ROOT_ENV,
        str(state_root),
    )
    monkeypatch.setenv(runner.PROOF_REUSE_STATE_ROOT_ENV, "")
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR",
        raising=False,
    )
    monkeypatch.setattr(runner, "_grok_cli_auth_path", lambda: source_auth)

    runtime = runner._prepare_provider_boundary_runtime(workspace=workspace)
    private_home = Path(runtime.environment["HOME"])
    try:
        private_grok_home = Path(runtime.environment["GROK_HOME"])
        projected_auth = private_grok_home / "auth.json"
        assert private_grok_home == private_home / ".grok"
        assert private_grok_home.stat().st_mode & 0o777 == 0o700
        assert projected_auth.is_file()
        assert not projected_auth.is_symlink()
        assert projected_auth.read_bytes() == payload
        projected_entry = projected_auth.stat()
        assert projected_entry.st_mode & 0o777 == 0o600
        assert projected_entry.st_nlink == 1
        assert projected_entry.st_ino != source_before[1]
        assert _source_identity(source_auth) == source_before
        assert str(operator_grok_home) not in "\0".join(
            runtime.command_prefix
        )
        assert str(empty_qualification_grok_home) not in "\0".join(
            runtime.command_prefix
        )
        assert not any(empty_qualification_grok_home.iterdir())
        assert payload.decode("utf-8").strip() not in repr(runtime.environment)
        assert runtime.receipt is not None
    finally:
        assert runtime.temporary_home is not None
        runtime.temporary_home.cleanup()
    assert not private_home.exists()


def test_unprotected_route_projects_operator_auth_without_replacing_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    qualification_home = tmp_path / "trusted-duckdb-home"
    qualification_home.mkdir(mode=0o700)
    empty_qualification_grok_home = qualification_home / ".grok"
    empty_qualification_grok_home.mkdir(mode=0o700)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(mode=0o700)
    xdg_config_home = tmp_path / "xdg-config"
    xdg_config_home.mkdir(mode=0o700)
    operator_home = tmp_path / "operator-home"
    operator_grok_home = operator_home / ".grok"
    operator_grok_home.mkdir(mode=0o700, parents=True)
    source_auth = operator_grok_home / "auth.json"
    payload = b'{"oauth":"private-grok-authority"}\n'
    source_auth.write_bytes(payload)
    source_auth.chmod(0o600)
    source_before = _source_identity(source_auth)

    monkeypatch.setenv("HOME", str(qualification_home))
    monkeypatch.delenv("GROK_HOME", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config_home))
    monkeypatch.delenv(
        runner.PROVIDER_PROTECTED_STATE_ROOT_ENV,
        raising=False,
    )
    monkeypatch.delenv(runner.PROOF_REUSE_STATE_ROOT_ENV, raising=False)
    monkeypatch.setattr(
        llm_router,
        "_operator_home_dir",
        lambda: operator_home,
    )

    runtime = runner._prepare_provider_boundary_runtime(workspace=workspace)
    private_grok_home = Path(runtime.environment["GROK_HOME"])
    private_home = private_grok_home.parent
    try:
        projected_auth = private_grok_home / "auth.json"
        assert runtime.command_prefix == ()
        assert runtime.receipt is None
        assert runtime.state_root is None
        assert runtime.environment["HOME"] == str(qualification_home)
        assert runtime.environment["CODEX_HOME"] == str(codex_home)
        assert runtime.environment["XDG_CONFIG_HOME"] == str(xdg_config_home)
        assert private_grok_home == private_home / ".grok"
        assert private_home.stat().st_mode & 0o777 == 0o700
        assert private_grok_home.stat().st_mode & 0o777 == 0o700
        assert projected_auth.read_bytes() == payload
        projected_entry = projected_auth.stat()
        assert projected_entry.st_mode & 0o777 == 0o600
        assert projected_entry.st_nlink == 1
        assert projected_entry.st_ino != source_before[1]
        assert _source_identity(source_auth) == source_before
        assert not any(empty_qualification_grok_home.iterdir())
        assert payload.decode("utf-8").strip() not in repr(runtime.environment)
    finally:
        assert runtime.temporary_home is not None
        runtime.temporary_home.cleanup()
    assert not private_home.exists()


def test_unprotected_route_without_file_auth_preserves_key_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    qualification_home = tmp_path / "trusted-duckdb-home"
    qualification_home.mkdir(mode=0o700)
    operator_home = tmp_path / "operator-home"
    operator_home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(qualification_home))
    monkeypatch.delenv("GROK_HOME", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "test-key-authority")
    monkeypatch.delenv(
        runner.PROVIDER_PROTECTED_STATE_ROOT_ENV,
        raising=False,
    )
    monkeypatch.delenv(runner.PROOF_REUSE_STATE_ROOT_ENV, raising=False)
    monkeypatch.setattr(
        llm_router,
        "_operator_home_dir",
        lambda: operator_home,
    )

    runtime = runner._prepare_provider_boundary_runtime(workspace=workspace)

    assert runtime.command_prefix == ()
    assert runtime.receipt is None
    assert runtime.temporary_home is None
    assert "GROK_HOME" not in runtime.environment
    assert runtime.environment["HOME"] == str(qualification_home)
    assert runtime.environment["XAI_API_KEY"] == "test-key-authority"


@pytest.mark.parametrize("unsafe_kind", ("symlink", "public", "hardlink"))
def test_private_grok_auth_projection_rejects_unsafe_source(
    unsafe_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_home = tmp_path / "private-home"
    private_home.mkdir(mode=0o700)
    source_home = tmp_path / "operator-home" / ".grok"
    source_home.mkdir(mode=0o700, parents=True)
    source_auth = source_home / "auth.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "real-auth.json"
        target.write_text('{"oauth":"authority"}\n', encoding="utf-8")
        target.chmod(0o600)
        source_auth.symlink_to(target)
    elif unsafe_kind == "public":
        source_auth.write_text('{"oauth":"authority"}\n', encoding="utf-8")
        source_auth.chmod(0o644)
    else:
        target = tmp_path / "linked-auth.json"
        target.write_text('{"oauth":"authority"}\n', encoding="utf-8")
        target.chmod(0o600)
        os.link(target, source_auth)
    monkeypatch.setattr(runner, "_grok_cli_auth_path", lambda: source_auth)

    with pytest.raises(runner.ValidationRuntimeError, match="Grok auth projection"):
        runner._project_private_grok_auth(private_home)

    assert not (private_home / ".grok").exists()
