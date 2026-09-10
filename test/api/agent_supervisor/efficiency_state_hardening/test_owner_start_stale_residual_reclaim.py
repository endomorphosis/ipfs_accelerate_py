"""R21 owner-start must reclaim a dead previous owner's marker before start."""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    OwnerMarker,
)
from scripts import run_agent_supervisor_efficiency_state_hardening as aseh_operator


def _r21_server(database: Path, owner: Path, *, port: int = 43210) -> SimpleNamespace:
    return SimpleNamespace(
        lifecycle=SimpleNamespace(value="created"),
        identity=None,
        config=SimpleNamespace(host="127.0.0.1", port=port),
        owner_marker_path=lambda: database.with_name(
            f".{database.name}.state-owner.json"
        ),
        typed_command_socket_path=lambda: owner / "typed-owner.sock",
        typed_command_token_path=lambda: owner / "typed-owner.token",
    )


def _write_marker(database: Path, birth: ProcessBirthIdentity) -> Path:
    marker_path = database.with_name(f".{database.name}.state-owner.json")
    marker = OwnerMarker(
        server_id="server:stale-residual",
        process_birth=birth,
        database_path=str(database),
        started_at="2026-09-09T14:06:32Z",
        fence_token="e436a45781656600589441abd711e485",
        generation=129,
    )
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
    marker_path.chmod(0o600)
    return marker_path


def test_r21_observation_reclaims_dead_owner_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"sealed-r21-owner-state")
    database.chmod(0o600)
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    marker_path = _write_marker(
        database,
        ProcessBirthIdentity(
            pid=2_147_483_647,
            start_time_ticks=1,
            boot_id="dead-boot",
            parent_pid=1,
        ),
    )
    assert marker_path.is_file()
    monkeypatch.setattr(
        aseh_operator,
        "_r21_listener_absence_samples",
        lambda *_args, **_kwargs: ["connection_refused", "connection_refused"],
    )

    observation = aseh_operator._r21_owner_start_contention_observation(
        paths={"database": database, "owner": owner},
        server=_r21_server(database, owner),
        expected_lifecycle="created",
    )

    assert not marker_path.exists()
    assert observation["residual_paths_absent"]["owner_marker"] is True
    assert observation["identity_absent"] is True
    assert stat.S_IMODE(database.stat().st_mode) == 0o600


def test_r21_observation_keeps_live_owner_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"sealed-r21-owner-state")
    database.chmod(0o600)
    owner = tmp_path / "owner"
    owner.mkdir(mode=0o700)
    marker_path = _write_marker(database, current_process_birth())
    monkeypatch.setattr(
        aseh_operator,
        "_r21_listener_absence_samples",
        lambda *_args, **_kwargs: ["connection_refused", "connection_refused"],
    )

    with pytest.raises(
        aseh_operator.OperatorError,
        match="residual authority is present",
    ):
        aseh_operator._r21_owner_start_contention_observation(
            paths={"database": database, "owner": owner},
            server=_r21_server(database, owner),
            expected_lifecycle="created",
        )

    assert marker_path.is_file()


def test_assert_witness_skips_status_when_foreign_index_lock_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    head = "a" * 40
    tree = "b" * 40
    witness = {
        "head": head,
        "tree": tree,
        "status_digest": aseh_operator._identity(b""),
        "branch_ref": "refs/heads/aseh",
        "index_entries_digest": "unused",
        "index_flags_digest": "unused",
        "head_reflog_digest": "unused",
        "branch_reflog_digest": "unused",
    }
    status_calls: list[tuple[str, ...]] = []

    def fake_git(*arguments: str, **_kwargs: object) -> str:
        if arguments and arguments[0] == "status":
            status_calls.append(arguments)
            raise subprocess.TimeoutExpired(("git",) + arguments, 60)
        if arguments[:2] == ("rev-parse", "HEAD"):
            return head
        if arguments[:2] == ("rev-parse", "HEAD^{tree}"):
            return tree
        raise AssertionError(arguments)

    monkeypatch.setattr(aseh_operator, "_ASEH_CANDIDATE_GIT_GUARD", None)
    monkeypatch.setattr(aseh_operator, "_git_guard_index_lock_is_held", lambda: True)
    monkeypatch.setattr(aseh_operator, "_git", fake_git)
    aseh_operator._assert_candidate_authorization_witness(
        witness,
        expected_head=head,
        expected_tree=tree,
        boundary="sealed child vs parent git-guard",
    )
    assert status_calls == []


def test_git_status_timeout_becomes_operator_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def expire(*_args: object, **_kwargs: object) -> object:
        raise __import__("subprocess").TimeoutExpired(
            cmd=("git", "status"),
            timeout=300,
        )

    monkeypatch.setattr(aseh_operator.subprocess, "run", expire)
    monkeypatch.setattr(
        aseh_operator,
        "_trusted_git_executable",
        lambda: "/usr/bin/git",
    )
    with pytest.raises(aseh_operator.OperatorError, match="timed out after 300"):
        aseh_operator._git("status", "--porcelain=v1", "--untracked-files=all")
