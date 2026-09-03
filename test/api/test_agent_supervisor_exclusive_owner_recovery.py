"""Fail-closed exclusive owner death recovery does not treat leftover markers as live."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.exclusive_owner_recovery import (
    admit_dead_exclusive_owner_recovery,
    observe_exclusive_owner_liveness,
    restore_archived_exclusive_owner_runtime,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import OwnerMarker


def _birth(*, pid: int = 545899, ticks: int = 428471, boot: str = "boot-1") -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=ticks,
        boot_id=boot,
        parent_pid=1,
    )


def _write_marker(path: Path, *, pid: int = 545899, server_id: str = "server:dead") -> None:
    marker = OwnerMarker(
        server_id=server_id,
        process_birth=_birth(pid=pid),
        database_path=str(path.parent / "control.duckdb"),
        started_at="2026-09-02T01:53:51Z",
        fence_token="fence-old",
        generation=1,
    )
    path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")


def test_leftover_marker_with_dead_pid_is_not_live(tmp_path: Path) -> None:
    marker_path = tmp_path / ".control.duckdb.state-owner.json"
    lock_path = tmp_path / ".control.duckdb.state-owner.lock"
    _write_marker(marker_path)

    observation = observe_exclusive_owner_liveness(
        marker_path=marker_path,
        listen_port=47831,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        port_probe=lambda _host, _port: False,
    )
    assert observation["state"] == "dead"
    assert observation["reason"] == "owner_dead"
    assert observation["has_marker"] is True

    admitted = admit_dead_exclusive_owner_recovery(
        marker_path=marker_path,
        lock_path=lock_path,
        listen_port=47831,
        live_runtime=tmp_path / "runtime",
        liveness=lambda _birth: OwnerLiveness.DEAD,
        port_probe=lambda _host, _port: False,
    )
    assert admitted["admitted"] is True
    assert admitted["relaunch"] is True
    assert admitted["reclaimed"] is True
    assert not marker_path.exists()


def test_live_owner_refuses_recovery(tmp_path: Path) -> None:
    marker_path = tmp_path / ".control.duckdb.state-owner.json"
    lock_path = tmp_path / ".control.duckdb.state-owner.lock"
    _write_marker(marker_path, pid=1269733, server_id="server:live")

    admitted = admit_dead_exclusive_owner_recovery(
        marker_path=marker_path,
        lock_path=lock_path,
        listen_port=47831,
        liveness=lambda _birth: OwnerLiveness.ALIVE,
        port_probe=lambda _host, _port: True,
    )
    assert admitted["admitted"] is False
    assert admitted["reason"] == "owner_alive"
    assert marker_path.exists()


def test_unknown_liveness_and_bound_port_fail_closed(tmp_path: Path) -> None:
    marker_path = tmp_path / ".control.duckdb.state-owner.json"
    lock_path = tmp_path / ".control.duckdb.state-owner.lock"
    _write_marker(marker_path)

    unknown = admit_dead_exclusive_owner_recovery(
        marker_path=marker_path,
        lock_path=lock_path,
        listen_port=47831,
        liveness=lambda _birth: OwnerLiveness.UNKNOWN,
        port_probe=lambda _host, _port: False,
    )
    assert unknown["admitted"] is False
    assert unknown["reason"] == "owner_liveness_unknown"
    assert marker_path.exists()

    port_held = admit_dead_exclusive_owner_recovery(
        marker_path=marker_path,
        lock_path=lock_path,
        listen_port=47831,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        port_probe=lambda _host, _port: True,
    )
    assert port_held["admitted"] is False
    assert port_held["reason"] == "listen_port_held"
    assert marker_path.exists()


def test_restores_pre_relaunch_archive_only_when_live_runtime_is_absent(
    tmp_path: Path,
) -> None:
    live = tmp_path / "proof_carrying_platform_qualification_and_release_v1"
    archive = tmp_path / (
        "proof_carrying_platform_qualification_and_release_v1.pre-relaunch-88ae5e48"
    )
    archive.mkdir()
    (archive / "control.duckdb").write_text("archived", encoding="utf-8")

    admitted = admit_dead_exclusive_owner_recovery(
        marker_path=live / ".control.duckdb.state-owner.json",
        lock_path=live / ".control.duckdb.state-owner.lock",
        listen_port=47831,
        live_runtime=live,
        port_probe=lambda _host, _port: False,
    )
    assert admitted["admitted"] is True
    assert admitted["restore_from"] == str(archive)

    restored = restore_archived_exclusive_owner_runtime(live)
    assert restored["restored"] is True
    assert live.is_dir()
    assert not archive.exists()
    assert (live / "control.duckdb").read_text(encoding="utf-8") == "archived"

    refused = restore_archived_exclusive_owner_runtime(live)
    assert refused["restored"] is False
    assert refused["reason"] == "live_runtime_present"
