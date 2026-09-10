"""Real socket and inode boundaries for interrupted native owner shutdown."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    stale_owner_endpoints as recovery,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    OWNER_LOCK_SUFFIX,
    QUACK_STATE_SERVER_SCHEMA,
    STATE_SERVER_IDENTITY_SCHEMA,
    QuackStateServerOwnershipError,
    acquire_exclusive_owner_lock,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
    TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
)


@pytest.fixture
def stopped_owner(tmp_path):
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"canonical database must remain unchanged")
    database.chmod(0o600)
    wal = tmp_path / "control.duckdb.wal"
    wal.write_bytes(b"canonical WAL must remain unchanged")
    wal.chmod(0o600)
    state = tmp_path / "q"
    state.mkdir(mode=0o700)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    server = SimpleNamespace(
        config=SimpleNamespace(
            database_path=database,
            state_dir=state,
            host="127.0.0.1",
            port=port,
            store_id="test-store",
        ),
        lifecycle=SimpleNamespace(value="created"),
        identity=None,
        owner_lock_path=lambda: database.with_name(
            f".{database.name}{OWNER_LOCK_SUFFIX}"
        ),
        owner_marker_path=lambda: database.with_name(
            f".{database.name}.state-owner.json"
        ),
        typed_command_socket_path=lambda: state / "typed-state-owner.sock",
        typed_command_token_path=lambda: state / "typed-state-owner.token",
        status_path=lambda: state / "quack-state-server.status.json",
    )
    sockets = [
        server.typed_command_socket_path(),
        state / TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME,
    ]
    directory_fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
    for path in sockets:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(f"/proc/self/fd/{directory_fd}/{path.name}")
            path.chmod(0o600)
    server.typed_command_token_path().write_text("x" * 64)
    server.typed_command_token_path().chmod(0o600)
    write_status(server, ProcessBirthIdentity(2_147_483_647, 1, "dead-boot", 1))
    with acquire_exclusive_owner_lock(server.owner_lock_path()) as handle:
        yield server, handle, directory_fd
    os.close(directory_fd)
    assert database.read_bytes() == b"canonical database must remain unchanged"
    assert wal.read_bytes() == b"canonical WAL must remain unchanged"


def write_status(server, birth):
    material = (
        f"{birth.pid}:{birth.start_time_ticks}:{birth.boot_id}:{birth.parent_pid}"
    )
    payload = {
        "schema": QUACK_STATE_SERVER_SCHEMA,
        "database_path": str(server.config.database_path),
        "state_dir": str(server.config.state_dir),
        "store_id": "test-store",
        "identity": {
            "schema": STATE_SERVER_IDENTITY_SCHEMA,
            "store_id": "test-store",
            "server_id": "server:test",
            "database_uuid": "database:test",
            "generation": 148,
            "process_birth": birth.to_dict(),
            "process_birth_id": "birth:"
            + hashlib.sha256(material.encode()).hexdigest()[:32],
            "listen_uri": f"quack:127.0.0.1:{server.config.port}",
        },
    }
    server.status_path().write_text(json.dumps(payload))
    server.status_path().chmod(0o600)


def reclaim(fixture):
    server, handle, directory_fd = fixture
    return recovery.reclaim_stale_typed_owner_endpoints(
        server=server,
        owner_handle=handle,
        state_directory_fd=directory_fd,
    )


def assert_preserved(server):
    assert server.typed_command_socket_path().exists()
    assert server.typed_command_token_path().exists()
    assert server.status_path().exists()


def test_reclaims_only_inert_endpoints_and_is_repeatable(stopped_owner):
    server = stopped_owner[0]
    status = server.status_path().read_bytes()
    result = reclaim(stopped_owner)
    assert result["reclaimed"] is True
    assert len(result["paths"]) == 3
    assert result["task_completion_authority"] is False
    assert server.status_path().read_bytes() == status
    assert reclaim(stopped_owner) == {"reclaimed": False, "reason": "endpoints_absent"}


def test_partial_prior_retirement_is_recoverable(stopped_owner):
    stopped_owner[0].typed_command_socket_path().unlink()
    assert len(reclaim(stopped_owner)["paths"]) == 2


@pytest.mark.parametrize("state", [OwnerLiveness.ALIVE, OwnerLiveness.UNKNOWN])
def test_live_or_unknown_owner_preserves_all_endpoints(
    stopped_owner, monkeypatch, state
):
    monkeypatch.setattr(recovery, "owner_liveness", lambda _: state)
    with pytest.raises(QuackStateServerOwnershipError, match="live or unknown"):
        reclaim(stopped_owner)
    assert_preserved(stopped_owner[0])


def test_real_current_birth_is_never_reclaimed(stopped_owner):
    write_status(stopped_owner[0], current_process_birth())
    with pytest.raises(QuackStateServerOwnershipError, match="live or unknown"):
        reclaim(stopped_owner)
    assert_preserved(stopped_owner[0])


@pytest.mark.parametrize("endpoint", ["tcp", "typed", "broker"])
def test_live_listener_blocks_cleanup_even_with_dead_status(stopped_owner, endpoint):
    server, _, directory_fd = stopped_owner
    family = socket.AF_INET if endpoint == "tcp" else socket.AF_UNIX
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        if endpoint == "tcp":
            listener.bind((server.config.host, server.config.port))
        else:
            name = (
                server.typed_command_socket_path().name
                if endpoint == "typed"
                else TYPED_STATE_OWNER_GRANT_BROKER_SOCKET_FILENAME
            )
            path = server.config.state_dir / name
            path.unlink()
            listener.bind(f"/proc/self/fd/{directory_fd}/{name}")
            path.chmod(0o600)
        listener.listen(1)
        with pytest.raises(
            QuackStateServerOwnershipError, match="listener is live or unknown"
        ):
            reclaim(stopped_owner)
        assert_preserved(server)


@pytest.mark.parametrize("kind", ["mode", "symlink", "hardlink", "size"])
def test_unsafe_token_is_preserved(stopped_owner, kind):
    server = stopped_owner[0]
    token = server.typed_command_token_path()
    if kind == "mode":
        token.chmod(0o644)
    elif kind == "symlink":
        token.unlink()
        token.symlink_to(server.status_path())
    elif kind == "hardlink":
        os.link(token, token.with_suffix(".other"))
    else:
        token.write_text("wrong length")
    before = token.lstat()
    with pytest.raises(QuackStateServerOwnershipError, match="inode is unsafe"):
        reclaim(stopped_owner)
    assert token.lstat() == before
    assert_preserved(server)


@pytest.mark.parametrize("target", ["status", "token"])
def test_inode_race_refuses_cleanup(stopped_owner, monkeypatch, target):
    server = stopped_owner[0]
    path = (
        server.status_path()
        if target == "status"
        else server.typed_command_token_path()
    )
    changed = False

    def race(_):
        nonlocal changed
        if not changed:
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.chmod(0o600)
            replacement.replace(path)
            changed = True
        return OwnerLiveness.DEAD

    monkeypatch.setattr(recovery, "owner_liveness", race)
    with pytest.raises(QuackStateServerOwnershipError, match="changed"):
        reclaim(stopped_owner)
    assert_preserved(server)


@pytest.mark.parametrize(
    "field", ["process_birth", "process_birth_id", "store_id", "listen_uri"]
)
def test_incomplete_or_foreign_identity_refuses_cleanup(stopped_owner, field):
    server = stopped_owner[0]
    payload = json.loads(server.status_path().read_text())
    payload["identity"][field] = {} if field == "process_birth" else "different"
    server.status_path().write_text(json.dumps(payload))
    with pytest.raises(QuackStateServerOwnershipError):
        reclaim(stopped_owner)
    assert_preserved(server)


def test_native_baseline_recovers_residuals_before_observation(stopped_owner):
    from scripts import run_agent_supervisor_efficiency_state_hardening as operator

    server, handle, _ = stopped_owner
    handle.close()  # The real native observer must acquire its own four locks.
    observed = operator._r21_owner_start_contention_observation(
        paths={
            "database": server.config.database_path,
            "owner": server.config.state_dir,
        },
        server=server,
        expected_lifecycle="created",
    )
    assert all(observed["residual_paths_absent"].values())
    assert observed["stale_endpoint_recovery"]["reclaimed"] is True
    assert observed["lock_order"] == ["owner", "migration", "intent", "database"]
