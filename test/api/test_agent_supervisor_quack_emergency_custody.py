"""Emergency shutdown requires closure proof before releasing real OS custody."""

import errno
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    ExclusiveOwnerLease,
    FakeQuackTransport,
    QuackStateServer,
    QuackStateServerConfig,
    QuackStateServerControlError,
    QuackStateServerOwnershipError,
    ServerLifecycle,
)


@pytest.mark.parametrize("uncertain_resource", ["endpoint", "database", "namespace"])
def test_emergency_cleanup_retains_owner_until_both_resources_are_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uncertain_resource: str,
) -> None:
    database = tmp_path / "database" / "control.duckdb"
    database.parent.mkdir()
    database.write_bytes(b"fixture namespace anchor; no database operations")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    server = QuackStateServer(
        QuackStateServerConfig(database_path=database, state_dir=state_dir),
        transport=FakeQuackTransport(),
    )
    owner = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    owner.acquire(
        server_id="server:emergency-fixture", process_birth=current_process_birth(),
        database_path=database, generation=1,
    )
    server._owner = owner
    server._lifecycle = ServerLifecycle.FAILED
    anchor = server._open_database_parent_anchor()
    server._bind_database_inode_after_migration()
    server._connection = object()
    server._transport_connection = object()
    closure = {"endpoint": uncertain_resource != "endpoint",
               "database": uncertain_resource != "database"}
    displaced = database.with_name("retained.duckdb")
    if uncertain_resource == "namespace":
        # Drift after the previous successful anchor observation must not be
        # hidden by its cached flag during otherwise successful shutdown.
        database.rename(displaced)
        database.write_bytes(b"replacement inode")
        assert server._database_namespace_drifted is False

    # Inject only the external closure observations. The held file lock,
    # process birth, marker, inode anchors and canonical socket fence are real.
    def stop_transport(**_kwargs):
        if not closure["endpoint"]:
            raise QuackStateServerControlError("endpoint closure unobserved")
        server._transport_connection = None

    def wait_transport():
        if not closure["endpoint"]:
            raise QuackStateServerControlError("endpoint remains unobserved")

    def close_database():
        if not closure["database"]:
            raise QuackStateServerControlError("database closure unobserved")
        server._connection = None

    monkeypatch.setattr(server, "_stop_transport_connection", stop_transport)
    monkeypatch.setattr(server, "_wait_for_transport_endpoint_closed", wait_transport)
    monkeypatch.setattr(server, "_close_authoritative_connection_observed", close_database)
    contender = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    try:
        server._emergency_cleanup()
        assert server._owner is owner
        assert owner._lock_open
        assert owner._state.canonical_path_fence.fileno() >= 0
        assert server._database_namespace_anchor is anchor
        assert anchor.directory_descriptor >= 0 and anchor.database_descriptor >= 0
        with pytest.raises(QuackStateServerOwnershipError, match="namespace is held"):
            contender.acquire(
                server_id="server:contender", process_birth=current_process_birth(),
                database_path=database, generation=2,
            )
        with pytest.raises(QuackStateServerOwnershipError, match="retains database namespace"):
            server.start()

        if displaced.exists():
            assert server._database_namespace_drifted is True
            database.unlink()
            displaced.rename(database)
        closure.update(endpoint=True, database=True)
        server._emergency_cleanup()
        assert server._owner is None
        assert not owner._lock_open
        assert owner._state.canonical_path_fence is None
        assert server._database_namespace_anchor is None
        contender.acquire(
            server_id="server:contender", process_birth=current_process_birth(),
            database_path=database, generation=2,
        )
    finally:
        if displaced.exists():
            database.unlink()
            displaced.rename(database)
        closure.update(endpoint=True, database=True)
        server._emergency_cleanup()
        contender.release()


@pytest.mark.parametrize("failed_handle", ["parent", "canonical_socket"])
def test_emergency_cleanup_retains_partial_close_for_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_handle: str,
) -> None:
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"namespace anchor; no database operations")
    server = QuackStateServer(
        QuackStateServerConfig(database_path=database, state_dir=tmp_path / "state"),
        transport=FakeQuackTransport(),
    )
    owner = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    owner.acquire(
        server_id="server:partial-close", process_birth=current_process_birth(),
        database_path=database, generation=1,
    )
    server._owner = owner
    server._lifecycle = ServerLifecycle.FAILED
    anchor = server._open_database_parent_anchor()
    server._bind_database_inode_after_migration()
    monkeypatch.setattr(server, "_stop_transport_connection", lambda **_kwargs: None)
    monkeypatch.setattr(server, "_close_authoritative_connection_observed", lambda: None)
    path_fence = owner._state.canonical_path_fence
    parent_fd = owner._state.parent_fd
    fail_close = True
    real_close = os.close
    parent_close_attempts = []

    def close_fd(descriptor):
        if descriptor == parent_fd:
            parent_close_attempts.append(descriptor)
            if fail_close:
                raise OSError(errno.EIO, "injected directory close failure")
        real_close(descriptor)

    class UncertainSocketClose:
        def close(self):
            if fail_close:
                raise OSError(errno.EIO, "injected socket close failure")
            path_fence.close()

        def fileno(self):
            return path_fence.fileno()

    if failed_handle == "parent":
        monkeypatch.setattr(os, "close", close_fd)
    else:
        owner._state.canonical_path_fence = UncertainSocketClose()
    contender = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    try:
        with pytest.raises(QuackStateServerControlError, match="(teardown incomplete|close outcome is unknown)"):
            server._emergency_cleanup()
        # The flock has closed, but the exact canonical fence still prevents a
        # second owner. Losing this owner reference would make retry impossible.
        assert owner._state.lock_fd is None
        assert owner._retained_local_handles is True
        assert server._owner is owner
        assert server._database_namespace_anchor is anchor
        assert path_fence.fileno() >= 0
        if failed_handle == "parent":
            assert owner._state.parent_fd == parent_fd
            os.fstat(parent_fd)
        with pytest.raises(QuackStateServerOwnershipError, match="namespace is held"):
            contender.acquire(
                server_id="server:contender", process_birth=current_process_birth(),
                database_path=database, generation=2,
            )

        fail_close = False
        if failed_handle == "parent":
            # A production close error cannot establish whether its integer
            # was released. Never retry it, even when the error stops recurring.
            assert owner._state.descriptor_close_uncertain is True
            with pytest.raises(QuackStateServerControlError, match="close outcome is unknown"):
                server._emergency_cleanup()
            assert parent_close_attempts == [parent_fd]
            assert server._owner is owner
            assert path_fence.fileno() >= 0
            return
        server._emergency_cleanup()
        assert server._owner is None
        assert owner._retained_local_handles is False
        assert server._database_namespace_anchor is None
        assert path_fence.fileno() == -1
        contender.acquire(
            server_id="server:contender", process_birth=current_process_birth(),
            database_path=database, generation=2,
        )
    finally:
        fail_close = False
        # This fixture knows its injected error occurred before any OS close.
        # Only release test resources; production exposes no uncertainty reset.
        owner._state.descriptor_close_uncertain = False
        server._emergency_cleanup()
        contender.release()


def test_stop_rechecks_database_namespace_after_observed_connection_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This existing fixture uses actual DuckDB/schema/gateway admission; only
    # the Quack network transport and capability probe are substituted.
    from test.api.test_agent_supervisor_quack_state_server import _real_database_server

    server = _real_database_server(tmp_path)
    server.start()
    owner = server._owner
    database = server.config.database_path
    retained = database.with_name("retained.duckdb")
    close_database = server._close_authoritative_connection_observed
    inject_swap = True

    def close_then_swap():
        close_database()
        if inject_swap:
            database.rename(retained)
            database.write_bytes(b"replacement after original writer closed")

    monkeypatch.setattr(server, "_close_authoritative_connection_observed", close_then_swap)
    contender = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    try:
        with pytest.raises(QuackStateServerOwnershipError, match="database name or inode changed"):
            server.stop()
        assert server._connection is None
        assert server._owner is owner
        assert owner._lock_open
        assert owner._state.canonical_path_fence.fileno() >= 0
        assert server.lifecycle is ServerLifecycle.STOPPING
        with pytest.raises(QuackStateServerOwnershipError, match="namespace is held"):
            contender.acquire(
                server_id="server:contender", process_birth=current_process_birth(),
                database_path=database, generation=2,
            )

        inject_swap = False
        database.unlink()
        retained.rename(database)
        assert server.stop()["stopped"] is True
        assert server._owner is None
    finally:
        inject_swap = False
        if retained.exists():
            database.unlink()
            retained.rename(database)
        server._emergency_cleanup()
        contender.release()


def test_uncertain_integer_close_never_closes_a_reused_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"namespace fixture")
    server = QuackStateServer(
        QuackStateServerConfig(database_path=database, state_dir=tmp_path / "state"),
        transport=FakeQuackTransport(),
    )
    lease = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    lease.acquire(
        server_id="server:fd-reuse", process_birth=current_process_birth(),
        database_path=database, generation=1,
    )
    parent_fd = lease._state.parent_fd
    unrelated = tmp_path / "unrelated-resource"
    unrelated.write_bytes(b"must remain open")
    unrelated_fd = os.open(unrelated, os.O_RDONLY)
    real_close = os.close
    attempts = []

    def close_then_reuse(descriptor):
        if descriptor == parent_fd:
            attempts.append(descriptor)
            real_close(descriptor)
            os.dup2(unrelated_fd, descriptor)
            raise OSError(errno.EIO, "close failed after releasing descriptor")
        real_close(descriptor)

    monkeypatch.setattr(os, "close", close_then_reuse)
    contender = ExclusiveOwnerLease(
        lock_path=server.owner_lock_path(), marker_path=server.owner_marker_path(),
    )
    try:
        with pytest.raises(QuackStateServerControlError):
            lease.release()
        with pytest.raises(QuackStateServerControlError, match="close outcome is unknown"):
            lease.emergency_close_after_owner_shutdown()
        assert attempts == [parent_fd]
        assert os.fstat(parent_fd).st_ino == os.fstat(unrelated_fd).st_ino
        assert lease._retained_local_handles
        assert lease._state.canonical_path_fence.fileno() >= 0
        with pytest.raises(QuackStateServerOwnershipError, match="namespace is held"):
            contender.acquire(
                server_id="server:contender", process_birth=current_process_birth(),
                database_path=database, generation=2,
            )
    finally:
        monkeypatch.setattr(os, "close", real_close)
        # The fixture owns both descriptors and knows the original directory
        # handle was already closed. This is teardown, not a runtime reset API.
        real_close(parent_fd)
        real_close(unrelated_fd)
        lease._state.parent_fd = None
        lease._state.descriptor_close_uncertain = False
        lease.emergency_close_after_owner_shutdown()
        contender.release()
