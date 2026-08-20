"""Tests for the loopback Quack state-owner service (DQP-006).

Acceptance:

* No token appears in argv, logs, status, exports, or provider environment
* A second owner fails closed
* Ready requires live query plus matching store/generation/schema/server identities
* Non-loopback bind requires a separately reviewed policy unavailable by default
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationExpiredError,
    LeaseState,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    DEFAULT_LOOPBACK_HOST,
    QUACK_STATE_SERVER_INTERFACE,
    STATE_SERVER_IDENTITY_INTERFACE,
    ExclusiveOwnerLease,
    FakeQuackTransport,
    OwnerMarker,
    QuackStateServer,
    QuackStateServerBindError,
    QuackStateServerCapabilityError,
    QuackStateServerConfig,
    QuackStateServerOwnershipError,
    QuackStateServerReadyError,
    QuackStateServerTokenError,
    RemoteBindPolicy,
    ServerLifecycle,
    StateServerIdentity,
    TokenVault,
    assert_bind_admitted,
    build_server,
    listen_uri,
    provider_safe_environment,
    reclaim_stale_owner_marker,
    sanitize_for_export,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationRunReport,
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DATABASE_TASK_SOURCE_SCHEMA,
    TaskPage,
    TaskRecord,
    TaskSourceSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnection,
    DuckDBConnectionPolicyError,
    open_quack_transport_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
    probe_quack_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_mutation import (
    QuackOwnerMutationEnvelopeError,
    build_mutation_request,
    parse_mutation_request,
    parse_mutation_result,
    read_envelope,
    write_envelope_atomic,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    database_program_from_daemon_namespace,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_args as parse_implementation_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    resolve_database_implementation_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_SCRIPT = REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "quack_state_server.py"
_DIGEST = "sha256:" + ("ab" * 32)
_UUID = "123e4567-e89b-12d3-a456-426614174000"


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _Result:
    def __init__(self, row: Any = None) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return [] if self._row is None else [self._row]


class FakeConnection:
    """Minimal DuckDB stand-in for hermetic state-owner tests."""

    def __init__(
        self,
        *,
        database_uuid: str = _UUID,
        schema_version: str = "1",
        schema_fingerprint: str = _DIGEST,
        max_generation: int = 0,
    ) -> None:
        self.database_uuid = database_uuid
        self.schema_version = schema_version
        self.schema_fingerprint = schema_fingerprint
        self.max_generation = max_generation
        self.statements: list[str] = []
        self.closed = False
        self._meta = {
            "database_uuid": database_uuid,
            "schema_version": schema_version,
            "schema_fingerprint": schema_fingerprint,
        }

    def execute(self, sql: str, params: Any = None) -> _Result:
        text = " ".join(str(sql).strip().split())
        self.statements.append(text)
        upper = text.upper()
        if "FROM CONTROL_PLANE_METADATA" in upper and "KEY" in upper:
            key = None
            if params:
                key = params[0] if not isinstance(params, dict) else params.get("key")
            if key is None and "KEY =" in upper:
                # Not parameterized in some paths
                pass
            return _Result((self._meta.get(str(key), ""),))
        if "FROM STORE_GENERATIONS" in upper or "MAX(GENERATION)" in upper:
            return _Result((self.max_generation,))
        if upper.startswith("SELECT 1"):
            return _Result((1,))
        if upper.startswith("CHECKPOINT"):
            return _Result()
        if upper.startswith("INSERT ") or upper.startswith("UPDATE "):
            return _Result()
        if upper.startswith("LOAD "):
            return _Result()
        return _Result()

    def close(self) -> None:
        self.closed = True


def _compatible_report(
    *,
    status: QuackCapabilityStatus = QuackCapabilityStatus.COMPATIBLE,
    fingerprint: str = _DIGEST,
) -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=status,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.2",
        duckdb_version_parsed=ParsedVersion(1, 5, 2, raw="1.5.2"),
        platform_name="Linux",
        platform_machine="x86_64",
        extension=ExtensionObservation(
            name="quack",
            installed=True,
            loaded=True,
            install_path="/tmp/quack.duckdb_extension",
            extension_version="0.1.0",
        ),
        extension_fingerprint=fingerprint,
        observed_functions=("quack_serve", "quack_query"),
        observed_surfaces=profile.required_surfaces,
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


def _migration_report() -> MigrationRunReport:
    return MigrationRunReport(
        from_version=0,
        to_version=1,
        receipts=(),
        schema_fingerprint=_DIGEST,
        catalog_fingerprint=_DIGEST,
        changed=True,
    )


def _birth(*, pid: int = 4242, ticks: int = 999, boot: str = "boot-1") -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=ticks,
        boot_id=boot,
        parent_pid=1,
    )


def _server(
    tmp_path: Path,
    *,
    host: str = DEFAULT_LOOPBACK_HOST,
    port: int = 0,
    remote_policy: RemoteBindPolicy | None = None,
    transport: FakeQuackTransport | None = None,
    capability: QuackCapabilityReport | None = None,
    schema_version: str = "1",
    liveness: OwnerLiveness = OwnerLiveness.DEAD,
    birth: ProcessBirthIdentity | None = None,
    secret_handle: str = "",
) -> QuackStateServer:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    connection = FakeConnection(schema_version=schema_version)
    live_map = {"value": liveness}

    def probe_liveness(_birth: ProcessBirthIdentity) -> OwnerLiveness:
        return live_map["value"]

    return build_server(
        database_path=db,
        state_dir=state,
        host=host,
        port=port,
        repository_id="repository:sha256:test",
        secret_handle=secret_handle,
        remote_bind_policy=remote_policy,
        transport=transport or FakeQuackTransport(),
        capability_probe=lambda **_kwargs: capability or _compatible_report(),
        migrate=lambda _path: _migration_report(),
        connection_factory=lambda _path: connection,
        process_birth_factory=lambda: birth or _birth(),
        owner_liveness_probe=probe_liveness,
    )


def _real_database_server(tmp_path: Path) -> QuackStateServer:
    """Use a real local DuckDB and fake only the network Quack transport."""

    db = tmp_path / "real-control.duckdb"
    state = tmp_path / "real-state"
    state.mkdir(parents=True, exist_ok=True)
    return build_server(
        database_path=db,
        state_dir=state,
        host="127.0.0.1",
        port=45124,
        repository_id="repository:sha256:mutation-test",
        store_id="apmc-mutation-test",
        secret_handle="handle:apmc-mutation-test",
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: _compatible_report(),
        migrate=lambda path: install_control_plane_schema(
            path,
            application_version="test",
            tool_version="test",
            owner_id="quack-mutation-test",
        ),
        connection_factory=lambda path: DuckDBConnection(path),
        process_birth_factory=lambda: _birth(),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )


class _OwnerMutationTaskSource:
    """Small read/CAS client for exercising the real owner mutation boundary."""

    def __init__(
        self,
        connection: DuckDBConnection,
        *,
        owner_lock: threading.RLock,
        first_claim_barrier: threading.Barrier,
    ) -> None:
        self._connection = connection
        self._owner_lock = owner_lock
        self._first_claim_barrier = first_claim_barrier
        self._barrier_lock = threading.Lock()
        self._raced_threads: set[int] = set()
        self.claim_receipts: list[dict[str, Any]] = []

    @staticmethod
    def _record(row: Any) -> TaskRecord:
        return TaskRecord(
            task_cid=str(row[0]),
            task_alias=str(row[1]),
            goal_cid=str(row[2]),
            ordinal=int(row[3]),
            status=str(row[4]),
            revision=int(row[5]),
            body=MappingProxyType({}),
        )

    def _select(self, *, task_cid: str = "") -> tuple[TaskRecord, ...]:
        sql = "SELECT task_cid, task_alias, goal_cid, ordinal, status, revision FROM tasks"
        params: list[Any] = []
        if task_cid:
            sql += " WHERE task_cid = ?"
            params.append(task_cid)
        sql += " ORDER BY ordinal, task_cid"
        with self._owner_lock:
            rows = self._connection.execute(sql, params).fetchall()
        return tuple(self._record(row) for row in rows)

    def ready_tasks(
        self,
        completed_ids: Any = (),
        blocked_ids: Any = (),
        limit: int = 1000,
    ) -> TaskPage:
        completed = {str(item) for item in completed_ids}
        blocked = {str(item) for item in blocked_ids}
        tasks = tuple(
            task
            for task in self._select()
            if task.status in {"todo", "ready", "open"}
            and task.task_cid not in completed
            and task.task_cid not in blocked
        )[: int(limit)]
        revision = max((task.revision for task in tasks), default=1)
        return TaskPage(tasks=tasks, revision=revision)

    def list_tasks(self, limit: int = 1000) -> TaskPage:
        tasks = self._select()
        bounded = tasks[: int(limit)]
        revision = max((task.revision for task in tasks), default=1)
        return TaskPage(
            tasks=bounded,
            revision=revision,
            next_cursor="more" if len(tasks) > len(bounded) else "",
        )

    def snapshot(self) -> TaskSourceSnapshot:
        tasks = self._select()
        projection_cid = content_identity(
            {"tasks": [task.to_dict() for task in tasks]}
        )
        terminal_statuses = {
            "completed",
            "skipped",
            "cancelled",
            "failed",
            "quarantined",
            "complete",
            "done",
        }
        return TaskSourceSnapshot(
            source_schema=DATABASE_TASK_SOURCE_SCHEMA,
            schema_version=1,
            plan_root_cid="plan:parallel",
            repository_tree_id="tree:parallel",
            projection_cid=projection_cid,
            formal_plan_id="plan:parallel",
            source_identity=content_identity(
                {"source": "owner-mutation-task-source"}
            ),
            revision=max((task.revision for task in tasks), default=1),
            event_cursor=max((task.revision for task in tasks), default=1),
            goal_count=len({task.goal_cid for task in tasks}),
            task_count=len(tasks),
            dependency_count=sum(len(task.dependencies) for task in tasks),
            terminal=bool(tasks)
            and all(task.status in terminal_statuses for task in tasks),
            objective_count=0,
            plan_count=1,
        )

    def get(self, task_cid: str) -> TaskRecord | None:
        rows = self._select(task_cid=str(task_cid))
        task = rows[0] if rows else None
        if task is not None and task.ordinal == 0:
            thread_id = threading.get_ident()
            with self._barrier_lock:
                should_race = thread_id not in self._raced_threads
                self._raced_threads.add(thread_id)
            if should_race:
                self._first_claim_barrier.wait(timeout=5.0)
        return task

    get_task = get

    def compare_and_set_status(
        self,
        task_cid: str,
        expected_revision: int,
        status: str,
        receipt: Any = None,
        *,
        evidence_digests: Any = None,
    ) -> Any:
        del evidence_digests
        if isinstance(receipt, dict):
            with self._barrier_lock:
                self.claim_receipts.append(dict(receipt))
        client = DuckDBConnection.wrap(object())
        client._default_catalog = "control_plane"  # noqa: SLF001
        result = client.execute(
            "UPDATE tasks SET status = ?, revision = revision + 1 "
            "WHERE task_cid = ? AND revision = ? "
            "AND status IN ('todo', 'ready', 'open') "
            "RETURNING revision",
            [str(status), str(task_cid), int(expected_revision)],
        )
        row = result.fetchone()
        if row is None:
            raise TaskSourceConflictError(
                f"task {task_cid} revision/status compare-and-set conflict"
            )
        return MappingProxyType({"changed": True, "revision": int(row[0])})

    cas_status = compare_and_set_status


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert QUACK_STATE_SERVER_INTERFACE == "QuackStateServer@1"
    assert STATE_SERVER_IDENTITY_INTERFACE == "StateServerIdentity@1"
    assert QuackStateServer.INTERFACE == QUACK_STATE_SERVER_INTERFACE
    assert StateServerIdentity.INTERFACE == STATE_SERVER_IDENTITY_INTERFACE


# ---------------------------------------------------------------------------
# Bind policy
# ---------------------------------------------------------------------------


def test_loopback_bind_admitted_by_default() -> None:
    assert_bind_admitted("127.0.0.1")
    assert_bind_admitted("::1")
    assert_bind_admitted("localhost")


def test_non_loopback_bind_requires_reviewed_policy() -> None:
    with pytest.raises(QuackStateServerBindError, match="separately reviewed"):
        assert_bind_admitted("0.0.0.0")
    with pytest.raises(QuackStateServerBindError, match="unavailable by default"):
        QuackStateServerConfig(
            database_path=Path("/tmp/control.duckdb"),
            state_dir=Path("/tmp/state"),
            host="0.0.0.0",
        )


def test_remote_policy_admits_listed_host_only() -> None:
    policy = RemoteBindPolicy(
        policy_id="policy:remote-1",
        reviewed_by="security-reviewer",
        review_receipt="receipt:sha256:deadbeef",
        allowed_hosts=("10.0.0.5",),
        require_tls=False,
    )
    assert_bind_admitted("10.0.0.5", remote_policy=policy)
    with pytest.raises(QuackStateServerBindError, match="not admitted"):
        assert_bind_admitted("10.0.0.6", remote_policy=policy)


def test_remote_policy_rejects_unimplemented_tls() -> None:
    policy = RemoteBindPolicy(
        policy_id="policy:remote-tls",
        reviewed_by="security-reviewer",
        review_receipt="receipt:sha256:deadbeef",
        allowed_hosts=("10.0.0.5",),
    )
    with pytest.raises(QuackStateServerBindError, match="TLS is not implemented"):
        assert_bind_admitted("10.0.0.5", remote_policy=policy)


def test_remote_policy_unavailable_without_receipt() -> None:
    with pytest.raises(QuackStateServerBindError, match="review_receipt"):
        RemoteBindPolicy(
            policy_id="policy:x",
            reviewed_by="rev",
            review_receipt="",
            allowed_hosts=("1.2.3.4",),
        )


# ---------------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------------


def test_token_vault_mints_handle_only_and_destroys(tmp_path: Path) -> None:
    vault = TokenVault(tmp_path)
    handle = vault.mint(secret_handle="handle:quack-token:test:g1", generation=1)
    assert handle.handle.startswith("handle:")
    token = vault.resolve()
    assert token
    assert token not in handle.handle
    status = {"secret_handle": handle.handle, "token": token}
    with pytest.raises(QuackStateServerTokenError):
        vault.assert_absent_from(status, surface_name="status")
    vault.destroy()
    with pytest.raises(QuackStateServerTokenError):
        vault.resolve()


def test_started_server_never_leaks_token_to_surfaces(tmp_path: Path) -> None:
    server = _server(tmp_path)
    identity = server.start()
    token = server._vault.resolve()  # noqa: SLF001 — test inspects vault

    status = server.status()
    export = server.export_identity()
    ready = server.ready()
    logs = server.logs()
    argv = server.argv_safe_launch_spec()
    provider_env = server.provider_environment(
        {
            "PATH": "/usr/bin",
            "QUACK_TOKEN": token,
            "AUTH_TOKEN": token,
            "NORMAL": "ok",
        }
    )

    surfaces = [status, export, ready, list(logs), argv, provider_env, identity.to_dict()]
    for surface in surfaces:
        blob = json.dumps(surface, default=str) if not isinstance(surface, str) else surface
        if isinstance(surface, (list, tuple)):
            blob = " ".join(str(item) for item in surface)
        assert token not in blob
        assert token not in json.dumps(status)

    assert "QUACK_TOKEN" not in provider_env
    assert "AUTH_TOKEN" not in provider_env
    assert provider_env.get("NORMAL") == "ok"
    assert identity.secret_handle
    assert "token" not in identity.secret_handle or identity.secret_handle.startswith("handle:")
    # Status may include secret_handle but never raw token keys with values.
    assert status.get("secret_handle") == identity.secret_handle
    assert status.get("token") in (None, "secret_material")
    server.stop()


def test_sanitize_for_export_redacts_token_keys() -> None:
    payload = {"auth_token": "super-secret", "server_id": "server:1"}
    out = sanitize_for_export(payload)
    assert out["auth_token"] == "secret_material"
    assert out["server_id"] == "server:1"


def test_provider_safe_environment_strips_credential_names() -> None:
    env = provider_safe_environment(
        {
            "PATH": "/usr/bin",
            "QUACK_TOKEN": "abc",
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR": "/private/mutations",
            "MY_SECRET": "x",
            "HOME": "/tmp",
        }
    )
    assert env == {"PATH": "/usr/bin", "HOME": "/tmp"}


# ---------------------------------------------------------------------------
# Exclusive ownership / second owner / stale recovery
# ---------------------------------------------------------------------------


def test_second_owner_fails_closed(tmp_path: Path) -> None:
    first = _server(tmp_path)
    first.start()
    # Second server on the same DB must refuse while first is live.
    second = _server(tmp_path, liveness=OwnerLiveness.ALIVE, birth=_birth(pid=9999))
    with pytest.raises(QuackStateServerOwnershipError, match="second state-owner"):
        second.start()
    first.stop()


def test_mutation_request_is_published_only_after_complete_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polling cannot observe a final request path while its bytes are partial."""

    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_owner_mutation,
    )

    token = "atomic-publication-token"
    request_id = "ac" * 16
    request = build_mutation_request(
        request_id=request_id,
        store_id="store:atomic-publication",
        generation=1,
        sql="UPDATE tasks SET revision = revision + 1 "
        "WHERE task_cid = ? AND revision = ? RETURNING revision",
        parameters=["task:atomic", 1],
        token=token,
    )
    target = tmp_path / f"{request_id}.request.json"
    publish_entered = threading.Event()
    allow_publish = threading.Event()
    real_publish = quack_owner_mutation._publish_without_replace  # noqa: SLF001

    def delayed_publish(source: Path, destination: Path) -> None:
        publish_entered.set()
        assert allow_publish.wait(timeout=5.0)
        real_publish(source, destination)

    monkeypatch.setattr(
        quack_owner_mutation,
        "_publish_without_replace",
        delayed_publish,
    )
    writer_errors: list[BaseException] = []

    def writer() -> None:
        try:
            write_envelope_atomic(target, request, replace=False)
        except BaseException as exc:  # pragma: no cover - asserted below
            writer_errors.append(exc)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    assert publish_entered.wait(timeout=5.0)
    try:
        assert not target.exists()
        temporary = list(tmp_path.glob(f".{target.name}.*.tmp"))
        assert len(temporary) == 1
        parsed_temporary = parse_mutation_request(
            read_envelope(temporary[0]),
            token=token,
            expected_request_id=request_id,
            expected_store_id="store:atomic-publication",
            expected_generation=1,
        )
        assert parsed_temporary["request_id"] == request_id
    finally:
        allow_publish.set()
        thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert not writer_errors
    assert target.stat().st_nlink == 1
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
    parsed = parse_mutation_request(
        read_envelope(target),
        token=token,
        expected_request_id=request_id,
        expected_store_id="store:atomic-publication",
        expected_generation=1,
    )
    assert parsed["request_id"] == request_id

    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_envelope_atomic(target, request, replace=False)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_mutation_request_publication_fails_closed_without_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources import (
        quack_owner_mutation,
    )

    request = build_mutation_request(
        request_id="ad" * 16,
        store_id="store:no-atomic-publication",
        generation=1,
        sql="UPDATE tasks SET revision = revision + 1 "
        "WHERE task_cid = ? AND revision = ? RETURNING revision",
        parameters=["task:no-atomic", 1],
        token="atomic-publication-token",
    )
    target = tmp_path / f"{'ad' * 16}.request.json"
    monkeypatch.setattr(quack_owner_mutation, "_RENAMEAT2", None)
    with pytest.raises(
        QuackOwnerMutationEnvelopeError,
        match="atomic no-replace publication is unavailable",
    ):
        write_envelope_atomic(target, request, replace=False)
    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB optional dependency unavailable")
def test_authenticated_owner_mutation_pump_returns_cas_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _real_database_server(tmp_path)
    identity = server.start()
    owner_connection = server._connection  # noqa: SLF001 - exact owner boundary
    owner_connection.execute(
        "CREATE TABLE mutation_probe (probe_id VARCHAR PRIMARY KEY, revision BIGINT NOT NULL)"
    )
    owner_connection.execute(
        "INSERT INTO mutation_probe(probe_id, revision) VALUES (?, ?)",
        ["probe:1", 1],
    )
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", identity.store_id)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(server.mutation_inbox_path()),
    )

    stop = threading.Event()

    def pump() -> None:
        while not stop.wait(0.005):
            server.process_mutation_inbox()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        # A wrapped attached client routes UPDATE through the owner before it
        # can touch its (unused) remote handle.
        client = DuckDBConnection.wrap(object())
        client._default_catalog = "control_plane"  # noqa: SLF001
        result = client.execute(
            "UPDATE mutation_probe SET revision = revision + 1 "
            "WHERE probe_id = ? AND revision = ? RETURNING revision",
            ["probe:1", 1],
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == 2
        stored = owner_connection.execute(
            "SELECT revision FROM mutation_probe WHERE probe_id = ?", ["probe:1"]
        ).fetchone()
        assert stored is not None
        assert stored[0] == 2
        assert not list(server.mutation_inbox_path().glob("*.request.json"))
        assert not list(server.mutation_inbox_path().glob("*.done.json"))
    finally:
        stop.set()
        thread.join(timeout=2.0)
        server.stop()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB optional dependency unavailable")
def test_owner_mutation_pump_does_not_reexecute_request_after_signed_result(
    tmp_path: Path,
) -> None:
    """A request left behind after result publication is idempotent."""

    server = _real_database_server(tmp_path)
    identity = server.start()
    owner_connection = server._connection  # noqa: SLF001 - exact owner boundary
    owner_connection.execute(
        "CREATE TABLE mutation_replay_probe "
        "(probe_id VARCHAR PRIMARY KEY, revision BIGINT NOT NULL)"
    )
    owner_connection.execute(
        "INSERT INTO mutation_replay_probe(probe_id, revision) VALUES (?, ?)",
        ["probe:replay", 1],
    )
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    request_id = "cd" * 16
    request = build_mutation_request(
        request_id=request_id,
        store_id=identity.store_id,
        generation=identity.generation,
        sql=(
            "UPDATE mutation_replay_probe SET revision = revision + 1 "
            "WHERE probe_id = ? RETURNING revision"
        ),
        parameters=["probe:replay"],
        token=token,
    )
    inbox = server.mutation_inbox_path()
    request_path = inbox / f"{request_id}.request.json"
    done_path = inbox / f"{request_id}.done.json"
    try:
        write_envelope_atomic(request_path, request, replace=False)
        first = server.process_mutation_inbox()
        assert first[0]["ok"] is True
        assert (
            owner_connection.execute(
                "SELECT revision FROM mutation_replay_probe WHERE probe_id = ?",
                ["probe:replay"],
            ).fetchone()[0]
            == 2
        )

        # Recreate the exact request as though request unlink failed.  The
        # authenticated result must suppress another execution.
        write_envelope_atomic(request_path, request, replace=False)
        replay = server.process_mutation_inbox()
        assert replay[0]["replayed"] is True
        assert replay[0]["ok"] is True
        assert (
            owner_connection.execute(
                "SELECT revision FROM mutation_replay_probe WHERE probe_id = ?",
                ["probe:replay"],
            ).fetchone()[0]
            == 2
        )
        parsed = parse_mutation_result(
            read_envelope(done_path),
            token=token,
            expected_request_id=request_id,
            expected_store_id=identity.store_id,
            expected_generation=identity.generation,
        )
        assert parsed["ok"] is True
    finally:
        server.stop()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB optional dependency unavailable")
def test_owner_mutation_pump_rejects_symlinked_inbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owner must not scan, chmod, write into, or unlink through a symlink."""

    server = _real_database_server(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    victim_request = victim / f"{'ef' * 16}.request.json"
    victim_request.write_text("not a mutation\n", encoding="utf-8")
    server.mutation_inbox_path().symlink_to(victim, target_is_directory=True)
    identity = server.start()
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", identity.store_id)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(server.mutation_inbox_path()),
    )
    victim_mode = victim.stat().st_mode
    try:
        client = DuckDBConnection.wrap(object())
        client._default_catalog = "control_plane"  # noqa: SLF001
        with pytest.raises(DuckDBConnectionPolicyError, match="safe owner directory"):
            client.execute(
                "UPDATE tasks SET status = ? WHERE task_cid = ?",
                ["in_progress", "task:missing"],
            )
        with pytest.raises(QuackStateServerReadyError, match="safe owner directory"):
            server.process_mutation_inbox()
        assert victim.stat().st_mode == victim_mode
        assert victim_request.read_text(encoding="utf-8") == "not a mutation\n"
        assert not list(victim.glob("*.done.json"))
    finally:
        server.stop()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB optional dependency unavailable")
@pytest.mark.parametrize("case", ["forged", "unknown_field"])
def test_owner_mutation_pump_rejects_forged_or_malformed_request(
    tmp_path: Path,
    case: str,
) -> None:
    server = _real_database_server(tmp_path)
    identity = server.start()
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    request_id = "ab" * 16
    request = dict(
        build_mutation_request(
            request_id=request_id,
            store_id=identity.store_id,
            generation=identity.generation,
            sql="UPDATE tasks SET status = ? WHERE task_cid = ?",
            parameters=["in_progress", "task:missing"],
            token="wrong-authentication-token" if case == "forged" else token,
        )
    )
    if case == "unknown_field":
        request["unexpected"] = "not-admitted"
    request_path = server.mutation_inbox_path() / f"{request_id}.request.json"
    done_path = server.mutation_inbox_path() / f"{request_id}.done.json"
    write_envelope_atomic(request_path, request, replace=False)
    request_bytes = request_path.read_bytes()
    assert token.encode("ascii") not in request_bytes
    try:
        summaries = server.process_mutation_inbox()
        assert len(summaries) == 1
        response = parse_mutation_result(
            read_envelope(done_path),
            token=token,
            expected_request_id=request_id,
            expected_store_id=identity.store_id,
            expected_generation=identity.generation,
        )
        assert response["ok"] is False
        assert response["error_code"] == (
            "authentication_failed" if case == "forged" else "malformed_envelope"
        )
        assert not request_path.exists()
    finally:
        server.stop()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB optional dependency unavailable")
def test_two_database_daemons_claim_distinct_tasks_through_one_quack_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote CAS loser releases its local lease and claims different work."""

    server = _real_database_server(tmp_path)
    identity = server.start()
    owner_connection = server._connection  # noqa: SLF001 - exact owner boundary
    for ordinal in range(2):
        owner_connection.execute(
            """
            INSERT INTO tasks(
                task_cid, task_alias, goal_cid, ordinal, status, revision,
                body_json
            ) VALUES (?, ?, ?, ?, 'ready', 1, '{}')
            """,
            [
                f"task:parallel:{ordinal}",
                f"APMC-PARALLEL-{ordinal}",
                "goal:parallel",
                ordinal,
            ],
        )
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", identity.store_id)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(server.mutation_inbox_path()),
    )

    owner_lock = threading.RLock()
    first_claim_barrier = threading.Barrier(2)
    task_sources = tuple(
        _OwnerMutationTaskSource(
            owner_connection,
            owner_lock=owner_lock,
            first_claim_barrier=first_claim_barrier,
        )
        for _ in range(2)
    )
    shared_store_id = "state/apmc/control.duckdb"
    lane_args = tuple(
        parse_implementation_daemon_args(
            [
                "--task-source-kind",
                "duckdb",
                "--authority-mode",
                "quack",
                "--endpoint-secret-handle",
                identity.secret_handle,
                "--quack-endpoint",
                identity.listen_uri,
                "--state-store-id",
                shared_store_id,
                "--state-store-generation",
                str(identity.generation),
                "--state-schema-revision",
                str(identity.schema_revision),
                "--state-dir",
                str(tmp_path / f"lane-{lane}"),
                "--task-shard-count",
                "2",
                "--task-shard-index",
                str(lane),
            ]
        )
        for lane in range(2)
    )
    programs = tuple(database_program_from_daemon_namespace(args) for args in lane_args)
    assert all(program is not None for program in programs)
    assert {program.store_id for program in programs if program is not None} == {shared_store_id}
    lane_paths = tuple(
        resolve_database_implementation_paths(args, authority_mode="quack") for args in lane_args
    )
    assert lane_paths[0]["database_path"] != lane_paths[1]["database_path"]
    assert all(
        str(paths["database_path"]).endswith("quack-lane-control.duckdb") for paths in lane_paths
    )

    daemons = tuple(
        DatabaseImplementationDaemon(
            database_path=lane_paths[lane]["database_path"],
            owner_session_id=f"parallel-lane-{lane}",
            authority_mode="quack",
            task_source_kind="database",
            quack_uri=identity.listen_uri,
            task_source=task_sources[lane],
        )
        for lane in range(2)
    )
    stop = threading.Event()

    def pump() -> None:
        while not stop.wait(0.002):
            with owner_lock:
                server.process_mutation_inbox()

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(daemon.claim_next) for daemon in daemons]
            attempts = tuple(future.result(timeout=10.0) for future in futures)

            # A losing lane's next lane-local claim has attempt_number=2.
            # That number is not shared authority and must not let it inherit
            # the other lane's durable in-progress task.
            for source in task_sources:
                source._first_claim_barrier = threading.Barrier(1)  # noqa: SLF001
            duplicate_futures = [executor.submit(daemon.claim_next) for daemon in daemons]
            duplicate_attempts = tuple(future.result(timeout=10.0) for future in duplicate_futures)

        assert all(attempt is not None for attempt in attempts)
        assert {attempt.task_cid for attempt in attempts if attempt is not None} == {
            "task:parallel:0",
            "task:parallel:1",
        }
        assert duplicate_attempts == (None, None)
        with owner_lock:
            rows = owner_connection.execute(
                "SELECT task_cid, status FROM tasks ORDER BY ordinal"
            ).fetchall()
        assert [(row[0], row[1]) for row in rows] == [
            ("task:parallel:0", "in_progress"),
            ("task:parallel:1", "in_progress"),
        ]
        winning_receipts = {
            receipt["attempt_id"]: receipt
            for source in task_sources
            for receipt in source.claim_receipts
            if receipt["attempt_id"] in {attempt.attempt_id for attempt in attempts}
        }
        assert set(winning_receipts) == {attempt.attempt_id for attempt in attempts}
        for attempt in attempts:
            receipt = winning_receipts[attempt.attempt_id]
            assert receipt["claim_id"] == attempt.claim_id
            assert receipt["lease_id"] == attempt.lease_id
            assert receipt["fencing_token"] == attempt.fencing_token
            assert receipt["fence_epoch"] == attempt.fence_epoch

        # Claims are bound to their own lane's accepted lease and cannot be
        # used in the other daemon's coordination store.
        for daemon, attempt in zip(daemons, attempts, strict=True):
            assert attempt is not None
            claim = daemon.coordinator.get_task_claim(attempt.claim_id)
            assert claim is not None
            assert claim.state is LeaseState.ACCEPTED
            assert claim.task_cid == attempt.task_cid
            assert claim.owner_session_id == daemon.owner_session_id
            assert claim.fencing_token == attempt.fencing_token
            assert claim.fence_epoch == attempt.fence_epoch
            lease = daemon.coordinator.get_lease(claim.lease_id)
            assert lease is not None
            assert lease.state is LeaseState.ACCEPTED

        with pytest.raises(
            (DatabaseImplementationAuthorityError, DatabaseImplementationConflictError),
            match="unknown execution attempt|no coordination claim",
        ):
            daemons[0]._protect_attempt_write(attempts[1])  # noqa: SLF001

        # Once the exact accepted lease is released, its prior fence cannot
        # authorize another write.
        for daemon, attempt in zip(daemons, attempts, strict=True):
            assert attempt is not None
            claim = daemon.coordinator.get_task_claim(attempt.claim_id)
            assert claim is not None
            lease = daemon.coordinator.get_lease(claim.lease_id)
            assert lease is not None
            daemon.coordinator.release(
                lease,
                reason="test_stale_fence",
                expected_fencing_token=attempt.fencing_token,
                expected_fence_epoch=attempt.fence_epoch,
            )
            with pytest.raises(DatabaseCoordinationExpiredError):
                daemon._protect_attempt_write(attempt)  # noqa: SLF001
    finally:
        stop.set()
        pump_thread.join(timeout=2.0)
        for daemon in daemons:
            daemon.close()
        server.stop()


def test_stale_marker_recovery_allows_new_owner(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    marker_path = db.with_name(f".{db.name}.state-owner.json")
    lock_path = db.with_name(f".{db.name}.state-owner.lock")
    dead = _birth(pid=111, ticks=1, boot="old")
    marker = OwnerMarker(
        server_id="server:dead",
        process_birth=dead,
        database_path=str(db),
        started_at="2020-01-01T00:00:00Z",
        fence_token="fence-old",
        generation=1,
    )
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")

    result = reclaim_stale_owner_marker(
        marker_path=marker_path,
        lock_path=lock_path,
        liveness=lambda _b: OwnerLiveness.DEAD,
    )
    assert result["reclaimed"] is True
    assert not marker_path.exists()

    # New owner can start after reclaim.
    server = _server(tmp_path, liveness=OwnerLiveness.DEAD)
    identity = server.start()
    assert identity.server_id != "server:dead"
    server.stop()


def test_stale_marker_not_reclaimed_when_owner_alive(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    marker_path = db.with_name(f".{db.name}.state-owner.json")
    lock_path = db.with_name(f".{db.name}.state-owner.lock")
    live = _birth(pid=222, ticks=2, boot="live")
    marker = OwnerMarker(
        server_id="server:live",
        process_birth=live,
        database_path=str(db),
        started_at="2020-01-01T00:00:00Z",
        fence_token="fence-live",
        generation=1,
    )
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
    result = reclaim_stale_owner_marker(
        marker_path=marker_path,
        lock_path=lock_path,
        liveness=lambda _b: OwnerLiveness.ALIVE,
    )
    assert result["reclaimed"] is False
    assert result["reason"] == "owner_alive"
    assert marker_path.exists()


def test_exclusive_owner_lease_fence_mismatch_on_release(tmp_path: Path) -> None:
    lock_path = tmp_path / "owner.lock"
    marker_path = tmp_path / "owner.json"
    lease = ExclusiveOwnerLease(
        lock_path=lock_path,
        marker_path=marker_path,
        liveness=lambda _b: OwnerLiveness.DEAD,
    )
    lease.acquire(
        server_id="server:1",
        process_birth=_birth(),
        database_path=tmp_path / "control.duckdb",
    )
    with pytest.raises(Exception, match="fence"):
        lease.release(fence_token="wrong-fence")
    lease.release()


# ---------------------------------------------------------------------------
# Ready / identity / migration / lifecycle
# ---------------------------------------------------------------------------


def test_start_ready_checkpoint_stop_lifecycle(tmp_path: Path) -> None:
    transport = FakeQuackTransport()
    server = _server(tmp_path, transport=transport, port=0)
    identity = server.start()

    assert server.lifecycle is ServerLifecycle.READY
    assert identity.listen_uri.startswith("quack:127.0.0.1:")
    assert identity.schema_fingerprint == _DIGEST
    assert identity.database_uuid == _UUID
    assert identity.generation >= 1
    assert identity.process_birth_id.startswith("birth:")
    assert transport.started is True

    ready = server.ready()
    assert ready["ready"] is True
    assert ready["store_id"] == identity.store_id
    assert ready["generation"] == identity.generation
    assert ready["schema_fingerprint"] == identity.schema_fingerprint
    assert ready["server_id"] == identity.server_id

    checkpoint = server.checkpoint()
    assert checkpoint["checkpointed"] is True

    stop = server.stop()
    assert stop["stopped"] is True
    assert server.lifecycle is ServerLifecycle.STOPPED
    assert transport.stopped is True
    assert not server.owner_marker_path().exists()


def test_ready_requires_live_query(tmp_path: Path) -> None:
    transport = FakeQuackTransport(fail_live_query=True)
    server = _server(tmp_path, transport=transport)
    server.start()
    with pytest.raises(QuackStateServerReadyError, match="live query"):
        server.ready()
    assert server.is_ready() is False
    # Clear failure for clean stop path
    transport.fail_live_query = False
    server.stop()


def test_ready_requires_matching_identities(tmp_path: Path) -> None:
    class DriftTransport(FakeQuackTransport):
        def live_query(self, connection, *, identity, token):  # type: ignore[no-untyped-def]
            del connection, token
            return {
                "live": True,
                "server_id": identity.server_id,
                "store_id": "wrong-store",
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
            }

    server = _server(tmp_path, transport=DriftTransport())
    server.start()
    with pytest.raises(QuackStateServerReadyError, match="do not match"):
        server.ready()
    server.stop()


def test_ready_requires_complete_live_identity_fields(tmp_path: Path) -> None:
    class IncompleteTransport(FakeQuackTransport):
        def live_query(self, connection, *, identity, token):  # type: ignore[no-untyped-def]
            del connection, token
            return {
                "live": True,
                "server_id": identity.server_id,
                # store_id intentionally omitted — must not fall back silently
                "database_uuid": identity.database_uuid,
                "schema_revision": identity.schema_revision,
                "schema_fingerprint": identity.schema_fingerprint,
                "generation": identity.generation,
                "process_birth_id": identity.process_birth_id,
            }

    server = _server(tmp_path, transport=IncompleteTransport())
    server.start()
    with pytest.raises(QuackStateServerReadyError, match="missing identity fields"):
        server.ready()
    server.stop()


def test_migration_required_before_ready(tmp_path: Path) -> None:
    server = _server(tmp_path, schema_version="0")
    with pytest.raises(Exception, match="migrated before ready|schema must be migrated"):
        server.start()


def test_capability_admission_fail_closed(tmp_path: Path) -> None:
    bad = _compatible_report(status=QuackCapabilityStatus.UNAVAILABLE)
    server = _server(tmp_path, capability=bad)
    with pytest.raises(QuackStateServerCapabilityError):
        server.start()


def test_whoami_process_birth_published(tmp_path: Path) -> None:
    birth = _birth(pid=7777, ticks=12345, boot="boot-xyz")
    server = _server(tmp_path, birth=birth)
    identity = server.start()
    assert identity.process_birth.pid == 7777
    assert identity.process_birth.start_time_ticks == 12345
    assert identity.process_birth.boot_id == "boot-xyz"
    status = server.status()
    assert status["identity"]["process_birth"]["pid"] == 7777
    # whoami-style export
    export = server.export_identity()
    assert export["identity"]["process_birth_id"] == identity.process_birth_id
    server.stop()


def test_graceful_stop_uses_fence_control_path(tmp_path: Path) -> None:
    server = _server(tmp_path)
    identity = server.start()
    request = server.request_stop()
    assert request["requested"] is True
    assert Path(request["control_path"]).is_file()
    control = json.loads(Path(request["control_path"]).read_text(encoding="utf-8"))
    assert control["server_id"] == identity.server_id
    assert control["fence_token"]
    # fence is ownership fence, not quack auth token
    token = server._vault.resolve()  # noqa: SLF001
    assert control["fence_token"] != token
    result = server.stop()
    assert result["stopped"] is True


def test_listen_uri_format() -> None:
    assert listen_uri("127.0.0.1", 4242) == "quack:127.0.0.1:4242"


# ---------------------------------------------------------------------------
# Ops CLI argv policy
# ---------------------------------------------------------------------------


def test_ops_script_rejects_token_argv() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(OPS_SCRIPT),
            "--token",
            "raw-secret",
            "start",
            "--database",
            "/tmp/x.duckdb",
            "--state-dir",
            "/tmp/state",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "refusing argv credential flag" in (proc.stderr + proc.stdout)


def test_ops_script_help_is_cold() -> None:
    proc = subprocess.run(
        [sys.executable, str(OPS_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "secret-handle" in proc.stdout
    assert "Never accepts raw auth tokens" in proc.stdout
    # Help must not advertise a raw-token flag as an option.
    assert "--token " not in proc.stdout
    assert "--token\n" not in proc.stdout


def test_ops_module_import_is_cold() -> None:
    # Importing the ops facade must not open a database.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, sys; "
                f"spec = importlib.util.spec_from_file_location('qss', {str(OPS_SCRIPT)!r}); "
                "mod = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(mod); "
                "print('ok')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0
    assert "ok" in proc.stdout


# ---------------------------------------------------------------------------
# Optional integration with real DuckDB (migration path)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for integration path")
def test_real_duckdb_migration_then_fake_transport_ready(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="test-owner",
    )

    # Use real connection factory via duckdb_state but fake transport/capability.

    server = build_server(
        database_path=db,
        state_dir=state,
        transport=FakeQuackTransport(),
        capability_probe=lambda **_k: _compatible_report(),
        # migrate is no-op / real install already done; still call real installer
        # which is replay-safe.
        process_birth_factory=lambda: _birth(pid=os.getpid()),
        owner_liveness_probe=lambda _b: OwnerLiveness.DEAD,
    )
    # Override connection to keep open across ready.
    # Default migrate+connection_factory use real duckdb.
    identity = server.start()
    assert identity.database_uuid
    assert identity.schema_revision >= 1
    assert identity.schema_fingerprint.startswith("sha256:")
    ready = server.ready()
    assert ready["ready"] is True
    export = server.export_identity()
    assert export["identity"]["server_id"] == identity.server_id
    server.checkpoint()
    server.stop()
    # Connection closed; marker gone.
    assert not server.owner_marker_path().exists()


def test_real_default_transport_requires_authenticated_remote_readiness(
    tmp_path: Path,
) -> None:
    if not duckdb_available():
        pytest.skip("DuckDB is unavailable")
    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    server = build_server(
        database_path=tmp_path / "control.duckdb",
        state_dir=tmp_path / "owner",
        port=0,
        store_id="test-real-quack-owner",
        secret_handle="handle:test-real-quack-owner",
    )
    identity = server.start()
    client = None
    try:
        assert identity.status == "ready"
        token = (tmp_path / "owner/handle_test-real-quack-owner.quack-token").read_text(
            encoding="utf-8"
        )
        client = open_quack_transport_connection(identity.listen_uri, token=token)
        assert client.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    finally:
        if client is not None:
            client.close()
        server.stop()


def test_config_rejects_raw_token_as_secret_handle(tmp_path: Path) -> None:
    with pytest.raises(QuackStateServerTokenError):
        QuackStateServerConfig(
            database_path=tmp_path / "control.duckdb",
            state_dir=tmp_path / "state",
            secret_handle="raw-not-a-handle",
        )
