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
from pathlib import Path
from typing import Any

import pytest

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
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    MigrationRunReport,
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
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
    )
    assert_bind_admitted("10.0.0.5", remote_policy=policy)
    with pytest.raises(QuackStateServerBindError, match="not admitted"):
        assert_bind_admitted("10.0.0.6", remote_policy=policy)


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
    assert "token" not in identity.secret_handle or identity.secret_handle.startswith(
        "handle:"
    )
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
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        open_duckdb_connection,
    )

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


def test_config_rejects_raw_token_as_secret_handle(tmp_path: Path) -> None:
    with pytest.raises(QuackStateServerTokenError):
        QuackStateServerConfig(
            database_path=tmp_path / "control.duckdb",
            state_dir=tmp_path / "state",
            secret_handle="raw-not-a-handle",
        )
