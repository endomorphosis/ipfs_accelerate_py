"""Focused recovery tests for fenced Quack owner lifecycle bookkeeping."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    OwnerMarker,
    QuackStateServerMigrationError,
    QuackStateServerOwnershipError,
    QuackStateServerReadyError,
    StateServerIdentity,
    build_server,
    inspect_state_server_lifecycle,
    offline_state_server_fence,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    probe_quack_capabilities,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_SCRIPT = REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "quack_state_server.py"
_DIGEST = "sha256:" + ("cd" * 32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _birth(pid: int = 918273, ticks: int = 4567) -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=ticks,
        boot_id="boot-stale-owner",
        parent_pid=1,
    )


def _birth_id(birth: ProcessBirthIdentity) -> str:
    return StateServerIdentity(
        server_id="server:birth-id-only",
        store_id="control.duckdb",
        database_uuid="database:birth-id-only",
        schema_revision=1,
        schema_fingerprint=_DIGEST,
        generation=1,
        fence_epoch=1,
        revision=0,
        process_birth=birth,
        listen_uri="quack:127.0.0.1:1",
        extension_fingerprint=_DIGEST,
        credential_generation=1,
        secret_handle="handle:test-birth-id",
    ).process_birth_id


def _seed_stale_server(
    tmp_path: Path,
    *,
    status: str,
    row_birth: ProcessBirthIdentity | None = None,
    row_database_uuid: str | None = None,
) -> tuple[Path, Path, OwnerMarker, str]:
    duckdb = pytest.importorskip("duckdb")
    database = tmp_path / "control.duckdb"
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    install_control_plane_schema(database, owner_id="lifecycle-recovery-test")
    connection = duckdb.connect(str(database))
    try:
        database_uuid = str(
            connection.execute(
                "SELECT value FROM control_plane_metadata WHERE key = 'database_uuid'"
            ).fetchone()[0]
        )
        stale_birth = _birth()
        effective_birth = row_birth or stale_birth
        effective_database_uuid = row_database_uuid or database_uuid
        stopped_at = "2026-08-30T00:01:00Z" if status == "stopped" else None
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at, extension_schema, extension_json
            ) VALUES (1, 1, 1, 0, ?, ?, '2026-08-30T00:00:00Z', '', '{}')
            """,
            [database_uuid, _birth_id(stale_birth)],
        )
        connection.execute(
            """
            INSERT INTO state_servers (
                server_id, store_id, database_uuid, process_birth_id,
                listen_uri, extension_fingerprint, schema_revision, generation,
                started_at, stopped_at, status, revision,
                extension_schema, extension_json
            ) VALUES (?, 'control.duckdb', ?, ?, 'quack:127.0.0.1:29999',
                      ?, 1, 1, '2026-08-30T00:00:00Z', ?, ?, 7, '', '{}')
            """,
            [
                "server:stale-prior",
                effective_database_uuid,
                _birth_id(effective_birth),
                _DIGEST,
                stopped_at,
                status,
            ],
        )
        connection.execute(
            """
            INSERT INTO server_epochs (
                server_id, epoch, fence_epoch, started_at, ended_at
            ) VALUES ('server:stale-prior', 1, 1,
                      '2026-08-30T00:00:00Z', ?)
            """,
            [stopped_at],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    marker = OwnerMarker(
        server_id="server:stale-prior",
        process_birth=stale_birth,
        database_path=str(database.resolve()),
        started_at="2026-08-30T00:00:00Z",
        fence_token="prior-fence-token",
        generation=1,
    )
    marker_path = database.with_name(f".{database.name}.state-owner.json")
    marker_path.write_text(json.dumps(marker.to_dict()), encoding="utf-8")
    return database, state_dir, marker, database_uuid


def _server(database: Path, state_dir: Path):
    return build_server(
        database_path=database,
        state_dir=state_dir,
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: probe_quack_capabilities(
            allow_network_install=False,
            allow_local_load=True,
            use_cache=False,
        ),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )


def _prior_row(database: Path) -> tuple[Any, ...]:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(
            """
            SELECT database_uuid, process_birth_id, status, stopped_at, revision
            FROM state_servers WHERE server_id = 'server:stale-prior'
            """
        ).fetchone()
    finally:
        connection.close()


def _prior_epochs(database: Path) -> list[tuple[Any, ...]]:
    duckdb = pytest.importorskip("duckdb")
    connection = duckdb.connect(str(database), read_only=True)
    try:
        return connection.execute(
            """
            SELECT epoch, fence_epoch, started_at, ended_at
            FROM server_epochs
            WHERE server_id = 'server:stale-prior'
            ORDER BY epoch, fence_epoch
            """
        ).fetchall()
    finally:
        connection.close()


@pytest.mark.parametrize("prior_status", ["starting", "ready"])
def test_dead_starting_or_ready_generation_is_reconciled_exactly(
    tmp_path: Path,
    prior_status: str,
) -> None:
    database, state_dir, marker, database_uuid = _seed_stale_server(
        tmp_path,
        status=prior_status,
    )
    server = _server(database, state_dir)
    identity = server.start()
    reconciliation = server.status()["lifecycle_reconciliation"]
    assert identity.generation == 2
    assert reconciliation["reconciled"] is True
    assert reconciliation["prior_server_id"] == marker.server_id
    assert reconciliation["prior_process_birth_id"] == _birth_id(marker.process_birth)
    assert reconciliation["database_uuid"] == database_uuid
    assert reconciliation["prior_status"] == prior_status
    assert reconciliation["status"] == "stopped"
    assert reconciliation["revision"] == 8
    assert reconciliation["fence_token_digest"].startswith("sha256:")
    current_marker = OwnerMarker.from_dict(
        json.loads(server.owner_marker_path().read_text(encoding="utf-8"))
    )
    assert current_marker.server_id == identity.server_id
    assert current_marker.generation == identity.generation == 2
    assert current_marker.started_at == identity.started_at
    server.stop()
    row = _prior_row(database)
    assert row[0] == database_uuid
    assert row[1] == _birth_id(marker.process_birth)
    assert row[2] == "stopped"
    assert row[3] is not None
    assert row[4] == 8


def test_already_stopped_prior_generation_is_idempotent(tmp_path: Path) -> None:
    database, state_dir, _marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="stopped",
    )
    before = _prior_row(database)
    server = _server(database, state_dir)
    server.start()
    reconciliation = server.status()["lifecycle_reconciliation"]
    assert reconciliation["reconciled"] is False
    assert reconciliation["reason"] == "prior_server_already_terminal"
    server.stop()
    assert _prior_row(database) == before


def test_post_publication_failure_is_closed_before_successful_retry(
    tmp_path: Path,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    database, state_dir, prior_marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    failing = build_server(
        database_path=database,
        state_dir=state_dir,
        transport=FakeQuackTransport(fail_live_query=True),
        capability_probe=lambda **_kwargs: probe_quack_capabilities(
            allow_network_install=False,
            allow_local_load=True,
            use_cache=False,
        ),
        process_birth_factory=lambda: _birth(pid=222_333, ticks=444),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )

    with pytest.raises(QuackStateServerReadyError):
        failing.start()

    failed_identity = failing.status()["identity"]
    assert failed_identity["generation"] == 2
    assert failed_identity["status"] == "stopped"
    marker_path = database.with_name(f".{database.name}.state-owner.json")
    restored = OwnerMarker.from_dict(json.loads(marker_path.read_text(encoding="utf-8")))
    assert restored.server_id == prior_marker.server_id
    connection = duckdb.connect(str(database), read_only=True)
    try:
        failed_row = connection.execute(
            """
            SELECT status, stopped_at, revision
            FROM state_servers WHERE server_id = ?
            """,
            [failed_identity["server_id"]],
        ).fetchone()
    finally:
        connection.close()
    assert failed_row[0] == "stopped"
    assert failed_row[1] is not None
    assert failed_row[2] >= 2

    retry = _server(database, state_dir)
    recovered = retry.start()
    assert recovered.generation == 3
    retry.stop()

    connection = duckdb.connect(str(database), read_only=True)
    try:
        nonterminal = connection.execute(
            """
            SELECT server_id, generation, status
            FROM state_servers
            WHERE status IN ('starting', 'ready') OR stopped_at IS NULL
            """
        ).fetchall()
    finally:
        connection.close()
    assert nonterminal == []


@pytest.mark.parametrize("epoch_corruption", ["missing", "ambiguous"])
def test_failed_start_closure_refuses_inexact_epoch_and_preserves_current_marker(
    tmp_path: Path,
    epoch_corruption: str,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    database, state_dir, prior_marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    failing = build_server(
        database_path=database,
        state_dir=state_dir,
        transport=FakeQuackTransport(fail_live_query=True),
        capability_probe=lambda **_kwargs: probe_quack_capabilities(
            allow_network_install=False,
            allow_local_load=True,
            use_cache=False,
        ),
        process_birth_factory=lambda: _birth(pid=333_444, ticks=555),
        owner_liveness_probe=lambda _birth: OwnerLiveness.DEAD,
    )
    original_publish = failing._publish_identity_rows

    def publish_with_corrupt_epoch(
        connection: Any,
        identity: StateServerIdentity,
        capability: Any,
    ) -> None:
        original_publish(connection, identity, capability)
        expected_epoch = int(identity.startup_epoch or identity.generation)
        if epoch_corruption == "missing":
            connection.execute(
                "DELETE FROM server_epochs WHERE server_id = ?",
                [identity.server_id],
            )
        else:
            connection.execute(
                """
                INSERT INTO server_epochs (
                    server_id, epoch, fence_epoch, started_at, ended_at
                ) VALUES (?, ?, ?, ?, NULL)
                """,
                [
                    identity.server_id,
                    expected_epoch + 1,
                    identity.fence_epoch,
                    identity.started_at,
                ],
            )

    failing._publish_identity_rows = publish_with_corrupt_epoch  # type: ignore[method-assign]

    with pytest.raises(QuackStateServerReadyError):
        failing.start()

    failed_identity = failing.status()["identity"]
    assert failed_identity["generation"] == 2
    assert failed_identity["status"] == "ready"
    retained = OwnerMarker.from_dict(
        json.loads(failing.owner_marker_path().read_text(encoding="utf-8"))
    )
    assert retained.server_id == failed_identity["server_id"]
    assert retained.server_id != prior_marker.server_id
    assert retained.generation == failed_identity["generation"]
    assert retained.started_at == failed_identity["started_at"]

    connection = duckdb.connect(str(database), read_only=True)
    try:
        failed_row = connection.execute(
            """
            SELECT status, stopped_at
            FROM state_servers WHERE server_id = ?
            """,
            [failed_identity["server_id"]],
        ).fetchone()
        failed_epochs = connection.execute(
            """
            SELECT epoch, fence_epoch, ended_at
            FROM server_epochs WHERE server_id = ? ORDER BY epoch
            """,
            [failed_identity["server_id"]],
        ).fetchall()
    finally:
        connection.close()
    assert failed_row == ("ready", None)
    assert all(row[2] is None for row in failed_epochs)
    assert len(failed_epochs) == (0 if epoch_corruption == "missing" else 2)


@pytest.mark.parametrize("mismatch", ["process_birth", "database_uuid"])
def test_reconciliation_identity_mismatch_fails_closed_and_restores_marker(
    tmp_path: Path,
    mismatch: str,
) -> None:
    row_birth = _birth(pid=111222, ticks=333) if mismatch == "process_birth" else None
    row_database_uuid = "database:wrong-authority" if mismatch == "database_uuid" else None
    database, state_dir, marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
        row_birth=row_birth,
        row_database_uuid=row_database_uuid,
    )
    before = _prior_row(database)
    server = _server(database, state_dir)
    with pytest.raises(QuackStateServerOwnershipError):
        server.start()
    assert _prior_row(database) == before
    restored = json.loads(server.owner_marker_path().read_text(encoding="utf-8"))
    assert restored == marker.to_dict()


def test_reconciliation_rejects_marker_generation_mismatch_without_mutation(
    tmp_path: Path,
) -> None:
    database, state_dir, marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    mismatched = OwnerMarker(
        server_id=marker.server_id,
        process_birth=marker.process_birth,
        database_path=marker.database_path,
        started_at=marker.started_at,
        fence_token=marker.fence_token,
        generation=2,
    )
    server = _server(database, state_dir)
    server.owner_marker_path().write_text(
        json.dumps(mismatched.to_dict()),
        encoding="utf-8",
    )
    before_row = _prior_row(database)
    before_epochs = _prior_epochs(database)

    with pytest.raises(
        QuackStateServerOwnershipError,
        match="marker generation differs",
    ):
        server.start()

    assert _prior_row(database) == before_row
    assert _prior_epochs(database) == before_epochs
    restored = OwnerMarker.from_dict(
        json.loads(server.owner_marker_path().read_text(encoding="utf-8"))
    )
    assert restored == mismatched


@pytest.mark.parametrize("epoch_corruption", ["missing", "wrong_fence", "ambiguous"])
def test_reconciliation_rejects_missing_or_ambiguous_epoch_without_mutation(
    tmp_path: Path,
    epoch_corruption: str,
) -> None:
    duckdb = pytest.importorskip("duckdb")
    database, state_dir, marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    connection = duckdb.connect(str(database))
    try:
        if epoch_corruption == "missing":
            connection.execute(
                "DELETE FROM server_epochs WHERE server_id = ?",
                [marker.server_id],
            )
        elif epoch_corruption == "wrong_fence":
            connection.execute(
                "UPDATE server_epochs SET fence_epoch = 2 WHERE server_id = ?",
                [marker.server_id],
            )
        else:
            connection.execute(
                """
                INSERT INTO server_epochs (
                    server_id, epoch, fence_epoch, started_at, ended_at
                ) VALUES (?, 2, 1, '2026-08-30T00:00:00Z', NULL)
                """,
                [marker.server_id],
            )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    before_row = _prior_row(database)
    before_epochs = _prior_epochs(database)
    server = _server(database, state_dir)

    with pytest.raises(
        QuackStateServerMigrationError,
        match="missing or ambiguous generation epoch",
    ):
        server.start()

    assert _prior_row(database) == before_row
    assert _prior_epochs(database) == before_epochs
    restored = OwnerMarker.from_dict(
        json.loads(server.owner_marker_path().read_text(encoding="utf-8"))
    )
    assert restored == marker


@pytest.mark.parametrize("liveness", [OwnerLiveness.ALIVE, OwnerLiveness.UNKNOWN])
def test_live_or_unknown_prior_owner_refuses_before_database_mutation(
    tmp_path: Path,
    liveness: OwnerLiveness,
) -> None:
    database, state_dir, marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    before_hash = _sha256(database)
    capability_calls: list[bool] = []
    server = build_server(
        database_path=database,
        state_dir=state_dir,
        transport=FakeQuackTransport(),
        capability_probe=lambda **_kwargs: capability_calls.append(True),
        owner_liveness_probe=lambda _birth: liveness,
    )
    with pytest.raises(QuackStateServerOwnershipError):
        server.start()
    assert capability_calls == []
    assert _sha256(database) == before_hash
    assert json.loads(server.owner_marker_path().read_text(encoding="utf-8")) == marker.to_dict()


@pytest.mark.parametrize("liveness", [OwnerLiveness.ALIVE, OwnerLiveness.UNKNOWN])
def test_lifecycle_inspector_never_opens_live_or_unknown_database(
    tmp_path: Path,
    liveness: OwnerLiveness,
) -> None:
    database, _state_dir, _marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    before_hash = _sha256(database)
    opens: list[Path] = []

    def forbidden_open(path: Path):
        opens.append(path)
        raise AssertionError("live database must not be opened")

    result = inspect_state_server_lifecycle(
        database_path=database,
        liveness=lambda _birth: liveness,
        connection_factory=forbidden_open,
    )
    assert result["available"] is False
    assert result["owner_liveness"] == liveness.value
    assert opens == []
    assert _sha256(database) == before_hash


def test_generic_live_status_defers_without_direct_database_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = importlib.util.spec_from_file_location("quack_status_ops", OPS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    database = tmp_path / "control.duckdb"
    database.write_bytes(b"not-opened")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    birth = current_process_birth()
    (state_dir / "quack-state-server.status.json").write_text(
        json.dumps(
            {
                "lifecycle": "ready",
                "identity": {
                    "status": "ready",
                    "process_birth": birth.to_dict(),
                },
            }
        ),
        encoding="utf-8",
    )
    import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server as runtime

    monkeypatch.setattr(
        runtime,
        "inspect_state_server_lifecycle",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("live status direct-opened DuckDB")
        ),
    )
    result = module.main(
        [
            "--database",
            str(database),
            "--state-dir",
            str(state_dir),
            "--json",
            "status",
        ]
    )
    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["authoritative_lifecycle"]["reason"] == (
        "deferred_to_authenticated_live_owner"
    )
    assert payload["authoritative_lifecycle"]["direct_database_file_open"] is False


@pytest.mark.parametrize("with_status_projection", [True, False])
def test_generic_status_marks_nonterminal_database_row_stale_without_live_owner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    with_status_projection: bool,
) -> None:
    database, state_dir, marker, database_uuid = _seed_stale_server(
        tmp_path,
        status="ready",
    )
    if with_status_projection:
        identity = {
            "server_id": marker.server_id,
            "store_id": "control.duckdb",
            "database_uuid": database_uuid,
            "process_birth_id": _birth_id(marker.process_birth),
            "process_birth": marker.process_birth.to_dict(),
            "listen_uri": "quack:127.0.0.1:29999",
            "extension_fingerprint": _DIGEST,
            "schema_revision": 1,
            "generation": 1,
            "started_at": "2026-08-30T00:00:00Z",
            "status": "ready",
            "revision": 7,
        }
        (state_dir / "quack-state-server.status.json").write_text(
            json.dumps(
                {"lifecycle": "ready", "identity": identity, "ready": True}
            ),
            encoding="utf-8",
        )
    spec = importlib.util.spec_from_file_location(
        f"quack_status_dead_owner_{with_status_projection}",
        OPS_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.main(
        [
            "--database",
            str(database),
            "--state-dir",
            str(state_dir),
            "--json",
            "status",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["lifecycle"] == "stale"
    assert payload["ready"] is False
    assert payload["lifecycle_consistent"] is False
    assert payload["reason_code"] == "authoritative_server_owner_not_live"
    assert payload["authoritative_lifecycle"]["latest"]["status"] == "ready"
    if with_status_projection:
        assert payload["owner_liveness"] == "dead"


def test_generic_status_rejects_database_json_lifecycle_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, state_dir, marker, database_uuid = _seed_stale_server(
        tmp_path,
        status="stopped",
    )
    status_path = state_dir / "quack-state-server.status.json"
    status_path.write_text(
        json.dumps(
            {
                "lifecycle": "ready",
                "identity": {
                    "server_id": marker.server_id,
                    "store_id": "control.duckdb",
                    "database_uuid": database_uuid,
                    "process_birth_id": _birth_id(marker.process_birth),
                    "process_birth": marker.process_birth.to_dict(),
                    "generation": 1,
                    "status": "ready",
                },
            }
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("quack_status_mismatch_ops", OPS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.main(
        [
            "--database",
            str(database),
            "--state-dir",
            str(state_dir),
            "--json",
            "status",
        ]
    )
    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["lifecycle_consistent"] is False
    assert payload["reason_code"] == "status_projection_lifecycle_mismatch"
    assert payload["authoritative_lifecycle"]["latest"]["status"] == "stopped"


@pytest.mark.parametrize("extension_matches", [True, False])
def test_generic_stopped_status_compares_full_lifecycle_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    extension_matches: bool,
) -> None:
    database, state_dir, marker, database_uuid = _seed_stale_server(
        tmp_path,
        status="stopped",
    )
    identity = {
        "server_id": marker.server_id,
        "store_id": "control.duckdb",
        "database_uuid": database_uuid,
        "process_birth_id": _birth_id(marker.process_birth),
        "process_birth": marker.process_birth.to_dict(),
        "listen_uri": "quack:127.0.0.1:29999",
        "extension_fingerprint": _DIGEST if extension_matches else "sha256:" + "ee" * 32,
        "schema_revision": 1,
        "generation": 1,
        "started_at": "2026-08-30T00:00:00Z",
        "status": "stopped",
        "revision": 7,
    }
    (state_dir / "quack-state-server.status.json").write_text(
        json.dumps({"lifecycle": "stopped", "identity": identity}),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        f"quack_status_full_identity_{extension_matches}",
        OPS_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.main(
        [
            "--database",
            str(database),
            "--state-dir",
            str(state_dir),
            "--json",
            "status",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == (0 if extension_matches else 1)
    assert payload["lifecycle_consistent"] is extension_matches


def test_lifecycle_inspection_does_not_touch_unrelated_supervisor_state(
    tmp_path: Path,
) -> None:
    database, _state_dir, _marker, _database_uuid = _seed_stale_server(
        tmp_path / "target",
        status="stopped",
    )
    unrelated = tmp_path / "other-supervisor" / "control.duckdb"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"unrelated-authority")
    before = _sha256(unrelated)
    result = inspect_state_server_lifecycle(
        database_path=database,
        liveness=lambda _birth: OwnerLiveness.DEAD,
    )
    assert result["available"] is True
    assert result["latest"]["server_id"] == "server:stale-prior"
    assert result["latest"]["listen_uri"] == "quack:127.0.0.1:29999"
    assert result["latest"]["extension_fingerprint"] == _DIGEST
    assert result["latest"]["schema_revision"] == 1
    assert result["latest"]["status"] == "stopped"
    assert result["latest"]["revision"] == 7
    assert result["latest"]["stopped_at"] == "2026-08-30T00:01:00Z"
    assert _sha256(unrelated) == before


def test_offline_projection_runs_on_same_connection_under_owner_lock(
    tmp_path: Path,
) -> None:
    database, _state_dir, _marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="stopped",
    )
    lock_path = database.with_name(f".{database.name}.state-owner.lock")

    def project(connection: Any) -> dict[str, Any]:
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        count = connection.execute("SELECT COUNT(*) FROM state_servers").fetchone()[0]
        return {"state_server_count": int(count)}

    result = inspect_state_server_lifecycle(
        database_path=database,
        liveness=lambda _birth: OwnerLiveness.DEAD,
        offline_inspect=project,
    )

    assert result["available"] is True
    assert result["offline_projection"] == {"state_server_count": 1}


def test_public_offline_fence_holds_owner_lock_for_entire_read_context(
    tmp_path: Path,
) -> None:
    database, _state_dir, _marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="stopped",
    )
    lock_path = database.with_name(f".{database.name}.state-owner.lock")
    with offline_state_server_fence(
        database_path=database,
        liveness=lambda _birth: OwnerLiveness.DEAD,
    ) as connection:
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        assert connection.execute("SELECT COUNT(*) FROM state_servers").fetchone()[0] == 1


@pytest.mark.parametrize(
    "unsafe_kind,expected_reason",
    [("symlink", "owner_marker_unsafe"), ("oversized", "owner_marker_too_large")],
)
def test_lifecycle_inspector_rejects_unsafe_marker_before_database_open(
    tmp_path: Path,
    unsafe_kind: str,
    expected_reason: str,
) -> None:
    database, _state_dir, _marker, _database_uuid = _seed_stale_server(
        tmp_path,
        status="stopped",
    )
    marker_path = database.with_name(f".{database.name}.state-owner.json")
    marker_path.unlink()
    if unsafe_kind == "symlink":
        target = tmp_path / "attacker-marker.json"
        target.write_text("{}\n", encoding="utf-8")
        marker_path.symlink_to(target)
    else:
        marker_path.write_bytes(b"{" + (b" " * (64 * 1024)) + b"}")

    def forbidden_open(_database: Path) -> Any:
        pytest.fail("unsafe owner marker must be rejected before database open")

    result = inspect_state_server_lifecycle(
        database_path=database,
        connection_factory=forbidden_open,
    )

    assert result["available"] is False
    assert result["owner_liveness"] == "unknown"
    assert result["reason"] == expected_reason
