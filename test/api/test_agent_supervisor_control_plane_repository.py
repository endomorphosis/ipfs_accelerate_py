"""Tests for path-independent control-plane state repositories (DQP-008).

Acceptance:

* Local (embedded) and Quack adapters pass the same conformance population
* Quack authority never silently falls back to direct file writes
* Imports can use embedded exclusive mode only under a maintenance lease

Evidence subset: tasks, events, leases, commands, snapshots, transactions,
schema verification, cold imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    ControlPlaneStoreIdentity,
    StateAuthorityClass,
    StateCommand,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    DEFAULT_MAINTENANCE_SCOPE,
    EMBEDDED_STATE_REPOSITORY_INTERFACE,
    QUACK_STATE_REPOSITORY_INTERFACE,
    STATE_REPOSITORY_INTERFACE,
    EmbeddedStateRepository,
    MaintenanceLease,
    QuackStateRepository,
    RepositoryAuthorityMode,
    StateRepositoryAuthorityError,
    StateRepositoryMaintenanceError,
    StateRepositoryNotOpenError,
    acquire_maintenance_lease,
    exclusive_embedded_repository,
    open_embedded_repository,
    open_quack_repository,
    open_state_repository,
    populations_equivalent,
    release_maintenance_lease,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackEndpoint,
    TransportMode,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane repository hermetic tests",
)

_DIGEST = "sha256:" + ("ab" * 32)
_UUID = "123e4567-e89b-12d3-a456-426614174000"


def _install(db: Path) -> None:
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="repository-test",
    )


def _seed_generation(
    db: Path,
    *,
    generation: int = 1,
    fence_epoch: int = 1,
    revision: int = 0,
    database_uuid: str = _UUID,
    birth_id: str = "birth:server-1",
) -> None:
    with open_duckdb_connection(db) as connection:
        connection.execute("DELETE FROM store_generations")
        connection.execute(
            """
            INSERT INTO store_generations (
                generation, schema_revision, fence_epoch, revision,
                database_uuid, birth_id, created_at
            ) VALUES (?, 1, ?, ?, ?, ?, ?)
            """,
            [
                generation,
                fence_epoch,
                revision,
                database_uuid,
                birth_id,
                "1970-01-01T00:00:00Z",
            ],
        )


def _seed_population(db: Path, *, task_count: int = 3) -> list[str]:
    """Seed goals, tasks, one lease, one domain event for conformance."""

    task_cids: list[str] = []
    with open_duckdb_connection(db) as connection:
        connection.execute(
            """
            INSERT INTO goals (
                goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                title, status, created_at, updated_at, revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "goal:root",
                "G-ROOT",
                "objective:test",
                "",
                1,
                "Root",
                "open",
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                0,
                "{}",
            ],
        )
        for index in range(task_count):
            task_cid = f"task:cid:{index + 1:03d}"
            task_cids.append(task_cid)
            connection.execute(
                """
                INSERT INTO tasks (
                    task_cid, task_alias, goal_cid, plan_cid, objective_id,
                    ordinal, status, revision, priority, created_at, updated_at,
                    identity_json, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    task_cid,
                    f"T-{index + 1:03d}",
                    "goal:root",
                    "",
                    "objective:test",
                    index + 1,
                    "ready",
                    0,
                    "P0",
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    "{}",
                    "{}",
                ],
            )
        # One lease on the first task.
        connection.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt, state,
                started_at_ms, release_reason, retry_not_before_ms,
                owner_session_id, fence_epoch, revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                task_cids[0],
                "claim:001",
                "resolution:001",
                "did:claimant:1",
                1,
                1,
                9_999_999_999,
                1,
                "held",
                0,
                None,
                0,
                "session:lease-owner",
                1,
                0,
            ],
        )
        connection.execute(
            """
            INSERT INTO domain_events (
                event_id, stream_id, sequence, global_sequence, event_type,
                task_cid, attempt_id, session_id, recorded_at, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "event:001",
                "stream:tasks",
                1,
                1,
                "task.seeded",
                task_cids[0],
                "",
                "session:seed",
                "1970-01-01T00:00:00Z",
                "{}",
            ],
        )
    return task_cids


def _quack_factory(db: Path):
    """Hermetic factory: Quack transport mode, embedded connection under the hood.

    Proves the repository stays on the Quack authority path (no file endpoint)
    while remaining runnable without a live Quack extension in validation.
    """

    def factory(endpoint: QuackEndpoint) -> Any:
        if endpoint.mode is not TransportMode.QUACK:
            raise AssertionError(
                f"quack hermetic factory requires TransportMode.QUACK, got {endpoint.mode!r}"
            )
        # Refuse smuggled filesystem authority even under the hermetic factory.
        if endpoint.database_path is not None:
            raise StateRepositoryAuthorityError(
                "hermetic Quack factory refuses endpoints that carry database_path"
            )
        return open_duckdb_connection(db, timeout_seconds=60.0)

    return factory


def test_interface_identities() -> None:
    assert STATE_REPOSITORY_INTERFACE == "StateRepository@1"
    assert EMBEDDED_STATE_REPOSITORY_INTERFACE == "EmbeddedStateRepository@1"
    assert QUACK_STATE_REPOSITORY_INTERFACE == "QuackStateRepository@1"
    assert EmbeddedStateRepository.INTERFACE == EMBEDDED_STATE_REPOSITORY_INTERFACE
    assert QuackStateRepository.INTERFACE == QUACK_STATE_REPOSITORY_INTERFACE
    assert set(item.value for item in RepositoryAuthorityMode) == {
        "quack",
        "embedded",
        "embedded_exclusive",
    }


def test_embedded_repository_reads_tasks_leases_events(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_population(db, task_count=3)

    with open_embedded_repository(
        db,
        owner_id="owner:embedded",
        seed_generation=False,
    ) as repo:
        assert repo.INTERFACE == EMBEDDED_STATE_REPOSITORY_INTERFACE
        assert isinstance(repo, EmbeddedStateRepository)
        assert repo.authority_mode is RepositoryAuthorityMode.EMBEDDED
        assert repo.is_open
        assert repo.count_tasks() == 3
        task = repo.get_task(task_cids[0])
        assert task is not None
        assert task["status"] == "ready"
        lease = repo.get_lease(task_cids[0])
        assert lease is not None
        assert lease["claim_cid"] == "claim:001"
        assert len(repo.list_leases()) == 1
        events = repo.list_events(cursor=0, limit=10)
        assert len(events.items) == 1
        assert events.items[0]["event_id"] == "event:001"
        assert repo.event_watermark() == 1
        generation = repo.load_generation()
        assert generation.database_uuid == _UUID
        identity = repo.store_identity()
        assert identity.store_id == "control.duckdb"
        schema = repo.verify_schema()
        assert schema["schema_fingerprint"].startswith("sha256:")
        assert "tasks" in schema["tables_ok"] or "task_columns_ok" in schema
        snap = repo.snapshot()
        assert snap.database_uuid == _UUID
        assert snap.event_watermark == 1
        assert snap.authority_class is StateAuthorityClass.AUTHORITATIVE
        with pytest.raises(StateRepositoryAuthorityError):
            repo.execute_sql("SELECT 1")


def test_embedded_cas_and_commands_surface(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_population(db, task_count=1)

    with open_embedded_repository(
        db, owner_id="owner:cas", seed_generation=False
    ) as repo:
        result = repo.cas_task_status(
            task_cid=task_cids[0],
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:repo-cas",
            command_id="cmd:repo-cas",
        )
        assert result.outcome is CommandOutcome.ACCEPTED
        assert result.changed is True
        task = repo.get_task(task_cids[0])
        assert task is not None
        assert task["status"] == "claimed"
        commands = repo.list_commands()
        assert any(row["idempotency_key"] == "idem:repo-cas" for row in commands)
        # Transaction handle is available for higher layers.
        txn = repo.transaction()
        assert txn is not None
        txn.rollback()


def test_quack_repository_refuses_filesystem_paths(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)

    with pytest.raises(StateRepositoryAuthorityError):
        QuackStateRepository(db, owner_id="owner:bad").attach()
    with pytest.raises(StateRepositoryAuthorityError):
        QuackStateRepository(str(db), owner_id="owner:bad").attach()

    with pytest.raises(StateRepositoryAuthorityError):
        QuackStateRepository(
            "quack:127.0.0.1:9",
            owner_id="owner:x",
            allow_embedded_fallback=True,
        )

    # Endpoint that smuggles a database_path is rejected without a factory.
    with pytest.raises(StateRepositoryAuthorityError):
        QuackStateRepository(
            QuackEndpoint(
                mode=TransportMode.QUACK,
                target="quack:127.0.0.1:9",
                quack_uri="quack:127.0.0.1:9",
                database_path=db,
            ),
            owner_id="owner:x",
        ).attach()


def test_quack_repository_no_silent_embedded_fallback(tmp_path: Path) -> None:
    """Unavailable Quack must fail closed — never open the duckdb file path.

    Use a fail-closed connection factory rather than a live Quack ATTACH.
    Real ATTACH can hang when the extension loads but the peer is absent;
    the authority contract is that transport failure never degrades to a
    direct filesystem open of ``control.duckdb``.
    """

    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)

    def _unavailable(_endpoint: QuackEndpoint) -> Any:
        raise RuntimeError("quack transport unavailable (forced)")

    # 1) Forced transport failure with an explicit factory: no embedded open.
    repo = QuackStateRepository(
        "quack:127.0.0.1:59999",
        owner_id="owner:no-fallback",
        connection_factory=_unavailable,
    )
    with pytest.raises(Exception):
        repo.attach()
    assert not repo.is_open
    assert repo.endpoint is None or repo.endpoint.mode is TransportMode.QUACK
    assert repo.endpoint is None or repo.endpoint.database_path is None

    # 2) Constructor-level rejection of the only silent-fallback flag.
    with pytest.raises(StateRepositoryAuthorityError):
        QuackStateRepository(
            "quack:127.0.0.1:59999",
            owner_id="owner:no-fallback",
            allow_embedded_fallback=True,
        )

    # 3) Filesystem targets never become Quack authority (no attach I/O).
    with pytest.raises(StateRepositoryAuthorityError):
        QuackStateRepository(db, owner_id="owner:no-fallback")


def test_quack_and_embedded_share_conformance_population(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_population(db, task_count=2)

    embedded = open_embedded_repository(
        db,
        owner_id="owner:parity-embedded",
        seed_generation=False,
        process_birth_id="birth:parity",
    )
    try:
        # Mutate once so commands appear in the population.
        cas = embedded.cas_task_status(
            task_cid=task_cids[0],
            expected_task_revision=0,
            new_status="claimed",
            idempotency_key="idem:parity",
            command_id="cmd:parity",
        )
        assert cas.outcome is CommandOutcome.ACCEPTED
        left = embedded.canonical_population()
        assert left.task_count == 2
        assert len(left.leases) == 1
        assert len(left.events) == 1
        assert any(cmd["idempotency_key"] == "idem:parity" for cmd in left.commands)
    finally:
        embedded.close()

    # Sequential open: exclusive lock released before Quack hermetic factory.
    quack = open_quack_repository(
        "quack:127.0.0.1:18080",
        owner_id="owner:parity-quack",
        connection_factory=_quack_factory(db),
        seed_generation=False,
        process_birth_id="birth:parity-quack",
    )
    try:
        assert quack.authority_mode is RepositoryAuthorityMode.QUACK
        assert quack.client.session is not None
        assert quack.client.session.transport_mode is TransportMode.QUACK
        right = quack.canonical_population()
        schema = quack.verify_schema()
        assert schema["transport_mode"] == "quack"
        assert schema["schema_fingerprint"].startswith("sha256:")
        assert populations_equivalent(left, right)
        assert left.content_id == right.content_id
        # Tasks match by cid/status after the shared CAS.
        by_cid = {row["task_cid"]: row for row in right.tasks}
        assert by_cid[task_cids[0]]["status"] == "claimed"
    finally:
        quack.close()


def test_open_state_repository_factory_dispatches_modes(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    _seed_population(db, task_count=1)

    emb = open_state_repository(
        authority_mode="embedded",
        target=db,
        owner_id="owner:factory",
        seed_generation=False,
    )
    try:
        assert isinstance(emb, EmbeddedStateRepository)
        assert emb.count_tasks() == 1
    finally:
        emb.close()

    quack = open_state_repository(
        authority_mode=RepositoryAuthorityMode.QUACK,
        target="quack:127.0.0.1:18081",
        owner_id="owner:factory-q",
        connection_factory=_quack_factory(db),
        seed_generation=False,
    )
    try:
        assert isinstance(quack, QuackStateRepository)
        assert quack.count_tasks() == 1
    finally:
        quack.close()


def test_embedded_exclusive_requires_maintenance_lease(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    _seed_population(db, task_count=1)

    with pytest.raises(StateRepositoryMaintenanceError):
        open_embedded_repository(
            db,
            owner_id="owner:exclusive",
            exclusive=True,
            seed_generation=False,
        )

    with pytest.raises(StateRepositoryMaintenanceError):
        EmbeddedStateRepository(
            db,
            exclusive=True,
            maintenance_lease=MaintenanceLease(
                lease_id="mlease:stale",
                scope=DEFAULT_MAINTENANCE_SCOPE,
                owner_session_id="session:x",
                process_birth_id="birth:x",
                fencing_token=1,
                fence_epoch=1,
                acquired_at="1970-01-01T00:00:00Z",
                expires_at="1970-01-01T00:00:00Z",
                state="released",
            ),
            seed_generation=False,
        ).attach()


def test_embedded_exclusive_with_live_lease_allows_cold_import(
    tmp_path: Path,
) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)

    lease = acquire_maintenance_lease(
        db,
        owner_session_id="session:import",
        process_birth_id="birth:import",
        scope=DEFAULT_MAINTENANCE_SCOPE,
        fencing_token=7,
        fence_epoch=1,
    )
    assert lease.active

    repo = open_embedded_repository(
        db,
        owner_id="owner:import",
        exclusive=True,
        maintenance_lease=lease,
        seed_generation=False,
    )
    try:
        assert repo.authority_mode is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE
        assert repo.maintenance_lease is not None
        assert repo.maintenance_lease.lease_id == lease.lease_id
        # Cold import surface: insert a goal/task through closed templates.
        repo.execute(
            "insert_goal",
            {
                "goal_cid": "goal:import",
                "goal_alias": "G-IMPORT",
                "objective_id": "objective:import",
                "parent_goal_cid": "",
                "ordinal": 1,
                "title": "Imported",
                "status": "open",
                "created_at": "1970-01-01T00:00:00Z",
                "updated_at": "1970-01-01T00:00:00Z",
                "revision": 0,
                "body_json": "{}",
            },
        )
        repo.execute(
            "insert_task",
            {
                "task_cid": "task:imported:001",
                "task_alias": "T-IMP-001",
                "goal_cid": "goal:import",
                "plan_cid": "",
                "objective_id": "objective:import",
                "ordinal": 1,
                "status": "ready",
                "revision": 0,
                "priority": "P0",
                "created_at": "1970-01-01T00:00:00Z",
                "updated_at": "1970-01-01T00:00:00Z",
                "identity_json": "{}",
                "body_json": "{}",
            },
        )
        # Commit session mutations via a trivial accepted path.
        assert repo.count_tasks() == 1
        assert repo.get_task("task:imported:001") is not None
        population = repo.canonical_population()
        assert population.task_count == 1
    finally:
        repo.close()
        released = release_maintenance_lease(db, lease)
        assert not released.active

    # Second exclusive open without lease still fails.
    with pytest.raises(StateRepositoryMaintenanceError):
        open_embedded_repository(db, exclusive=True, seed_generation=False)


def test_exclusive_embedded_repository_context_manager(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    with exclusive_embedded_repository(
        db,
        owner_id="owner:ctx",
        install_schema=True,
        seed_generation=True,
    ) as repo:
        assert repo.authority_mode is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE
        assert repo.is_open
        schema = repo.verify_schema()
        assert schema["schema_fingerprint"].startswith("sha256:")
        # Seed a task for the cold-import path inside the leased session.
        repo.execute(
            "insert_goal",
            {
                "goal_cid": "goal:ctx",
                "goal_alias": "G-CTX",
                "objective_id": "objective:ctx",
                "parent_goal_cid": "",
                "ordinal": 1,
                "title": "Ctx",
                "status": "open",
                "created_at": "1970-01-01T00:00:00Z",
                "updated_at": "1970-01-01T00:00:00Z",
                "revision": 0,
                "body_json": "{}",
            },
        )
        repo.execute(
            "insert_task",
            {
                "task_cid": "task:ctx:001",
                "task_alias": "T-CTX",
                "goal_cid": "goal:ctx",
                "plan_cid": "",
                "objective_id": "objective:ctx",
                "ordinal": 1,
                "status": "ready",
                "revision": 0,
                "priority": "P1",
                "created_at": "1970-01-01T00:00:00Z",
                "updated_at": "1970-01-01T00:00:00Z",
                "identity_json": "{}",
                "body_json": "{}",
            },
        )
        assert repo.count_tasks() == 1

    # Lease released; ordinary embedded can read the imported rows.
    with open_embedded_repository(
        db, owner_id="owner:after", seed_generation=False
    ) as repo:
        assert repo.count_tasks() == 1


def test_closed_repository_rejects_ops(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    repo = open_embedded_repository(db, seed_generation=False)
    repo.close()
    with pytest.raises(StateRepositoryNotOpenError):
        repo.count_tasks()
    with pytest.raises(StateRepositoryNotOpenError):
        repo.attach()


def test_submit_command_through_repository(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db)
    task_cids = _seed_population(db, task_count=1)

    with open_embedded_repository(
        db, owner_id="owner:cmd", seed_generation=False
    ) as repo:
        live = repo.load_generation()
        session = repo.client.session
        assert session is not None
        command = StateCommand(
            command_id="cmd:repo-submit",
            command_kind=CommandKind.CLAIM,
            store_id="control.duckdb",
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key="idem:repo-submit",
            parameters={
                "task_cid": task_cids[0],
                "expected_task_revision": 0,
                "status": "running",
            },
        )
        result = repo.submit_command(command)
        assert result.outcome is CommandOutcome.ACCEPTED
        task = repo.get_task(task_cids[0])
        assert task is not None
        assert task["status"] == "running"


def test_install_schema_on_open(tmp_path: Path) -> None:
    db = tmp_path / "fresh.duckdb"
    with open_embedded_repository(
        db,
        owner_id="owner:install",
        install_schema=True,
        seed_generation=True,
    ) as repo:
        report = repo.verify_schema()
        assert report["schema_fingerprint"].startswith("sha256:")
        assert repo.load_generation().generation >= 1


def test_expected_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    _seed_generation(db, database_uuid=_UUID)
    expected = ControlPlaneStoreIdentity(
        repository_id="repository:sha256:test",
        database_uuid="00000000-0000-4000-8000-000000000099",
        store_id="control.duckdb",
        schema_revision=1,
        generation=1,
        schema_fingerprint=_DIGEST,
        authority_class=StateAuthorityClass.AUTHORITATIVE,
    )
    with pytest.raises(Exception):
        open_embedded_repository(
            db,
            owner_id="owner:id",
            expected_identity=expected,
            seed_generation=False,
        )
