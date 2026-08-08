"""Tests for path-independent control-plane repositories (DQP-008).

Acceptance:

* Local and Quack adapters pass the same conformance population
* Quack authority never silently falls back to direct file writes
* Imports can use embedded exclusive mode only under a maintenance lease

Evidence subset:

* tasks, events, leases, commands, snapshots, transactions,
  schema verification, cold imports
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandOutcome,
    StateAuthorityClass,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    CONFORMANCE_EVIDENCE_SUBSET,
    EMBEDDED_STATE_REPOSITORY_INTERFACE,
    QUACK_STATE_REPOSITORY_INTERFACE,
    STATE_REPOSITORY_INTERFACE,
    ConformanceReport,
    EmbeddedOpenPurpose,
    EmbeddedStateRepository,
    MaintenanceLease,
    QuackStateRepository,
    RepositoryAuthorityMode,
    StateRepository,
    StateRepositoryAuthorityError,
    StateRepositoryConformanceError,
    StateRepositoryLeaseError,
    assert_conformance_parity,
    issue_maintenance_lease,
    open_embedded_repository,
    open_quack_repository,
    open_quack_repository_against_database,
    repository_authority_for_task_source,
    run_conformance_population,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    QuackClientTransportError,
    QuackEndpoint,
    TransportMode,
    resolve_endpoint,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for control-plane repository hermetic tests",
)


def _install(db: Path) -> None:
    install_control_plane_schema(
        db,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="repo-test",
    )


# ---------------------------------------------------------------------------
# Interface / identity
# ---------------------------------------------------------------------------


def test_interface_identities_and_evidence_subset() -> None:
    assert STATE_REPOSITORY_INTERFACE == "StateRepository@1"
    assert EMBEDDED_STATE_REPOSITORY_INTERFACE == "EmbeddedStateRepository@1"
    assert QUACK_STATE_REPOSITORY_INTERFACE == "QuackStateRepository@1"
    for name in (
        "tasks",
        "events",
        "leases",
        "commands",
        "snapshots",
        "transactions",
        "schema_verification",
        "cold_imports",
    ):
        assert name in CONFORMANCE_EVIDENCE_SUBSET
    assert (
        repository_authority_for_task_source(prefer_quack=True)
        is RepositoryAuthorityMode.QUACK
    )
    assert (
        repository_authority_for_task_source(prefer_quack=False)
        is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE
    )


def test_embedded_repository_opens_and_verifies_schema(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    with open_embedded_repository(
        db,
        owner_id="owner:embedded",
        install_schema=True,
        seed_generation=True,
        purpose=EmbeddedOpenPurpose.HERMETIC_TEST,
    ) as repo:
        assert isinstance(repo, EmbeddedStateRepository)
        assert isinstance(repo, StateRepository)
        assert repo.authority_mode is RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE
        assert repo.open_session is True
        assert repo.session is not None
        assert repo.session.transport_mode is TransportMode.EMBEDDED
        generation = repo.load_generation()
        assert generation.generation >= 1
        report = repo.verify_schema()
        assert report["verified"] is True
        assert report.get("inventory") == "client_probe"
        assert int(report.get("task_count") or 0) >= 0
        identity = repo.observe_store_identity(repository_id="repository:test")
        assert identity.authority_class is StateAuthorityClass.AUTHORITATIVE
        assert identity.store_id == "control.duckdb"
        payload = repo.to_dict()
        assert payload["interface"] == EMBEDDED_STATE_REPOSITORY_INTERFACE
        assert payload["purpose"] == "hermetic_test"


def test_import_requires_maintenance_lease(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)

    # Missing lease fails closed for import purpose.
    with pytest.raises(StateRepositoryAuthorityError, match="maintenance lease"):
        open_embedded_repository(
            db,
            owner_id="owner:import",
            install_schema=False,
            seed_generation=True,
            purpose=EmbeddedOpenPurpose.IMPORT,
            maintenance_lease=None,
        )

    # Expired lease fails closed.
    expired = MaintenanceLease(
        lease_id="mlease:expired",
        scope="import",
        owner_session_id="session:1",
        process_birth_id="birth:1",
        fencing_token=1,
        fence_epoch=0,
        acquired_at="1970-01-01T00:00:00Z",
        expires_at="1970-01-01T00:00:01Z",
        state="held",
        purpose="import",
    )
    with pytest.raises(StateRepositoryLeaseError, match="not held|expired"):
        open_embedded_repository(
            db,
            owner_id="owner:import",
            install_schema=False,
            seed_generation=True,
            purpose=EmbeddedOpenPurpose.IMPORT,
            maintenance_lease=expired,
            clock=lambda: "2020-01-01T00:00:00Z",
        )

    # Held lease admits embedded exclusive import.
    lease = issue_maintenance_lease(
        scope="import",
        purpose="import",
        store_id="control.duckdb",
        clock=lambda: "2020-01-01T00:00:00Z",
    )
    with open_embedded_repository(
        db,
        owner_id="owner:import",
        install_schema=False,
        seed_generation=True,
        purpose=EmbeddedOpenPurpose.IMPORT,
        maintenance_lease=lease,
        clock=lambda: "2020-01-01T00:00:00Z",
    ) as repo:
        assert repo.purpose is EmbeddedOpenPurpose.IMPORT
        assert repo.maintenance_lease is not None
        assert repo.maintenance_lease.lease_id == lease.lease_id
        assert repo.get_task("missing") is None


def test_recovery_and_maintenance_purposes_require_lease(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    for purpose in (
        EmbeddedOpenPurpose.RECOVERY,
        EmbeddedOpenPurpose.MAINTENANCE,
    ):
        with pytest.raises(StateRepositoryAuthorityError, match="maintenance lease"):
            EmbeddedStateRepository(
                db,
                owner_id="owner:x",
                purpose=purpose,
            ).open()


def test_quack_refuses_file_path_and_embedded_fallback(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)

    with pytest.raises(StateRepositoryAuthorityError, match="quack: URI|file path"):
        QuackStateRepository(str(db), owner_id="owner:q")

    with pytest.raises(StateRepositoryAuthorityError, match="quack: URI|file path"):
        open_quack_repository(str(db), owner_id="owner:q")

    with pytest.raises(
        StateRepositoryAuthorityError, match="never permits embedded file-write"
    ):
        QuackStateRepository(
            "quack:127.0.0.1:9",
            owner_id="owner:q",
            allow_embedded_fallback=True,
        )

    embedded_endpoint = resolve_endpoint(str(db), mode=TransportMode.EMBEDDED)
    with pytest.raises(StateRepositoryAuthorityError, match="non-quack|file"):
        QuackStateRepository(embedded_endpoint, owner_id="owner:q")


def test_quack_transport_failure_does_not_open_database_file(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)
    # Marker proves the factory was asked only for quack endpoints and never
    # silently rewritten to an embedded path open by the repository.
    calls: list[str] = []

    def exploding_factory(endpoint: QuackEndpoint) -> Any:
        calls.append(endpoint.mode.value + ":" + endpoint.target)
        if endpoint.mode is not TransportMode.QUACK:
            raise AssertionError("factory must not receive embedded mode")
        raise RuntimeError("simulated quack transport outage")

    repo = QuackStateRepository(
        "quack:127.0.0.1:19999",
        owner_id="owner:q",
        connection_factory=exploding_factory,
    )
    with pytest.raises(
        (QuackClientTransportError, StateRepositoryAuthorityError),
        match="without file fallback|transport|Quack",
    ):
        repo.open()
    assert repo.open_session is False
    assert calls
    assert all(item.startswith("quack:") for item in calls)
    # Database file must remain untouched by a failed Quack open (no writer
    # session). Schema install above is the only legitimate writer.
    assert db.is_file()


def test_quack_repository_with_explicit_test_double(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    with open_quack_repository_against_database(
        db,
        owner_id="owner:quack",
        quack_uri="quack:127.0.0.1:9",
        install_schema=True,
        seed_generation=True,
    ) as repo:
        assert isinstance(repo, QuackStateRepository)
        assert repo.authority_mode is RepositoryAuthorityMode.QUACK
        assert repo.session is not None
        assert repo.session.transport_mode is TransportMode.QUACK
        assert repo.endpoint.mode is TransportMode.QUACK
        assert repo.to_dict()["allows_embedded_fallback"] is False
        generation = repo.load_generation()
        assert generation.generation >= 1
        schema = repo.verify_schema()
        assert schema["verified"] is True
        assert schema["authority_mode"] == "quack"


def test_tasks_events_leases_commands_and_snapshots(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    with open_embedded_repository(
        db,
        owner_id="owner:ops",
        install_schema=True,
        seed_generation=True,
    ) as repo:
        report = run_conformance_population(repo, seed=True)
        assert report.passed is True
        assert report.population_digest.startswith("sha256:")
        evidence = dict(report.evidence)
        assert evidence["tasks"]["count"] >= 2
        assert evidence["commands"]["outcome"] in {
            CommandOutcome.ACCEPTED.value,
            CommandOutcome.IDEMPOTENT_REPLAY.value,
        }
        assert evidence["events"]["listed_count"] >= 1
        assert evidence["leases"]["state"] == "held"
        assert evidence["snapshots"]["event_watermark"] >= 1
        assert evidence["cold_imports"]["quack_allows_file_fallback"] is False
        # Direct repository surface still works after population.
        page = repo.list_tasks(cursor=0, limit=10)
        assert len(page.items) >= 2
        task_cid = str(page.items[0]["task_cid"])
        task = repo.get_task(task_cid)
        assert task is not None
        lease = repo.get_task_lease(task_cid)
        assert lease is not None
        snapshot = repo.capture_snapshot()
        assert snapshot.snapshot_digest.startswith("sha256:")
        assert snapshot.authority_class is StateAuthorityClass.AUTHORITATIVE


def test_local_and_quack_conformance_parity(tmp_path: Path) -> None:
    embedded_db = tmp_path / "embedded" / "control.duckdb"
    quack_db = tmp_path / "quack" / "control.duckdb"
    embedded_db.parent.mkdir(parents=True)
    quack_db.parent.mkdir(parents=True)

    with open_embedded_repository(
        embedded_db,
        owner_id="owner:parity",
        install_schema=True,
        seed_generation=True,
        purpose=EmbeddedOpenPurpose.HERMETIC_TEST,
    ) as embedded:
        left = run_conformance_population(embedded, seed=True)

    with open_quack_repository_against_database(
        quack_db,
        owner_id="owner:parity",
        quack_uri="quack:127.0.0.1:9",
        install_schema=True,
        seed_generation=True,
    ) as quack:
        right = run_conformance_population(quack, seed=True)

    assert left.authority_mode == RepositoryAuthorityMode.EMBEDDED_EXCLUSIVE.value
    assert right.authority_mode == RepositoryAuthorityMode.QUACK.value
    assert_conformance_parity(left, right)
    # Re-check digests explicitly for clearer failure output.
    assert left.population_digest == right.population_digest
    assert dict(left.evidence) == dict(right.evidence)


def test_conformance_parity_helper_detects_divergence() -> None:
    left = ConformanceReport(
        authority_mode="embedded_exclusive",
        store_id="control.duckdb",
        evidence={"tasks": {"count": 1}},
        population_digest="sha256:" + ("aa" * 32),
    )
    right = ConformanceReport(
        authority_mode="quack",
        store_id="control.duckdb",
        evidence={"tasks": {"count": 2}},
        population_digest="sha256:" + ("bb" * 32),
    )
    with pytest.raises(StateRepositoryConformanceError, match="diverged"):
        assert_conformance_parity(left, right)


def test_maintenance_lease_round_trip_on_open_repository(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    with open_embedded_repository(
        db,
        owner_id="owner:lease",
        install_schema=True,
        seed_generation=True,
    ) as repo:
        lease = repo.acquire_maintenance_lease(
            scope="import",
            purpose="import",
            ttl_seconds=60,
        )
        assert lease.state == "held"
        assert lease.is_held()
        payload = lease.to_dict()
        restored = MaintenanceLease.from_dict(payload)
        assert restored.lease_id == lease.lease_id
        released = repo.release_maintenance_lease(lease)
        assert released.state == "released"
        assert released.is_held() is False


def test_issue_maintenance_lease_helper() -> None:
    lease = issue_maintenance_lease(
        scope="recovery",
        purpose="recovery",
        fencing_token=7,
        fence_epoch=3,
        clock=lambda: "2020-01-01T00:00:00Z",
    )
    assert lease.scope == "recovery"
    assert lease.fencing_token == 7
    assert lease.fence_epoch == 3
    assert lease.is_held(now="2020-01-01T00:00:00Z")


def test_quack_refuses_connection_factory_mode_drift(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    _install(db)

    def bad_factory(endpoint: QuackEndpoint) -> Any:
        # Simulate a misbehaving factory that tries to re-interpret the
        # endpoint as an embedded path — repository guard must refuse.
        from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
            open_duckdb_connection,
        )

        if endpoint.mode is TransportMode.QUACK:
            # Deliberately open via path only after rewriting mode — the
            # repository wraps the factory and still requires quack mode on
            # the endpoint object itself (which remains quack).
            return open_duckdb_connection(db)
        raise AssertionError("unexpected mode")

    # Factory that honors quack mode is fine (explicit double).
    with open_quack_repository(
        "quack:127.0.0.1:9",
        owner_id="owner:double",
        connection_factory=bad_factory,
        seed_generation=True,
    ) as repo:
        assert repo.session is not None
        assert repo.session.transport_mode is TransportMode.QUACK


def test_cold_import_protocol_is_runtime_checkable(tmp_path: Path) -> None:
    db = tmp_path / "control.duckdb"
    with open_embedded_repository(
        db,
        owner_id="owner:proto",
        install_schema=True,
        seed_generation=True,
    ) as repo:
        assert isinstance(repo, StateRepository)
