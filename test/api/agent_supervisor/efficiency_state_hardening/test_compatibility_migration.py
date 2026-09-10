"""ASEH-061: compatibility adapters route through IntentRepository to the typed owner."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    OwnerRestartSnapshot,
    RecoveryFault,
    SupervisorRecovery,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    FakeQuackTransport,
    QuackStateServerCompatibilityError,
    build_server,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    CALLER_REPLACEMENTS,
    PRODUCTION_AUTHORITY_PATH,
    PRODUCTION_CUTOVER_TASK_ID,
    SUPPORTED_LEGACY_OPERATIONS,
    UNSUPPORTED_LEGACY_OPERATIONS,
    IntentEventType,
    IntentRepository,
    IntentRepositoryCompatibilityWarning,
    IntentRepositoryUnsupportedPathError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for ASEH-061 compatibility migration tests",
)

_DIGEST = "sha256:" + ("ab" * 32)
_MIGRATION_DOC = (
    Path(__file__).resolve().parents[4]
    / "docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_STATE_HARDENING_MIGRATION.md"
)


def _seed(path: Path, *, status: str = "ready") -> IntentRepository:
    repository = IntentRepository(
        path,
        owner_id="intent-repository:aseh-061",
        session_id="session:aseh-061",
        authentication_subject_id="supervisor:alpha",
        authentication_binding_id="grant-binding:durable",
        repository_id="repository:aseh-061",
        tree_id="tree:aseh-061",
    )
    repository.upsert_goal(
        goal_cid="goal:aseh-061",
        goal_alias="GOAL-ASEH-061",
        objective_id="objective:aseh-061",
        title="compatibility",
    )
    repository.upsert_task(
        task_cid="task:aseh-061",
        task_alias="TASK-ASEH-061",
        goal_cid="goal:aseh-061",
        status=status,
    )
    return repository


def _compatible_report() -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=QuackCapabilityStatus.COMPATIBLE,
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
        extension_fingerprint=_DIGEST,
        observed_functions=("quack_serve", "quack_query"),
        observed_surfaces=profile.required_surfaces,
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


def _birth() -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=os.getpid(),
        start_time_ticks=1,
        boot_id="boot:aseh-061",
        parent_pid=1,
    )


def test_supported_legacy_cas_matches_canonical_cas(tmp_path: Path) -> None:
    canonical = _seed(tmp_path / "canonical.duckdb")
    legacy = _seed(tmp_path / "legacy.duckdb")

    direct = canonical.cas_task_status(
        task_cid="task:aseh-061",
        expected_revision=1,
        new_status="in_progress",
    )
    with pytest.warns(IntentRepositoryCompatibilityWarning, match="typed Quack owner"):
        routed = legacy.route_legacy_api(
            "compare_and_set_status",
            "task:aseh-061",
            1,
            "in_progress",
            caller="DuckDBTaskSource.compare_and_set_status",
        )

    left = canonical.get_task("task:aseh-061")
    right = legacy.get_task("task:aseh-061")
    assert left is not None and right is not None
    assert left["status"] == right["status"] == "in_progress"
    assert left["revision"] == right["revision"] == 2
    assert direct.changed is routed.changed is True
    assert direct.event_type == routed.event_type == IntentEventType.TASK_STATUS_CHANGED.value
    assert direct.revision == routed.revision == 2


def test_legacy_transition_warns_and_routes_through_cas(tmp_path: Path) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    before = repository.event_watermark()

    with pytest.warns(IntentRepositoryCompatibilityWarning, match="compatibility adapter"):
        receipt = repository.apply_legacy_transition(
            task_cid="task:aseh-061",
            expected_revision=1,
            new_status="in_progress",
        )

    task = repository.get_task("task:aseh-061")
    assert task is not None
    assert task["status"] == "in_progress"
    assert task["revision"] == 2
    assert receipt.changed is True
    assert repository.event_watermark() == before + 1
    assert repository.legacy_route_count == 1


def test_single_write_legacy_cas_appends_one_event(tmp_path: Path) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    before = repository.event_watermark()

    with pytest.warns(IntentRepositoryCompatibilityWarning):
        repository.compare_and_set_status("TASK-ASEH-061", 1, "blocked")

    events = repository.list_events(after_global_sequence=before, limit=10)
    assert len(events) == 1
    assert events[0]["event_type"] == IntentEventType.TASK_STATUS_CHANGED.value
    task = repository.get_task("task:aseh-061")
    assert task is not None
    assert task["status"] == "blocked"
    assert task["revision"] == 2

    with pytest.raises(Exception, match="CAS is stale"):
        with pytest.warns(IntentRepositoryCompatibilityWarning):
            repository.route_legacy_api(
                "cas_status",
                "task:aseh-061",
                1,
                "in_progress",
            )
    assert repository.event_watermark() == before + 1


def test_bound_owner_connection_is_typed_quack_owner_path(tmp_path: Path) -> None:
    path = tmp_path / "control.duckdb"
    seed = _seed(path)
    seed.close()

    connection = open_duckdb_connection(path)
    try:
        bound = IntentRepository(
            path,
            bound_connection=connection,
            owner_id="quack-state-owner",
            session_id="quack-owner-1",
            install_schema=False,
        )
        try:
            assert bound.uses_typed_quack_owner is True
            assert bound.uses_bound_connection is True
            with pytest.warns(IntentRepositoryCompatibilityWarning):
                bound.route_legacy_api(
                    "transition",
                    task_cid="task:aseh-061",
                    expected_revision=1,
                    new_status="in_progress",
                    caller="TaskTransitionService.transition",
                )
            task = bound.get_task("task:aseh-061")
            assert task is not None
            assert task["status"] == "in_progress"
            catalog = bound.compatibility_catalog()
            assert catalog["independent_writer"] is False
            assert catalog["uses_typed_quack_owner"] is True
            assert catalog["plan_delta_cannot_waive_production_integration"] is True
        finally:
            bound.close()
    finally:
        connection.close()


def test_restart_rebuilds_identical_projection_and_recovers(tmp_path: Path) -> None:
    path = tmp_path / "intent.duckdb"
    repository = _seed(path)
    with pytest.warns(IntentRepositoryCompatibilityWarning):
        repository.route_legacy_api(
            "compare_and_set_status",
            "task:aseh-061",
            1,
            "in_progress",
        )
    repository.assert_projection_matches_events()
    exported = repository.export_owner_restart_snapshot()
    watermark = repository.event_watermark()
    status = repository.get_task("task:aseh-061")
    repository.close()

    restarted = IntentRepository(
        path,
        owner_id="intent-repository:aseh-061",
        session_id="session:aseh-061",
        install_schema=False,
        authentication_subject_id="supervisor:alpha",
        authentication_binding_id="grant-binding:durable",
        repository_id="repository:aseh-061",
        tree_id="tree:aseh-061",
    )
    rebuilt = restarted.rebuild_owner_restart_projection(exported)
    restarted.assert_projection_matches_events()
    task = restarted.get_task("task:aseh-061")
    assert task is not None
    assert task["status"] == status["status"] == "in_progress"
    assert task["revision"] == status["revision"] == 2
    assert restarted.event_watermark() == watermark
    assert rebuilt["task_state"]["task:aseh-061"]["status"] == "in_progress"


def test_supervisor_recovery_uses_intent_repository_authority(tmp_path: Path) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    with pytest.warns(IntentRepositoryCompatibilityWarning):
        repository.apply_legacy_transition(
            task_cid="task:aseh-061",
            expected_revision=1,
            new_status="in_progress",
        )
    payload = repository.export_owner_restart_snapshot()
    snapshot = OwnerRestartSnapshot.from_dict(payload)
    assert snapshot.cursor.stream_id
    EventCursor.from_dict(payload["cursor"])

    class _Authority:
        def __init__(self, repo: IntentRepository, current: OwnerRestartSnapshot) -> None:
            self.repo = repo
            self.snapshot = current
            self.takeovers = 0

        def rebuild(self, expected: OwnerRestartSnapshot) -> OwnerRestartSnapshot:
            rebuilt = self.repo.rebuild_owner_restart_projection(expected.to_dict())
            return OwnerRestartSnapshot.from_dict(rebuilt)

        def take_over(self, expected: OwnerRestartSnapshot, owner: str) -> OwnerRestartSnapshot:
            self.takeovers += 1
            taken = self.repo.take_over_owner_session(expected.to_dict(), owner)
            return OwnerRestartSnapshot.from_dict(taken)

        def authenticated(self, current: OwnerRestartSnapshot) -> bool:
            return self.repo.owner_restart_authenticated(current.to_dict())

    authority = _Authority(repository, snapshot)
    recovery = SupervisorRecovery(tmp_path / "recovery")
    recovery.checkpoint_owner_restart(
        snapshot,
        accepted_merged_tree_evidence=("receipt:accepted-merge",),
    )
    receipt = recovery.recover_owner_restart(
        incident_id="restart:aseh-061",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id=snapshot.repository_id,
        tree_id=snapshot.tree_id,
        owner_session_id=snapshot.owner_session_id,
        rebuild=authority.rebuild,
        verify_authenticated=authority.authenticated,
    )
    assert receipt.resulting_state_root == snapshot.state_root
    assert receipt.takeover is False
    assert authority.takeovers == 0
    task = repository.get_task("task:aseh-061")
    assert task is not None
    assert task["status"] == "in_progress"


def test_owner_takeover_advances_fence_without_rewriting_tasks(tmp_path: Path) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    with pytest.warns(IntentRepositoryCompatibilityWarning):
        repository.route_legacy_api(
            "compare_and_set_status",
            "task:aseh-061",
            1,
            "in_progress",
        )
    expected = repository.export_owner_restart_snapshot()
    taken = repository.take_over_owner_session(expected, "session:owner-two")
    assert taken["fencing_epoch"] == int(expected["fencing_epoch"]) + 1
    assert taken["owner_session_id"] == "session:owner-two"
    assert taken["task_state"] == expected["task_state"]
    task = repository.get_task("task:aseh-061")
    assert task is not None
    assert task["status"] == "in_progress"
    assert task["revision"] == 2


def test_caller_replacement_catalog_and_migration_doc() -> None:
    text = _MIGRATION_DOC.read_text(encoding="utf-8")
    assert PRODUCTION_CUTOVER_TASK_ID in text
    assert "IntentRepository" in text
    assert "typed Quack owner" in text
    assert "Rollback" in text
    assert "plan delta cannot waive" in text.lower() or "cannot waive this production integration" in text
    assert "same" in text and "credential handle" in text
    for caller, replacement in CALLER_REPLACEMENTS.items():
        assert caller
        assert replacement
    assert "compare_and_set_status" in SUPPORTED_LEGACY_OPERATIONS
    assert "transition" in SUPPORTED_LEGACY_OPERATIONS
    assert "plan_delta_waive_production_integration" in UNSUPPORTED_LEGACY_OPERATIONS
    assert PRODUCTION_AUTHORITY_PATH == (
        "IntentRepository@1",
        "TypedStateOwnerCommandGateway@1",
        "QuackStateServer@1",
    )


def test_rollback_plan_does_not_restore_independent_writer(tmp_path: Path) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    plan = repository.rollback_plan()
    assert plan["restores_independent_writer"] is False
    assert plan["public_api_deletion"] is False
    assert plan["preserve_observed_effects_and_receipts"] is True
    assert any("same external credential handle" in step for step in plan["procedure"])
    catalog = repository.compatibility_catalog()
    assert catalog["independent_writer"] is False
    assert catalog["plan_delta_cannot_waive_production_integration"] is True

    before = repository.get_task("task:aseh-061")
    with pytest.warns(IntentRepositoryCompatibilityWarning):
        with pytest.raises(IntentRepositoryUnsupportedPathError, match="failed closed"):
            repository.route_legacy_api("direct_sql", caller="raw DuckDB SQL")
    after = repository.get_task("task:aseh-061")
    assert after == before


@pytest.mark.parametrize(
    "operation",
    sorted(UNSUPPORTED_LEGACY_OPERATIONS),
)
def test_unsupported_paths_warn_then_fail_closed_without_writing(
    tmp_path: Path,
    operation: str,
) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    before = repository.event_watermark()
    with pytest.warns(IntentRepositoryCompatibilityWarning, match="compatibility adapter"):
        with pytest.raises(IntentRepositoryUnsupportedPathError, match="failed closed"):
            repository.route_legacy_api(operation, caller="legacy-test")
    assert repository.get_task("task:aseh-061")["status"] == "ready"
    assert repository.event_watermark() == before


def test_plan_delta_cannot_waive_production_integration(tmp_path: Path) -> None:
    repository = _seed(tmp_path / "intent.duckdb")
    with pytest.warns(IntentRepositoryCompatibilityWarning):
        with pytest.raises(IntentRepositoryUnsupportedPathError, match="plan delta"):
            repository.reject_unsupported_legacy_path(
                operation="plan_delta_waive_production_integration",
                caller="plan delta",
            )
    assert repository.caller_replacement("DuckDBTaskSource.compare_and_set_status") == (
        "IntentRepository.cas_task_status"
    )
    with pytest.raises(IntentRepositoryUnsupportedPathError, match="no admitted replacement"):
        repository.caller_replacement("invented-second-writer")


def test_quack_state_server_routes_legacy_api_through_bound_repository(
    tmp_path: Path,
) -> None:
    database = tmp_path / "control.duckdb"
    state = tmp_path / "state"
    state.mkdir()
    install_control_plane_schema(
        database,
        application_version="0.0.45",
        tool_version="1.5.2",
        owner_id="quack-state-owner",
    )
    seed = IntentRepository(database, owner_id="quack-state-owner", install_schema=False)
    try:
        seed.upsert_goal(
            goal_cid="goal:aseh-061",
            goal_alias="GOAL-ASEH-061",
            title="compatibility",
        )
        seed.upsert_task(
            task_cid="task:aseh-061",
            task_alias="TASK-ASEH-061",
            goal_cid="goal:aseh-061",
            status="ready",
        )
    finally:
        seed.close()

    server = build_server(
        database_path=database,
        state_dir=state,
        transport=FakeQuackTransport(),
        capability_probe=lambda **_k: _compatible_report(),
        process_birth_factory=_birth,
        owner_liveness_probe=lambda _b: OwnerLiveness.DEAD,
        connection_factory=lambda path: open_duckdb_connection(path),
    )
    server.start()
    try:
        assert server.production_authority_path() == PRODUCTION_AUTHORITY_PATH
        assert server.lifecycle.value == "ready"
        bound = server.bound_intent_repository()
        try:
            assert bound.uses_typed_quack_owner is True
        finally:
            bound.close()
        with pytest.warns(IntentRepositoryCompatibilityWarning, match="typed Quack owner"):
            server.route_legacy_api(
                "compare_and_set_status",
                "task:aseh-061",
                1,
                "in_progress",
                caller="DuckDBTaskSource.compare_and_set_status",
            )
        paused = server.pause_for_maintenance()
        assert paused["paused"] is True
        with pytest.raises(QuackStateServerCompatibilityError, match="paused"):
            server.route_legacy_api(
                "compare_and_set_status",
                "task:aseh-061",
                2,
                "blocked",
            )
        reconciled = server.reconcile_after_authenticated_restart()
        assert reconciled["projection_cid"]
        assert reconciled["generation"] >= 1
        resumed = server.resume_after_reconciliation(
            secret_handle=str(paused["secret_handle"])
        )
        assert resumed["resumed"] is True
        with pytest.warns(IntentRepositoryCompatibilityWarning):
            with pytest.raises(QuackStateServerCompatibilityError, match="failed closed"):
                server.route_legacy_api("dual_write", caller="compatibility dual-write")
        task = IntentRepository(
            database,
            bound_connection=server._connection,
            install_schema=False,
        )
        try:
            current = task.get_task("task:aseh-061")
            assert current is not None
            assert current["status"] == "in_progress"
            assert current["revision"] == 2
            task.assert_projection_matches_events()
        finally:
            task.close()
    finally:
        server.stop()
