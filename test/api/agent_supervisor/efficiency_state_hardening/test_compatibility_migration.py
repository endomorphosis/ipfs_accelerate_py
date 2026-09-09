"""ASEH-061: production cutover through IntentRepository to the typed Quack owner."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    OwnerRestartSnapshot,
    RecoveryFault,
    SupervisorRecovery,
)
from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
    PRODUCTION_CUTOVER_TASK_ID,
    QuackCompatibilityWarning,
    QuackStateServer,
    QuackStateServerConfig,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    CALLER_REPLACEMENTS,
    PRODUCTION_INTEGRATION_TASK_ID,
    SUPPORTED_LEGACY_OPERATIONS,
    TYPED_QUACK_OWNER_INTERFACE,
    UNSUPPORTED_LEGACY_OPERATIONS,
    IntentCompatibilityWarning,
    IntentIndependentWriterError,
    IntentRepository,
    IntentRepositoryConflictError,
    IntentUnsupportedPathError,
    production_authority_contract,
    reject_plan_delta_production_waiver,
)


ROOT = Path(__file__).resolve().parents[4]
MIGRATION_DOC = (
    ROOT / "docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_STATE_HARDENING_MIGRATION.md"
)


def _seed(repository: IntentRepository, *, suffix: str = "compat") -> None:
    repository.upsert_goal(
        goal_cid=f"goal:{suffix}",
        goal_alias=f"GOAL-{suffix.upper()}",
        objective_id=f"objective:{suffix}",
        title=suffix,
    )
    repository.upsert_task(
        task_cid=f"task:{suffix}",
        task_alias=f"TASK-{suffix.upper()}",
        goal_cid=f"goal:{suffix}",
        status="ready",
    )


def _repository(tmp_path: Path, *, suffix: str = "compat") -> IntentRepository:
    repository = IntentRepository(tmp_path / f"{suffix}.duckdb")
    _seed(repository, suffix=suffix)
    return repository


def _server(tmp_path: Path) -> QuackStateServer:
    return QuackStateServer(
        QuackStateServerConfig(
            database_path=tmp_path / "control.duckdb",
            state_dir=tmp_path / "quack-owner",
            store_id="aseh-061",
            secret_handle="handle:aseh-061",
        )
    )


def test_production_authority_contract_cannot_waive_integration() -> None:
    contract = production_authority_contract()
    assert contract["task_id"] == "ASEH-061"
    assert contract["writable_owner"] == TYPED_QUACK_OWNER_INTERFACE
    assert contract["adapter"] == "IntentRepository@1"
    assert contract["host"] == "QuackStateServer@1"
    assert contract["independent_writer"] is False
    assert contract["public_api_deletion_supported"] is False
    assert contract["plan_delta_may_waive_production_integration"] is False
    assert PRODUCTION_CUTOVER_TASK_ID == PRODUCTION_INTEGRATION_TASK_ID == "ASEH-061"
    assert "DuckDBTaskSource" in CALLER_REPLACEMENTS
    assert CALLER_REPLACEMENTS["DuckDBTaskSource"] == "DatabaseTaskSource@1"
    assert "transition_legacy" in UNSUPPORTED_LEGACY_OPERATIONS
    assert "compare_and_set_status" in SUPPORTED_LEGACY_OPERATIONS

    with pytest.warns(IntentCompatibilityWarning, match="plan_delta_waiver"):
        with pytest.raises(IntentUnsupportedPathError, match="cannot waive"):
            reject_plan_delta_production_waiver(
                {"waive_production_integration": True, "task_id": "ASEH-061"}
            )


def test_supported_legacy_cas_warns_and_matches_direct_authority(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    assert repository.writes_through_typed_owner is False
    assert repository.production_authority()["independent_writer"] is False

    before = repository.get_task("task:compat")
    assert before is not None
    events_before = repository.list_events()

    with pytest.warns(IntentCompatibilityWarning, match="compare_and_set_status"):
        routed = repository.route_supported_legacy_api(
            "compare_and_set_status",
            {
                "task_cid": "task:compat",
                "expected_revision": int(before["revision"]),
                "status": "in_progress",
            },
            caller="DuckDBTaskSource.compare_and_set_status",
        )

    after = repository.get_task("task:compat")
    assert after is not None
    assert after["status"] == "in_progress"
    assert int(after["revision"]) == int(before["revision"]) + 1
    assert routed.changed is True
    assert routed.revision == after["revision"]
    events_after = repository.list_events(after_global_sequence=0, limit=1000)
    assert len(events_after) == len(events_before) + 1
    assert events_after[-1]["event_id"] == routed.event_id


def test_admitted_transition_is_single_write_with_event_parity(tmp_path: Path) -> None:
    repository = _repository(tmp_path, suffix="transition")
    before = repository.get_task("task:transition")
    assert before is not None
    watermark = len(repository.list_events())

    receipt = repository.apply_admitted_transition(
        task_cid="task:transition",
        expected_revision=int(before["revision"]),
        new_status="in_progress",
    )
    after = repository.get_task("task:transition")
    assert after is not None
    assert after["status"] == "in_progress"
    assert receipt.revision == after["revision"]
    events = repository.list_events(after_global_sequence=0, limit=1000)
    assert len(events) == watermark + 1
    payload = events[-1]["body"]["body"]
    assert payload["task_cid"] == "task:transition"
    assert int(payload["revision"]) == after["revision"]

    with pytest.raises(IntentRepositoryConflictError, match="CAS is stale"):
        repository.apply_admitted_transition(
            task_cid="task:transition",
            expected_revision=int(before["revision"]),
            new_status="blocked",
        )
    unchanged = repository.get_task("task:transition")
    assert unchanged is not None
    assert unchanged["status"] == "in_progress"
    assert unchanged["revision"] == after["revision"]


def test_terminalization_requires_lease_fence_and_does_not_write(tmp_path: Path) -> None:
    repository = _repository(tmp_path, suffix="terminal")
    repository.record_validation_result(
        task_cid="task:terminal",
        outcome="passed",
        evidence_digest="sha256:" + ("ab" * 32),
        argv=["pytest"],
    )
    before = repository.get_task("task:terminal")
    assert before is not None
    events_before = len(repository.list_events())

    with pytest.raises(Exception, match="lease, fence, and claim revision"):
        repository.apply_admitted_transition(
            task_cid="task:terminal",
            expected_revision=int(before["revision"]),
            new_status="completed",
            evidence_digests=("sha256:" + ("ab" * 32),),
        )
    after = repository.get_task("task:terminal")
    assert after is not None
    assert after["status"] == "ready"
    assert after["revision"] == before["revision"]
    assert len(repository.list_events()) == events_before

    completed = repository.apply_admitted_transition(
        task_cid="task:terminal",
        expected_revision=int(before["revision"]),
        new_status="completed",
        evidence_digests=("sha256:" + ("ab" * 32),),
        lease_id="lease:terminal",
        fencing_token=1,
        claim_revision=1,
    )
    done = repository.get_task("task:terminal")
    assert done is not None
    assert done["status"] == "completed"
    assert completed.changed is True


def test_quack_host_routes_supported_legacy_api_through_intent_repository(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, suffix="host")
    server = _server(tmp_path)
    before = repository.get_task("task:host")
    assert before is not None
    events_before = len(repository.list_events())

    with pytest.warns(QuackCompatibilityWarning, match="legacy API"):
        receipt = server.route_supported_legacy_api(
            "cas_task_status",
            {
                "task_cid": "task:host",
                "expected_revision": int(before["revision"]),
                "new_status": "blocked",
            },
            caller="legacy-scheduler",
            repository=repository,
        )

    after = repository.get_task("task:host")
    assert after is not None
    assert after["status"] == "blocked"
    assert receipt.revision == after["revision"]
    assert len(repository.list_events()) == events_before + 1
    assert server.production_authority()["writable_owner"] == TYPED_QUACK_OWNER_INTERFACE
    assert server.production_authority()["independent_writer"] is False


def test_unsupported_paths_warn_then_fail_closed_without_writing(tmp_path: Path) -> None:
    repository = _repository(tmp_path, suffix="deny")
    server = _server(tmp_path)
    before = repository.get_task("task:deny")
    assert before is not None
    events_before = len(repository.list_events())

    for operation in (
        "direct_sql",
        "transition_legacy",
        "delete_public_api",
        "silent_fallback",
    ):
        with pytest.warns(IntentCompatibilityWarning):
            with pytest.raises(IntentUnsupportedPathError):
                repository.reject_unsupported_legacy_path(
                    caller="legacy-adapter", operation=operation
                )
        with pytest.warns(QuackCompatibilityWarning):
            with pytest.raises(IntentUnsupportedPathError):
                server.reject_unsupported_legacy_path(
                    caller="legacy-adapter",
                    operation=operation,
                    repository=repository,
                )
    for operation in ("independent_duckdb_write", "dual_write"):
        with pytest.warns(IntentCompatibilityWarning):
            with pytest.raises(IntentIndependentWriterError):
                repository.reject_unsupported_legacy_path(
                    caller="legacy-adapter", operation=operation
                )
        with pytest.warns(QuackCompatibilityWarning):
            with pytest.raises(IntentIndependentWriterError):
                server.reject_unsupported_legacy_path(
                    caller="legacy-adapter",
                    operation=operation,
                    repository=repository,
                )

    with pytest.warns(IntentCompatibilityWarning, match="legacy API"):
        with pytest.raises(IntentUnsupportedPathError, match="not an admitted"):
            repository.route_supported_legacy_api(
                "DELETE FROM tasks", {}, caller="sql-client"
            )

    after = repository.get_task("task:deny")
    assert after is not None
    assert after["status"] == before["status"]
    assert after["revision"] == before["revision"]
    assert len(repository.list_events()) == events_before


def test_restart_reconciliation_preserves_materialized_state_and_events(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path, suffix="restart")
    repository.apply_admitted_transition(
        task_cid="task:restart",
        expected_revision=1,
        new_status="in_progress",
    )
    expected = repository.get_task("task:restart")
    assert expected is not None
    expected_events = [dict(item) for item in repository.list_events()]

    report = repository.reconcile_after_authenticated_restart()
    assert report["task_id"] == "ASEH-061"
    assert report["authority"] == TYPED_QUACK_OWNER_INTERFACE
    assert report["independent_writer"] is False
    settled = repository.get_task("task:restart")
    assert settled is not None
    assert settled["status"] == expected["status"]
    assert settled["revision"] == expected["revision"]
    assert [dict(item) for item in repository.list_events()] == expected_events
    snapshot = report["snapshot"]
    assert snapshot["projection_cid"]
    assert snapshot["event_watermark"] >= 1

    server = _server(tmp_path)
    hosted = server.reconcile_typed_owner_authority(repository=repository)
    assert hosted["authority"] == TYPED_QUACK_OWNER_INTERFACE
    assert hosted["snapshot"]["projection_cid"] == snapshot["projection_cid"]


def test_owner_restart_recovery_still_uses_existing_recovery_authority(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import EventCursor

    snapshot = OwnerRestartSnapshot(
        repository_id="repository:aseh-061",
        tree_id="tree:aseh-061",
        generation=1,
        cursor=EventCursor(
            stream_id="events:aseh-061",
            position=4,
            last_event_id="event:4",
            snapshot_id="tree:aseh-061",
        ),
        task_state={"task:compat": {"revision": 2, "status": "in_progress"}},
        event_state={"head_event_id": "event:4", "event_count": 4},
        lease_state={
            "lease_id": "lease:owner:one",
            "owner_session_id": "owner:one",
            "fencing_epoch": 3,
            "claim_revision": 1,
        },
        idempotency_state={},
        reconciliation_state={},
        owner_session_id="owner:one",
        fencing_epoch=3,
        authenticated=True,
        authentication_subject_id="supervisor:alpha",
        authentication_binding_id="grant-binding:durable",
    )

    class _Authority:
        def __init__(self) -> None:
            self.snapshot = snapshot

        def rebuild(self, expected: OwnerRestartSnapshot) -> OwnerRestartSnapshot:
            return OwnerRestartSnapshot.from_dict(self.snapshot.to_dict())

        def authenticated(self, item: OwnerRestartSnapshot) -> bool:
            return item.authenticated and item.authentication_binding_id == (
                "grant-binding:durable"
            )

    authority = _Authority()
    recovery = SupervisorRecovery(tmp_path / "recovery")
    recovery.checkpoint_owner_restart(
        authority.snapshot,
        accepted_merged_tree_evidence=("receipt:accepted-merge",),
    )
    receipt = recovery.recover_owner_restart(
        incident_id="restart:aseh-061",
        fault=RecoveryFault.PROCESS_CRASH,
        repository_id="repository:aseh-061",
        tree_id="tree:aseh-061",
        owner_session_id="owner:one",
        rebuild=authority.rebuild,
        verify_authenticated=authority.authenticated,
    )
    assert receipt.resulting_state_root == authority.snapshot.state_root
    assert receipt.resulting_fencing_epoch == 3


def test_migration_document_records_replacement_rollback_and_no_deletion() -> None:
    text = MIGRATION_DOC.read_text(encoding="utf-8")
    required = (
        "ASEH-061",
        "TypedStateOwnerCommandGateway@1",
        "IntentRepository",
        "QuackStateServer",
        "DatabaseTaskSource",
        "DuckDBTaskSource",
        "IntentCompatibilityWarning",
        "rollback",
        "plan delta",
        "cannot waive",
        "public APIs are not removed",
        "reconcile_after_authenticated_restart",
        "owner-paused",
        "same external credential handle",
    )
    missing = [item for item in required if item.lower() not in text.lower()]
    assert missing == []
    assert "independent writer" in text.lower() or "write independently" in text.lower()


def test_caller_replacement_and_public_api_retention(tmp_path: Path) -> None:
    repository = _repository(tmp_path, suffix="replace")
    server = _server(tmp_path)
    assert CALLER_REPLACEMENTS["TaskTransitionService.transition"] == (
        "IntentRepository.apply_admitted_transition"
    )
    assert CALLER_REPLACEMENTS["TaskTransitionService.transition_legacy"].endswith(
        "reject_unsupported_legacy_path"
    )
    assert repository.production_authority()["public_api_deletion_supported"] is False
    assert server.production_authority()["public_api_deletion_supported"] is False
    with pytest.warns(QuackCompatibilityWarning):
        with pytest.raises(IntentUnsupportedPathError, match="cannot waive"):
            server.reject_plan_delta_production_waiver({"skip_cutover": True})

    with pytest.warns(IntentCompatibilityWarning):
        repository.route_supported_legacy_api(
            "DuckDBTaskSource.compare_and_set_status",
            {
                "task_cid": "task:replace",
                "expected_revision": 1,
                "status": "in_progress",
            },
            caller="DuckDBTaskSource",
        )
    task = repository.get_task("task:replace")
    assert task is not None
    assert task["status"] == "in_progress"
