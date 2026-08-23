"""Tests for DatabaseImplementationDaemon@1 cutover (DQP-018).

Evidence subset: ready selection, strict shards, lost response, provider
capacity, hard quota, timeout, cancellation, crash, restart, stale worker,
status parity.

Acceptance: Four daemon processes claim distinct work; no task status is
updated in Markdown under database authority; JSON queue/status/events/PID
projections can be absent; crash/restart resumes from committed phase and does
not duplicate provider/effect work.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationError,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DATABASE_PROGRAM_JSON_ENV,
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    DuckDBConnectionPolicyError,
    connect_duckdb_with_policy,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon_runner as daemon_runner,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalValidationRetry,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_COMPLETE,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_PROVIDER,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_database_implementation_daemon_from_args,
    build_portal_implementation_daemon_from_args,
    resolve_database_implementation_paths,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for database implementation daemon tests",
)


def test_provider_cold_execution_schema_installer_matches_daemon_contract(
    tmp_path: Path,
) -> None:
    """The bootstrap DDL stays provider-cold and is the daemon's exact DDL."""

    database_path = tmp_path / "execution.duckdb"
    program = """
import json
import sys
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema import (
    install_database_execution_schema,
)

receipt = install_database_execution_schema(
    Path(sys.argv[1]),
    metadata={
        "authority_mode": "embedded",
        "logical_owner_session_id": "session:test:logical-owner",
        "process_instance_id": "process:test:bootstrap",
        "state_schema_revision": "datasets-authoritative-operational-v1",
        "control_schema_profile_id": "profile:test",
        "control_schema_fingerprint": "sha256:" + "a" * 64,
    },
)
forbidden = sorted(
    name
    for name in sys.modules
    if name == "urllib.request"
    or "llm_router" in name
    or ".providers." in name
    or name.split(".", 1)[0] in {"anthropic", "openai"}
)
print(json.dumps({"forbidden": forbidden, "receipt": receipt}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(database_path)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    observed = json.loads(completed.stdout)
    assert observed["forbidden"] == []
    assert observed["receipt"]["tables"] == [
        "daemon_execution_metadata",
        "database_task_attempts",
        "attempt_phases",
        "provider_invocations",
        "effect_claims",
        "daemon_execution_events",
    ]

    schema_module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.database_execution_schema"
    )
    daemon_module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert daemon_module._DAEMON_EXECUTION_SQL == schema_module.DAEMON_EXECUTION_SQL

    duckdb = pytest.importorskip("duckdb")
    connection = connect_duckdb_with_policy(
        duckdb,
        database_path,
        read_only=True,
    )
    try:
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM daemon_execution_metadata ORDER BY key"
            ).fetchall()
        )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
    finally:
        connection.close()
    assert metadata == observed["receipt"]["metadata"]
    assert tables == set(observed["receipt"]["tables"])


def _population(task_count: int = 4) -> dict[str, object]:
    tasks = []
    for index in range(1, task_count + 1):
        tasks.append(
            {
                "task_cid": f"task:cid:{index:03d}",
                "task_id": f"DQP-T{index:03d}",
                "goal_cid": "goal:cid:root",
                "status": "ready",
                "priority": "P0",
                "ordinal": index,
                "title": f"Task {index}",
            }
        )
    return {
        "repository_tree_id": "tree:dqp-018",
        "objectives": [
            {
                "objective_id": "objective:dqp-018",
                "objective_alias": "DQP-O018",
                "title": "Daemon cutover",
                "goal_cid": "goal:cid:root",
                "goal_alias": "DQP-G030",
                "status": "open",
            }
        ],
        "tasks": tasks,
    }


def _open_daemon(
    tmp_path: Path,
    *,
    session: str = "",
    provider_calls: list[str] | None = None,
    effect_calls: list[str] | None = None,
    markdown_path: Path | None = None,
    provider_fn: Callable[[DatabaseTaskAttempt], dict[str, object]] | None = None,
    lease_ms: int = 60_000,
    max_task_attempts: int = 0,
    clock_ms: Callable[[], int] | None = None,
) -> DatabaseImplementationDaemon:
    database_path = tmp_path / "control.duckdb"
    coordination_path = tmp_path / "coordination.duckdb"
    execution_path = tmp_path / "execution.duckdb"

    def default_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if provider_calls is not None:
            provider_calls.append(attempt.task_cid)
        return {
            "status": "succeeded",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    def effect(
        attempt: DatabaseTaskAttempt, provider_result: dict[str, object]
    ) -> dict[str, object]:
        if effect_calls is not None:
            effect_calls.append(attempt.task_cid)
        return {
            "status": "applied",
            "task_cid": attempt.task_cid,
            "provider_result": dict(provider_result),
        }

    def validation(
        attempt: DatabaseTaskAttempt, effect_result: dict[str, object]
    ) -> dict[str, object]:
        return {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "a" * 64,
            "argv": ["focused-database-validation", attempt.task_cid],
            "effect_result": dict(effect_result),
        }

    return DatabaseImplementationDaemon(
        database_path=database_path,
        coordination_path=coordination_path,
        execution_path=execution_path,
        owner_session_id=session,
        authority_mode="embedded",
        task_source_kind="duckdb",
        markdown_path=markdown_path,
        # Projections intentionally absent.
        state_path=None,
        strategy_path=None,
        events_path=None,
        pid_path=None,
        queue_path=None,
        lease_ms=lease_ms,
        max_task_attempts=max_task_attempts,
        provider_fn=provider_fn or default_provider,
        effect_fn=effect,
        validation_fn=validation,
        require_real_execution=True,
        clock_ms=clock_ms,
    )


def _validation_retry_receipt(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
        "disposition": "retry",
        "reason": "declared_validation_failed",
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "lease_id": attempt.lease_id,
        "attempt_number": attempt.attempt_number,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "portal_attempt": 1,
        "typed_retry_generation": 1,
        "retry_budget_basis": "portal_attempt",
        "legacy_database_attempts_excluded": True,
        "max_task_attempts": daemon.max_task_attempts,
        "remaining_task_attempts": (
            daemon.max_task_attempts - 1
        ),
        "attempt_consumed": True,
        "provider_dispatched": True,
        "backoff_seconds": 0,
        "implementation_commit": "a" * 40,
        "rescue_branch": "rescue/dqp-t001-attempt-1-failed-validation",
        "binding_id": "sha256:" + "2" * 64,
        "events_digest": "sha256:" + "3" * 64,
        "event_stream_id": "event-log:validation-retry",
        "expected_output_event_id": "sha256:" + "1" * 64,
        "proposal_event_id": "sha256:" + "4" * 64,
        "preservation_event_id": "sha256:" + "5" * 64,
        "implementation_event_id": "sha256:" + "6" * 64,
        "proposal_id": "proposal:validation-retry",
        "proposal_receipt_id": "proposal-receipt:validation-retry",
        "proposal_policy_id": "proposal-policy:validation-retry",
        "validation_receipt_id": "validation-dag:validation-retry",
        "failure_review_receipt_id": "failure-review:validation-retry",
        "changed_paths": ["implementation.py", "test_implementation.py"],
        "authoritative_validation_executed": True,
        "proposal_policy_accepted": True,
        "output_policy_passed": True,
        "denial_findings": [],
    }
    receipt["receipt_id"] = daemon._database_portal_evidence_digest(receipt)
    return receipt


def test_interface_identities() -> None:
    assert DATABASE_IMPLEMENTATION_DAEMON_INTERFACE == (
        "DatabaseImplementationDaemon@1"
    )
    assert DATABASE_TASK_ATTEMPT_INTERFACE == "DatabaseTaskAttempt@1"
    assert DatabaseImplementationDaemon.INTERFACE == (
        DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
    )
    assert DatabaseTaskAttempt.INTERFACE == DATABASE_TASK_ATTEMPT_INTERFACE
    assert is_database_authority_mode(authority_mode="embedded")
    assert is_database_authority_mode(task_source_kind="duckdb")
    assert not is_database_authority_mode(
        authority_mode="legacy_markdown", task_source_kind="legacy-markdown"
    )


def test_direct_selector_never_bypasses_cooldown_when_all_ready_are_cooled() -> None:
    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon.merge_queue = SimpleNamespace(
        has_pending_for_task=lambda _task_cid: False
    )
    daemon.degradation_state = SimpleNamespace(
        degraded_submodules=lambda: []
    )
    daemon.task_queue = SimpleNamespace(
        is_cooled_down=lambda _task_cid: True,
        record_selection=lambda _task_cid: pytest.fail(
            "cooled task was selected"
        ),
    )
    daemon._canonical_ref = lambda task: f"task:cid:{task.task_id}"
    daemon._inflight_submodule_paths = lambda: set()
    task = SimpleNamespace(
        task_id="COOLED-001",
        priority="P0",
        track="implementation",
        depends_on=[],
        metadata={},
    )

    selected = daemon._select_next_task(
        [task],
        {task.task_id: "ready"},
        {},
        {},
        {},
    )

    assert selected is None


def test_four_daemon_processes_claim_distinct_work(tmp_path: Path) -> None:
    markdown = tmp_path / "board.md"
    markdown.write_text(
        "# Board\n\n## DQP-T001 Sample\n\n- Status: todo\n",
        encoding="utf-8",
    )
    original_markdown = markdown.read_text(encoding="utf-8")

    seed = _open_daemon(tmp_path, session="session:seed", markdown_path=markdown)
    try:
        seed.materialize_population(_population(4))
    finally:
        seed.close()

    claimed: list[str] = []
    for index in range(1, 5):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:{index}",
            markdown_path=markdown,
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"session {index} failed to claim"
            claimed.append(attempt.task_cid)
            assert attempt.owner_session_id == f"session:{index}"
            assert attempt.committed_phase == "claimed"
        finally:
            daemon.close()

    assert len(claimed) == 4
    assert len(set(claimed)) == 4

    idle = _open_daemon(tmp_path, session="session:extra", markdown_path=markdown)
    try:
        assert idle.claim_next() is None
        assert idle.markdown_status_write_count == 0
        assert markdown.read_text(encoding="utf-8") == original_markdown
    finally:
        idle.close()


def test_materialization_projects_completed_prerequisites_into_coordination(
    tmp_path: Path,
) -> None:
    population = _population(2)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["status"] = "completed"
    tasks[1]["dependencies"] = ["task:cid:001"]
    daemon = _open_daemon(tmp_path, session="session:bootstrap")
    try:
        receipt = daemon.materialize_population(population)
        assert receipt["bootstrap_completed_task_cids"] == ["task:cid:001"]
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["counts"]["logical_completions"] == 1

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:002"
    finally:
        daemon.close()


def test_claim_next_preserves_canonical_ready_order_for_late_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:ordered")
    try:
        population = _population(1)
        first = population["tasks"]
        assert isinstance(first, list)
        first[0]["ordinal"] = 20
        daemon.materialize_population(population)

        # The successor task enters the coordination registry later, but its
        # canonical plan ordinal places it first.  Registration time must not
        # override the intent repository's ready-order authority.
        daemon.task_source._intent.upsert_task(
            task_cid="task:cid:late-preferred",
            task_alias="DQP-LATE-PREFERRED",
            goal_cid="goal:cid:root",
            ordinal=1,
            status="ready",
            priority="P0",
            body={"title": "Late but plan-preferred"},
            identity={"task_cid": "task:cid:late-preferred"},
            dependencies=(),
            outputs=(),
            acceptance=(),
            validations=(),
            expected_revision=0,
        )

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:late-preferred"
    finally:
        daemon.close()


def test_no_markdown_status_update_under_database_authority(tmp_path: Path) -> None:
    markdown = tmp_path / "tasks.md"
    markdown.write_text(
        "# Tasks\n\n## DQP-T001 Work\n\n- Status: todo\n",
        encoding="utf-8",
    )
    before = markdown.read_text(encoding="utf-8")
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:md",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        markdown_path=markdown,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["markdown_status_writes"] == 0
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"
        with pytest.raises(DatabaseImplementationAuthorityError, match="Markdown"):
            daemon.write_markdown_task_status("DQP-T001", "completed")
        assert markdown.read_text(encoding="utf-8") == before
        assert "- Status: completed" not in markdown.read_text(encoding="utf-8")
    finally:
        daemon.close()


def test_json_projections_can_be_absent(tmp_path: Path) -> None:
    daemon = open_database_implementation_daemon(
        tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:proj",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        assert daemon.projections_required() is False
        assert daemon.state_path is None
        assert daemon.strategy_path is None
        assert daemon.events_path is None
        assert daemon.pid_path is None
        assert daemon.queue_path is None
        # No projection files created by open/materialize/run.
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["unchanged"] is True
        assert result["write_count"] == 0
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "ready"
        assert daemon.list_running_attempts() == []
        assert not (tmp_path / "task_state.json").exists()
        assert not (tmp_path / "events.jsonl").exists()
        assert not (tmp_path / "task_queue.json").exists()
        assert not list(tmp_path.glob("*.pid"))
    finally:
        daemon.close()


def test_database_observer_without_real_execution_never_resumes_or_claims(
    tmp_path: Path,
) -> None:
    """Inherited store authority cannot replace the explicit execution permit."""

    seed = _open_daemon(tmp_path)
    try:
        seed.materialize_population(_population(2))
        running = seed.claim_next()
        assert running is not None
        running = seed.commit_phase(
            running,
            "context",
            body={"source": "pre-reload-real-execution"},
        )
        running_attempt_id = running.attempt_id
    finally:
        seed.close()

    observer = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
        # Exact failed-reload shape: the database environment survives but
        # programmatic argv lost --implement, so no callbacks or execution
        # permit are present.
        require_real_execution=False,
    )
    try:
        before_tasks = {
            task_cid: observer.task_source.get(task_cid).to_dict()
            for task_cid in ("task:cid:001", "task:cid:002")
        }
        before_attempt = observer.get_attempt(running_attempt_id)
        assert before_attempt is not None
        before_count_row = observer._require_connection().execute(
            """
            SELECT
                (SELECT COUNT(*) FROM database_task_attempts),
                (SELECT COUNT(*) FROM provider_invocations),
                (SELECT COUNT(*) FROM effect_claims)
            """
        ).fetchone()
        before_counts = tuple(
            int(before_count_row[index]) for index in range(3)
        )

        result = observer.run_once()

        assert result["unchanged"] is True
        assert result["write_count"] == 0
        assert result["execution_authorized"] is False
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        assert result["implementation_result"] is None
        assert result["completion_reconciliations"] == []
        assert result["expired_attempt_reconciliations"] == []
        assert result["terminal_retry_reconciliations"] == []
        assert result["terminal_portal_reconciliations"] == []

        after_tasks = {
            task_cid: observer.task_source.get(task_cid).to_dict()
            for task_cid in ("task:cid:001", "task:cid:002")
        }
        after_attempt = observer.get_attempt(running_attempt_id)
        after_count_row = observer._require_connection().execute(
            """
            SELECT
                (SELECT COUNT(*) FROM database_task_attempts),
                (SELECT COUNT(*) FROM provider_invocations),
                (SELECT COUNT(*) FROM effect_claims)
            """
        ).fetchone()
        after_counts = tuple(
            int(after_count_row[index]) for index in range(3)
        )
        assert after_tasks == before_tasks
        assert after_attempt == before_attempt
        assert after_counts == before_counts == (1, 0, 0)
        assert after_tasks["task:cid:001"]["status"] == "in_progress"
        assert after_tasks["task:cid:002"]["status"] == "ready"

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="explicit real-execution authority",
        ):
            observer.claim_next()
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="explicit real-execution authority",
        ):
            observer.resume_attempt(running_attempt_id)
        guarded_mutations = (
            (
                "attempt phase commit",
                lambda: observer.commit_phase(running_attempt_id, "provider"),
            ),
            (
                "provider phase",
                lambda: observer.run_provider(before_attempt),
            ),
            (
                "effect phase",
                lambda: observer.run_effect(before_attempt, {}),
            ),
            (
                "task completion",
                lambda: observer.complete_attempt(before_attempt),
            ),
            (
                "prepared completion reconciliation",
                observer.reconcile_prepared_task_completions,
            ),
            (
                "expired attempt reconciliation",
                observer.reconcile_expired_running_attempts,
            ),
            (
                "terminal retry reconciliation",
                observer.reconcile_terminal_retry_states,
            ),
            (
                "terminal failure reconciliation",
                observer.reconcile_terminal_portal_failures,
            ),
        )
        for operation, mutation in guarded_mutations:
            with pytest.raises(
                DatabaseImplementationAuthorityError,
                match=operation,
            ):
                mutation()
        final_count = observer._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts"
        ).fetchone()
        assert int(final_count[0]) == 1
    finally:
        observer.close()


def test_datasets_authoritative_open_requires_preinstalled_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    control_path = tmp_path / "missing-control.duckdb"
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="preinstalled by the trusted materializer",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not control_path.exists()
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_full_control_plane_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "full-control.duckdb"
    install_control_plane_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    with open_duckdb_connection(control_path) as connection:
        names = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
    assert "proof_obligations" in names
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_rejects_tampered_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "tampered-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    with open_duckdb_connection(control_path) as connection:
        connection.execute(
            "UPDATE schema_migrations SET checksum = 'sha256:tampered'"
        )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="not the verified datasets-authoritative operational profile",
    ):
        DatabaseImplementationDaemon(
            database_path=control_path,
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
    assert not (tmp_path / "execution.duckdb").exists()


def test_datasets_authoritative_open_verifies_existing_operational_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "operational-control.duckdb"
    install_datasets_authoritative_operational_schema(control_path)
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        "datasets-authoritative-operational-v1",
    )
    daemon = DatabaseImplementationDaemon(
        database_path=control_path,
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        evidence = dict(daemon.control_schema_evidence)
        assert evidence["state_schema_revision"] == (
            "datasets-authoritative-operational-v1"
        )
        assert evidence["verified"] is True
        assert evidence["profile_id"]
        assert evidence["schema_fingerprint"]
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "ready"
    finally:
        daemon.close()


def test_crash_restart_resumes_without_duplicating_provider_or_effect(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    first = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == ["task:cid:001"]
        assert attempt.committed_phase == ATTEMPT_PHASE_PROVIDER
        # Crash boundary: process dies after provider commits, before effect.
        assert effect_calls == []
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:resume",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        running = second.list_running_attempts()
        assert len(running) == 1
        assert running[0].attempt_id == attempt_id
        assert running[0].committed_phase == ATTEMPT_PHASE_PROVIDER
        result = second.resume_attempt(running[0])
        assert result["resumed"] is True
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["committed_phase"] == ATTEMPT_PHASE_COMPLETE
        assert result["status"] == "succeeded"
        task = second.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "completed"

        # Second resume of a finished attempt is a no-op for provider/effect.
        finished = second.get_attempt(attempt_id)
        assert finished is not None
        again = second.resume_attempt(finished)
        assert again["resumed"] is False
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        second.close()


def test_implicit_embedded_owner_is_store_scoped_and_restart_stable(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first_owner = first.owner_session_id
        assert first_owner.startswith("embedded-store:")
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, _, duplicated = first.run_provider(attempt)
        assert duplicated is False
        assert provider_calls == [attempt.task_cid]
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        assert second.owner_session_id == first_owner
        result = second.run_once()
        assert result["implementation_result"]["provider_duplicated"] is True
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == [attempt.task_cid]
    finally:
        second.close()


def test_implicit_embedded_owner_is_distinct_for_different_stores(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path / "first")
    second = _open_daemon(tmp_path / "second")
    try:
        assert first.owner_session_id.startswith("embedded-store:")
        assert second.owner_session_id.startswith("embedded-store:")
        assert first.owner_session_id != second.owner_session_id
    finally:
        second.close()
        first.close()


def test_embedded_writer_lock_rejects_a_concurrent_same_store_opener(
    tmp_path: Path,
) -> None:
    first = _open_daemon(tmp_path)
    try:
        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="active database writer",
        ):
            _open_daemon(tmp_path)
        first.materialize_population(_population(1))
        assert first.claim_next() is not None
    finally:
        first.close()

    replacement = _open_daemon(tmp_path)
    try:
        assert replacement.owner_session_id == first.owner_session_id
    finally:
        replacement.close()


def test_effect_phase_resume_skips_both_provider_and_effect(tmp_path: Path) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        attempt = first.commit_phase(attempt, "context")
        attempt, provider_result, _ = first.run_provider(attempt)
        attempt, effect_result, _ = first.run_effect(attempt, provider_result)
        assert attempt.committed_phase == ATTEMPT_PHASE_EFFECT
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    second = _open_daemon(
        tmp_path,
        session="session:effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        attempt = second.get_attempt(attempt_id)
        assert attempt is not None
        result = second.resume_attempt(attempt)
        assert result["provider_duplicated"] is True
        assert result["effect_duplicated"] is True
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
        assert result["status"] == "succeeded"
    finally:
        second.close()


def test_provider_heartbeat_renews_exact_task_claim(tmp_path: Path) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    observed_revisions: list[int] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        initial = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert initial is not None
        observed_revisions.append(int(initial.revision))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            current = daemon.coordinator.get_task_claim(attempt.claim_id)
            assert current is not None
            if int(current.revision) > int(initial.revision):
                observed_revisions.append(int(current.revision))
                break
            time.sleep(0.005)
        assert len(observed_revisions) == 2, "background lease renewal did not run"
        return {
            "status": "succeeded",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    daemon = _open_daemon(
        tmp_path,
        session="session:heartbeat",
        provider_fn=provider,
        lease_ms=5_000,
    )
    holder["daemon"] = daemon
    daemon._lease_heartbeat_interval_seconds = 0.01
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        updated, _, duplicated = daemon.run_provider(attempt)
        assert duplicated is False
        assert updated.committed_phase == ATTEMPT_PHASE_PROVIDER
        assert observed_revisions[1] > observed_revisions[0]
    finally:
        daemon.close()


def test_portal_failure_terminal_cas_refetches_advanced_attempt(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    provider_revisions: list[int] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        provider_revisions.append(attempt.revision)
        daemon._record_event(
            "portal_progress_before_failure",
            attempt_id=attempt.attempt_id,
            task_cid=attempt.task_cid,
            body={"provider_revision": attempt.revision},
        )
        raise DatabasePortalBridgeError("portal validation failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-failure-cas",
        provider_fn=provider,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()

        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is False
        assert implementation["portal_terminal_failure"] is True
        assert implementation["status"] == "failed"
        assert implementation["deferred"] is False
        assert implementation["attempt_consumed"] == "unknown"
        assert implementation["provider_dispatched"] == "unknown"
        assert implementation["backoff_seconds"] == 0
        assert "fail_error" not in implementation
        assert provider_revisions == [2]

        stored = daemon.get_attempt(result["attempt_id"])
        assert stored is not None
        assert stored.status == "failed"
        assert stored.committed_phase == "failed"
        assert stored.revision == 3
        assert [
            (phase["phase"], phase["revision"])
            for phase in daemon.phase_history(stored.attempt_id)
        ] == [("claimed", 1), ("context", 2), ("failed", 3)]

        event_count = daemon._require_connection().execute(
            """
            SELECT COUNT(*) FROM daemon_execution_events
            WHERE attempt_id = ? AND event_type = ?
            """,
            [stored.attempt_id, "portal_progress_before_failure"],
        ).fetchone()
        assert event_count is not None
        assert int(event_count[0]) == 1
        task = daemon.task_source.get(stored.task_cid)
        assert task is not None
        assert task.status == "blocked"
        queue_entry = daemon.task_source.get_queue_entry(stored.task_cid)
        assert queue_entry is None
        assert implementation["terminal_state"]["status"] == "blocked"
    finally:
        daemon.close()


def test_typed_post_dispatch_validation_failure_retries_with_attempt_budget(
    tmp_path: Path,
) -> None:
    holder: dict[str, DatabaseImplementationDaemon] = {}
    now = {"ms": 1_000}

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        receipt = _validation_retry_receipt(holder["daemon"], attempt)
        raise DatabasePortalValidationRetry(receipt)

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-validation-retry",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()

        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is True
        assert implementation["portal_terminal_failure"] is False
        assert implementation["deferred"] is False
        assert implementation["attempt_consumed"] is True
        assert implementation["provider_dispatched"] is True
        assert implementation["typed_deferral_slot_consumed"] is False
        assert implementation["retry_budget_exhausted"] is False
        assert implementation["retry_state"]["status"] == "retrying"

        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None
        failed = daemon.phase_history(attempt.attempt_id)[-1]["body"]
        assert failed["typed_validation_retry"]["remaining_task_attempts"] == 2
        evidence = daemon._terminal_retry_evidence(attempt)
        assert evidence is not None
        assert evidence["typed_deferral_budget"] is None
        assert evidence["typed_validation_retry"]["attempt_consumed"] is True
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        retry_seed = task.body["completion_receipt"]["validation_retry_seed"]
        assert retry_seed["receipt_id"] == failed["typed_validation_retry"][
            "receipt_id"
        ]

        # The retrying->in_progress claim CAS carries the verified seed into
        # the exact successor record consumed by the fresh Portal bridge.
        now["ms"] = 7_000
        successor = daemon.claim_next()
        assert successor is not None
        assert successor.attempt_number == 2
        claimed = daemon.task_source.get(attempt.task_cid)
        assert claimed is not None
        claim_receipt = claimed.body["completion_receipt"]
        assert claim_receipt["operation"] == "database_claim"
        assert claim_receipt["validation_retry_seed"] == retry_seed
        assert claim_receipt["validation_retry_source_attempt_id"] == (
            attempt.attempt_id
        )
        assert claim_receipt["attempt_number"] == successor.attempt_number
        assert claim_receipt["fencing_token"] == successor.fencing_token
        assert claim_receipt["fence_epoch"] == successor.fence_epoch
        assert claim_receipt["lease_id"] == successor.lease_id
    finally:
        daemon.close()


def test_blocked_generic_validation_failure_has_idempotent_typed_recovery(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        receipt = _validation_retry_receipt(daemon, attempt)
        tampered = dict(receipt)
        tampered["denial_findings"] = ["denied_effect"]
        tampered.pop("receipt_id")
        tampered["receipt_id"] = daemon._database_portal_evidence_digest(
            tampered
        )
        with pytest.raises(DatabaseImplementationAuthorityError):
            daemon.recover_blocked_portal_validation_retry(
                attempt,
                retry_evidence=tampered,
            )
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        recovered = daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=receipt,
        )
        assert recovered["changed"] is True
        assert recovered["status"] == "retrying"
        assert recovered["validation_retry_evidence"] == receipt
        assert recovered["coordination"]["attempt_id"] == attempt.attempt_id
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"

        repeated = daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=receipt,
        )
        assert repeated["changed"] is False
        assert repeated["status"] == "retrying"
        assert repeated["validation_retry_evidence"] == receipt
        assert daemon.reconcile_terminal_portal_failures() == []
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


@pytest.mark.parametrize("control_status", ("completed", "todo"))
def test_terminal_portal_reconciliation_skips_settled_control_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_status: str,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-reconcile-skip",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        original_get = daemon.task_source.get

        def projected_get(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != attempt.task_cid or task is None:
                return task
            return SimpleNamespace(
                status=control_status,
                revision=task.revision,
                body=dict(task.body),
            )

        monkeypatch.setattr(daemon.task_source, "get", projected_get)
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_terminal_portal_reconciliation_accepts_board_unstall_retrying(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:board-unstall-retrying",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'retrying', revision = revision + 1, "
                "updated_at = '2026-08-22T22:00:00Z' WHERE task_cid = ?",
                [attempt.task_cid],
            )
        retried = daemon.task_source.get(attempt.task_cid)
        assert retried is not None
        assert retried.status == "retrying"
        receipt = (retried.body or {}).get("completion_receipt")
        assert not isinstance(receipt, dict) or receipt.get("operation") != (
            "database_portal_validation_retry_recovery"
        )
        assert daemon.reconcile_terminal_portal_failures() == []
        still = daemon.task_source.get(attempt.task_cid)
        assert still is not None
        assert still.status == "retrying"
    finally:
        daemon.close()


def test_proposal_gate_failure_retries_instead_of_blocking(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("proposal_gate_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:proposal-gate-retry",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        implementation = failed_result.get("implementation_result") or {}
        assert implementation.get("portal_retryable_failure") is True
        assert implementation.get("portal_terminal_failure") is False
        assert implementation.get("reason") == "proposal_gate_failed"
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
    finally:
        daemon.close()


def test_quack_attach_contention_defers_instead_of_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QuackTransportContentionError,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-defer",
        max_task_attempts=3,
    )
    try:
        def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise QuackTransportContentionError(
                "quack control-plane attach contended: Authentication failed"
            )

        monkeypatch.setattr(daemon, "_run_once_impl", boom)
        result = daemon.run_once()
        assert result.get("deferred") is True
        assert result.get("skipped") is True
        assert result.get("reason") == "quack_attach_contended"
        assert result.get("portal_retryable_failure") is True
        assert result.get("portal_terminal_failure") is False
        assert result.get("attempt_consumed") is False
        assert result.get("provider_dispatched") is False
    finally:
        daemon.close()


def test_quack_attach_contention_requests_owner_board_unstall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QuackTransportContentionError,
    )

    inbox = tmp_path / "mutations"
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", str(inbox))
    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-unstall",
        max_task_attempts=3,
    )
    try:
        def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
            raise QuackTransportContentionError(
                "quack control-plane attach contended: Authentication failed"
            )

        monkeypatch.setattr(daemon, "_run_once_impl", boom)
        result = daemon.run_once()
        assert result.get("board_unstall_request", {}).get("requested") is True
        requests = list(inbox.glob("*.request.json"))
        assert len(requests) == 1
        payload = json.loads(requests[0].read_text(encoding="utf-8"))
        assert payload["op"] == "board_unstall"
        assert payload["stale_seconds"] == 600
        second = daemon.run_once()
        assert second.get("board_unstall_request", {}).get("requested") is False
        assert second.get("board_unstall_request", {}).get("skipped") == (
            "bounce_already_pending"
        )
        assert list(inbox.glob("*.request.json")) == requests
    finally:
        daemon.close()


def test_run_once_unstalls_stale_in_progress_gate_and_claims(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    daemon = _open_daemon(
        tmp_path,
        session="session:unstall-gate",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        stale = (datetime.now(timezone.utc) - timedelta(hours=12)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [stale, "task:cid:001"],
            )
        stuck = daemon.task_source.get("task:cid:001")
        assert stuck is not None
        assert stuck.status == "in_progress"
        idle = daemon.claim_next()
        assert idle is None
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert [item["task_cid"] for item in unstalled] == ["task:cid:001"]
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_stale_in_progress_unstall_leaves_live_attempts_alone(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    daemon = _open_daemon(
        tmp_path,
        session="session:unstall-live",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        recent = (datetime.now(timezone.utc) - timedelta(seconds=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [recent, "task:cid:001"],
            )
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert unstalled == []
        live = daemon.task_source.get("task:cid:001")
        assert live is not None
        assert live.status == "in_progress"
    finally:
        daemon.close()


def test_abandoned_in_progress_gate_unstalls_without_running_attempt(
    tmp_path: Path,
) -> None:
    from datetime import datetime, timedelta, timezone

    daemon = _open_daemon(
        tmp_path,
        session="session:unstall-abandoned",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        abandoned = (datetime.now(timezone.utc) - timedelta(minutes=15)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with daemon.task_source._intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = 'in_progress', updated_at = ? "
                "WHERE task_cid = ?",
                [abandoned, "task:cid:001"],
            )
        assert daemon.list_running_attempts() == []
        unstalled = daemon.reconcile_stale_in_progress_gates()
        assert [item["task_cid"] for item in unstalled] == ["task:cid:001"]
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_claim_next_proceeds_when_owner_recycle_lands_cas_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-cas-timeout",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        original = daemon._cas_task_status_database

        def land_then_timeout(*args: object, **kwargs: object) -> None:
            original(*args, **kwargs)
            raise DuckDBConnectionPolicyError(
                "timed out waiting for quack state-owner to apply mutation"
            )

        monkeypatch.setattr(daemon, "_cas_task_status_database", land_then_timeout)
        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:001"
        assert daemon.task_source.get("task:cid:001").status == "in_progress"
        running = daemon.list_running_attempts()
        assert [item.attempt_id for item in running] == [attempt.attempt_id]
    finally:
        daemon.close()


def test_quack_attach_contention_still_expires_running_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QuackTransportContentionError,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:quack-attach-expire",
        max_task_attempts=3,
    )
    try:
        def boom(*_args: object, **_kwargs: object) -> list[object]:
            raise QuackTransportContentionError(
                "quack control-plane attach contended: Authentication failed"
            )

        expired = [
            {
                "status": "expired",
                "reason": "coordination_lease_expired_before_completion",
            }
        ]
        seen = {"expired": False}

        def expire() -> list[dict[str, object]]:
            seen["expired"] = True
            return expired

        monkeypatch.setattr(daemon, "reconcile_prepared_task_completions", boom)
        monkeypatch.setattr(daemon, "reconcile_expired_running_attempts", expire)
        monkeypatch.setattr(daemon, "reconcile_terminal_portal_failures", boom)
        monkeypatch.setattr(daemon, "reconcile_terminal_retry_states", lambda: [])
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(daemon, "claim_next", lambda: None)
        result = daemon.run_once()
        assert seen["expired"] is True
        assert result.get("expired_attempt_reconciliations") == expired
        assert result.get("selection_idle_reason") == "no_ready_tasks"
    finally:
        daemon.close()


def test_inflight_process_failure_retries_instead_of_blocking(
    tmp_path: Path,
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("inflight_process")

    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-process-retry",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        implementation = failed_result.get("implementation_result") or {}
        assert implementation.get("portal_retryable_failure") is True
        assert implementation.get("portal_terminal_failure") is False
        assert implementation.get("reason") == "inflight_process"
        assert daemon.task_source.get(attempt.task_cid).status == "retrying"
    finally:
        daemon.close()


def test_inflight_process_deferral_does_not_exhaust_typed_budget(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalBridgeDeferred,
    )

    now = {"ms": 1_000}
    calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred("inflight_process", backoff_seconds=30)

    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-deferral-budget",
        provider_fn=provider,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        for _ in range(12):
            daemon.run_once()
            now["ms"] += 31_000
            task = daemon.task_source.get("task:cid:001")
            assert task is not None
            assert task.status != "blocked"
        assert len(calls) >= 4
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "retrying"
    finally:
        daemon.close()


def test_reconcile_reopens_inflight_deferral_budget_block(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:inflight-deferral-unstall",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            task.task_cid,
            expected_revision=int(task.revision),
            status="blocked",
            receipt={
                "operation": "database_portal_typed_deferral_budget_exhausted",
                "retry_budget": {
                    "matching_attempts": [
                        {"reason": "inflight_process"},
                        {"reason": "inflight_process"},
                        {"reason": "inflight_process"},
                    ]
                },
            },
        )
        blocked = daemon.task_source.get("task:cid:001")
        assert blocked is not None
        assert blocked.status == "blocked"
        outcomes = daemon.reconcile_inflight_deferral_blocks()
        assert [item["task_cid"] for item in outcomes] == ["task:cid:001"]
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
    finally:
        daemon.close()


def test_reconcile_reopens_mixed_capacity_and_inflight_budget_block(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:capacity-inflight-unstall",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            task.task_cid,
            expected_revision=int(task.revision),
            status="blocked",
            receipt={
                "operation": "database_portal_typed_deferral_budget_exhausted",
                "retry_budget": {
                    "matching_attempts": [
                        {"reason": "provider_capacity_exhausted"},
                        {"reason": "inflight_process"},
                        {"reason": "quack_attach_contended"},
                    ]
                },
            },
        )
        outcomes = daemon.reconcile_inflight_deferral_blocks()
        assert [item["task_cid"] for item in outcomes] == ["task:cid:001"]
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
    finally:
        daemon.close()


def test_reconcile_reopens_provider_failed_terminal_block(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:provider-failed-unstall",
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            task.task_cid,
            expected_revision=int(task.revision),
            status="blocked",
            receipt={
                "operation": "database_portal_terminal_failure",
                "reason": "portal_provider_failed",
            },
        )
        outcomes = daemon.reconcile_inflight_deferral_blocks()
        assert [item["task_cid"] for item in outcomes] == ["task:cid:001"]
        retried = daemon.task_source.get("task:cid:001")
        assert retried is not None
        assert retried.status == "retrying"
    finally:
        daemon.close()


def test_stale_fence_expiry_does_not_crash_reconciliation(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationStaleFenceError,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:stale-fence-reconcile",
        max_task_attempts=3,
    )
    try:
        assert daemon._is_stale_coordination_fence(
            DatabaseCoordinationStaleFenceError(
                "task claim is not the latest fencing epoch and token"
            )
        )
        assert daemon._run_reconciliation_step(
            lambda: (_ for _ in ()).throw(
                DatabaseCoordinationStaleFenceError(
                    "task claim is not the latest fencing epoch and token"
                )
            )
        ) == []
        result = daemon.run_once()
        assert result.get("selection_idle_reason") == "no_ready_tasks"
    finally:
        daemon.close()


def test_owner_mutation_reject_is_attach_contention(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        DuckDBConnectionPolicyError,
    )

    daemon = _open_daemon(
        tmp_path,
        session="session:mutation-contention",
        max_task_attempts=3,
    )
    try:
        exc = DuckDBConnectionPolicyError(
            "quack owner mutation failed: FatalException: mutation rejected"
        )
        assert daemon._is_quack_attach_contention(exc) is True
        assert daemon._database_portal_reason(str(exc)) == "quack_attach_contended"
        result = daemon.run_once()
        # no exception: run_once maps owner mutation/attach misses to a deferral
        assert result.get("reason") in (None, "quack_attach_contended", "no_ready_tasks") or (
            result.get("selection_idle_reason") == "no_ready_tasks"
        )
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("missing_seed", DatabaseImplementationAuthorityError),
        ("wrong_seed_receipt", DatabaseImplementationAuthorityError),
    ],
)
def test_terminal_portal_reconciliation_rejects_foreign_retrying_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_type: type[Exception],
) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-forgery",
        provider_fn=provider,
        max_task_attempts=3,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        retry_evidence = _validation_retry_receipt(daemon, attempt)
        daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=retry_evidence,
        )
        persisted = daemon.task_source.get(attempt.task_cid)
        assert persisted is not None
        receipt = dict(persisted.body["completion_receipt"])
        if mutation == "missing_seed":
            receipt.pop("validation_retry_seed")
        else:
            seed = dict(receipt["validation_retry_seed"])
            seed["receipt_id"] = "sha256:" + "0" * 64
            receipt["validation_retry_seed"] = seed

        original_get = daemon.task_source.get

        def projected_get(task_cid: str) -> object:
            task = original_get(task_cid)
            if task_cid != attempt.task_cid or task is None:
                return task
            return SimpleNamespace(
                status=task.status,
                revision=task.revision,
                body={**dict(task.body), "completion_receipt": receipt},
            )

        monkeypatch.setattr(daemon.task_source, "get", projected_get)
        with pytest.raises(error_type):
            daemon.reconcile_terminal_portal_failures()
    finally:
        daemon.close()


def test_terminal_portal_recovery_projection_rejects_newer_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-newer-fence",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        daemon.recover_blocked_portal_validation_retry(
            attempt,
            retry_evidence=_validation_retry_receipt(daemon, attempt),
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"

        source_claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert source_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(source_claim, now_ms=now["ms"])
        newer = daemon.coordinator.claim_ready_task(
            owner_session_id="session:newer-validation-retry-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert newer is not None
        assert newer.fencing_token > attempt.fencing_token

        with pytest.raises(DatabaseCoordinationError):
            daemon.reconcile_terminal_portal_failures()
        unchanged = daemon.task_source.get(attempt.task_cid)
        assert unchanged is not None
        assert unchanged.status == "retrying"
    finally:
        daemon.close()


def test_restart_accepts_exact_validation_retry_recovery_projection(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    first = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-before-restart",
        provider_fn=provider,
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        failed_result = first.run_once()
        source = first.get_attempt(failed_result["attempt_id"])
        assert source is not None
        first.recover_blocked_portal_validation_retry(
            source,
            retry_evidence=_validation_retry_receipt(first, source),
        )
    finally:
        first.close()

    now["ms"] = 7_000
    restarted = _open_daemon(
        tmp_path,
        session="session:validation-retry-recovery-after-restart",
        max_task_attempts=3,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = restarted.run_once()
        assert result["terminal_portal_reconciliations"] == []
        assert result["implementation_result"] is not None
        assert result["implementation_result"]["status"] == "succeeded"
        successor = restarted.get_attempt(result["attempt_id"])
        assert successor is not None
        assert successor.attempt_number > source.attempt_number
    finally:
        restarted.close()


def test_restart_finishes_terminal_portal_failure_control_cas(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-portal-recovery",
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "untyped_portal_integrity_failure",
                "portal_retryable_failure": False,
                "portal_terminal_failure": True,
            },
        )

        result = daemon.run_once()

        assert result["implementation_result"] is None
        assert len(result["terminal_portal_reconciliations"]) == 1
        blocked = daemon.task_source.get(failed_attempt.task_cid)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_typed_portal_deferral_honors_canonical_cooldown_after_lease_expiry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-portal-deferral",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()

        assert len(provider_attempts) == 1
        assert first["implementation_result"]["deferred"] is True
        assert first["implementation_result"]["attempt_consumed"] is False
        assert first["implementation_result"]["provider_dispatched"] is False
        assert first["implementation_result"]["backoff_seconds"] == 300
        task_cid = str(first["claimed_task_cid"])
        queue_entry = daemon.task_source.get_queue_entry(task_cid)
        assert queue_entry is not None
        assert queue_entry.retry_not_before_ms == 301_000
        task = daemon.task_source.get(task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert task.revision == 3

        # The coordination lease is expired, but the canonical queue deadline
        # remains authoritative and no replacement attempt is constructed.
        now["ms"] = 7_000
        cooled = daemon.run_once()
        assert cooled["selection_idle_reason"] == "no_ready_tasks"
        assert cooled["implementation_result"] is None
        assert len(provider_attempts) == 1
        assert daemon.task_source.get_queue_entry(
            task_cid
        ).retry_not_before_ms == 301_000

        now["ms"] = 301_001
        retried = daemon.run_once()
        assert len(provider_attempts) == 2
        assert retried["attempt_id"] != first["attempt_id"]
        assert retried["implementation_result"]["deferred"] is True
        retried_task = daemon.task_source.get(task_cid)
        assert retried_task is not None
        assert retried_task.status == "retrying"
        assert retried_task.revision == 5
    finally:
        daemon.close()


def test_typed_portal_deferral_budget_blocks_before_fourth_dispatch(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:typed-deferral-budget",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=3,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        task_cid = str(first["claimed_task_cid"])
        assert first["implementation_result"]["retry_budget_exhausted"] is False

        now["ms"] = 301_001
        second = daemon.run_once()
        assert second["implementation_result"]["retry_budget_exhausted"] is False

        now["ms"] = 601_002
        third = daemon.run_once()
        implementation = third["implementation_result"]
        assert implementation["retry_budget_exhausted"] is True
        assert implementation["attempt_consumed"] is False
        assert implementation["typed_deferral_slot_consumed"] is True
        assert implementation["retry_state"] is None
        terminal = implementation["terminal_state"]
        assert terminal["status"] == "blocked"
        assert terminal["reason"] == "typed_portal_deferral_budget_exhausted"
        budget = terminal["retry_budget"]
        assert budget["typed_deferral_count"] == 3
        assert budget["max_task_attempts"] == 3
        assert budget["exhausted"] is True
        assert len(budget["matching_attempts"]) == 3

        task = daemon.task_source.get(task_cid)
        assert task is not None
        assert task.status == "blocked"
        assert len(provider_attempts) == 3

        # Even after every prior cooldown and lease deadline, the blocked
        # control task cannot construct or dispatch attempt four.
        now["ms"] = 1_000_000
        idle = daemon.run_once()
        assert idle["implementation_result"] is None
        assert idle["selection_idle_reason"] == "no_ready_tasks"
        assert len(provider_attempts) == 3
    finally:
        daemon.close()


def test_legacy_failed_claim_does_not_consume_typed_deferral_budget(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:legacy-deferral-migration",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=1,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        legacy = daemon.claim_next()
        assert legacy is not None
        legacy = daemon.commit_phase(legacy, "context")
        legacy = daemon.commit_phase(
            legacy,
            "failed",
            body={
                "reason": "validation_project_dependency_preflight_failed",
                "portal_retryable_failure": True,
                # Deliberately pre-fix: no explicit deferred disposition or
                # identity-bound typed-deferral receipt.
            },
        )

        recovered = daemon.reconcile_terminal_retry_states()
        assert len(recovered) == 1
        assert recovered[0]["status"] == "retrying"
        assert daemon.task_source.get(legacy.task_cid).status == "retrying"

        # The first patch-era closed typed deferral consumes slot one and
        # blocks.  The legacy claim did not pre-exhaust the migration budget.
        now["ms"] = 301_001
        typed = daemon.run_once()
        assert len(provider_attempts) == 1
        assert typed["implementation_result"]["retry_budget_exhausted"] is True
        assert typed["implementation_result"]["terminal_state"][
            "retry_budget"
        ]["typed_deferral_count"] == 1
    finally:
        daemon.close()


def test_restart_reconciles_exhausted_typed_deferral_without_new_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    first = _open_daemon(
        tmp_path,
        session="session:typed-budget-restart",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        initial = first.run_once()
        task_cid = str(initial["claimed_task_cid"])
        now["ms"] = 301_001

        def crash_before_block(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("simulated crash before exhausted control CAS")

        monkeypatch.setattr(
            first,
            "_persist_typed_deferral_budget_exhausted",
            crash_before_block,
        )
        interrupted = first.run_once()
        assert "simulated crash" in interrupted["implementation_result"][
            "fail_error"
        ]
        control = first.task_source.get(task_cid)
        assert control is not None
        assert control.status == "in_progress"
        assert len(provider_attempts) == 2
    finally:
        first.close()

    monkeypatch.undo()
    now["ms"] = 307_000
    replacement = _open_daemon(
        tmp_path,
        session="session:typed-budget-restart",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        reconciled = replacement.run_once()
        assert reconciled["implementation_result"] is None
        assert len(reconciled["terminal_retry_reconciliations"]) == 1
        terminal = reconciled["terminal_retry_reconciliations"][0]
        assert terminal["status"] == "blocked"
        assert terminal["retry_budget"]["typed_deferral_count"] == 2
        assert replacement.task_source.get(task_cid).status == "blocked"
        assert len(provider_attempts) == 2
    finally:
        replacement.close()


def test_exhaustion_blocks_already_retrying_task_and_bounds_evidence_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeDeferred(
            "validation_project_dependency_preflight_failed",
            backoff_seconds=300,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:retrying-budget-reconciliation",
        provider_fn=provider,
        lease_ms=5_000,
        max_task_attempts=2,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        task_cid = str(first["claimed_task_cid"])
        now["ms"] = 301_001

        # Model a pre-budget writer that durably persisted the second typed
        # deferral and queue/CAS but crashed before checking exhaustion.
        original_observation = daemon._typed_deferral_budget_observation
        monkeypatch.setattr(
            daemon,
            "_typed_deferral_budget_observation",
            lambda _attempt: None,
        )
        second = daemon.run_once()
        assert second["implementation_result"]["retry_state"][
            "status"
        ] == "retrying"
        assert daemon.task_source.get(task_cid).status == "retrying"
        assert daemon.task_source.get_queue_entry(task_cid) is not None

        monkeypatch.setattr(
            daemon,
            "_typed_deferral_budget_observation",
            original_observation,
        )
        implementation_daemon_module = importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
        )
        monkeypatch.setattr(
            implementation_daemon_module,
            "_MAX_TYPED_DEFERRAL_ATTEMPT_PREVIEW",
            1,
        )
        reconciled = daemon.reconcile_terminal_retry_states()

        assert len(reconciled) == 1
        terminal = reconciled[0]
        assert terminal["status"] == "blocked"
        assert terminal["control_previous_status"] == "retrying"
        assert terminal["prior_queue_entry_preserved_inactive"] is True
        budget = terminal["retry_budget"]
        assert budget["typed_deferral_count"] == 2
        assert budget["verified_typed_deferral_count"] == 2
        assert budget["verified_count_complete"] is True
        assert len(budget["matching_attempts"]) == 1
        assert budget["matching_attempts_truncated"] is True
        assert budget["omitted_matching_attempt_count"] == 1
        assert budget["matching_attempts_digest"].startswith("sha256:")
        assert daemon.task_source.get(task_cid).status == "blocked"
    finally:
        daemon.close()


def test_typed_deferral_from_old_state_schema_does_not_consume_current_budget(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:typed-budget-schema-generation",
        max_task_attempts=1,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        daemon.state_schema_revision = "state-schema-old"
        typed = daemon._typed_deferral_receipt(
            attempt,
            reason="typed_schema_migration_deferral",
        )
        attempt = daemon.commit_phase(
            attempt,
            "failed",
            body={
                "reason": "typed_schema_migration_deferral",
                "portal_retryable_failure": True,
                "portal_terminal_failure": False,
                "deferred": True,
                "attempt_consumed": False,
                "provider_dispatched": False,
                "typed_deferral_slot_consumed": True,
                "backoff_seconds": 300,
                "typed_deferral": typed,
            },
        )

        daemon.state_schema_revision = "state-schema-new"
        evidence = daemon._terminal_retry_evidence(attempt)
        assert evidence is not None
        assert evidence["typed_deferral_budget"] is None
    finally:
        daemon.close()


def test_restart_reconciles_failed_execution_and_expired_coordination_claim(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    first = _open_daemon(
        tmp_path,
        session="session:terminal-retry-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        failed_attempt = first.claim_next()
        assert failed_attempt is not None
        failed_attempt = first.commit_phase(failed_attempt, "context")
        failed_attempt = first.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "validation_project_dependency_preflight_failed",
                "portal_retryable_failure": True,
                # Pre-fix receipt: no typed backoff_seconds field.
            },
        )
        task_before = first.task_source.get(failed_attempt.task_cid)
        assert task_before is not None
        assert task_before.status == "in_progress"
        assert first.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        first.close()

    # The legacy 300-second window elapsed while the supervisor was down.
    now["ms"] = 401_000
    replacement = _open_daemon(
        tmp_path,
        session="session:terminal-retry-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        reconciliations = replacement.reconcile_terminal_retry_states()
        assert len(reconciliations) == 1
        reconciliation = reconciliations[0]
        assert reconciliation["backoff_ms"] == 0
        assert reconciliation["retry_not_before_ms"] == 401_000
        assert reconciliation["control_previous_status"] == "in_progress"
        assert reconciliation["control_previous_revision"] == 2
        assert reconciliation["control_new_status"] == "retrying"
        assert reconciliation["control_new_revision"] == 3
        assert reconciliation["coordination"]["expired_now"] is True
        assert reconciliation["coordination"]["claim_state"] == "expired"
        assert reconciliation["coordination"][
            "coordination_attempt_status"
        ] == "expired"

        recovered = replacement.task_source.get(failed_attempt.task_cid)
        assert recovered is not None
        assert recovered.status == "retrying"
        queue_entry = replacement.task_source.get_queue_entry(
            failed_attempt.task_cid
        )
        assert queue_entry is not None
        queue_attempt = queue_entry.attempt

        # Reconciliation is idempotent after both durable writes landed.
        assert replacement.reconcile_terminal_retry_states() == []
        repeated_entry = replacement.task_source.get_queue_entry(
            failed_attempt.task_cid
        )
        assert repeated_entry is not None
        assert repeated_entry.attempt == queue_attempt
        assert repeated_entry.retry_not_before_ms == 401_000

        replacement_attempt = replacement.claim_next()
        assert replacement_attempt is not None
        assert replacement_attempt.attempt_id != failed_attempt.attempt_id
        reclaimed = replacement.task_source.get(failed_attempt.task_cid)
        assert reclaimed is not None
        assert reclaimed.status == "in_progress"
        assert reclaimed.revision == 4
    finally:
        replacement.close()


def test_retry_reconciliation_reuses_attempt_bound_queue_after_cas_crash(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:queue-cas-crash",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        queue_reason = (
            f"database_portal_retry:{failed_attempt.attempt_id}:typed_deferral"
        )
        daemon.task_source.record_queue_backoff(
            task_cid=failed_attempt.task_cid,
            delay_ms=300_000,
            reason=queue_reason,
        )
        before = daemon.task_source.get_queue_entry(failed_attempt.task_cid)
        assert before is not None

        # Simulate restart after queue commit but before control CAS.
        now["ms"] = 2_000
        reconciliations = daemon.reconcile_terminal_retry_states()
        assert len(reconciliations) == 1
        assert reconciliations[0]["queue_reused"] is True
        after = daemon.task_source.get_queue_entry(failed_attempt.task_cid)
        assert after is not None
        assert after.attempt == before.attempt
        assert after.retry_not_before_ms == before.retry_not_before_ms
        task = daemon.task_source.get(failed_attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert task.revision == 3
    finally:
        daemon.close()


def test_retry_reconciliation_repairs_retrying_without_queue(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:retrying-without-queue",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        control = daemon.task_source.get(failed_attempt.task_cid)
        assert control is not None
        daemon._cas_task_status_database(
            failed_attempt.task_cid,
            expected_revision=int(control.revision),
            new_status="retrying",
            receipt={"operation": "simulated_cas_before_queue_crash"},
        )
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None

        repaired = daemon.reconcile_terminal_retry_states()

        assert len(repaired) == 1
        assert repaired[0]["status"] == "retrying"
        assert repaired[0]["changed"] is False
        entry = daemon.task_source.get_queue_entry(failed_attempt.task_cid)
        assert entry is not None
        assert entry.retry_not_before_ms == 301_000
    finally:
        daemon.close()


def test_retry_reconciliation_rejects_superseded_coordination_fence(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:superseded-retry-fence",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        failed_attempt = daemon.claim_next()
        assert failed_attempt is not None
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )
        old_claim = daemon.coordinator.get_task_claim(failed_attempt.claim_id)
        assert old_claim is not None
        now["ms"] = 7_000
        daemon.coordinator.expire_task_claim(old_claim, now_ms=now["ms"])
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:newer-fence",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.fencing_token > failed_attempt.fencing_token

        with pytest.raises(DatabaseCoordinationError):
            daemon.reconcile_terminal_retry_states()

        control = daemon.task_source.get(failed_attempt.task_cid)
        assert control is not None
        assert control.status == "in_progress"
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_retry_reconciliation_rejects_manual_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:manual-retry-rejection",
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["completion"] = "manual"
        daemon.materialize_population(population)
        control = daemon.task_source.get("task:cid:001")
        assert control is not None
        claim = daemon.coordinator.claim_ready_task(
            owner_session_id=daemon.owner_session_id,
            lease_ms=daemon.lease_ms,
            now_ms=daemon._now_ms(),
        )
        assert claim is not None
        daemon._protect_new_claim(claim)
        daemon._cas_task_status_database(
            control.task_cid,
            expected_revision=int(control.revision),
            new_status="in_progress",
            receipt={"operation": "simulated_legacy_manual_claim"},
        )
        failed_attempt = daemon._insert_attempt_from_claim(
            claim,
            task_alias=control.task_alias,
        )
        failed_attempt = daemon.commit_phase(failed_attempt, "context")
        failed_attempt = daemon.commit_phase(
            failed_attempt,
            "failed",
            body={
                "reason": "typed_deferral",
                "portal_retryable_failure": True,
                "backoff_seconds": 300,
            },
        )

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="manual/review-only",
        ):
            daemon.reconcile_terminal_retry_states()

        unchanged = daemon.task_source.get(failed_attempt.task_cid)
        assert unchanged is not None
        assert unchanged.status == "in_progress"
        assert daemon.task_source.get_queue_entry(failed_attempt.task_cid) is None
    finally:
        daemon.close()


def test_provider_result_is_rejected_after_fenced_takeover(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    holder: dict[str, DatabaseImplementationDaemon] = {}
    replacement_claim_ids: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        daemon = holder["daemon"]
        # Cross the renewed deadline and let another session claim the same
        # ready coordination task before this provider result is returned.
        now["ms"] = 7_000
        replacement = daemon.coordinator.claim_ready_task(
            owner_session_id="session:replacement",
            lease_ms=5_000,
            now_ms=now["ms"],
        )
        assert replacement is not None
        assert replacement.task_cid == attempt.task_cid
        replacement_claim_ids.append(replacement.claim_id)
        return {
            "status": "succeeded",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    daemon = _open_daemon(
        tmp_path,
        session="session:stale-provider",
        provider_fn=provider,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(attempt, "context")
        with pytest.raises(DatabaseCoordinationError):
            daemon.run_provider(attempt)
        assert replacement_claim_ids
        assert (
            daemon.provider_invocation_recorded(
                attempt.attempt_id,
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            is None
        )
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "context"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_expired_attempt_cannot_commit_logical_completion(tmp_path: Path) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:expired-completion",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)
        now["ms"] = 6_000
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "a" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        assert daemon.coordinator.claimability(attempt.task_cid)["claimable"] is True
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.committed_phase == "validation"
        assert stored.status == "running"
    finally:
        daemon.close()


def test_restart_retires_prepared_absent_expired_attempt_then_refences_retry(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        old_attempt = first.claim_next()
        assert old_attempt is not None
        old_attempt = first.commit_phase(old_attempt, "context")
        old_attempt, _, duplicated = first.run_provider(old_attempt)
        assert duplicated is False
        old_owner = first.owner_session_id
    finally:
        first.close()

    # No intervening coordinator mutation performs an expiry sweep.
    now["ms"] = 7_000
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        assert replacement.owner_session_id == old_owner
        result = replacement.run_once()
        reconciliations = result["expired_attempt_reconciliations"]
        assert len(reconciliations) == 1
        assert reconciliations[0]["status"] == "expired"
        assert reconciliations[0]["provider_evidence_reused"] is False
        assert reconciliations[0]["effect_evidence_reused"] is False
        assert result["attempt_id"] != old_attempt.attempt_id
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [old_attempt.task_cid, old_attempt.task_cid]
        assert effect_calls == [old_attempt.task_cid]
        retired = replacement.get_attempt(old_attempt.attempt_id)
        assert retired is not None
        assert retired.status == "failed"
        assert retired.committed_phase == "failed"
        replacement_claim = replacement.coordinator.get_task_claim(
            result["claim_id"]
        )
        assert replacement_claim is not None
        assert replacement_claim.fencing_token > old_attempt.fencing_token
    finally:
        replacement.close()


def test_completed_control_cas_is_recovered_from_prepared_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry after control CAS cannot expose an uncoordinated completion."""

    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-recovery",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_complete = daemon.coordinator.complete_task_claim

        def expire_at_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            kwargs["now_ms"] = now["ms"]
            return original_complete(*args, **kwargs)

        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            expire_at_promotion,
        )
        with pytest.raises(DatabaseCoordinationError):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "b" * 64,
                    "argv": ["focused-validation"],
                },
            )

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        assert task.revision == 3
        readiness = daemon.coordinator.claimability(attempt.task_cid)
        assert readiness["claimable"] is False
        assert readiness["completion_status"] == "prepared"
        prepared = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert prepared is not None
        assert prepared["attempt_id"] == attempt.attempt_id
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        # Restore the ordinary method.  The next pass proves the exact control
        # receipt, promotes and settles the expired preparation, and repairs
        # the execution projection without rerunning provider/effect work.
        monkeypatch.setattr(
            daemon.coordinator,
            "complete_task_claim",
            original_complete,
        )
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = daemon.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        assert recovered.committed_phase == ATTEMPT_PHASE_COMPLETE
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
    finally:
        daemon.close()


def test_restart_recovers_prepared_control_completion_without_prior_expiry_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = first.commit_phase(attempt, phase)

        def crash_before_promotion(*args: object, **kwargs: object) -> object:
            now["ms"] = 7_000
            raise RuntimeError("simulated crash before coordination promotion")

        monkeypatch.setattr(
            first.coordinator,
            "complete_task_claim",
            crash_before_promotion,
        )
        with pytest.raises(RuntimeError, match="before coordination promotion"):
            first.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "e" * 64,
                    "argv": ["focused-validation"],
                },
            )
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "completed"
        unswept = first.coordinator.get_task_claim(attempt.claim_id)
        assert unswept is not None
        assert unswept.state.value == "accepted"
    finally:
        first.close()

    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["recovered"] is True
        recovered = replacement.get_attempt(attempt.attempt_id)
        assert recovered is not None
        assert recovered.status == "succeeded"
        claim = replacement.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "completed"
        assert provider_calls == []
        assert effect_calls == []
    finally:
        replacement.close()


def test_promoted_completion_replays_after_local_phase_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:promotion-replay",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_commit_phase = daemon.commit_phase

        def lose_local_complete(
            current: DatabaseTaskAttempt | str,
            phase: str,
            **kwargs: object,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_COMPLETE:
                raise RuntimeError("simulated local COMPLETE outage")
            return original_commit_phase(current, phase, **kwargs)

        monkeypatch.setattr(daemon, "commit_phase", lose_local_complete)
        with pytest.raises(RuntimeError, match="simulated local COMPLETE outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "d" * 64,
                    "argv": ["focused-validation"],
                },
            )
        promoted = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert promoted is not None
        assert promoted["status"] == "succeeded"
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert claim.state.value == "accepted"
        stored = daemon.get_attempt(attempt.attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"

        monkeypatch.setattr(daemon, "commit_phase", original_commit_phase)
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert result["implementation_result"] is None
        assert len(result["completion_reconciliations"]) == 1
        repaired = daemon.get_attempt(attempt.attempt_id)
        assert repaired is not None
        assert repaired.status == "succeeded"
        assert repaired.committed_phase == ATTEMPT_PHASE_COMPLETE
        assert provider_calls == []
        assert effect_calls == []
        settled = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert settled is not None
        assert settled.state.value == "released"
    finally:
        daemon.close()


def test_expired_preparation_without_control_cas_is_aborted_and_requeued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:prepared-abort",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        for phase in ("context", "provider", "effect", "validation"):
            attempt = daemon.commit_phase(attempt, phase)

        original_cas = daemon._cas_task_status_database

        def reject_control_completion(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated control CAS outage")

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            reject_control_completion,
        )
        with pytest.raises(RuntimeError, match="simulated control CAS outage"):
            daemon.complete_attempt(
                attempt,
                validation_result={
                    "outcome": "passed",
                    "evidence_digest": "sha256:" + "c" * 64,
                    "argv": ["focused-validation"],
                },
            )
        assert daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"

        monkeypatch.setattr(daemon, "_cas_task_status_database", original_cas)
        now["ms"] = 7_000
        result = daemon.run_once()
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "aborted"
        assert result["implementation_result"]["status"] == "succeeded"
        assert result["attempt_id"] != attempt.attempt_id
        old_attempt = daemon.get_attempt(attempt.attempt_id)
        assert old_attempt is not None
        assert old_attempt.status == "failed"
        final_completion = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert final_completion is not None
        assert final_completion["status"] == "succeeded"
        assert final_completion["attempt_id"] == result["attempt_id"]
        completed = daemon.task_source.get(attempt.task_cid)
        assert completed is not None
        assert completed.status == "completed"
    finally:
        daemon.close()


def test_task_claim_settlement_authority_loss_is_not_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:settlement-loss")
    try:
        daemon.materialize_population(_population(1))

        def reject_settlement(*args: object, **kwargs: object) -> object:
            raise DatabaseCoordinationError("simulated settlement authority loss")

        monkeypatch.setattr(
            daemon.coordinator,
            "settle_task_claim",
            reject_settlement,
        )
        with pytest.raises(
            DatabaseCoordinationError,
            match="simulated settlement authority loss",
        ):
            daemon.run_once()
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("restart_ms", "expected_claim_state"),
    ((2_000, "released"), (7_000, "completed")),
)
def test_restart_settles_promoted_completion_after_local_complete_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restart_ms: int,
    expected_claim_state: str,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))

        def crash_before_settlement(*args: object, **kwargs: object) -> object:
            raise RuntimeError("simulated crash before claim settlement")

        monkeypatch.setattr(
            first.coordinator,
            "settle_task_claim",
            crash_before_settlement,
        )
        with pytest.raises(RuntimeError, match="before claim settlement"):
            first.run_once()
        row = first._require_connection().execute(
            """
            SELECT attempt_id, claim_id FROM database_task_attempts
            WHERE status = 'succeeded'
            """
        ).fetchone()
        assert row is not None
        attempt_id, claim_id = str(row[0]), str(row[1])
        unsettled = first.coordinator.get_task_claim(claim_id)
        assert unsettled is not None
        assert unsettled.state.value == "accepted"
    finally:
        first.close()

    now["ms"] = restart_ms
    replacement = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        result = replacement.run_once()
        assert result["unchanged"] is False
        assert result["write_count"] == 1
        assert len(result["completion_reconciliations"]) == 1
        assert result["completion_reconciliations"][0]["status"] == "succeeded"
        settled = replacement.coordinator.get_task_claim(claim_id)
        assert settled is not None
        assert settled.state.value == expected_claim_state
        local = replacement.get_attempt(attempt_id)
        assert local is not None
        assert local.status == "succeeded"
        assert replacement.coordinator.list_unsettled_task_completions() == []
        assert provider_calls == ["task:cid:001"]
        assert effect_calls == ["task:cid:001"]
    finally:
        replacement.close()


def test_automatic_run_once_never_claims_manual_or_review_only_task(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        population = _population(2)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["completion"] = "manual"
        tasks[1]["review_only"] = True
        daemon.materialize_population(population)
        result = daemon.run_once()
        assert result["unchanged"] is True
        assert result["selection_idle_reason"] == "no_ready_tasks"
        assert daemon.list_running_attempts() == []
        assert daemon.coordinator.get_task_claim("claim:missing") is None
        for task_cid in ("task:cid:001", "task:cid:002"):
            task = daemon.task_source.get(task_cid)
            assert task is not None
            assert task.status == "ready"

        # The coordinator still exposes the task to a separately authorized
        # trusted manual-seal path; only automatic daemon dispatch is excluded.
        direct = daemon.coordinator.claim_task(
            task_cid="task:cid:001",
            owner_session_id="session:trusted-manual-seal",
            now_ms=daemon._now_ms(),
        )
        assert direct.task_cid == "task:cid:001"
    finally:
        daemon.close()


def test_parse_args_accepts_database_authority_flags() -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            "/tmp/control.duckdb",
            "--owner-session-id",
            "session:cli",
            "--once",
        ]
    )
    assert args.task_source_kind == "duckdb"
    assert args.authority_mode == "embedded"
    assert Path(args.database_path) == Path("/tmp/control.duckdb")
    assert args.owner_session_id == "session:cli"
    paths = resolve_database_implementation_paths(args)
    assert paths["database_path"] == Path("/tmp/control.duckdb")


def test_runner_builds_database_daemon_without_json_projections(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "unused.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon = build_database_implementation_daemon_from_args(
        args,
        owner_session_id="session:runner",
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.state_path is None
        assert daemon.events_path is None
        assert daemon.max_task_attempts == 3
        assert daemon.projections_required() is False
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["authority_mode"] == "embedded"
        assert result["markdown_status_writes"] == 0
        assert result["unchanged"] is True
        assert result["write_count"] == 0
        assert result["execution_authorized"] is False
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "ready"
        assert daemon.list_running_attempts() == []
    finally:
        daemon.close()


def test_portal_builder_with_inherited_database_program_without_implement_is_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A replacement child must not infer execution authority from its DB env."""

    monkeypatch.chdir(tmp_path)
    program = DatabaseProgramConfig(
        authority_mode="embedded",
        task_source_kind="duckdb",
        store_id="control.duckdb",
        store_generation="generation-reload-regression",
        schema_revision="reload-regression-v1",
    )
    monkeypatch.setenv(
        DATABASE_PROGRAM_JSON_ENV,
        json.dumps(program.to_dict(), separators=(",", ":"), sort_keys=True),
    )
    args = parse_args(
        [
            *program.daemon_cli_args(),
            "--todo-path",
            str(tmp_path / "wrong-default-board.md"),
            "--state-dir",
            str(tmp_path / "wrong-default-state"),
            "--state-prefix",
            "wrong-default",
            "--once",
            # Deliberately no --implement: this is the failed reload shape.
        ]
    )
    bind_results: list[object | None] = []
    real_bind = daemon_runner.bind_database_portal_execution_from_args

    def record_bind(*bind_args: object, **bind_kwargs: object) -> object | None:
        result = real_bind(*bind_args, **bind_kwargs)
        bind_results.append(result)
        return result

    monkeypatch.setattr(
        daemon_runner,
        "bind_database_portal_execution_from_args",
        record_bind,
    )
    daemon, _context = daemon_runner.build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.require_real_execution is False
        assert daemon.execution_callbacks_bound is False
        assert bind_results == [None]
        daemon.materialize_population(_population(1))
        before = daemon.task_source.get("task:cid:001")
        assert before is not None

        result = daemon.run_once()

        after = daemon.task_source.get("task:cid:001")
        assert after is not None
        assert result["execution_authorized"] is False
        assert result["write_count"] == 0
        assert result["selection_idle_reason"] == (
            "database_execution_not_authorized"
        )
        assert after.to_dict() == before.to_dict()
        assert after.status == "ready"
        assert daemon.list_running_attempts() == []
        counts = daemon._require_connection().execute(
            """
            SELECT
                (SELECT COUNT(*) FROM database_task_attempts),
                (SELECT COUNT(*) FROM provider_invocations),
                (SELECT COUNT(*) FROM effect_claims)
            """
        ).fetchone()
        assert tuple(int(counts[index]) for index in range(3)) == (0, 0, 0)
    finally:
        daemon.close()


def test_runner_portal_builder_selects_database_daemon(tmp_path: Path) -> None:
    database_path = tmp_path / "control.duckdb"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(database_path),
            "--todo-path",
            str(tmp_path / "board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "dqp",
            "--max-task-attempts",
            "4",
            "--implement",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.max_task_attempts == 4
        assert context.state_path.name.startswith("dqp_")
        daemon.materialize_population(_population(2))
        first = daemon.claim_next()
        second = daemon.claim_next()
        # Single session claims one at a time via claim_ready; second claim is
        # a different task while the first remains leased.
        assert first is not None
        assert second is not None
        assert first.task_cid != second.task_cid
    finally:
        daemon.close()
