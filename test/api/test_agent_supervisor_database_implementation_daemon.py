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

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceConflictError as DatabaseTaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
    DatabasePortalBridgeConsumedNoProgressError,
    DatabasePortalBridgeDeferred,
    database_portal_consumed_no_progress_fingerprint,
    database_portal_task_contract_digest,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_BLOCKED,
    ATTEMPT_PHASE_COMPLETE,
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_FAILED,
    ATTEMPT_PHASE_PROVIDER,
    ATTEMPT_PHASE_VALIDATION,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA,
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


def _apmc_bootstrap_frontier_population() -> dict[str, object]:
    completed = tuple(f"APMC-{index:03d}" for index in range(6)) + ("APMC-018",)
    dependencies = {
        "APMC-001": ("APMC-000",),
        "APMC-002": ("APMC-001",),
        "APMC-003": ("APMC-001",),
        "APMC-004": ("APMC-001", "APMC-003"),
        "APMC-005": ("APMC-002", "APMC-004"),
        "APMC-018": ("APMC-000",),
        "APMC-006": ("APMC-001",),
        "APMC-012": ("APMC-002", "APMC-005"),
        "APMC-014": ("APMC-002", "APMC-004"),
    }

    def task(task_id: str, *, ordinal: int, status: str) -> dict[str, object]:
        return {
            "task_cid": f"task:cid:{task_id}",
            "task_id": task_id,
            "goal_cid": "goal:cid:apmc",
            "status": status,
            "priority": "P0",
            "ordinal": ordinal,
            "title": task_id,
            "dependencies": [
                f"task:cid:{dependency}"
                for dependency in dependencies.get(task_id, ())
            ],
        }

    frontier = ("APMC-006", "APMC-012", "APMC-014")
    return {
        "repository_tree_id": "tree:apmc-qualified-bootstrap",
        "objectives": [
            {
                "objective_id": "APMC-G000",
                "objective_alias": "APMC-G000",
                "title": "Autonomous meta-controller",
                "goal_cid": "goal:cid:apmc",
                "goal_alias": "APMC-G000",
                "status": "open",
            }
        ],
        "tasks": [
            *(
                task(task_id, ordinal=index, status="completed")
                for index, task_id in enumerate(completed, start=1)
            ),
            *(
                task(task_id, ordinal=index, status="ready")
                for index, task_id in enumerate(frontier, start=len(completed) + 1)
            ),
        ],
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
    clock_ms: Callable[[], int] | None = None,
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
    control_path: Path | None = None,
) -> DatabaseImplementationDaemon:
    database_path = control_path or (tmp_path / "control.duckdb")
    coordination_path = tmp_path / "coordination.duckdb"
    execution_path = tmp_path / "execution.duckdb"

    def default_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if provider_calls is not None:
            provider_calls.append(attempt.task_cid)
        return {"status": "ok", "task_cid": attempt.task_cid}

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
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        provider_fn=provider_fn or default_provider,
        effect_fn=effect,
        clock_ms=clock_ms,
    )


def _alias_home(task_alias: str, shard_count: int) -> int:
    digest = hashlib.sha256(task_alias.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % shard_count


def _consumed_no_progress_evidence(
    daemon: DatabaseImplementationDaemon,
    attempt: DatabaseTaskAttempt,
    *,
    tag: str,
) -> dict[str, object]:
    task = daemon.task_source.get(attempt.task_cid)
    assert task is not None
    snapshot = daemon.task_source.snapshot()
    evidence: dict[str, object] = {
        "schema": DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
        "failure_kind": "consumed_no_progress",
        "diagnostic_failure_id": f"baguq-failure-{tag}",
        "diagnostic_receipt_id": f"baguq-diagnostic-{tag}",
        "diagnostic_receipt_digest": "sha256:" + "c" * 64,
        "diagnostic_receipt_size": 512,
        "context_receipt_id": f"baguq-context-{tag}",
        "context_receipt_digest": "sha256:" + "d" * 64,
        "context_receipt_size": 1024,
        "log_digest": "sha256:" + hashlib.sha256(tag.encode()).hexdigest(),
        "log_size": len(tag.encode()),
        "repository_id": "repository:database-daemon-test",
        "tree_id": "tree:portal-baseline",
        "control_repository_tree_id": snapshot.repository_tree_id,
        "task_cid": attempt.task_cid,
        "task_contract_digest": database_portal_task_contract_digest(
            task,
        ),
        "database_binding_id": "sha256:" + "b" * 64,
        "database_attempt_id": attempt.attempt_id,
        "database_claim_id": attempt.claim_id,
        "database_lease_id": attempt.lease_id,
        "database_fencing_token": int(attempt.fencing_token),
        "database_fence_epoch": int(attempt.fence_epoch),
        "portal_task_id": attempt.task_alias,
        "portal_attempt_number": 1,
        "returncode": 1,
        "attempt_consumed": True,
        "portal_provider_dispatched": True,
        "provider_effect_state": "unknown_may_have_started",
        "implementation_commit_present": False,
        "implementation_candidate_present": False,
        "validation_state": "not_run",
    }
    evidence["failure_fingerprint"] = (
        database_portal_consumed_no_progress_fingerprint(evidence)
    )
    return evidence


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


def test_strict_database_lane_claims_only_alias_hash_home_tasks(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:strict-lane-0",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(8))
        claimed_aliases: list[str] = []
        while True:
            attempt = daemon.claim_next()
            if attempt is None:
                break
            claimed_aliases.append(attempt.task_alias)

        expected = {
            f"DQP-T{index:03d}"
            for index in range(1, 9)
            if _alias_home(f"DQP-T{index:03d}", 2) == 0
        }
        assert set(claimed_aliases) == expected
        assert claimed_aliases
        assert all(_alias_home(alias, 2) == 0 for alias in claimed_aliases)
    finally:
        daemon.close()


def test_non_strict_database_lane_preserves_cross_shard_claiming(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:non-strict-lane-0",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_alias == "DQP-T001"
        assert _alias_home(attempt.task_alias, 2) == 1
    finally:
        daemon.close()


def test_strict_restart_resumes_exact_in_home_claim(
    tmp_path: Path,
) -> None:
    first = _open_daemon(
        tmp_path,
        session="session:strict-in-home",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(3))
        attempt = first.claim_next()
        assert attempt is not None
        assert attempt.task_alias == "DQP-T003"
        assert _alias_home(attempt.task_alias, 2) == 0
    finally:
        first.close()

    provider_calls: list[str] = []
    restarted = _open_daemon(
        tmp_path,
        session="session:strict-in-home",
        provider_calls=provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()["implementation_result"]
        assert result["resumed"] is True
        assert result["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
    finally:
        restarted.close()


def test_strict_restart_requeues_pre_provider_out_of_home_attempt(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        assert attempt.task_alias == "DQP-T001"
        assert _alias_home(attempt.task_alias, 2) == 1
    finally:
        first.close()

    restart_provider_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0",
        provider_calls=restart_provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()
        implementation = result["implementation_result"]
        assert implementation["reason"] == "strict_resume_not_admitted"
        assert implementation["task_requeued"] is True
        assert implementation["task_quarantined"] is False
        assert restart_provider_calls == []
        failed = restarted.get_attempt(attempt.attempt_id)
        assert failed is not None
        assert failed.status == "failed"
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "ready"
        assert task.body["completion_receipt"]["operation"] == (
            "database_strict_resume_requeue"
        )
        claim = restarted.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        assert str(getattr(claim.state, "value", claim.state)) == "released"
    finally:
        restarted.close()

    home = _open_daemon(
        tmp_path / "lane-1",
        control_path=control_path,
        session="session:lane-1",
        task_shard_count=2,
        task_shard_index=1,
        strict_task_sharding=True,
    )
    try:
        admitted = home.claim_next()
        assert admitted is not None
        assert admitted.task_cid == attempt.task_cid
        assert admitted.task_alias == "DQP-T001"
    finally:
        home.close()


def test_strict_restart_quarantines_effect_committed_out_of_home_attempt(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-effect",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        current = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"test": "strict-restart"},
        )
        current, provider_result, _duplicated = first.run_provider(current)
        current, _effect_result, _duplicated = first.run_effect(
            current,
            provider_result,
        )
        assert current.committed_phase == "effect"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == [attempt.task_cid]
    finally:
        first.close()

    restart_provider_calls: list[str] = []
    restart_effect_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-effect",
        provider_calls=restart_provider_calls,
        effect_calls=restart_effect_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()
        implementation = result["implementation_result"]
        assert implementation["reason"] == "strict_resume_not_admitted"
        assert implementation["task_requeued"] is False
        assert implementation["task_quarantined"] is True
        assert restart_provider_calls == []
        assert restart_effect_calls == []
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        assert task.body["completion_receipt"]["operation"] == (
            "database_strict_resume_quarantine"
        )
        assert restarted.claim_next() is None
    finally:
        restarted.close()


@pytest.mark.parametrize(
    "provider_idempotency_key",
    ["", "provider:custom-crash-key"],
    ids=["canonical-key", "custom-key"],
)
def test_strict_restart_quarantines_provider_receipt_before_phase_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_idempotency_key: str,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane-0"
    provider_calls: list[str] = []
    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-provider-crash",
        provider_calls=provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=False,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        current = first.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"test": "provider-receipt-crash"},
        )
        original_commit_phase = first.commit_phase

        def crash_before_provider_phase(
            current_attempt: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_PROVIDER:
                raise RuntimeError("injected crash after provider receipt")
            return original_commit_phase(current_attempt, phase, body=body)

        monkeypatch.setattr(first, "commit_phase", crash_before_provider_phase)
        with pytest.raises(
            RuntimeError,
            match="injected crash after provider receipt",
        ):
            first.run_provider(
                current,
                idempotency_key=provider_idempotency_key,
            )
        persisted = first.get_attempt(attempt.attempt_id)
        assert persisted is not None
        assert persisted.committed_phase == ATTEMPT_PHASE_CONTEXT
        recorded_key = (
            provider_idempotency_key
            or f"provider:{attempt.attempt_id}"
        )
        assert first.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=recorded_key,
        ) is not None
        assert first.provider_invocation_exists(attempt.attempt_id) is True
        assert provider_calls == [attempt.task_cid]
    finally:
        first.close()

    restart_provider_calls: list[str] = []
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:lane-0-provider-crash",
        provider_calls=restart_provider_calls,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        result = restarted.run_once()["implementation_result"]
        assert result["reason"] == "strict_resume_not_admitted"
        assert result["task_requeued"] is False
        assert result["task_quarantined"] is True
        assert restart_provider_calls == []
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == "database_strict_resume_quarantine"
        assert receipt["provider_phase_committed"] is False
        assert receipt["provider_invocation_receipt_present"] is True
    finally:
        restarted.close()


@pytest.mark.parametrize("raced_alias", ["", "DQP-T001", "DQP-T004"])
def test_strict_database_lane_rechecks_authoritative_alias_after_local_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raced_alias: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:strict-race",
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(3))
        original_projection = daemon._stable_authoritative_task_projection
        reads = 0

        def raced_projection() -> tuple[tuple[object, ...], frozenset[str]]:
            nonlocal reads
            tasks, ready_cids = original_projection()
            reads += 1
            if reads < 3:
                return tasks, ready_cids
            return (
                tuple(
                    replace(task, task_alias=raced_alias)
                    if task.task_alias == "DQP-T003"
                    else task
                    for task in tasks
                ),
                ready_cids,
            )

        monkeypatch.setattr(
            daemon,
            "_stable_authoritative_task_projection",
            raced_projection,
        )

        assert daemon.claim_next() is None
        assert reads >= 3
        task = daemon.task_source.get_task("task:cid:003")
        assert task is not None
        assert task.status == "ready"
        projection = daemon.coordinator.coordination_registry_projection()
        claim = next(
            row
            for row in projection["task_claims"]
            if row["task_cid"] == "task:cid:003"
        )
        assert claim["state"] == "released"
    finally:
        daemon.close()


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


def test_apmc_bootstrap_completions_unlock_exact_frontier_across_lane_sidecars(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "apmc-control.duckdb"
    seed = DatabaseImplementationDaemon(
        database_path=database_path,
        coordination_path=tmp_path / "seed-coordination.duckdb",
        execution_path=tmp_path / "seed-execution.duckdb",
        owner_session_id="apmc-seed",
        authority_mode="embedded",
        task_source_kind="duckdb",
    )
    try:
        seed.materialize_population(_apmc_bootstrap_frontier_population())
    finally:
        seed.close()

    expected_ready = {
        "task:cid:APMC-006",
        "task:cid:APMC-012",
        "task:cid:APMC-014",
    }
    claimed: set[str] = set()
    for lane in range(3):
        coordination_path = tmp_path / f"lane-{lane}-coordination.duckdb"
        execution_path = tmp_path / f"lane-{lane}-execution.duckdb"
        daemon = DatabaseImplementationDaemon(
            database_path=database_path,
            coordination_path=coordination_path,
            execution_path=execution_path,
            owner_session_id=f"apmc-lane-{lane}",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
        try:
            ready = set(daemon.sync_ready_tasks_into_coordination())
            assert ready == expected_ready - claimed
            for task_cid in ready:
                assert daemon.coordinator.claimability(task_cid)["claimable"] is True
            if lane == 0:
                first_projection = daemon.coordinator.coordination_registry_projection()
                assert set(daemon.sync_ready_tasks_into_coordination()) == ready
                assert (
                    daemon.coordinator.coordination_registry_projection()
                    == first_projection
                )
        finally:
            daemon.close()

        # Reopening the exact lane sidecars is an idempotent projection replay.
        daemon = DatabaseImplementationDaemon(
            database_path=database_path,
            coordination_path=coordination_path,
            execution_path=execution_path,
            owner_session_id=f"apmc-lane-{lane}",
            authority_mode="embedded",
            task_source_kind="duckdb",
        )
        try:
            assert set(daemon.sync_ready_tasks_into_coordination()) == ready
            attempt = daemon.claim_next()
            assert attempt is not None
            assert attempt.task_cid in ready
            claimed.add(attempt.task_cid)
        finally:
            daemon.close()

    assert claimed == expected_ready


def test_removed_authoritative_task_is_excluded_without_idle_growth(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:removed-task")
    authoritative_cid = "task:cid:001"
    removed_cid = "task:cid:removed"
    try:
        daemon.materialize_population(_population(1))
        daemon.coordinator.register_task(
            task_cid=removed_cid,
            task_id="REMOVED",
            body={"status": "ready"},
        )
        assert daemon.coordinator.claimability(removed_cid)["claimable"] is True
        assert daemon.sync_ready_tasks_into_coordination() == [authoritative_cid]
        before = daemon.coordinator.coordination_registry_projection()

        for _pass in range(2):
            assert daemon.claim_next(exclude_task_cids=(authoritative_cid,)) is None
            assert daemon.coordinator.coordination_registry_projection() == before
    finally:
        daemon.close()


def test_portal_deferral_refreshes_failed_revision_and_releases_exact_lease(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []

    def defer_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        if len(provider_calls) == 1:
            raise DatabasePortalBridgeDeferred(
                "validation_project_dependency_preflight_failed"
            )
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-deferral",
        provider_fn=defer_provider,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None

        result = daemon._resume_attempt_without_process_crash(attempt)

        assert provider_calls == [attempt.attempt_id]
        assert result["status"] == "failed"
        assert "fail_error" not in result
        failed = daemon.get_attempt(attempt.attempt_id)
        assert failed is not None
        assert failed.committed_phase == "failed"
        assert failed.status == "failed"
        assert failed.revision > attempt.revision
        projection = daemon.coordinator.coordination_registry_projection()
        assert next(
            row["state"]
            for row in projection["task_claims"]
            if row["claim_id"] == attempt.claim_id
        ) == "released"
        assert next(
            row["status"]
            for row in projection["task_attempts"]
            if row["attempt_id"] == attempt.attempt_id
        ) == "released"
        assert next(
            row["state"]
            for row in projection["fenced_leases"]
            if row["lease_id"] == attempt.lease_id
        ) == "released"

        retry = daemon.claim_next()
        assert retry is not None
        assert retry.task_cid == attempt.task_cid
        assert retry.attempt_number == 2
        resumed = daemon.resume_attempt(retry)
        assert resumed["resumed"] is True
        assert resumed["status"] == "succeeded"
        assert provider_calls == [attempt.attempt_id, retry.attempt_id]
    finally:
        daemon.close()


def test_consumed_no_progress_quarantines_and_abstains_after_restart(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:deterministic-preflight",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempted = first.claim_next()
        assert attempted is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempted,
                tag="preflight-symbol-drift",
            )
        )
        fingerprint = str(failure_evidence["failure_fingerprint"])
        first_result = first._resume_attempt_without_process_crash(attempted)

        assert first_result["status"] == "blocked"
        assert first_result["portal_retryable_failure"] is False
        assert first_result["portal_replay_suppressed"] is True
        assert first_result["task_quarantined"] is True
        assert first_result["root_cause_required"] is True
        assert first_result["failure_fingerprint"] == fingerprint
        assert len(provider_calls) == 1

        blocked = first.get_attempt(attempted.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.committed_phase == ATTEMPT_PHASE_BLOCKED
        assert first.provider_invocation_exists(blocked.attempt_id) is True
        phases = first.phase_history(blocked.attempt_id)
        blocked_phase = next(
            item for item in phases if item["phase"] == ATTEMPT_PHASE_BLOCKED
        )
        assert blocked_phase["body"]["portal_replay_suppressed"] is True
        assert (
            blocked_phase["body"]["failure_evidence"]["failure_fingerprint"]
            == fingerprint
        )

        task = first.task_source.get(blocked.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == (
            "database_portal_neutral_failure_quarantine"
        )
        assert receipt["failure_fingerprint"] == fingerprint
        assert receipt["retry_suppressed"] is True
        assert receipt["circuit_breaker_key"] == fingerprint
        assert receipt["provider_effect_state"] == "unknown_may_have_started"
        intent = first.provider_invocation_recorded(
            blocked.attempt_id,
            idempotency_key=f"provider:{blocked.attempt_id}",
        )
        assert intent is not None
        assert intent["database_binding_id"] == failure_evidence[
            "database_binding_id"
        ]
        assert intent["portal_failure_fingerprint"] == fingerprint
        assert receipt["provider_callback_intent_fingerprint"] == intent[
            "failure_fingerprint"
        ]
        projection = first.coordinator.coordination_registry_projection()
        claims = [
            row
            for row in projection["task_claims"]
            if row["task_cid"] == blocked.task_cid
        ]
        assert len(claims) == 1
        assert claims[0]["state"] == "released"
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:deterministic-preflight",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        assert replay["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [blocked.attempt_id]
        assert restarted.claim_next() is None
        assert restarted.list_running_attempts() == []
        assert restarted.get_attempt(blocked.attempt_id) == blocked
        projection = restarted.coordinator.coordination_registry_projection()
        assert len(
            [
                row
                for row in projection["task_claims"]
                if row["task_cid"] == blocked.task_cid
            ]
        ) == 1
    finally:
        restarted.close()


def test_consumed_no_progress_quarantine_replays_after_commit_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-commit-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempt,
                tag="commit-crash",
            )
        )
        fingerprint = str(failure_evidence["failure_fingerprint"])
        original_commit_phase = first.commit_phase

        def crash_before_blocked_phase(
            current: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_BLOCKED:
                raise RuntimeError("injected crash before blocked phase")
            return original_commit_phase(current, phase, body=body)

        monkeypatch.setattr(first, "commit_phase", crash_before_blocked_phase)
        with pytest.raises(
            RuntimeError,
            match="injected crash before blocked phase",
        ):
            first._resume_attempt_without_process_crash(attempt)

        running = first.get_attempt(attempt.attempt_id)
        assert running is not None and running.status == "running"
        task = first.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        assert task.body["completion_receipt"]["failure_fingerprint"] == (
            fingerprint
        )
        assert first.provider_invocation_exists(attempt.attempt_id) is True
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-commit-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        reconciled = replay["expired_attempt_reconciliations"]
        assert len(reconciled) == 1
        assert reconciled[0]["reason"] == (
            "portal_neutral_failure"
        )
        assert reconciled[0]["disposition"] == "quarantined"
        assert provider_calls == [attempt.attempt_id]
        terminal = restarted.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        assert restarted.claim_next() is None
        claims = [
            row
            for row in restarted.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["task_cid"] == attempt.task_cid
        ]
        assert len(claims) == 1 and claims[0]["state"] == "released"
    finally:
        restarted.close()


def test_cold_restart_rejects_rebound_neutral_receipt_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-neutral-receipt",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempt,
                tag="rebound-neutral-receipt",
            )
        )
        original_commit_phase = first.commit_phase

        def crash_before_blocked_phase(
            current: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase == ATTEMPT_PHASE_BLOCKED:
                raise RuntimeError("injected crash before rebound replay")
            return original_commit_phase(current, phase, body=body)

        monkeypatch.setattr(first, "commit_phase", crash_before_blocked_phase)
        with pytest.raises(
            RuntimeError,
            match="injected crash before rebound replay",
        ):
            first._resume_attempt_without_process_crash(attempt)

        task = first.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        forged_body = dict(task.body)
        forged_receipt = dict(forged_body["completion_receipt"])
        forged_evidence = dict(forged_receipt["failure_evidence"])
        forged_evidence["task_contract_digest"] = "sha256:" + "9" * 64
        forged_evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(
                forged_evidence
            )
        )
        forged_receipt["failure_evidence"] = forged_evidence
        forged_receipt["failure_fingerprint"] = forged_evidence[
            "failure_fingerprint"
        ]
        forged_receipt["circuit_breaker_key"] = forged_evidence[
            "failure_fingerprint"
        ]
        evidence_bytes = json.dumps(
            forged_evidence,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        forged_receipt["failure_evidence_digest"] = (
            "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
        )
        forged_body["completion_receipt"] = forged_receipt
        with first.task_source.intent._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [
                    json.dumps(
                        forged_body,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    ),
                    attempt.task_cid,
                ],
            )
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-neutral-receipt",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        running = restarted.get_attempt(attempt.attempt_id)
        task = restarted.task_source.get(attempt.task_cid)
        assert running is not None and running.status == "running"
        assert task is not None and task.status == "quarantined"
        assert (
            restarted._strict_resume_rejection_receipt_matches(
                task,
                running,
            )
            is False
        )
        assert restarted.reconcile_expired_running_attempts() == []
        claim = restarted.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "accepted"
        assert provider_calls == [attempt.attempt_id]
    finally:
        restarted.close()


def test_neutral_blocked_claim_release_replays_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-blocked-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempt,
                tag="blocked-release-crash",
            )
        )

        def crash_before_exact_release(*args: object, **kwargs: object) -> object:
            raise RuntimeError("injected crash after blocked phase")

        monkeypatch.setattr(
            first.coordinator,
            "release",
            crash_before_exact_release,
        )
        with pytest.raises(
            RuntimeError,
            match="injected crash after blocked phase",
        ):
            first._resume_attempt_without_process_crash(attempt)

        blocked = first.get_attempt(attempt.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.committed_phase == ATTEMPT_PHASE_BLOCKED
        assert first.provider_invocation_exists(attempt.attempt_id) is True
        task = first.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        claims = [
            row
            for row in first.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "accepted"
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:preflight-blocked-crash",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        reconciled = replay["expired_attempt_reconciliations"]
        assert len(reconciled) == 1
        assert reconciled[0]["status"] == "blocked"
        assert reconciled[0]["reason"] == (
            "portal_neutral_failure"
        )
        assert reconciled[0]["disposition"] == "quarantined"
        assert provider_calls == [attempt.attempt_id]
        terminal = restarted.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "blocked"
        claims = [
            row
            for row in restarted.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "released"
        second_replay = restarted.run_once()
        assert second_replay["expired_attempt_reconciliations"] == []
        assert second_replay["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [attempt.attempt_id]
    finally:
        restarted.close()


def test_provider_callback_hard_crash_abstains_after_cold_restart(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise SimulatedProcessCrash("injected hard callback crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None

        with pytest.raises(
            SimulatedProcessCrash,
            match="injected hard callback crash",
        ):
            first._resume_attempt_without_process_crash(attempt)

        current = first.get_attempt(attempt.attempt_id)
        assert current is not None
        assert current.status == "running"
        assert current.committed_phase == ATTEMPT_PHASE_CONTEXT
        intent = first.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent is not None
        assert intent["schema"] == DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA
        assert intent["callback_state"] == "started_outcome_unknown"
        assert intent["provider_effect_state"] == "unknown_may_have_started"
        assert intent["database_binding_id"] == ""
        assert intent["portal_failure_fingerprint"] == ""
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()

        assert replay["expired_attempt_reconciliations"] == []
        result = replay["implementation_result"]
        assert result["status"] == "blocked"
        assert result["reason"] == "portal_neutral_failure"
        assert result["failure_kind"] == "provider_callback_outcome_unknown"
        assert result["portal_replay_suppressed"] is True
        assert provider_calls == [attempt.attempt_id]
        blocked = restarted.get_attempt(attempt.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        assert blocked.committed_phase == ATTEMPT_PHASE_BLOCKED
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["provider_effect_state"] == "unknown_may_have_started"
        assert receipt["failure_kind"] == "provider_callback_outcome_unknown"
        claims = [
            row
            for row in restarted.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "released"
    finally:
        restarted.close()


def test_provider_callback_hard_crash_after_expiry_never_redispatches(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    now = {"ms": 1_000}
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    provider_calls: list[str] = []

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        raise SimulatedProcessCrash("injected expired callback crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:expired-callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(
            SimulatedProcessCrash,
            match="injected expired callback crash",
        ):
            first._resume_attempt_without_process_crash(attempt)
        intent = first.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent is not None
        assert intent["schema"] == DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA
        assert provider_calls == [attempt.attempt_id]
    finally:
        first.close()

    now["ms"] = 7_000
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:expired-callback-hard-crash",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        replay = restarted.run_once()
        reconciled = replay["expired_attempt_reconciliations"]
        assert len(reconciled) == 1
        assert reconciled[0]["reason"] == "portal_neutral_failure"
        assert reconciled[0]["disposition"] == "quarantined"
        assert reconciled[0]["retry_required"] is False
        assert replay["implementation_result"] is None
        assert replay["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [attempt.attempt_id]

        terminal = restarted.get_attempt(attempt.attempt_id)
        assert terminal is not None
        assert terminal.status == "failed"
        assert terminal.committed_phase == ATTEMPT_PHASE_FAILED
        task = restarted.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        receipt = task.body["completion_receipt"]
        assert receipt["failure_kind"] == "provider_callback_outcome_unknown"
        claim = restarted.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None and claim.state.value == "expired"

        second = restarted.run_once()
        assert second["expired_attempt_reconciliations"] == []
        assert second["selection_idle_reason"] == "no_ready_tasks"
        assert provider_calls == [attempt.attempt_id]
    finally:
        restarted.close()


def test_blocked_response_replay_rejects_different_failure_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:blocked-response-mismatch",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="blocked-response-mismatch",
            )
        )
        original_commit_phase = daemon.commit_phase

        def lose_mismatched_blocked_response(
            current: DatabaseTaskAttempt,
            phase: str,
            *,
            body: dict[str, object] | None = None,
        ) -> DatabaseTaskAttempt:
            if phase != ATTEMPT_PHASE_BLOCKED:
                return original_commit_phase(current, phase, body=body)
            forged_body = dict(body or {})
            forged_body["failure_fingerprint"] = "sha256:" + "0" * 64
            original_commit_phase(current, phase, body=forged_body)
            raise RuntimeError("injected lost mismatched blocked response")

        monkeypatch.setattr(
            daemon,
            "commit_phase",
            lose_mismatched_blocked_response,
        )
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="different immutable failure evidence",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        blocked = daemon.get_attempt(attempt.attempt_id)
        assert blocked is not None and blocked.status == "blocked"
        phase = next(
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == ATTEMPT_PHASE_BLOCKED
        )
        assert phase["body"]["failure_fingerprint"] == "sha256:" + "0" * 64
        claims = [
            row
            for row in daemon.coordinator.coordination_registry_projection()[
                "task_claims"
            ]
            if row["claim_id"] == attempt.claim_id
        ]
        assert len(claims) == 1 and claims[0]["state"] == "accepted"
    finally:
        daemon.close()


@pytest.mark.parametrize(
    "terminal_phase",
    [ATTEMPT_PHASE_FAILED, ATTEMPT_PHASE_BLOCKED, ATTEMPT_PHASE_COMPLETE],
)
def test_terminal_phase_evidence_is_immutable(
    tmp_path: Path,
    terminal_phase: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:terminal-immutable:{terminal_phase}",
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        attempt = daemon.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"step": ATTEMPT_PHASE_CONTEXT},
        )
        if terminal_phase == ATTEMPT_PHASE_COMPLETE:
            for phase in (
                ATTEMPT_PHASE_PROVIDER,
                ATTEMPT_PHASE_EFFECT,
                ATTEMPT_PHASE_VALIDATION,
            ):
                attempt = daemon.commit_phase(
                    attempt,
                    phase,
                    body={"step": phase},
                )

        original_body = {
            "failure_fingerprint": "sha256:" + "1" * 64,
            "failure_evidence_digest": "sha256:" + "2" * 64,
        }
        terminal = daemon.commit_phase(
            attempt,
            terminal_phase,
            body=original_body,
        )
        replay = daemon.commit_phase(
            terminal,
            terminal_phase,
            body=dict(original_body),
        )
        assert replay == terminal

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="different immutable evidence",
        ):
            daemon.commit_phase(
                terminal,
                terminal_phase,
                body={
                    **original_body,
                    "failure_evidence_digest": "sha256:" + "3" * 64,
                },
            )

        terminal_rows = [
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == terminal_phase
        ]
        assert len(terminal_rows) == 1
        assert terminal_rows[0]["body"] == original_body
    finally:
        daemon.close()


@pytest.mark.parametrize("succeeded", [False, True], ids=["failed", "complete"])
def test_reconciled_terminal_evidence_is_immutable(
    tmp_path: Path,
    succeeded: bool,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:reconciled-terminal:{succeeded}",
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        prepared = claim.to_dict()
        prepared["preparation_digest"] = "sha256:" + "4" * 64
        reconciliation = {
            "reason": "first-authoritative-reconciliation",
            "evidence_digest": "sha256:" + "5" * 64,
        }

        terminal = daemon._commit_reconciled_attempt_terminal(
            prepared,
            succeeded=succeeded,
            reconciliation=reconciliation,
        )
        assert terminal is not None
        replay = daemon._commit_reconciled_attempt_terminal(
            prepared,
            succeeded=succeeded,
            reconciliation=dict(reconciliation),
        )
        assert replay == terminal

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="different immutable terminal evidence",
        ):
            daemon._commit_reconciled_attempt_terminal(
                prepared,
                succeeded=succeeded,
                reconciliation={
                    **reconciliation,
                    "evidence_digest": "sha256:" + "6" * 64,
                },
            )

        expected_phase = (
            ATTEMPT_PHASE_COMPLETE if succeeded else ATTEMPT_PHASE_FAILED
        )
        rows = [
            item
            for item in daemon.phase_history(attempt.attempt_id)
            if item["phase"] == expected_phase
        ]
        assert len(rows) == 1
        assert rows[0]["body"]["reconciliation"] == reconciliation
    finally:
        daemon.close()


def test_consumed_failure_stale_task_contract_does_not_quarantine(
    tmp_path: Path,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-stale-task-binding",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="stale-task-binding",
            )
        )
        failure_evidence["task_contract_digest"] = "sha256:" + "9" * 64
        failure_evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(
                failure_evidence
            )
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="fresh task-bound evaluation is required",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "in_progress"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase != ATTEMPT_PHASE_BLOCKED
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_consumed_failure_structured_validation_race_does_not_quarantine(
    tmp_path: Path,
) -> None:
    failure_evidence: dict[str, object] = {}
    holder: dict[str, DatabaseImplementationDaemon] = {}
    provider_calls: list[str] = []

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        daemon = holder["daemon"]
        with daemon.task_source.intent._connection(write=True) as connection:
            connection.execute(
                """
                UPDATE task_validations
                SET argv_json = ?
                WHERE task_cid = ? AND ordinal = 0
                """,
                [json.dumps(["pytest", "changed-contract"]), attempt.task_cid],
            )
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:structured-task-contract-race",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    holder["daemon"] = daemon
    try:
        daemon.materialize_population(_population(1))
        with daemon.task_source.intent._connection(write=True) as connection:
            connection.execute(
                """
                INSERT INTO task_validations(
                    task_cid, ordinal, argv_json, policy_json
                ) VALUES (?, 0, ?, ?)
                """,
                [
                    "task:cid:001",
                    json.dumps(["pytest", "original-contract"]),
                    "{}",
                ],
            )
        attempt = daemon.claim_next()
        assert attempt is not None
        task_before = daemon.task_source.get(attempt.task_cid)
        assert task_before is not None
        old_revision = task_before.revision
        old_body = dict(task_before.body)
        assert task_before.validations[0]["argv"] == [
            "pytest",
            "original-contract",
        ]
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="structured-task-contract-race",
            )
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="provider callback intent is stale or rebound",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task_after = daemon.task_source.get(attempt.task_cid)
        assert task_after is not None
        assert task_after.status == "in_progress"
        assert task_after.revision == old_revision
        assert dict(task_after.body) == old_body
        assert task_after.validations[0]["argv"] == [
            "pytest",
            "changed-contract",
        ]
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase == ATTEMPT_PHASE_CONTEXT
        assert provider_calls == [attempt.attempt_id]
    finally:
        daemon.close()


def test_consumed_failure_stale_repository_tree_does_not_quarantine(
    tmp_path: Path,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-stale-tree-binding",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="stale-tree-binding",
            )
        )
        failure_evidence["control_repository_tree_id"] = "tree:stale"
        failure_evidence["failure_fingerprint"] = (
            database_portal_consumed_no_progress_fingerprint(
                failure_evidence
            )
        )

        with pytest.raises(
            DatabaseImplementationConflictError,
            match="fresh task-bound evaluation is required",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase != ATTEMPT_PHASE_BLOCKED
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_consumed_failure_mutated_exception_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    failure_holder: dict[str, Exception] = {}

    def mutated_failure(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise failure_holder["failure"]

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-mutated-evidence",
        provider_fn=mutated_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        evidence = _consumed_no_progress_evidence(
            daemon,
            attempt,
            tag="mutated-after-construction",
        )
        failure = DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=evidence,
        )
        failure.failure_evidence["tree_id"] = "tree:mutated-after-construction"
        failure_holder["failure"] = failure

        with pytest.raises(
            DatabaseImplementationAuthorityError,
            match="failure evidence is invalid",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_neutral_failure_cas_replay_rejects_different_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:preflight-cas-evidence-conflict",
        provider_fn=consumed_failure,
        strict_task_sharding=True,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempt,
                tag="expected-cas-evidence",
            )
        )
        real_cas = daemon._cas_task_status_database

        def conflicting_cas(
            task_cid: str,
            *,
            expected_revision: int,
            new_status: str,
            receipt: object = None,
            evidence_digests: object = None,
        ) -> object:
            assert isinstance(receipt, dict)
            forged_evidence = dict(receipt["failure_evidence"])
            forged_evidence["diagnostic_failure_id"] = "failure:different"
            forged_evidence["diagnostic_receipt_id"] = "diagnostic:different"
            forged_evidence["failure_fingerprint"] = (
                database_portal_consumed_no_progress_fingerprint(
                    forged_evidence
                )
            )
            forged_receipt = dict(receipt)
            forged_receipt["failure_evidence"] = forged_evidence
            forged_receipt["failure_fingerprint"] = forged_evidence[
                "failure_fingerprint"
            ]
            forged_receipt["circuit_breaker_key"] = forged_evidence[
                "failure_fingerprint"
            ]
            evidence_bytes = json.dumps(
                forged_evidence,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            forged_receipt["failure_evidence_digest"] = (
                "sha256:" + hashlib.sha256(evidence_bytes).hexdigest()
            )
            real_cas(
                task_cid,
                expected_revision=expected_revision,
                new_status=new_status,
                receipt=forged_receipt,
                evidence_digests=(
                    str(forged_evidence["failure_fingerprint"]),
                    str(forged_evidence["diagnostic_receipt_digest"]),
                    str(forged_receipt["failure_evidence_digest"]),
                ),
            )
            raise DatabaseTaskSourceConflictError(
                "injected CAS response conflict with different evidence"
            )

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            conflicting_cas,
        )
        with pytest.raises(
            DatabaseTaskSourceConflictError,
            match="different evidence",
        ):
            daemon._resume_attempt_without_process_crash(attempt)

        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "quarantined"
        assert task.body["completion_receipt"]["failure_fingerprint"] != (
            failure_evidence["failure_fingerprint"]
        )
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase == ATTEMPT_PHASE_CONTEXT
        assert daemon.provider_invocation_exists(attempt.attempt_id) is True
    finally:
        daemon.close()


def test_authoritative_dependency_reopen_invalidates_stale_lane_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:dependency-reopen")
    dependency_cid = "task:cid:dependency"
    dependent_cid = "task:cid:dependent"
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:dependency-reopen",
                "objectives": [
                    {
                        "objective_id": "objective:dependency-reopen",
                        "goal_cid": "goal:cid:root",
                        "status": "open",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": dependency_cid,
                        "task_id": "DEP",
                        "goal_cid": "goal:cid:root",
                        "status": "completed",
                        "ordinal": 1,
                    },
                    {
                        "task_cid": dependent_cid,
                        "task_id": "WORK",
                        "goal_cid": "goal:cid:root",
                        "status": "ready",
                        "ordinal": 2,
                        "dependencies": [dependency_cid],
                    },
                ],
            }
        )
        assert daemon.sync_ready_tasks_into_coordination() == [dependent_cid]
        assert daemon.coordinator.claimability(dependent_cid)["claimable"] is True

        real_claim_ready_task = daemon.coordinator.claim_ready_task
        reopened = False

        def claim_then_reopen_dependency(**kwargs: object) -> object:
            nonlocal reopened
            claim = real_claim_ready_task(**kwargs)
            if claim is not None and not reopened:
                dependency = daemon.task_source.get(dependency_cid)
                assert dependency is not None
                daemon.task_source.compare_and_set_status(
                    dependency_cid,
                    expected_revision=int(dependency.revision),
                    status="ready",
                )
                reopened = True
            return claim

        monkeypatch.setattr(
            daemon.coordinator,
            "claim_ready_task",
            claim_then_reopen_dependency,
        )
        assert daemon.claim_next(exclude_task_cids=(dependency_cid,)) is None
        assert reopened is True
        assert daemon.list_running_attempts() == []
        dependent = daemon.task_source.get(dependent_cid)
        assert dependent is not None
        assert dependent.status == "ready"
        assert dependent.revision == 1

        projection = daemon.coordinator.coordination_registry_projection()
        assert {
            (edge["task_cid"], edge["dependency_task_cid"])
            for edge in projection["dependency_edges"]
        } >= {(dependent_cid, dependency_cid)}
        rejected_claims = [
            claim
            for claim in projection["task_claims"]
            if claim["task_cid"] == dependent_cid
        ]
        assert len(rejected_claims) == 1
        assert rejected_claims[0]["state"] == "released"

        assert daemon.sync_ready_tasks_into_coordination() == [dependency_cid]
        blocked = daemon.coordinator.claimability(dependent_cid)
        assert blocked["claimable"] is False
        assert blocked["blocked_dependency_task_cids"] == [dependency_cid]
        assert daemon.claim_next(exclude_task_cids=(dependency_cid,)) is None
    finally:
        daemon.close()


def test_fenced_retry_cannot_bypass_dependency_reopen_after_local_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"ms": 1_000}
    daemon = _open_daemon(
        tmp_path,
        session="session:fenced-retry-dependency",
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    dependency_cid = "task:cid:retry-dependency"
    dependent_cid = "task:cid:retry-dependent"
    try:
        assert daemon._automatic_claim_forbidden(object()) is False
        assert daemon._shared_claim_binding_for_this_owner(object()) is None
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:fenced-retry-dependency",
                "objectives": [
                    {
                        "objective_id": "objective:fenced-retry-dependency",
                        "goal_cid": "goal:cid:root",
                        "status": "open",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": dependency_cid,
                        "task_id": "RETRY-DEP",
                        "goal_cid": "goal:cid:root",
                        "status": "completed",
                        "ordinal": 1,
                    },
                    {
                        "task_cid": dependent_cid,
                        "task_id": "RETRY-WORK",
                        "goal_cid": "goal:cid:root",
                        "status": "ready",
                        "ordinal": 2,
                        "dependencies": [dependency_cid],
                    },
                ],
            }
        )
        first_attempt = daemon.claim_next(exclude_task_cids=(dependency_cid,))
        assert first_attempt is not None
        assert first_attempt.task_cid == dependent_cid
        current = daemon.task_source.get(dependent_cid)
        assert current is not None
        assert current.status == "in_progress"
        assert current.revision == 2

        now["ms"] = 7_000
        real_claim_ready_task = daemon.coordinator.claim_ready_task
        reopened = False

        def retry_then_reopen_dependency(**kwargs: object) -> object:
            nonlocal reopened
            claim = real_claim_ready_task(**kwargs)
            if claim is not None and not reopened:
                assert claim.task_cid == dependent_cid
                assert claim.attempt_number == 2
                dependency = daemon.task_source.get(dependency_cid)
                assert dependency is not None
                daemon.task_source.compare_and_set_status(
                    dependency_cid,
                    expected_revision=int(dependency.revision),
                    status="ready",
                )
                reopened = True
            return claim

        monkeypatch.setattr(
            daemon.coordinator,
            "claim_ready_task",
            retry_then_reopen_dependency,
        )
        assert daemon.claim_next(exclude_task_cids=(dependency_cid,)) is None
        assert reopened is True
        assert [attempt.attempt_id for attempt in daemon.list_running_attempts()] == [
            first_attempt.attempt_id
        ]
        unchanged = daemon.task_source.get(dependent_cid)
        assert unchanged is not None
        assert unchanged.status == "in_progress"
        assert unchanged.revision == 2

        projection = daemon.coordinator.coordination_registry_projection()
        assert {
            (edge["task_cid"], edge["dependency_task_cid"])
            for edge in projection["dependency_edges"]
        } >= {(dependent_cid, dependency_cid)}
        retry_claims = [
            claim
            for claim in projection["task_claims"]
            if claim["task_cid"] == dependent_cid
            and int(claim["attempt_number"]) == 2
        ]
        assert len(retry_claims) == 1
        assert retry_claims[0]["state"] == "released"

        evidence_digest = "sha256:" + "d" * 64
        daemon.task_source.record_validation_result(
            task_cid=dependency_cid,
            outcome="passed",
            evidence_digest=evidence_digest,
            argv=("dependency-recompleted",),
        )
        reopened_dependency = daemon.task_source.get(dependency_cid)
        assert reopened_dependency is not None
        daemon.task_source.compare_and_set_status(
            dependency_cid,
            expected_revision=int(reopened_dependency.revision),
            status="completed",
            evidence_digests=(evidence_digest,),
        )
        converged_retry = daemon.claim_next(exclude_task_cids=(dependency_cid,))
        assert converged_retry is not None
        assert converged_retry.task_cid == dependent_cid
        assert converged_retry.attempt_number == 3
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
        daemon.run_once()
        assert not (tmp_path / "task_state.json").exists()
        assert not (tmp_path / "events.jsonl").exists()
        assert not (tmp_path / "task_queue.json").exists()
        assert not list(tmp_path.glob("*.pid"))
    finally:
        daemon.close()


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
        return {"status": "ok", "task_cid": attempt.task_cid}

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
        return {"status": "ok", "task_cid": attempt.task_cid}

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
        intent = daemon.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert intent is not None
        assert intent["schema"] == DATABASE_PROVIDER_CALLBACK_UNKNOWN_SCHEMA
        assert intent["callback_state"] == "started_outcome_unknown"
        assert intent["provider_effect_state"] == "unknown_may_have_started"
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
            "--task-shard-count",
            "4",
            "--task-shard-index",
            "3",
            "--strict-task-sharding",
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
        assert daemon.projections_required() is False
        assert daemon.task_shard_count == 4
        assert daemon.task_shard_index == 3
        assert daemon.strict_task_sharding is True
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["authority_mode"] == "embedded"
        assert result["markdown_status_writes"] == 0
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("field_name", "malformed", "message"),
    [
        ("task_shard_count", True, "positive integer"),
        ("task_shard_index", False, "range"),
        ("strict_task_sharding", 1, "boolean"),
    ],
)
def test_database_runner_preserves_exact_shard_types_for_constructor_guard(
    tmp_path: Path,
    field_name: str,
    malformed: object,
    message: str,
) -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "unused.md"),
            "--once",
        ]
    )
    setattr(args, field_name, malformed)
    with pytest.raises(ValueError, match=message):
        build_database_implementation_daemon_from_args(args)


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
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "1",
            "--strict-task-sharding",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert context.state_path.name.startswith("dqp_")
        assert daemon.task_shard_count == 2
        assert daemon.task_shard_index == 1
        assert daemon.strict_task_sharding is True
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
