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
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationError,
    open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
    install_datasets_authoritative_operational_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    TaskSourceConflictError as DatabaseTaskSourceConflictError,
    TaskSourceUnknownOutcomeError as DatabaseTaskSourceUnknownOutcomeError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    ATTEMPT_PHASE_COMPLETE,
    ATTEMPT_PHASE_CONTEXT,
    ATTEMPT_PHASE_EFFECT,
    ATTEMPT_PHASE_PROVIDER,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_RETRY_BUDGET_BACKPRESSURE_SCHEMA,
    DATABASE_RETRY_BUDGET_SCHEMA,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT,
    DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    _canonical_mapping_matches,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
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


def _open_daemon(
    tmp_path: Path,
    *,
    session: str = "",
    provider_calls: list[str] | None = None,
    effect_calls: list[str] | None = None,
    markdown_path: Path | None = None,
    provider_fn: Callable[[DatabaseTaskAttempt], dict[str, object]] | None = None,
    effect_fn: Callable[[DatabaseTaskAttempt, object], dict[str, object]] | None = None,
    validation_fn: Callable[
        [DatabaseTaskAttempt, object], dict[str, object]
    ]
    | None = None,
    lease_ms: int = 60_000,
    clock_ms: Callable[[], int] | None = None,
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
    task_prefix: str = "",
    max_task_attempts: int = 0,
    callbacks_bound: bool = True,
) -> DatabaseImplementationDaemon:
    database_path = tmp_path / "control.duckdb"
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
        provider_fn=(provider_fn or default_provider) if callbacks_bound else None,
        effect_fn=(effect_fn or effect) if callbacks_bound else None,
        validation_fn=validation_fn if callbacks_bound else None,
        clock_ms=clock_ms,
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        task_prefix=task_prefix,
        max_task_attempts=max_task_attempts,
    )


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
    finally:
        idle.close()
    assert markdown.read_text(encoding="utf-8") == original_markdown


def test_retry_budget_caps_provider_across_fresh_database_portal_epochs(
    tmp_path: Path,
) -> None:
    provider_attempts: list[str] = []

    def failing_portal_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:bounded-portals",
        provider_fn=failing_portal_provider,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))

        first = daemon.run_once()
        assert first["implementation_result"]["status"] == "failed"
        assert first["implementation_result"]["retry_exhausted"] is False
        first_task = daemon.task_source.get("task:cid:001")
        assert first_task is not None
        assert first_task.status == "retrying"
        assert first_task.body["completion_receipt"]["schema"] == (
            DATABASE_RETRY_BUDGET_SCHEMA
        )
        assert first_task.body["completion_receipt"]["attempts_used"] == 1

        second = daemon.run_once()
        assert second["attempt_id"] != first["attempt_id"]
        assert second["implementation_result"]["status"] == "retry_exhausted"
        assert second["implementation_result"]["retry_exhausted"] is True
        second_task = daemon.task_source.get("task:cid:001")
        assert second_task is not None
        assert second_task.status == "blocked"
        assert second_task.body["completion_receipt"]["attempts_used"] == 2

        backpressure = daemon.run_once()
        assert backpressure["implementation_result"] is None
        assert backpressure["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        assert backpressure["retry_exhausted_task_cids"] == ["task:cid:001"]
        assert backpressure["retry_budget_backpressure"]["schema"] == (
            DATABASE_RETRY_BUDGET_BACKPRESSURE_SCHEMA
        )
        assert backpressure["retry_budget_backpressure"]["tasks"][0][
            "attempts_used"
        ] == 2

        # Each failed database claim gets a fresh private Portal directory in
        # production.  The canonical database receipt, not that disposable
        # directory, owns the total provider budget.
        assert len(provider_attempts) == 2
        assert len(set(provider_attempts)) == 2
    finally:
        daemon.close()


def test_validation_spec_repair_opens_one_fresh_bounded_retry_epoch(
    tmp_path: Path,
) -> None:
    provider_attempts: list[str] = []

    def failing_portal_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-repair",
        provider_fn=failing_portal_provider,
        max_task_attempts=1,
    )
    try:
        daemon.materialize_population(_population(1))
        exhausted = daemon.run_once()
        assert exhausted["implementation_result"]["retry_exhausted"] is True
        assert len(provider_attempts) == 1

        repaired = _population(1)
        repaired_task = repaired["tasks"][0]
        assert isinstance(repaired_task, dict)
        repaired_task["status"] = "retrying"
        repaired_task["validation_commands"] = [
            {"argv": ["python", "-m", "pytest", "fixed_validation.py"]}
        ]
        daemon.materialize_population(repaired)

        retried = daemon.run_once()
        assert retried["implementation_result"]["retry_exhausted"] is True
        assert len(provider_attempts) == 2
        assert provider_attempts[0] != provider_attempts[1]
    finally:
        daemon.close()


def test_retry_budget_is_global_across_lane_local_coordination_stores(
    tmp_path: Path,
) -> None:
    provider_attempts: list[tuple[str, str]] = []

    def failing_portal_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append((attempt.owner_session_id, attempt.attempt_id))
        raise DatabasePortalBridgeError("declared_validation_failed")

    seed = _open_daemon(tmp_path / "seed", session="session:seed")
    lane_coordinators = []
    lanes: list[DatabaseImplementationDaemon] = []
    try:
        seed.materialize_population(_population(1))
        for lane_index in range(2):
            lane_root = tmp_path / f"lane-{lane_index}"
            coordinator = open_database_coordinator(
                lane_root / "coordination.duckdb"
            )
            lane_coordinators.append(coordinator)
            lane = DatabaseImplementationDaemon(
                database_path=seed.database_path,
                coordination_path=lane_root / "coordination.duckdb",
                execution_path=lane_root / "execution.duckdb",
                owner_session_id=f"session:lane-{lane_index}",
                authority_mode="embedded",
                task_source_kind="duckdb",
                task_source=seed.task_source,
                coordinator=coordinator,
                provider_fn=failing_portal_provider,
                max_task_attempts=2,
            )
            lanes.append(lane)

        first = lanes[0].run_once()
        assert first["implementation_result"]["retry_exhausted"] is False
        second = lanes[1].run_once()
        assert second["implementation_result"]["retry_exhausted"] is True

        for lane in lanes:
            idle = lane.run_once()
            assert idle["implementation_result"] is None
            assert idle["selection_idle_reason"] == (
                "all_selectable_ready_tasks_reached_max_task_attempts"
            )
        assert [owner for owner, _attempt_id in provider_attempts] == [
            "session:lane-0",
            "session:lane-1",
        ]
        assert len({attempt_id for _owner, attempt_id in provider_attempts}) == 2
    finally:
        for lane in lanes:
            lane.close()
        for coordinator in lane_coordinators:
            coordinator.close()
        seed.close()


@pytest.mark.parametrize("replacement_cap", [0, 1, 3])
def test_persisted_retry_policy_rejects_lane_cap_mismatch(
    tmp_path: Path,
    replacement_cap: int,
) -> None:
    first_calls: list[str] = []

    def fail_first(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        first_calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    first = _open_daemon(
        tmp_path,
        session="session:policy-origin",
        provider_fn=fail_first,
        max_task_attempts=2,
    )
    try:
        first.materialize_population(_population(1))
        result = first.run_once()
        assert result["implementation_result"]["retry_exhausted"] is False
        persisted = first.task_source.get("task:cid:001")
        assert persisted is not None
        assert persisted.body["completion_receipt"]["max_task_attempts"] == 2
    finally:
        first.close()

    replacement_calls: list[str] = []
    replacement = _open_daemon(
        tmp_path,
        session=f"session:policy-mismatch:{replacement_cap}",
        provider_calls=replacement_calls,
        max_task_attempts=replacement_cap,
    )
    try:
        claims_before = replacement.coordinator.coordination_registry_projection()[
            "task_claim_state_counts"
        ]
        blocked = replacement.run_once()
        assert blocked["implementation_result"] is None
        assert blocked["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        entry = blocked["retry_budget_backpressure"]["tasks"][0]
        assert entry["policy_mismatch"] is True
        assert entry["max_task_attempts"] == 2
        assert entry["configured_max_task_attempts"] == replacement_cap
        assert replacement_calls == []
        assert len(first_calls) == 1
        projection = replacement.coordinator.coordination_registry_projection()
        assert projection["counts"]["active_task_claims"] == 0
        assert projection["task_claim_state_counts"] == claims_before
        repeated = replacement.run_once()
        assert repeated["unchanged"] is True
        assert repeated["write_count"] == 0
        assert (
            replacement.coordinator.coordination_registry_projection()[
                "task_claim_state_counts"
            ]
            == claims_before
        )
    finally:
        replacement.close()


def test_canonical_cross_lane_claim_cas_loss_is_benign_and_never_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_attempts: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:cas-loser",
        provider_calls=provider_attempts,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))

        def lose_canonical_cas(*_args: object, **_kwargs: object) -> object:
            raise DatabaseTaskSourceConflictError("simulated other-lane winner")

        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lose_canonical_cas,
        )
        assert daemon.claim_next() is None
        assert provider_attempts == []
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["counts"]["active_task_claims"] == 0
        assert projection["task_claim_state_counts"] == [
            {"state": "released", "count": 1}
        ]
        assert daemon.list_running_attempts() == []
    finally:
        daemon.close()


def test_identical_materializer_replay_cannot_rearm_exhausted_budget(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:materializer-replay",
        provider_fn=fail,
        max_task_attempts=1,
    )
    population = _population(1)
    try:
        daemon.materialize_population(population)
        daemon.run_once()
        assert len(calls) == 1
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        assert task.status == "blocked"
        assert task.body["completion_receipt"]["attempts_used"] == 1
        idle = daemon.run_once()
        assert idle["implementation_result"] is None
        assert len(calls) == 1
    finally:
        daemon.close()


def test_unreviewed_nested_portal_rearm_evidence_is_claim_fenced() -> None:
    """An empty shortcut identity is invalid, never merely inapplicable."""

    shortcut = SimpleNamespace(
        task_cid="task:cid:pctdd-034",
        revision=9,
        status="retrying",
        body={
            "completion_receipt": {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "nested-portal-rearm-evidence@1"
                ),
                "evidence_id": "",
            }
        },
    )

    assert (
        DatabaseImplementationDaemon._no_provider_rearm_fence_state(shortcut)
        == "invalid"
    )
    assert DatabaseImplementationDaemon._automatic_claim_forbidden(shortcut)

    nested = SimpleNamespace(
        **{
            **vars(shortcut),
            "body": {
                "completion_receipt": {
                    "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                    "operation": DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION,
                    "no_provider_rearm_evidence_id": "",
                    "no_provider_rearm_evidence": {
                        "schema": "nested-portal-rearm-evidence@1",
                        "evidence_id": "",
                    },
                }
            },
        }
    )
    assert (
        DatabaseImplementationDaemon._no_provider_rearm_fence_state(nested)
        == "invalid"
    )
    assert DatabaseImplementationDaemon._automatic_claim_forbidden(nested)


def test_identical_materializer_replay_preserves_live_claim_revision(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:live-materializer-replay",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    population = _population(1)
    try:
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        before = daemon.task_source.get(attempt.task_cid)
        assert before is not None and before.status == "in_progress"
        before_receipt = dict(before.body["completion_receipt"])

        daemon.materialize_population(population)

        after = daemon.task_source.get(attempt.task_cid)
        assert after is not None and after.status == "in_progress"
        assert after.revision == before.revision
        assert after.body["completion_receipt"] == before_receipt
        result = daemon.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [attempt.task_cid]
    finally:
        daemon.close()


def test_materializer_revision_cas_cannot_overwrite_concurrent_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:materializer-claim-race",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        original_upsert = daemon.task_source._intent.upsert_task
        injected: dict[str, DatabaseTaskAttempt] = {}

        def claim_before_stale_upsert(**kwargs: object) -> object:
            if kwargs.get("task_cid") == "task:cid:001" and not injected:
                attempt = daemon.claim_next()
                assert attempt is not None
                injected["attempt"] = attempt
            return original_upsert(**kwargs)

        monkeypatch.setattr(
            daemon.task_source._intent,
            "upsert_task",
            claim_before_stale_upsert,
        )
        changed = _population(1)
        changed_task = changed["tasks"][0]
        assert isinstance(changed_task, dict)
        changed_task["title"] = "Changed during claim race"

        with pytest.raises(DatabaseTaskSourceConflictError):
            daemon.materialize_population(changed)

        attempt = injected["attempt"]
        current = daemon.task_source.get(attempt.task_cid)
        assert current is not None and current.status == "in_progress"
        assert current.body["completion_receipt"]["attempt_id"] == attempt.attempt_id
        assert current.body["completion_receipt"]["attempts_used"] == 1
    finally:
        daemon.close()


def test_identical_singular_validation_replay_preserves_exhausted_budget(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fail(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:singular-validation-replay",
        provider_fn=fail,
        max_task_attempts=1,
    )
    population = _population(1)
    population_task = population["tasks"][0]
    assert isinstance(population_task, dict)
    population_task["validation"] = "python -m pytest singular_validation.py"
    try:
        daemon.materialize_population(population)
        exhausted = daemon.run_once()
        assert exhausted["implementation_result"]["retry_exhausted"] is True
        before = daemon.task_source.get("task:cid:001")
        assert before is not None and before.status == "blocked"
        before_receipt = dict(before.body["completion_receipt"])

        daemon.materialize_population(population)

        after = daemon.task_source.get("task:cid:001")
        assert after is not None and after.status == "blocked"
        assert after.body["completion_receipt"] == before_receipt
        idle = daemon.run_once()
        assert idle["implementation_result"] is None
        assert calls == [before_receipt["attempt_id"]]
    finally:
        daemon.close()


def test_unexpected_provider_exception_never_replays_same_attempt(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def explode(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise RuntimeError("provider exploded")

    daemon = _open_daemon(
        tmp_path,
        session="session:provider-exception",
        provider_fn=explode,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        second = daemon.run_once()
        idle = daemon.run_once()
        assert first["implementation_result"]["retry_exhausted"] is False
        assert second["implementation_result"]["retry_exhausted"] is True
        assert idle["implementation_result"] is None
        assert len(calls) == 2
        assert len(set(calls)) == 2
    finally:
        daemon.close()


def test_effect_exception_is_unknown_and_blocks_without_replay(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def effect(attempt: DatabaseTaskAttempt, _result: object) -> dict[str, object]:
        effect_calls.append(attempt.attempt_id)
        raise RuntimeError("effect outcome unknown")

    daemon = _open_daemon(
        tmp_path,
        session="session:effect-exception",
        provider_calls=provider_calls,
        effect_fn=effect,
        max_task_attempts=3,
    )

    try:
        daemon.materialize_population(_population(1))
        failed = daemon.run_once()
        assert failed["implementation_result"]["retry_exhausted"] is True
        task = daemon.task_source.get("task:cid:001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["forced_block"] is True
        daemon.run_once()
        assert len(provider_calls) == 1
        assert len(effect_calls) == 1
        same_session = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert same_session == []
        assert daemon.task_source.get("task:cid:001").status == "blocked"
    finally:
        daemon.close()


def test_later_session_does_not_rearm_effect_unknown_outcome(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def effect(attempt: DatabaseTaskAttempt, _result: object) -> dict[str, object]:
        effect_calls.append(attempt.attempt_id)
        raise RuntimeError("effect outcome unknown")

    blocker = _open_daemon(
        tmp_path,
        session="session:effect-exception-block",
        provider_calls=provider_calls,
        effect_fn=effect,
        max_task_attempts=3,
    )
    try:
        blocker.materialize_population(_population(1))
        failed = blocker.run_once()
        assert failed["implementation_result"]["retry_exhausted"] is True
        assert blocker.task_source.get("task:cid:001").status == "blocked"
    finally:
        blocker.close()

    successor_calls: list[str] = []
    successor = _open_daemon(
        tmp_path,
        session="session:effect-exception-rearm",
        provider_calls=successor_calls,
        max_task_attempts=3,
    )
    try:
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert rearms == []
        task = successor.task_source.get("task:cid:001")
        assert task is not None and task.status == "blocked"
        receipt = task.body["completion_receipt"]
        assert receipt["operation"] == "database_unknown_outcome_blocked"
        assert receipt["reason"] == "callback_authority_incomplete_blocked"
        assert receipt["retry_exhausted"] is True
        assert successor.reconcile_blocked_unknown_outcome_tasks() == []
        idle = successor.run_once()
        assert idle["implementation_result"] is None
        attempts = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            ["task:cid:001"],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert successor_calls == []
        assert len(provider_calls) == 1
        assert len(effect_calls) == 1
    finally:
        successor.close()


def test_crash_reconciler_preserves_dispatch_process_for_automatic_rearm(
    tmp_path: Path,
) -> None:
    first = _open_daemon(
        tmp_path,
        session="session:unknown-dispatch-crash",
        max_task_attempts=2,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        dispatch_process = first.process_instance_id
        first._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        claimed = first.task_source.get(attempt.task_cid)
        assert claimed is not None and claimed.status == "in_progress"
        assert (
            claimed.body["completion_receipt"]["process_instance_id"]
            == dispatch_process
        )
    finally:
        first.close()

    successor = _open_daemon(
        tmp_path,
        session="session:unknown-dispatch-crash",
        max_task_attempts=2,
    )
    try:
        running = successor.get_attempt(attempt.attempt_id)
        assert running is not None and running.status == "running"
        assert successor.process_instance_id != dispatch_process

        failed, receipt = successor._finalize_failed_attempt(
            running,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )

        assert failed.status == "failed"
        failed_phase = next(
            phase
            for phase in successor.phase_history(failed.attempt_id)
            if phase["phase"] == "failed"
        )
        assert "terminal_reconciliation" not in failed_phase["body"]
        assert receipt["process_instance_id"] == dispatch_process
        assert receipt["reconciled_by_process_instance_id"] == (
            successor.process_instance_id
        )
        blocked = successor.task_source.get(attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"

        rearms = successor.reconcile_blocked_unknown_outcome_tasks()

        assert len(rearms) == 1
        assert rearms[0]["task_cid"] == attempt.task_cid
        rearmed = successor.task_source.get(attempt.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        assert rearmed.body["completion_receipt"]["operation"] == (
            DATABASE_UNKNOWN_OUTCOME_REARM_OPERATION
        )
    finally:
        successor.close()


def test_exact_interrupted_implementation_refunds_attempt_two_without_widening_unknown_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match the live attempt-2/count-1 recovery without a generic rearm."""

    first = _open_daemon(
        tmp_path,
        session="session:interrupted-refund",
        max_task_attempts=2,
    )
    try:
        first.materialize_population(_population(1))
        attempt_one = first.claim_next()
        assert attempt_one is not None
        first._begin_callback_dispatch(
            attempt_one,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_one.attempt_id}",
        )
        first._finalize_failed_attempt(
            attempt_one,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
    finally:
        first.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:interrupted-refund",
        max_task_attempts=2,
    )
    callback_calls: list[str] = []
    try:
        generic = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(generic) == 1
        assert generic[0]["unknown_outcome_rearm_count"] == 1
        attempt_two = daemon.claim_next()
        assert attempt_two is not None
        assert attempt_two.attempt_number == 2
        daemon._begin_callback_dispatch(
            attempt_two,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_two.attempt_id}",
        )
        daemon._record_callback_dispatch_outcome(
            attempt_two,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_two.attempt_id}",
            outcome="raised",
            body={
                "exception_type": "DatabasePortalBridgeError",
                "message": "interrupted nested implementation recovered",
            },
        )
        failed, receipt = daemon._finalize_failed_attempt(
            attempt_two,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
        )
        assert failed.status == "failed"
        assert receipt["attempt_number"] == 2
        assert receipt["attempts_used"] == 1
        assert receipt["unknown_outcome_rearm_count"] == 1

        task = daemon.task_source.get(attempt_two.task_cid)
        assert task is not None and task.status == "blocked"
        evidence = {
            "schema": (
                DATABASE_PORTAL_INTERRUPTED_IMPLEMENTATION_REARM_EVIDENCE_SCHEMA
            ),
            "attempt_id": attempt_two.attempt_id,
            "claim_id": attempt_two.claim_id,
            "task_cid": attempt_two.task_cid,
            "task_alias": attempt_two.task_alias,
            "attempt_number": attempt_two.attempt_number,
            "owner_session_id": attempt_two.owner_session_id,
            "lease_id": attempt_two.lease_id,
            "fencing_token": attempt_two.fencing_token,
            "fence_epoch": attempt_two.fence_epoch,
            "attempt_root_key": hashlib.sha256(
                attempt_two.attempt_id.encode("utf-8")
            ).hexdigest()[:24],
            "attempt_authority_root_digest": "sha256:" + "1" * 64,
            "attempt_root_digest": "sha256:" + "2" * 64,
            "binding_id": "sha256:" + "3" * 64,
            "binding_admission_id": content_identity(
                {"binding-admission": "current"}
            ),
            "binding_admission_digest": "sha256:" + "4" * 64,
            "projection_immutable_digest": "sha256:" + "5" * 64,
            "nested_task_cid": "nested:task:current",
            "nested_attempt": 1,
            "terminal_reconciliation_evidence_id": (
                "baguqeeras3b36epxxn7yv5nfylhmk5uahqhimjgn2452lpf7a523gkgkcfrq"
            ),
            "first_clear_receipt_id": "sha256:" + "6" * 64,
            "interrupted_retry_evidence_id": "sha256:" + "7" * 64,
            "interrupted_retry_id": content_identity(
                {"interrupted-retry": "current"}
            ),
            "state_recovery_event_id": "sha256:" + "8" * 64,
            "claim_release_receipt_id": content_identity(
                {"claim-release": "current"}
            ),
            "prepared_reconciliation_receipt_id": "sha256:" + "9" * 64,
            "commit_barrier_receipt_id": "sha256:" + "b" * 64,
            "state_digest": "sha256:" + "a" * 64,
            "outer_block_receipt_digest": (
                daemon._database_no_provider_rearm_digest(receipt)
            ),
            "provider_dispatched": False,
            "implementation_dispatched": False,
            "validation_attempted": False,
            "commit_created": False,
            "merge_attempted": False,
            "acceptance_inferred": False,
            "recovery_terminal": True,
            "retained_candidate_disposition": "preserved_unvalidated",
        }
        authorization = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-implementation-"
                "rearm-authorization@1"
            ),
            **{
                name: evidence[name]
                for name in (
                    "attempt_id",
                    "claim_id",
                    "task_cid",
                    "attempt_number",
                    "owner_session_id",
                    "lease_id",
                    "fencing_token",
                    "fence_epoch",
                    "binding_id",
                    "binding_admission_id",
                    "binding_admission_digest",
                    "projection_immutable_digest",
                    "nested_task_cid",
                    "nested_attempt",
                    "terminal_reconciliation_evidence_id",
                    "first_clear_receipt_id",
                    "interrupted_retry_evidence_id",
                    "interrupted_retry_id",
                    "state_recovery_event_id",
                    "claim_release_receipt_id",
                    "prepared_reconciliation_receipt_id",
                    "commit_barrier_receipt_id",
                    "state_digest",
                    "outer_block_receipt_digest",
                )
            },
        }
        evidence["rearm_authorization_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                authorization,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        evidence["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                evidence,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert daemon._valid_no_provider_rearm_evidence(
            evidence,
            task=task,
            original=receipt,
            expected_evidence_id=evidence["evidence_id"],
        )
        for content_id_field in (
            "binding_admission_id",
            "interrupted_retry_id",
            "claim_release_receipt_id",
        ):
            wrong_profile = {**evidence, content_id_field: "sha256:" + "e" * 64}
            wrong_authorization = {
                **authorization,
                content_id_field: wrong_profile[content_id_field],
            }
            wrong_profile["rearm_authorization_id"] = (
                "sha256:"
                + hashlib.sha256(
                    json.dumps(
                        wrong_authorization,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
            )
            wrong_unsigned = dict(wrong_profile)
            wrong_unsigned.pop("evidence_id")
            wrong_profile["evidence_id"] = "sha256:" + hashlib.sha256(
                json.dumps(
                    wrong_unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            assert not daemon._valid_no_provider_rearm_evidence(
                wrong_profile,
                task=task,
                original=receipt,
                expected_evidence_id=wrong_profile["evidence_id"],
            )
        tampered_authorization = {
            **evidence,
            "rearm_authorization_id": "sha256:" + "c" * 64,
        }
        tampered_unsigned = dict(tampered_authorization)
        tampered_unsigned.pop("evidence_id")
        tampered_authorization["evidence_id"] = "sha256:" + hashlib.sha256(
            json.dumps(
                tampered_unsigned,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        assert not daemon._valid_no_provider_rearm_evidence(
            tampered_authorization,
            task=task,
            original=receipt,
            expected_evidence_id=tampered_authorization["evidence_id"],
        )
        monkeypatch.setattr(
            daemon,
            "_database_portal_no_provider_rearm_evidence",
            lambda _task, _receipt: dict(evidence),
        )
        monkeypatch.setattr(
            daemon,
            "_resume_attempt_without_process_crash",
            lambda _attempt: callback_calls.append("callback"),
        )

        refunded = daemon.run_once()

        assert refunded["selection_idle_reason"] == (
            "database_unknown_outcomes_rearmed"
        )
        assert refunded["implementation_result"] is None
        assert callback_calls == []
        assert len(refunded["unknown_outcome_rearms"]) == 1
        outcome = refunded["unknown_outcome_rearms"][0]
        assert outcome["previous_attempt_id"] == attempt_two.attempt_id
        assert outcome["unknown_outcome_rearm_count"] == 1
        assert outcome["nested_event_head_id"] == (
            evidence["state_recovery_event_id"]
        )
        rearmed = daemon.task_source.get(attempt_two.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        rearm_receipt = rearmed.body["completion_receipt"]
        assert rearm_receipt["attempts_used"] == 0
        assert rearm_receipt["retry_exhausted"] is False
        # The exact no-provider refund is not a second generic allowance.
        assert rearm_receipt["unknown_outcome_rearm_count"] == 1
        assert rearm_receipt["no_provider_rearm_evidence_id"] == (
            evidence["evidence_id"]
        )
        for numeric_field in ("attempts_used", "attempt_number"):
            for malformed in (None, "not-an-integer", True, [], {}):
                malformed_receipt = dict(rearm_receipt)
                malformed_original = dict(
                    malformed_receipt["no_provider_rearm_original_block_receipt"]
                )
                malformed_original[numeric_field] = malformed
                malformed_receipt[
                    "no_provider_rearm_original_block_receipt"
                ] = malformed_original
                malformed_task = SimpleNamespace(
                    task_cid=rearmed.task_cid,
                    task_alias=rearmed.task_alias,
                    revision=rearmed.revision,
                    status=rearmed.status,
                    body={
                        **dict(rearmed.body),
                        "completion_receipt": malformed_receipt,
                    },
                )
                assert (
                    daemon._no_provider_rearm_fence_state(malformed_task)
                    == "invalid"
                )
                assert daemon._automatic_claim_forbidden(malformed_task)
        assert daemon.reconcile_blocked_unknown_outcome_tasks() == []
        assert callback_calls == []
    finally:
        daemon.close()


def test_interrupted_rearm_evidence_binds_terminal_barrier_and_recovery_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recomputed outer link cannot detach the immutable recovery barrier."""

    attempt = SimpleNamespace(
        attempt_id="attempt:interrupted:exact",
        claim_id="claim:interrupted:exact",
        task_cid="task:cid:pctdd-034",
        task_alias="PCTDD-034",
        attempt_number=2,
        owner_session_id="session:interrupted:exact",
        lease_id="lease:interrupted:exact",
        fencing_token=7,
        fence_epoch=3,
        status="failed",
        committed_phase="failed",
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths = bridge._paths(attempt)
    state_digest = "sha256:" + "1" * 64
    binding = {
        "task_alias": attempt.task_alias,
        "binding_id": "sha256:" + "2" * 64,
        "projection_immutable_digest": "sha256:" + "3" * 64,
    }
    exact_attempt = {
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "attempt_number": attempt.attempt_number,
        "owner_session_id": attempt.owner_session_id,
        "lease_id": attempt.lease_id,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
    }
    durable_binding = {
        **exact_attempt,
        **binding,
        "stage": "portal_entered",
        "record_id": content_identity({"binding-admission": "exact"}),
    }
    released_attempt = {
        "released_from": 1,
        "released_to": 0,
        "event_id": "sha256:" + "5" * 64,
    }
    replay = {
        "reconciled": True,
        "blocked": False,
        "reason": "interrupted_implementation_recovered_for_retry",
        "task_id": attempt.task_alias,
        "canonical_task_cid": "nested:task:current",
        "attempt": 1,
        "task_claim_reconciliation": {
            "reconciled": True,
            "blocked": False,
            "reason": "quiesced_task_claim_released",
            "task_id": attempt.task_alias,
            "task_status": "todo",
            "released_unfinished_retry_id": content_identity(
                {"interrupted-retry": "exact"}
            ),
            "receipt_id": content_identity({"claim-release": "exact"}),
            "released_unfinished_attempt": released_attempt,
        },
        "provider_dispatched": False,
        "implementation_dispatched": False,
        "acceptance_inferred": False,
        "retained_candidate_disposition": "preserved_unvalidated",
        "stale_lock_cleared": False,
    }
    recovery = {
        **replay,
        "provider_forbidden_terminal_recovery": {
            "applicable": False,
            "blocked": False,
            "implementation_dispatched": False,
            "provider_dispatched": False,
            "reason": "provider_forbidden_terminal_recovery_not_applicable",
            "reconciled": False,
        },
    }
    prepared_id = "sha256:" + "8" * 64
    barrier_id = "sha256:" + "9" * 64
    trigger = "database_daemon_startup"
    prepared = {
        "receipt_id": prepared_id,
        "binding_id": binding["binding_id"],
        "intended_database_disposition": "blocked_unknown_outcome",
        "reason": "nested_portal_attempt_reconciled",
        "reconciled": True,
        "blocked": False,
        "trigger": trigger,
        "nested_state": {
            "present": True,
            "active": False,
            "active_task_id": "",
            "active_attempt": 0,
            "active_phase": "",
            "state_path": str(paths.state),
            "state_digest": state_digest,
        },
        "provider_runner_fence": {
            "safe_to_restart": True,
            "applicable": False,
            "fenced": False,
            "reason": "ordinary_provider_runner_receipt_absent",
        },
        "portal_reconciliation": recovery,
    }
    barrier = {
        **prepared,
        "receipt_id": barrier_id,
        "prepared_reconciliation_receipt_id": prepared_id,
    }
    source_receipt = {
        "binding_id": binding["binding_id"],
        "receipt_id": "sha256:" + "a" * 64,
    }
    source = {
        "reconciliation_receipt": source_receipt,
        "evidence_id": "sha256:" + "b" * 64,
    }
    link = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-terminal-reconciliation-link@1"
        ),
        **exact_attempt,
        "binding_id": binding["binding_id"],
        "nested_state_digest": state_digest,
        "nested_reason": "nested_portal_attempt_reconciled",
        "nested_reconciled": True,
        "trigger": trigger,
        "intended_database_disposition": "blocked_unknown_outcome",
        "prepared_reconciliation_receipt_id": prepared_id,
        "commit_barrier_receipt_id": barrier_id,
    }
    link["evidence_id"] = content_identity(link)
    calls: list[str] = []
    replay_holder: dict[str, object] = {}

    class RetryOnlyPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, object]:
            return dict(replay_holder)

        def reconcile_provider_forbidden_terminal_result(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": (
                    "provider_forbidden_terminal_recovery_not_applicable"
                ),
                "reconciled": False,
            }

        def reconcile_interrupted_database_implementation_attempt(
            self,
            evidence: object,
        ) -> dict[str, object]:
            assert evidence == source
            calls.append("recovery")
            return dict(replay)

        def close_event_runtime(self) -> None:
            return None

    bridge.portal_factory = lambda _paths, _alias: RetryOnlyPortal()
    bridge._binding_lookup = lambda _attempt: dict(durable_binding)
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _value: None)
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(prepared) if receipt_id == prepared_id else dict(barrier)
        ),
    )
    monkeypatch.setattr(
        bridge,
        "_interrupted_implementation_retry_evidence",
        lambda _attempt, _binding: dict(source),
    )
    state = {
        "implementation_in_progress": False,
        "active_task_id": "",
        "active_attempt": 0,
        "active_phase": "",
    }
    monkeypatch.setattr(
        bridge,
        "_strict_state_record",
        lambda _path: (dict(state), state_digest),
    )

    outer_receipt = {**exact_attempt, "terminal_reconciliation": link}
    evidence = bridge._interrupted_implementation_rearm_evidence(
        attempt,
        outer_receipt,
    )

    assert evidence is not None
    assert evidence["rearm_authorization_id"].startswith("sha256:")
    assert evidence["terminal_reconciliation_evidence_id"] == link["evidence_id"]
    assert evidence["commit_barrier_receipt_id"] == barrier_id
    assert calls == ["recovery"]
    assert DatabaseImplementationDaemon._valid_no_provider_rearm_evidence(
        evidence,
        task=attempt,
        original=outer_receipt,
        expected_evidence_id=evidence["evidence_id"],
    )

    for numeric_field in ("attempt_number", "fencing_token", "fence_epoch"):
        numeric_alias = dict(link)
        numeric_alias[numeric_field] = float(numeric_alias[numeric_field])
        assert bridge._interrupted_implementation_rearm_evidence(
            attempt,
            {**exact_attempt, "terminal_reconciliation": numeric_alias},
        ) is None
    assert calls == ["recovery"]

    malformed_nested = {
        **prepared,
        "nested_state": {
            **dict(prepared["nested_state"]),
            "active_attempt": False,
        },
    }
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(malformed_nested)
            if receipt_id == prepared_id
            else {
                **dict(malformed_nested),
                "receipt_id": barrier_id,
                "prepared_reconciliation_receipt_id": prepared_id,
            }
        ),
    )
    assert bridge._interrupted_implementation_rearm_evidence(
        attempt,
        outer_receipt,
    ) is None
    assert calls == ["recovery"]
    monkeypatch.setattr(
        bridge,
        "load_reconciliation_receipt",
        lambda _attempt, receipt_id, required_stage="": (
            dict(prepared) if receipt_id == prepared_id else dict(barrier)
        ),
    )

    tampered_link = {**link, "trigger": "tampered-trigger"}
    tampered_link.pop("evidence_id")
    tampered_link["evidence_id"] = content_identity(tampered_link)
    assert bridge._interrupted_implementation_rearm_evidence(
        attempt,
        {**exact_attempt, "terminal_reconciliation": tampered_link},
    ) is None
    assert calls == ["recovery"]

    persisted_bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "persisted-barrier-attempts",
        portal_factory=lambda _paths, _alias: RetryOnlyPortal(),
    )
    persisted_paths = persisted_bridge._paths(attempt)
    persisted_paths.reconciliation.mkdir(parents=True)
    persisted_nested = {
        "present": True,
        "active": False,
        "active_task_id": "",
        "active_attempt": 0,
        "active_phase": "",
        "state_path": str(persisted_paths.state),
        "state_digest": state_digest,
    }
    persisted_core = {
        "binding_id": binding["binding_id"],
        "reason": "nested_portal_attempt_reconciled",
        "reconciled": True,
        "blocked": False,
        "nested_state": persisted_nested,
        "provider_runner_fence": prepared["provider_runner_fence"],
        "portal_reconciliation": recovery,
        "terminal_provider_evidence": False,
    }
    persisted_prepared = persisted_bridge.persist_reconciliation_receipt(
        attempt,
        {
            **persisted_core,
            "stage": "prepared",
            "trigger": trigger,
            "intended_database_disposition": "blocked_unknown_outcome",
            "reconciled_at": "2026-01-01T00:00:00+00:00",
        },
    )
    persisted_bridge._binding_lookup = lambda _attempt: dict(durable_binding)
    monkeypatch.setattr(
        persisted_bridge,
        "_read_binding",
        lambda _path: dict(binding),
    )
    monkeypatch.setattr(
        persisted_bridge,
        "_verify_binding_identity",
        lambda _value: None,
    )
    monkeypatch.setattr(
        persisted_bridge,
        "_interrupted_implementation_retry_evidence",
        lambda _attempt, _binding: dict(source),
    )
    monkeypatch.setattr(
        persisted_bridge,
        "_strict_state_record",
        lambda _path: (dict(state), state_digest),
    )
    for changed_field, changed_value in (
        ("binding_id", "sha256:" + "f" * 64),
        ("reason", "contradictory-reason"),
        ("reconciled", False),
        ("blocked", True),
        (
            "nested_state",
            {**persisted_nested, "state_digest": "sha256:" + "e" * 64},
        ),
        (
            "provider_runner_fence",
            {**prepared["provider_runner_fence"], "safe_to_restart": False},
        ),
        (
            "portal_reconciliation",
            {**recovery, "reason": "contradictory-recovery"},
        ),
        ("reconciled_at", "2026-01-01T00:00:01+00:00"),
    ):
        contradictory_core = {
            **persisted_core,
            changed_field: changed_value,
        }
        contradictory_barrier = (
            persisted_bridge.persist_reconciliation_receipt(
                attempt,
                {
                    **contradictory_core,
                    "stage": "commit_barrier",
                    "trigger": trigger,
                    "intended_database_disposition": (
                        "blocked_unknown_outcome"
                    ),
                    "prepared_reconciliation_receipt_id": (
                        persisted_prepared["receipt_id"]
                    ),
                    "reconciled_at": (
                        changed_value
                        if changed_field == "reconciled_at"
                        else "2026-01-01T00:00:00+00:00"
                    ),
                },
            )
        )
        contradictory_link = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            ),
            **exact_attempt,
            "binding_id": binding["binding_id"],
            "nested_state_digest": state_digest,
            "nested_reason": "nested_portal_attempt_reconciled",
            "nested_reconciled": True,
            "trigger": trigger,
            "intended_database_disposition": "blocked_unknown_outcome",
            "prepared_reconciliation_receipt_id": persisted_prepared[
                "receipt_id"
            ],
            "commit_barrier_receipt_id": contradictory_barrier["receipt_id"],
        }
        contradictory_link["evidence_id"] = content_identity(
            contradictory_link
        )
        assert persisted_bridge._interrupted_implementation_rearm_evidence(
            attempt,
            {
                **exact_attempt,
                "terminal_reconciliation": contradictory_link,
            },
        ) is None
    assert calls == ["recovery"]


def test_interrupted_rearm_uses_production_run_once_link_and_terminal_saga(
    tmp_path: Path,
) -> None:
    """The real verifier admits one linked refund without widening budget."""

    seed = _open_daemon(
        tmp_path,
        session="session:production-interrupted-refund",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        attempt_one = seed.claim_next()
        assert attempt_one is not None
        seed._begin_callback_dispatch(
            attempt_one,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt_one.attempt_id}",
        )
        seed._finalize_failed_attempt(
            attempt_one,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
    finally:
        seed.close()

    daemon = _open_daemon(
        tmp_path,
        session="session:production-interrupted-refund",
        max_task_attempts=2,
        callbacks_bound=False,
    )
    replay_holder: dict[str, object] = {}
    recovery_calls: list[str] = []

    class RetryOnlyPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, object]:
            return dict(replay_holder)

        def reconcile_provider_forbidden_terminal_result(
            self,
            **_kwargs: object,
        ) -> dict[str, object]:
            return {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": (
                    "provider_forbidden_terminal_recovery_not_applicable"
                ),
                "reconciled": False,
            }

        def reconcile_interrupted_database_implementation_attempt(
            self,
            evidence: object,
        ) -> dict[str, object]:
            assert isinstance(evidence, dict)
            assert evidence.get("schema") == (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-interrupted-implementation-retry@1"
            )
            recovery_calls.append("recovery")
            return dict(replay_holder)

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=daemon.task_source,
        attempt_root=tmp_path / "portal-attempts",
        portal_factory=lambda _paths, _alias: RetryOnlyPortal(),
    )
    daemon.bind_execution_callbacks(
        provider_fn=bridge.run_provider,
        effect_fn=bridge.apply_effect,
        validation_fn=bridge.validate_effect,
    )
    daemon.bind_database_portal_bridge(bridge)
    try:
        generic = daemon.reconcile_blocked_unknown_outcome_tasks()
        assert len(generic) == 1
        assert generic[0]["unknown_outcome_rearm_count"] == 1

        attempt = daemon.claim_next()
        assert attempt is not None and attempt.attempt_number == 2
        attempt = daemon.commit_phase(
            attempt,
            ATTEMPT_PHASE_CONTEXT,
            body={"context_cid": content_identity({"attempt": attempt.attempt_id})},
        )
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        record = daemon.task_source.get(attempt.task_cid)
        assert record is not None
        paths, binding = bridge._ensure_attempt_projection(
            attempt,
            record,
            admit_before_publish=True,
        )
        daemon._record_callback_dispatch_outcome(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
            outcome="raised",
            body={
                "exception_type": "DatabasePortalBridgeError",
                "message": "interrupted nested implementation recovered",
            },
        )
        assert bridge._binding_recorder is not None
        bridge._binding_recorder(attempt, binding, "portal_entered")

        quiescent_state = {
            "implementation_in_progress": False,
            "active_task_id": "",
            "active_attempt": 0,
            "active_phase": "",
        }
        state_bytes = json.dumps(
            quiescent_state,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        paths.state.write_bytes(state_bytes)
        paths.state.chmod(0o600)
        state_digest = "sha256:" + hashlib.sha256(state_bytes).hexdigest()
        nested_task_cid = "nested:task:current"
        nested_attempt = 1
        workspace = str(tmp_path / "retained-candidate")
        first_clear = bridge.persist_reconciliation_receipt(
            attempt,
            {
                "stage": "blocked",
                "trigger": "supervisor_signal_shutdown",
                "reconciled_at": "2026-01-01T00:00:00+00:00",
                "reconciled": False,
                "blocked": True,
                "reason": "nested_portal_attempt_reconciliation_blocked",
                "binding_id": binding["binding_id"],
                "task_alias": attempt.task_alias,
                "nested_state": {
                    "present": True,
                    "active": True,
                    "active_task_id": attempt.task_alias,
                    "active_attempt": nested_attempt,
                    "active_phase": "implementing",
                    "active_worktree_path": workspace,
                    "active_branch": "implementation/test-attempt-1",
                    "state_path": str(paths.state),
                    "state_digest": "sha256:" + "0" * 64,
                },
                "provider_runner_fence": {
                    "applicable": True,
                    "fenced": True,
                    "safe_to_restart": True,
                    "pid": 4242,
                    "reason": "ordinary_provider_runner_exact_birth_fenced",
                },
                "provider_runner_reconciliation_authority": (
                    "ordinary_provider_runner_fence"
                ),
                "portal_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "task_claim_reconciliation_blocked",
                    "reconciled_at": "2026-01-01T00:00:00+00:00",
                    "protected_path_reconciliation": {
                        "blocked": False,
                        "reason": "crash_reconciliation_unchanged",
                        "task_id": attempt.task_alias,
                        "attempt": nested_attempt,
                        "workspace_path": workspace,
                    },
                    "worktree_lifecycle_reconciliation": {
                        "blocked": False,
                        "reconciled": True,
                        "state": "terminal",
                        "task_id": attempt.task_alias,
                        "attempt": nested_attempt,
                        "workspace_path": workspace,
                        "record_id": content_identity({"lifecycle": "terminal"}),
                        "fence": 1,
                    },
                    "task_claim_reconciliation": {
                        "blocked": True,
                        "reconciled": False,
                        "reason": "canonical_task_not_terminal",
                        "observed_task_status": "todo",
                        "task_id": attempt.task_alias,
                        "canonical_task_cid": nested_task_cid,
                    },
                    "attempt_recovery": {
                        "consumed": False,
                        "attempt": nested_attempt,
                        "task_id": attempt.task_alias,
                        "canonical_task_cid": nested_task_cid,
                        "previous_display_count": nested_attempt,
                        "previous_cid_count": nested_attempt,
                    },
                },
                "terminal_provider_evidence": False,
            },
        )
        interrupted_retry_id = content_identity(
            {"interrupted-retry": attempt.attempt_id}
        )
        released_attempt = {
            "released_from": nested_attempt,
            "released_to": nested_attempt - 1,
            "event_id": "sha256:" + "2" * 64,
        }
        replay_holder.update(
            {
                "reconciled": True,
                "blocked": False,
                "reason": "interrupted_implementation_recovered_for_retry",
                "task_id": attempt.task_alias,
                "canonical_task_cid": nested_task_cid,
                "attempt": nested_attempt,
                "task_claim_reconciliation": {
                    "reconciled": True,
                    "blocked": False,
                    "reason": "quiesced_task_claim_released",
                    "task_id": attempt.task_alias,
                    "task_status": "todo",
                    "released_unfinished_retry_id": interrupted_retry_id,
                    "receipt_id": content_identity(
                        {"claim-release": attempt.attempt_id}
                    ),
                    "released_unfinished_attempt": released_attempt,
                },
                "provider_dispatched": False,
                "implementation_dispatched": False,
                "acceptance_inferred": False,
                "retained_candidate_disposition": "preserved_unvalidated",
                "stale_lock_cleared": False,
            }
        )
        recovery = {
            **replay_holder,
            "provider_forbidden_terminal_recovery": {
                "applicable": False,
                "blocked": False,
                "implementation_dispatched": False,
                "provider_dispatched": False,
                "reason": "provider_forbidden_terminal_recovery_not_applicable",
                "reconciled": False,
            },
        }
        terminal_core = {
            "reconciled": True,
            "blocked": False,
            "reason": "nested_portal_attempt_reconciled",
            "binding_id": binding["binding_id"],
            "nested_state": {
                "present": True,
                "active": False,
                "active_task_id": "",
                "active_attempt": 0,
                "active_phase": "",
                "state_path": str(paths.state),
                "state_digest": state_digest,
            },
            "provider_runner_fence": {
                "safe_to_restart": True,
                "applicable": False,
                "fenced": False,
                "reason": "ordinary_provider_runner_receipt_absent",
            },
            "portal_reconciliation": recovery,
            "terminal_provider_evidence": False,
        }
        trigger = "database_daemon_startup"
        prepared = bridge.persist_reconciliation_receipt(
            attempt,
            {
                **terminal_core,
                "stage": "prepared",
                "trigger": trigger,
                "intended_database_disposition": "blocked_unknown_outcome",
                "reconciled_at": "2026-01-01T00:00:01+00:00",
            },
        )
        barrier = bridge.persist_reconciliation_receipt(
            attempt,
            {
                **terminal_core,
                "stage": "commit_barrier",
                "trigger": trigger,
                "intended_database_disposition": "blocked_unknown_outcome",
                "prepared_reconciliation_receipt_id": prepared["receipt_id"],
                "reconciled_at": "2026-01-01T00:00:01+00:00",
            },
        )
        link = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-link@1"
            ),
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "task_cid": attempt.task_cid,
            "attempt_number": attempt.attempt_number,
            "owner_session_id": attempt.owner_session_id,
            "lease_id": attempt.lease_id,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "binding_id": binding["binding_id"],
            "nested_state_digest": state_digest,
            "nested_reason": "nested_portal_attempt_reconciled",
            "nested_reconciled": True,
            "trigger": trigger,
            "intended_database_disposition": "blocked_unknown_outcome",
            "prepared_reconciliation_receipt_id": prepared["receipt_id"],
            "commit_barrier_receipt_id": barrier["receipt_id"],
        }
        link["evidence_id"] = content_identity(link)
        daemon._record_database_portal_terminal_reconciliation_barrier(
            attempt,
            link,
        )
        failed, receipt = daemon._finalize_failed_attempt(
            attempt,
            reason="callback_authority_incomplete_blocked",
            force_block=True,
            unknown_authority=True,
            reconciliation_evidence=link,
        )
        assert receipt["attempts_used"] == 1
        assert receipt["unknown_outcome_rearm_count"] == 1
        repaired = daemon._repair_database_portal_terminal_receipts(
            bridge=bridge,
            trigger="test_terminal_repair",
            exact_attempt=failed,
        )
        assert len(repaired) == 1 and repaired[0]["reconciled"] is True
        saga = daemon._database_portal_terminal_reconciliation_saga(failed)
        assert saga is not None and saga["stage"] == "terminal"

        failed_phase = daemon.phase_history(failed.attempt_id)[-1]["body"]
        mismatched_phase = dict(failed_phase)
        mismatched_phase["terminal_reconciliation"] = {
            **link,
            "trigger": "different-trigger",
        }
        assert daemon._database_interrupted_failed_phase_link(
            mismatched_phase,
            receipt,
        ) is None
        assert recovery_calls == []

        blocked_task = daemon.task_source.get(attempt.task_cid)
        assert blocked_task is not None
        assert daemon._database_interrupted_failed_phase_link(
            daemon.phase_history(failed.attempt_id)[-1]["body"],
            receipt,
        ) is not None
        assert [
            phase["phase"] for phase in daemon.phase_history(failed.attempt_id)
        ] == ["claimed", "context", "failed"]
        provider_dispatch = daemon._dispatch_journal_entry(
            failed,
            dispatch_kind="provider",
            idempotency_key=f"provider:{failed.attempt_id}",
        )
        assert provider_dispatch is not None
        assert provider_dispatch["outcome"] == "raised"
        assert provider_dispatch["body"]["exception_type"] == (
            "DatabasePortalBridgeError"
        )
        terminal_claim = daemon.coordinator.get_task_claim(failed.claim_id)
        assert terminal_claim is not None
        assert str(terminal_claim.state.value) == "released"
        control_claim = dict(failed.body["control_claim"])
        assert blocked_task.revision == int(control_claim["revision"]) + 1
        assert control_claim["task_cid"] == failed.task_cid
        assert control_claim["execution_spec_cid"] == (
            daemon._task_execution_spec_cid(blocked_task)
        )
        assert control_claim["validation_spec_cid"] == (
            daemon._retry_budget_validation_spec_cid(blocked_task)
        )
        budget_state = daemon._retry_budget_state(blocked_task)
        assert budget_state["malformed"] is False
        assert budget_state["policy_mismatch"] is False
        assert receipt["max_task_attempts"] == budget_state["max_task_attempts"]

        result = daemon.run_once()

        assert result["selection_idle_reason"] == (
            "database_unknown_outcomes_rearmed"
        )
        assert len(result["unknown_outcome_rearms"]) == 1
        assert recovery_calls == ["recovery"]
        rearmed = daemon.task_source.get(attempt.task_cid)
        assert rearmed is not None and rearmed.status == "retrying"
        rearm_receipt = rearmed.body["completion_receipt"]
        assert rearm_receipt["attempts_used"] == 0
        assert rearm_receipt["unknown_outcome_rearm_count"] == 1
        assert rearm_receipt["no_provider_rearm_evidence"][
            "first_clear_receipt_id"
        ] == first_clear["receipt_id"]
        assert daemon.reconcile_blocked_unknown_outcome_tasks() == []
        assert daemon._unresolved_database_no_provider_rearm_sagas() == ()
        assert recovery_calls == ["recovery"]
    finally:
        daemon.close()


def test_unknown_outcome_rearm_limit_survives_claim_and_block_revisions(
    tmp_path: Path,
) -> None:
    population = _population(1)
    seed = _open_daemon(
        tmp_path,
        session="session:durable-unknown-rearm-limit",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(population)
    finally:
        seed.close()

    for expected_count in range(1, DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT + 1):
        blocker = _open_daemon(
            tmp_path,
            session="session:durable-unknown-rearm-limit",
            max_task_attempts=2,
        )
        try:
            attempt = blocker.claim_next()
            assert attempt is not None
            blocker._begin_callback_dispatch(
                attempt,
                dispatch_kind="provider",
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            _failed, receipt = blocker._finalize_failed_attempt(
                attempt,
                reason="provider_dispatch_outcome_unknown",
                force_block=True,
                unknown_authority=True,
            )
            prior_count = expected_count - 1
            if prior_count:
                assert receipt["unknown_outcome_rearm_count"] == prior_count
        finally:
            blocker.close()

        successor = _open_daemon(
            tmp_path,
            session="session:durable-unknown-rearm-limit",
            max_task_attempts=2,
        )
        try:
            rearms = successor.reconcile_blocked_unknown_outcome_tasks()
            assert len(rearms) == 1
            task = successor.task_source.get("task:cid:001")
            assert task is not None and task.status == "retrying"
            assert task.body["completion_receipt"][
                "unknown_outcome_rearm_count"
            ] == expected_count
        finally:
            successor.close()

    final_blocker = _open_daemon(
        tmp_path,
        session="session:durable-unknown-rearm-limit",
        max_task_attempts=2,
    )
    try:
        final_attempt = final_blocker.claim_next()
        assert final_attempt is not None
        final_blocker._begin_callback_dispatch(
            final_attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{final_attempt.attempt_id}",
        )
        _failed, final_receipt = final_blocker._finalize_failed_attempt(
            final_attempt,
            reason="provider_dispatch_outcome_unknown",
            force_block=True,
            unknown_authority=True,
        )
        assert final_receipt["unknown_outcome_rearm_count"] == (
            DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
        )
    finally:
        final_blocker.close()

    terminal = _open_daemon(
        tmp_path,
        session="session:durable-unknown-rearm-limit",
        max_task_attempts=2,
    )
    try:
        assert terminal.reconcile_blocked_unknown_outcome_tasks() == []
        task = terminal.task_source.get("task:cid:001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"][
            "unknown_outcome_rearm_count"
        ] == DATABASE_UNKNOWN_OUTCOME_REARM_LIMIT
    finally:
        terminal.close()


def test_later_process_rearms_exhausted_portal_provider_failure(
    tmp_path: Path,
) -> None:
    seed = _open_daemon(
        tmp_path,
        session="session:portal-exhausted-seed",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(1))
        task = seed.task_source.get("task:cid:001")
        assert task is not None
        receipt = seed._retry_budget_receipt(
            task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason="portal_provider_failed",
        )
        receipt["retry_exhausted"] = True
        seed._cas_task_status_database(
            task.task_cid,
            expected_revision=int(task.revision),
            new_status="blocked",
            receipt=receipt,
        )
        assert seed.reconcile_blocked_unknown_outcome_tasks() == []
        assert seed.task_source.get("task:cid:001").status == "blocked"
    finally:
        seed.close()

    successor_calls: list[str] = []
    successor = _open_daemon(
        tmp_path,
        session="session:portal-exhausted-rearm",
        provider_calls=successor_calls,
        max_task_attempts=2,
    )
    try:
        rearms = successor.reconcile_blocked_unknown_outcome_tasks()
        assert len(rearms) == 1
        task = successor.task_source.get("task:cid:001")
        assert task is not None and task.status == "retrying"
        assert task.body["completion_receipt"]["attempts_used"] == 0
        claimed = successor.run_once()
        assert claimed["claimed_task_cid"] == "task:cid:001"
        assert successor_calls == ["task:cid:001"]
    finally:
        successor.close()


def test_malformed_terminal_candidate_does_not_starve_unrelated_rearm(
    tmp_path: Path,
) -> None:
    seed = _open_daemon(
        tmp_path,
        session="session:terminal-selector-liveness-seed",
        max_task_attempts=2,
    )
    try:
        seed.materialize_population(_population(2))
        terminal_task = seed.task_source.get("task:cid:001")
        generic_task = seed.task_source.get("task:cid:002")
        assert terminal_task is not None and generic_task is not None
        terminal_receipt = seed._retry_budget_receipt(
            terminal_task,
            attempts_used=2,
            operation="database_unknown_outcome_blocked",
            reason="callback_authority_incomplete_blocked",
        )
        terminal_receipt.update(
            {
                "authority_outcome": "unknown",
                "forced_block": True,
                "terminal_reconciliation": "malformed",
            }
        )
        seed._cas_task_status_database(
            terminal_task.task_cid,
            expected_revision=int(terminal_task.revision),
            new_status="blocked",
            receipt=terminal_receipt,
        )
        generic_receipt = seed._retry_budget_receipt(
            generic_task,
            attempts_used=2,
            operation="database_retry_exhausted",
            reason="portal_provider_failed",
        )
        generic_receipt["retry_exhausted"] = True
        seed._cas_task_status_database(
            generic_task.task_cid,
            expected_revision=int(generic_task.revision),
            new_status="blocked",
            receipt=generic_receipt,
        )
    finally:
        seed.close()

    successor = _open_daemon(
        tmp_path,
        session="session:terminal-selector-liveness-successor",
        max_task_attempts=2,
    )
    try:
        # The malformed link is rejected before this sentinel can be used.
        successor._database_portal_bridge = object()
        outcomes = successor.reconcile_blocked_unknown_outcome_tasks()

        assert any(
            item.get("task_cid") == "task:cid:001"
            and item.get("reason") == "terminal_landed_candidate_link_invalid"
            and item.get("blocked") is True
            for item in outcomes
        )
        assert any(
            item.get("task_cid") == "task:cid:002"
            and item.get("operation") == "database_unknown_outcome_rearmed"
            for item in outcomes
        )
        terminal = successor.task_source.get("task:cid:001")
        generic = successor.task_source.get("task:cid:002")
        assert terminal is not None and terminal.status == "blocked"
        assert generic is not None and generic.status == "retrying"
    finally:
        successor.close()


def test_terminal_candidate_quarantine_covers_full_bounded_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:terminal-selector-full-page",
        max_task_attempts=2,
    )
    terminal_tasks = []
    for index in range(128):
        task_cid = f"task:cid:terminal:{index:03d}"
        terminal_tasks.append(
            SimpleNamespace(
                task_cid=task_cid,
                task_alias=f"PCTDD-TERMINAL-{index:03d}",
                revision=1,
                status="blocked",
                body={
                    "completion_receipt": {
                        "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                        "operation": "database_unknown_outcome_blocked",
                        "reason": "callback_authority_incomplete_blocked",
                        "forced_block": True,
                        "authority_outcome": "unknown",
                        "terminal_reconciliation": {"schema": "candidate"},
                    }
                },
            )
        )
    overflow = SimpleNamespace(
        task_cid="task:cid:terminal:overflow",
        task_alias="PCTDD-TERMINAL-OVERFLOW",
        revision=1,
        status="blocked",
        body={
            "completion_receipt": {
                "schema": DATABASE_RETRY_BUDGET_SCHEMA,
                "operation": "database_retry_exhausted",
                "reason": "portal_provider_failed",
                "retry_exhausted": True,
                "terminal_reconciliation": {"schema": "malformed"},
            }
        },
    )
    tasks = (*terminal_tasks, overflow)
    cas_calls: list[str] = []
    try:
        daemon._database_portal_bridge = object()
        monkeypatch.setattr(
            daemon.task_source,
            "list_tasks",
            lambda **_kwargs: SimpleNamespace(tasks=tasks),
        )
        monkeypatch.setattr(
            daemon,
            "_automatic_claim_forbidden",
            lambda _task: False,
        )
        monkeypatch.setattr(
            daemon,
            "_reconcile_one_blocked_terminal_landed_task",
            lambda *, task, bridge: {
                "task_cid": str(task.task_cid),
                "task_alias": str(task.task_alias),
                "operation": "database_terminal_landed_completion",
                "recovered": False,
                "rearmed": False,
                "blocked": True,
                "reason": "terminal_landed_candidate_recovery_blocked",
            },
        )
        monkeypatch.setattr(daemon, "list_running_attempts", lambda: [])
        monkeypatch.setattr(
            daemon,
            "_cas_task_status_database",
            lambda task_cid, **_kwargs: cas_calls.append(str(task_cid)),
        )

        outcomes = daemon.reconcile_blocked_unknown_outcome_tasks()

        assert len(outcomes) == 129
        overflow_outcomes = [
            item
            for item in outcomes
            if item.get("task_cid") == overflow.task_cid
        ]
        assert len(overflow_outcomes) == 1
        assert overflow_outcomes[0]["reason"] == (
            "terminal_landed_candidate_policy_invalid"
        )
        assert overflow_outcomes[0]["rearmed"] is False
        assert overflow.task_cid not in cas_calls
    finally:
        daemon.close()


def test_post_effect_dispatch_journal_failure_blocks_without_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    effect_calls: list[str] = []

    def effect(attempt: DatabaseTaskAttempt, _result: object) -> dict[str, object]:
        effect_calls.append(attempt.attempt_id)
        return {"status": "applied", "effect_key": "external:one"}

    daemon = _open_daemon(
        tmp_path,
        session="session:effect-post-return",
        effect_fn=effect,
        max_task_attempts=3,
    )

    original_record = daemon._record_callback_dispatch_outcome

    def fail_after_effect(*args: object, **kwargs: object) -> None:
        if kwargs.get("dispatch_kind") == "effect" and kwargs.get("outcome") == "returned":
            raise RuntimeError("lost effect return journal")
        original_record(*args, **kwargs)

    monkeypatch.setattr(daemon, "_record_callback_dispatch_outcome", fail_after_effect)
    try:
        daemon.materialize_population(_population(1))
        failed = daemon.run_once()
        assert failed["implementation_result"]["retry_exhausted"] is True
        monkeypatch.setattr(
            daemon, "_record_callback_dispatch_outcome", original_record
        )
        daemon.run_once()
        assert len(effect_calls) == 1
    finally:
        daemon.close()


@pytest.mark.parametrize("dispatch_kind", ["provider", "effect"])
@pytest.mark.parametrize("failure_site", ["phase_event", "committed_journal"])
def test_post_phase_callback_bookkeeping_failure_resumes_exact_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dispatch_kind: str,
    failure_site: str,
) -> None:
    provider_attempts: list[str] = []
    effect_attempts: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_attempts.append(attempt.attempt_id)
        return {"status": "ok", "attempt_id": attempt.attempt_id}

    def effect(
        attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        effect_attempts.append(attempt.attempt_id)
        return {
            "status": "applied",
            "effect_key": "external:post-phase",
            "attempt_id": attempt.attempt_id,
        }

    daemon = _open_daemon(
        tmp_path,
        session=f"session:post-phase:{dispatch_kind}:{failure_site}",
        provider_fn=provider,
        effect_fn=effect,
        max_task_attempts=2,
    )
    injected = {"done": False}
    if failure_site == "phase_event":
        original_event = daemon._record_event
        target_phase = (
            ATTEMPT_PHASE_PROVIDER
            if dispatch_kind == "provider"
            else ATTEMPT_PHASE_EFFECT
        )

        def fail_committed_phase_event(
            event_type: str,
            *args: object,
            **kwargs: object,
        ) -> None:
            body = kwargs.get("body")
            if (
                not injected["done"]
                and event_type == "attempt_phase_committed"
                and isinstance(body, dict)
                and body.get("phase") == target_phase
            ):
                injected["done"] = True
                raise RuntimeError("lost committed phase event response")
            original_event(event_type, *args, **kwargs)

        monkeypatch.setattr(daemon, "_record_event", fail_committed_phase_event)
    else:
        original_dispatch = daemon._record_callback_dispatch_outcome

        def fail_committed_dispatch_journal(
            *args: object,
            **kwargs: object,
        ) -> None:
            if (
                not injected["done"]
                and kwargs.get("dispatch_kind") == dispatch_kind
                and kwargs.get("outcome") == "committed"
            ):
                injected["done"] = True
                raise RuntimeError("lost committed dispatch journal response")
            original_dispatch(*args, **kwargs)

        monkeypatch.setattr(
            daemon,
            "_record_callback_dispatch_outcome",
            fail_committed_dispatch_journal,
        )

    try:
        daemon.materialize_population(_population(1))
        pending = daemon.run_once()
        pending_result = pending["implementation_result"]
        assert pending_result["status"] == (
            "callback_commit_reconciliation_pending"
        )
        assert pending_result["dispatch_kind"] == dispatch_kind
        assert pending_result["retry_budget_consumed"] is False
        attempt_id = pending_result["attempt_id"]
        attempt = daemon.get_attempt(attempt_id)
        assert attempt is not None and attempt.status == "running"
        assert attempt.phase_committed(
            ATTEMPT_PHASE_PROVIDER
            if dispatch_kind == "provider"
            else ATTEMPT_PHASE_EFFECT
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
        assert task.body["completion_receipt"]["attempts_used"] == 1

        recovered = daemon.run_once()
        recovered_result = recovered["implementation_result"]
        assert recovered_result["status"] == "succeeded"
        assert recovered_result["attempt"]["attempt_id"] == attempt_id
        assert provider_attempts == [attempt_id]
        assert effect_attempts == [attempt_id]
        assert injected["done"] is True
    finally:
        daemon.close()


@pytest.mark.parametrize(
    ("max_attempts", "expected_status"),
    [(2, "retrying"), (1, "blocked")],
)
def test_missing_claim_history_rearms_or_blocks_canonical_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_attempts: int,
    expected_status: str,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session=f"session:missing-claim:{max_attempts}",
        max_task_attempts=max_attempts,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        monkeypatch.setattr(daemon.coordinator, "get_task_claim", lambda _claim: None)
        outcomes = daemon.reconcile_expired_running_attempts()
        assert outcomes[0]["authority_outcome"] == "unknown"
        assert outcomes[0]["status"] == expected_status
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None and task.status == expected_status
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
    finally:
        daemon.close()


def test_failure_status_cas_outage_recovers_without_provider_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fail(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        calls.append(attempt.attempt_id)
        raise DatabasePortalBridgeError("declared_validation_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:failure-cas",
        provider_fn=fail,
        max_task_attempts=2,
    )
    original_cas = daemon._cas_task_status_database
    injected = {"done": False}

    def fail_retry_cas(*args: object, **kwargs: object) -> object:
        if kwargs.get("new_status") in {"retrying", "blocked"} and not injected["done"]:
            injected["done"] = True
            raise RuntimeError("injected retry CAS outage")
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(daemon, "_cas_task_status_database", fail_retry_cas)
    try:
        daemon.materialize_population(_population(1))
        pending = daemon.run_once()
        assert pending["implementation_result"]["status"] == (
            "failure_reconciliation_pending"
        )
        assert len(calls) == 1
        recovered = daemon.run_once()
        assert recovered["implementation_result"]["status"] == "failed"
        assert len(calls) == 1
    finally:
        daemon.close()


def test_claim_insert_failure_is_compensated_before_any_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-insert",
        provider_calls=calls,
        max_task_attempts=2,
    )
    original_insert = daemon._insert_attempt_from_claim
    monkeypatch.setattr(
        daemon,
        "_insert_attempt_from_claim",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("insert outage")),
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["implementation_result"] is None
        task = daemon.task_source.get("task:cid:001")
        assert task is not None and task.status == "retrying"
        assert calls == []
        monkeypatch.setattr(daemon, "_insert_attempt_from_claim", original_insert)
        daemon.run_once()
        assert calls == ["task:cid:001"]
    finally:
        daemon.close()


def test_claim_insert_and_compensation_cas_failure_recovers_next_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-insert-compensation",
        provider_calls=calls,
        max_task_attempts=2,
    )
    original_insert = daemon._insert_attempt_from_claim
    original_cas = daemon._cas_task_status_database
    injected = {"insert": False, "compensation": False}

    def fail_first_insert(*args: object, **kwargs: object) -> object:
        if not injected["insert"]:
            injected["insert"] = True
            raise RuntimeError("insert outage")
        return original_insert(*args, **kwargs)

    def fail_first_compensation(*args: object, **kwargs: object) -> object:
        if (
            kwargs.get("new_status") in {"retrying", "blocked"}
            and not injected["compensation"]
        ):
            injected["compensation"] = True
            raise RuntimeError("compensation CAS outage")
        return original_cas(*args, **kwargs)

    monkeypatch.setattr(daemon, "_insert_attempt_from_claim", fail_first_insert)
    monkeypatch.setattr(daemon, "_cas_task_status_database", fail_first_compensation)
    try:
        daemon.materialize_population(_population(1))
        pending = daemon.run_once()
        assert pending["implementation_result"] is None
        stranded = daemon.task_source.get("task:cid:001")
        assert stranded is not None and stranded.status == "in_progress"
        assert daemon.list_running_attempts() == []
        assert calls == []

        # The next pass consumes the durable database_claim marker, restores
        # retryable state, releases the old claim, and opens one fresh attempt.
        recovered = daemon.run_once()
        assert recovered["implementation_result"]["status"] == "succeeded"
        assert calls == ["task:cid:001"]
        assert recovered["write_count"] >= 2
    finally:
        daemon.close()


def test_nonpassing_validation_consumes_only_global_retry_budget(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    validation_calls: list[str] = []

    def reject(attempt: DatabaseTaskAttempt, _effect: object) -> dict[str, object]:
        validation_calls.append(attempt.attempt_id)
        return {"outcome": "failed", "evidence_digest": "sha256:rejected"}

    daemon = _open_daemon(
        tmp_path,
        session="session:validation-nonpass",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        validation_fn=reject,
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        first = daemon.run_once()
        second = daemon.run_once()
        idle = daemon.run_once()
        assert first["implementation_result"]["retry_exhausted"] is False
        assert second["implementation_result"]["retry_exhausted"] is True
        assert idle["implementation_result"] is None
        assert len(provider_calls) == len(effect_calls) == len(validation_calls) == 2
        assert len(set(validation_calls)) == 2
    finally:
        daemon.close()


def test_malformed_retry_receipt_fails_closed_without_dispatch(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:malformed-retry",
        provider_calls=provider_calls,
        max_task_attempts=2,
    )
    try:
        population = _population(1)
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        malformed = _population(1)
        malformed_task = malformed["tasks"][0]
        assert isinstance(malformed_task, dict)
        malformed_task["completion_receipt"] = {
            "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            "task_cid": task.task_cid,
            "validation_spec_cid": daemon._retry_budget_validation_spec_cid(task),
            "attempts_used": "not-an-integer",
        }
        daemon.materialize_population(malformed)

        blocked = daemon.run_once()
        assert blocked["implementation_result"] is None
        assert blocked["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        entry = blocked["retry_budget_backpressure"]["tasks"][0]
        assert entry["malformed"] is True
        assert provider_calls == []
    finally:
        daemon.close()


def test_orphan_recovery_preserves_malformed_retry_latch(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:malformed-orphan",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        daemon.sync_ready_tasks_into_coordination()
        claim = daemon.coordinator.claim_ready_task(
            owner_session_id=daemon.owner_session_id,
            lease_ms=daemon.lease_ms,
            now_ms=daemon._now_ms(),
        )
        assert claim is not None
        task = daemon.task_source.get(claim.task_cid)
        assert task is not None
        malformed_receipt = daemon._retry_budget_receipt(
            task,
            attempts_used=1,
            operation="database_claim",
            attempt=claim,
        )
        malformed_receipt["attempts_used"] = "not-an-integer"
        daemon._cas_task_status_database(
            task.task_cid,
            expected_revision=task.revision,
            new_status="in_progress",
            receipt=malformed_receipt,
        )

        outcomes = daemon.reconcile_orphaned_canonical_claims()
        assert outcomes[0]["status"] == "blocked"
        recovered = daemon.task_source.get(task.task_cid)
        assert recovered is not None and recovered.status == "blocked"
        assert recovered.body["completion_receipt"]["malformed"] is True
        assert recovered.body["completion_receipt"]["retry_exhausted"] is True

        idle = daemon.run_once()
        assert idle["selection_idle_reason"] == (
            "all_selectable_ready_tasks_reached_max_task_attempts"
        )
        entry = idle["retry_budget_backpressure"]["tasks"][0]
        assert entry["task_cid"] == task.task_cid
        assert entry["malformed"] is True
    finally:
        daemon.close()


def test_retry_backpressure_distinguishes_some_from_all_selectable_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:partial-backpressure",
        task_prefix="DQP-T",
        max_task_attempts=1,
    )
    try:
        population = _population(2)
        daemon.materialize_population(population)
        first = daemon.task_source.get("task:cid:001")
        assert first is not None
        population_task = population["tasks"][0]
        assert isinstance(population_task, dict)
        population_task["completion_receipt"] = {
            "schema": DATABASE_RETRY_BUDGET_SCHEMA,
            "task_cid": first.task_cid,
            "validation_spec_cid": daemon._retry_budget_validation_spec_cid(first),
            "attempts_used": 1,
        }
        daemon.materialize_population(population)
        monkeypatch.setattr(daemon, "claim_next", lambda: None)

        result = daemon.run_once()
        assert result["selection_idle_reason"] == (
            "some_selectable_tasks_reached_max_task_attempts"
        )
        backpressure = result["retry_budget_backpressure"]
        assert backpressure["any_eligible_exhausted"] is True
        assert backpressure["all_eligible_exhausted"] is False
        assert backpressure["eligible_ready_task_cids"] == ["task:cid:002"]
        assert result["retry_exhausted_task_cids"] == ["task:cid:001"]
    finally:
        daemon.close()


def test_claim_cas_unknown_response_is_reconciled_without_release_or_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:claim-unknown",
        provider_calls=calls,
        max_task_attempts=2,
    )
    original_cas = daemon._cas_task_status_database
    injected = {"done": False}

    def commit_then_unknown(*args: object, **kwargs: object) -> object:
        result = original_cas(*args, **kwargs)
        if kwargs.get("new_status") == "in_progress" and not injected["done"]:
            injected["done"] = True
            raise DatabaseTaskSourceUnknownOutcomeError("lost CAS response")
        return result

    monkeypatch.setattr(daemon, "_cas_task_status_database", commit_then_unknown)
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        assert result["implementation_result"]["status"] == "succeeded"
        assert calls == ["task:cid:001"]
    finally:
        daemon.close()


def test_replacement_task_body_revokes_old_claim_before_provider(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session="session:task-replacement",
        provider_calls=calls,
        max_task_attempts=2,
    )
    try:
        original = _population(1)
        daemon.materialize_population(original)
        old = daemon.claim_next()
        assert old is not None
        replacement = _population(1)
        replacement_task = replacement["tasks"][0]
        assert isinstance(replacement_task, dict)
        replacement_task["title"] = "Replacement task body"
        replacement_task["validation_commands"] = ["python -m pytest replacement.py"]
        daemon.materialize_population(replacement)
        revoked = daemon.run_once()
        assert revoked["implementation_result"]["callback_failure"] is True
        assert calls == []
        current = daemon.task_source.get(old.task_cid)
        assert current is not None and current.status == "ready"
        daemon.run_once()
        assert calls == [old.task_cid]
    finally:
        daemon.close()


def test_attempt_control_claim_rejects_numeric_revision_aliases(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(
        tmp_path,
        session="session:control-claim-numeric-alias",
        max_task_attempts=2,
    )
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        claim = dict(attempt.body["control_claim"])
        assert daemon._database_attempt_has_exact_control(attempt, task)
        for numeric_alias in (True, float(claim["revision"])):
            tampered = SimpleNamespace(
                task_cid=attempt.task_cid,
                body={
                    **dict(attempt.body),
                    "control_claim": {
                        **claim,
                        "revision": numeric_alias,
                    },
                },
            )
            assert not daemon._database_attempt_has_exact_control(
                tampered,
                task,
            )
    finally:
        daemon.close()


@pytest.mark.parametrize("numeric_alias", (True, 1.0))
def test_rearm_snapshot_comparison_rejects_numeric_aliases(
    numeric_alias: object,
) -> None:
    assert not _canonical_mapping_matches(
        {"body": {"revision": 1}},
        {"body": {"revision": numeric_alias}},
    )


def test_old_validation_epoch_failure_cannot_charge_replacement_claim(
    tmp_path: Path,
) -> None:
    old_lane = _open_daemon(
        tmp_path / "seed",
        session="session:old-validation-epoch",
        max_task_attempts=2,
    )
    new_coordinator = None
    new_lane = None
    try:
        old_lane.materialize_population(_population(1))
        old_attempt = old_lane.claim_next()
        assert old_attempt is not None

        replacement = _population(1)
        replacement_task = replacement["tasks"][0]
        assert isinstance(replacement_task, dict)
        replacement_task["validation_commands"] = [
            {"argv": ["python", "-m", "pytest", "replacement.py"]}
        ]
        old_lane.materialize_population(replacement)

        lane_root = tmp_path / "replacement-lane"
        new_coordinator = open_database_coordinator(
            lane_root / "coordination.duckdb"
        )
        new_lane = DatabaseImplementationDaemon(
            database_path=old_lane.database_path,
            coordination_path=lane_root / "coordination.duckdb",
            execution_path=lane_root / "execution.duckdb",
            owner_session_id="session:new-validation-epoch",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_source=old_lane.task_source,
            coordinator=new_coordinator,
            max_task_attempts=2,
        )
        new_attempt = new_lane.claim_next()
        assert new_attempt is not None
        before = new_lane.task_source.get(new_attempt.task_cid)
        assert before is not None and before.status == "in_progress"
        new_receipt = dict(before.body["completion_receipt"])
        assert new_receipt["attempt_id"] == new_attempt.attempt_id
        assert new_receipt["attempts_used"] == 1

        _failed, stale_receipt = old_lane._finalize_failed_attempt(
            old_attempt,
            reason="old validation callback failed after replacement",
        )
        assert stale_receipt["operation"] == (
            "database_superseded_attempt_revoked"
        )
        after = new_lane.task_source.get(new_attempt.task_cid)
        assert after is not None and after.status == "in_progress"
        assert after.body["completion_receipt"] == new_receipt
        assert old_lane.get_attempt(old_attempt.attempt_id).status == "failed"
    finally:
        if new_lane is not None:
            new_lane.close()
        if new_coordinator is not None:
            new_coordinator.close()
        old_lane.close()


def test_strict_shards_claim_only_home_lane_tasks(tmp_path: Path) -> None:
    seed = _open_daemon(tmp_path, session="session:seed")
    try:
        seed.materialize_population(_population(8))
    finally:
        seed.close()

    claimed: dict[int, str] = {}
    for index in range(4):
        daemon = _open_daemon(
            tmp_path,
            session=f"session:shard-{index}",
            task_shard_count=4,
            task_shard_index=index,
            strict_task_sharding=True,
            task_prefix="DQP-T",
        )
        try:
            attempt = daemon.claim_next()
            assert attempt is not None, f"shard {index} found no home-lane work"
            alias = str(attempt.task_alias or "")
            home = daemon._task_home_shard_index(alias)
            assert home == index, f"{alias} home={home} claimed by shard {index}"
            claimed[index] = alias
        finally:
            daemon.close()

    assert len(set(claimed.values())) == 4


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
        expiry_result = replacement.run_once()
        reconciliations = expiry_result["expired_attempt_reconciliations"]
        assert len(reconciliations) == 1
        assert reconciliations[0]["status"] == "expired"
        assert reconciliations[0]["provider_evidence_reused"] is False
        assert reconciliations[0]["effect_evidence_reused"] is False
        assert expiry_result["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        # Expiry and a freshly fenced retry are separate durable passes.
        result = replacement.run_once()
        assert result["attempt_id"] != old_attempt.attempt_id
        assert result["implementation_result"]["status"] == "succeeded"
        assert provider_calls == [old_attempt.task_cid]
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


@pytest.mark.parametrize("callback_kind", ("provider", "effect"))
@pytest.mark.parametrize("result_state", ("missing", "corrupt"))
def test_expired_committed_callback_with_invalid_result_never_redispatches(
    tmp_path: Path,
    callback_kind: str,
    result_state: str,
) -> None:
    now = {"ms": 1_000}
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    predecessor = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        predecessor.materialize_population(_population(1))
        attempt = predecessor.claim_next()
        assert attempt is not None
        attempt = predecessor.commit_phase(attempt, "context")
        attempt, provider_result, duplicated = predecessor.run_provider(attempt)
        assert duplicated is False
        table = "provider_invocations"
        if callback_kind == "effect":
            attempt, _effect_result, duplicated = predecessor.run_effect(
                attempt,
                provider_result,
            )
            assert duplicated is False
            table = "effect_claims"
        connection = predecessor._require_connection()
        if result_state == "missing":
            connection.execute(
                f"DELETE FROM {table} WHERE attempt_id = ?",
                [attempt.attempt_id],
            )
        else:
            connection.execute(
                f"UPDATE {table} SET result_json = ? WHERE attempt_id = ?",
                ["{not-strict-json", attempt.attempt_id],
            )
        owner_session_id = predecessor.owner_session_id
    finally:
        predecessor.close()

    now["ms"] = 7_000
    successor = _open_daemon(
        tmp_path,
        provider_calls=provider_calls,
        effect_calls=effect_calls,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        assert successor.owner_session_id == owner_session_id
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        assert expired["callback_authority_incomplete"] is True
        for _pass in range(2):
            later = successor.run_once()
            assert later["implementation_result"] is None
        count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts"
        ).fetchone()
        assert count is not None and int(count[0]) == 1
        terminal = successor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert provider_calls == [attempt.task_cid]
        assert effect_calls == (
            [attempt.task_cid] if callback_kind == "effect" else []
        )
    finally:
        successor.close()


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


@pytest.mark.parametrize(
    "failure_window",
    ["promotion_response_loss", "local_complete_outage"],
)
def test_run_once_preserves_promoted_completion_for_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_window: str,
) -> None:
    """The non-crashing wrapper must not turn durable success into FAILED."""

    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session=f"session:wrapper-{failure_window}",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.materialize_population(_population(1))
        if failure_window == "promotion_response_loss":
            original_complete_claim = daemon.coordinator.complete_task_claim

            def lose_promotion_response(
                *args: object,
                **kwargs: object,
            ) -> object:
                original_complete_claim(*args, **kwargs)
                raise RuntimeError("simulated promotion response loss")

            monkeypatch.setattr(
                daemon.coordinator,
                "complete_task_claim",
                lose_promotion_response,
            )
        else:
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

        first = daemon.run_once()
        pending = first["implementation_result"]
        assert pending["status"] == "completion_reconciliation_pending"
        assert pending["retry_budget_consumed"] is False
        attempt_id = str(first["attempt_id"])
        claim_id = str(first["claim_id"])
        stored = daemon.get_attempt(attempt_id)
        assert stored is not None
        assert stored.status == "running"
        assert stored.committed_phase == "validation"
        promoted = daemon.coordinator.get_prepared_task_completion(
            stored.task_cid
        )
        assert promoted is not None
        assert promoted["status"] == "succeeded"

        if failure_window == "promotion_response_loss":
            monkeypatch.setattr(
                daemon.coordinator,
                "complete_task_claim",
                original_complete_claim,
            )
        else:
            monkeypatch.setattr(daemon, "commit_phase", original_commit_phase)

        second = daemon.run_once()
        assert len(second["completion_reconciliations"]) == 1
        repaired = daemon.get_attempt(attempt_id)
        assert repaired is not None
        assert repaired.status == "succeeded"
        assert repaired.committed_phase == ATTEMPT_PHASE_COMPLETE
        settled = daemon.coordinator.get_task_claim(claim_id)
        assert settled is not None
        assert settled.state.value in {"completed", "released"}
        assert provider_calls == [stored.task_cid]
        assert effect_calls == [stored.task_cid]
    finally:
        daemon.close()


def test_expired_preparation_without_control_cas_is_permanently_blocked(
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
        assert result["completion_reconciliations"][0]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert result["completion_reconciliations"][0]["retry_required"] is False
        assert result["implementation_result"] is None
        old_attempt = daemon.get_attempt(attempt.attempt_id)
        assert old_attempt is not None
        assert old_attempt.status == "failed"
        final_completion = daemon.coordinator.get_prepared_task_completion(
            attempt.task_cid
        )
        assert final_completion is None
        blocked = daemon.task_source.get(attempt.task_cid)
        assert blocked is not None and blocked.status == "blocked"
        assert blocked.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        daemon.close()

        callback_calls: list[str] = []

        def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
            callback_calls.append("callback")
            raise AssertionError("post-validation barrier was redispatched")

        daemon = _open_daemon(
            tmp_path,
            session="session:prepared-abort",
            provider_fn=forbidden,
            effect_fn=forbidden,
            validation_fn=forbidden,
            lease_ms=5_000,
            clock_ms=lambda: now["ms"],
        )
        for _pass in range(2):
            restarted = daemon.run_once()
            assert restarted["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_calls == []
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


def test_quack_runner_resolves_lane_private_database_paths(tmp_path: Path) -> None:
    def lane_args(index: int):
        return parse_args(
            [
                "--task-source-kind",
                "duckdb",
                "--authority-mode",
                "quack",
                "--database-path",
                str(tmp_path / "shared-control.duckdb"),
                "--coordination-path",
                str(tmp_path / "shared-coordination.duckdb"),
                "--quack-endpoint",
                "quack:127.0.0.1:45671",
                "--state-dir",
                str(tmp_path / f"lane-{index}"),
                "--state-prefix",
                f"pctdd_lane_{index}",
                "--once",
            ]
        )

    first = resolve_database_implementation_paths(lane_args(0))
    second = resolve_database_implementation_paths(lane_args(1))
    assert first["database_path"] == (
        tmp_path / "lane-0" / "quack-lane-control.duckdb"
    )
    assert second["database_path"] == (
        tmp_path / "lane-1" / "quack-lane-control.duckdb"
    )
    assert first["database_path"] != second["database_path"]
    assert first["coordination_path"] == (
        tmp_path / "lane-0" / "quack-lane-coordination.duckdb"
    )
    assert second["coordination_path"] == (
        tmp_path / "lane-1" / "quack-lane-coordination.duckdb"
    )
    assert first["coordination_path"] != second["coordination_path"]

    daemons = [
        DatabaseImplementationDaemon(
            database_path=paths["database_path"],
            coordination_path=paths["coordination_path"],
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri="quack:127.0.0.1:45671",
            control_store_id="store:quack-lane-test",
            control_store_generation="generation:quack-lane-test",
            install_schema=False,
        )
        for paths in (first, second)
    ]
    try:
        assert daemons[0].execution_path != daemons[1].execution_path
        assert daemons[0].coordination_path != daemons[1].coordination_path
        daemons[0]._acquire_embedded_writer_lock()
        daemons[1]._acquire_embedded_writer_lock()
    finally:
        for daemon in daemons:
            daemon.close()


def test_quack_builder_ignores_shared_database_keyword_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )

    captured: dict[str, object] = {}

    def fake_daemon(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(daemon_module, "DatabaseImplementationDaemon", fake_daemon)
    state_dir = tmp_path / "lane-2"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--database-path",
            str(tmp_path / "shared-cli.duckdb"),
            "--coordination-path",
            str(tmp_path / "shared-coordination.duckdb"),
            "--quack-endpoint",
            "quack:127.0.0.1:45671",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "generation-test",
            "--state-schema-revision",
            "schema-test",
            "--state-dir",
            str(state_dir),
            "--once",
        ]
    )

    build_database_implementation_daemon_from_args(
        args,
        database_path=tmp_path / "shared-keyword.duckdb",
    )
    assert captured["database_path"] == state_dir / "quack-lane-control.duckdb"
    assert captured["coordination_path"] == (
        state_dir / "quack-lane-coordination.duckdb"
    )


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
            "--task-prefix",
            "DQP-",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon, context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.max_task_attempts == 3
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
