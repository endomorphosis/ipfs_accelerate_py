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
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
    checkout_mutation_lock_path,
)
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
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
    TaskSourceConflictError as DatabaseTaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    connect_duckdb_with_policy,
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    COMPLETED_STATUSES as TASK_SOURCE_COMPLETED_STATUSES,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon_runner as daemon_runner,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_CONSUMED_NO_PROGRESS_SCHEMA,
    DATABASE_PORTAL_VALIDATION_RETRY_SCHEMA,
    DatabasePortalBridgeConsumedNoProgressError,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalCandidateRetry,
    DatabasePortalValidationRetry,
    database_portal_consumed_no_progress_fingerprint,
    database_portal_task_contract_digest,
    is_protected_checkout_setup_block,
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
    DatabaseImplementationCoordinationDriftError,
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
    max_task_attempts: int = 0,
    clock_ms: Callable[[], int] | None = None,
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
    control_path: Path | None = None,
    repo_root: Path | None = None,
    merge_target_ref: str = "HEAD",
    task_prefix: str = "",
) -> DatabaseImplementationDaemon:
    database_path = control_path or (tmp_path / "control.duckdb")
    coordination_path = tmp_path / "coordination.duckdb"
    execution_path = tmp_path / "execution.duckdb"

    def default_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        if provider_calls is not None:
            provider_calls.append(attempt.task_cid)
        return {
            "status": "ok",
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
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        provider_fn=provider_fn or default_provider,
        effect_fn=effect,
        validation_fn=validation,
        require_real_execution=True,
        clock_ms=clock_ms,
        repo_root=repo_root,
        merge_target_ref=merge_target_ref,
        task_prefix=task_prefix,
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


def test_database_completion_seed_vocabulary_matches_task_source() -> None:
    daemon_module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert daemon_module._DATABASE_SUCCESSFUL_CONTROL_STATUSES is (
        TASK_SOURCE_COMPLETED_STATUSES
    )
    assert TASK_SOURCE_COMPLETED_STATUSES == frozenset(
        {"completed", "complete", "done", "skipped"}
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


def test_fresh_strict_lane_seeds_completed_dependency_before_claim(
    tmp_path: Path,
) -> None:
    """A lane-private coordinator reconstructs dependency evidence from control."""

    population = _population(4)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0].update(
        {
            "task_id": "DQP-COMPLETE",
            "status": "skipped",
        }
    )
    tasks[1].update(
        {
            "task_id": "DQP-DEPENDENT",
            "dependencies": ["task:cid:001"],
        }
    )
    tasks[2]["task_id"] = "DQP-OTHER-1"
    tasks[3].update(
        {
            "task_id": "DQP-BLOCKED",
            "status": "blocked",
        }
    )

    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(population, repository_tree_id="tree:dqp-fresh-lane")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane-0" / "coordination.duckdb",
        execution_path=tmp_path / "lane-0" / "execution.duckdb",
        owner_session_id="session:fresh-strict-lane",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        task_shard_count=2,
        task_shard_index=0,
        strict_task_sharding=True,
        require_real_execution=True,
    )
    try:
        # The prerequisite and off-shard task hash to lane 1; the dependent
        # and blocked task hash to lane 0.  Only the ready dependent is exposed
        # through the lane's Quack-authoritative eligibility sequence.
        assert daemon._task_home_shard_index("DQP-COMPLETE") == 1
        assert daemon._task_home_shard_index("DQP-DEPENDENT") == 0
        assert daemon._task_home_shard_index("DQP-OTHER-1") == 1
        assert daemon._task_home_shard_index("DQP-BLOCKED") == 0
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:002"]

        projection = daemon.coordinator.coordination_registry_projection()
        assert {item["task_cid"] for item in projection["tasks"]} == {
            "task:cid:001",
            "task:cid:002",
        }
        assert projection["logical_completions"] == [
            {
                "task_cid": "task:cid:001",
                "status": "succeeded",
                "body": {
                    "authority": "database_task_source",
                    "source_status": "skipped",
                    "task_alias": "DQP-COMPLETE",
                    "task_revision": 1,
                },
            }
        ]
        assert daemon.sync_ready_tasks_into_coordination() == ["task:cid:002"]
        assert daemon.coordinator.coordination_registry_projection() == projection

        attempt = daemon.claim_next()
        assert attempt is not None
        assert attempt.task_cid == "task:cid:002"
        # Neither the same-shard blocked task nor the ready task owned by the
        # other strict shard may be claimed from this private coordinator.
        assert daemon.claim_next() is None
        blocked = source.get_task("task:cid:004")
        off_shard = source.get_task("task:cid:003")
        assert blocked is not None and blocked.status == "blocked"
        assert off_shard is not None and off_shard.status == "ready"
    finally:
        daemon.close()
        source.close()


def test_conflicting_local_completion_refuses_coordination_drift(
    tmp_path: Path,
) -> None:
    population = _population(2)
    tasks = population["tasks"]
    assert isinstance(tasks, list)
    tasks[0]["status"] = "skipped"
    tasks[1]["dependencies"] = ["task:cid:001"]

    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(population, repository_tree_id="tree:dqp-local-drift")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane" / "coordination.duckdb",
        execution_path=tmp_path / "lane" / "execution.duckdb",
        owner_session_id="session:local-drift",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        require_real_execution=True,
    )
    try:
        daemon.coordinator.register_task(
            task_cid="task:cid:001",
            task_id="DQP-T001",
        )
        daemon.coordinator.mark_task_complete(
            "task:cid:001",
            status="failed",
            body={"source": "stale-lane-local-evidence"},
        )

        with pytest.raises(
            DatabaseImplementationCoordinationDriftError,
            match="lane-local completion contradicts",
        ):
            daemon.claim_next()
        assert daemon.list_running_attempts() == []
        dependent = source.get_task("task:cid:002")
        assert dependent is not None and dependent.status == "ready"
    finally:
        daemon.close()
        source.close()


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


def _git_repo_with_output(tmp_path: Path, relative: str = "landed.py") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Daemon Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "daemon-test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    output = repo / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("landed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", relative],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "landed output"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def test_landed_quarantined_task_with_outputs_is_completed(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_output(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "landed.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt={
                "operation": "database_portal_neutral_failure_quarantine",
                "retry_suppressed": True,
            },
        )
        result = daemon.run_once()
        repaired = result["landed_merge_reconciliations"]
        assert repaired
        assert repaired[0]["completed"] is True
        assert repaired[0]["task_cid"] == "task:cid:001"
        completed = daemon.task_source.get("task:cid:001")
        assert completed is not None
        assert completed.status == "completed"
        assert result["selection_idle_reason"] == "no_ready_tasks"
    finally:
        daemon.close()


def test_consumed_no_progress_completes_when_declared_outputs_already_landed(
    tmp_path: Path,
) -> None:
    repo = _git_repo_with_output(tmp_path)
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    daemon = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:landed-outputs",
        provider_fn=consumed_failure,
        repo_root=repo,
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "landed.py"}]
        daemon.materialize_population(population)
        attempted = daemon.claim_next()
        assert attempted is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                daemon,
                attempted,
                tag="already-landed-outputs",
            )
        )
        result = daemon._resume_attempt_without_process_crash(attempted)
        assert result["task_quarantined"] is False
        assert result["landed_outputs_completed"] is True
        assert result["status"] == "succeeded"
        task = daemon.task_source.get(attempted.task_cid)
        assert task is not None
        assert task.status == "completed"
    finally:
        daemon.close()


def test_reopened_quarantine_retires_stale_blocked_attempt(
    tmp_path: Path,
) -> None:
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    failure_evidence: dict[str, object] = {}

    def consumed_failure(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeConsumedNoProgressError(
            "portal_consumed_no_progress",
            failure_evidence=failure_evidence,
        )

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:reopen-stale-block",
        provider_fn=consumed_failure,
    )
    try:
        first.materialize_population(_population(1))
        attempted = first.claim_next()
        assert attempted is not None
        failure_evidence.update(
            _consumed_no_progress_evidence(
                first,
                attempted,
                tag="reopen-stale-block",
            )
        )
        first._resume_attempt_without_process_crash(attempted)
        blocked = first.get_attempt(attempted.attempt_id)
        assert blocked is not None
        assert blocked.status == "blocked"
        task = first.task_source.get(blocked.task_cid)
        assert task is not None
        assert task.status == "quarantined"
        first.task_source.compare_and_set_status(
            blocked.task_cid,
            int(task.revision),
            "todo",
            receipt={
                "operation": "reopen_unimplemented_unknown_callback_quarantine",
                "reason": "declared_outputs_missing_on_merge_target",
            },
        )
        reopened = first.task_source.get(blocked.task_cid)
        assert reopened is not None
        assert reopened.status == "todo"

        outcomes = first.reconcile_expired_running_attempts()
        retired = [
            item
            for item in outcomes
            if item.get("reason") == "control_task_left_quarantine"
        ]
        assert retired
        assert retired[0]["attempt_id"] == blocked.attempt_id
        assert retired[0]["status"] == "failed"
        stale = first.get_attempt(blocked.attempt_id)
        assert stale is not None
        assert stale.status == "failed"
        current = first.task_source.get(blocked.task_cid)
        assert current is not None
        assert current.status == "todo"
        assert first.list_running_attempts() == []
    finally:
        first.close()


def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Daemon Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "daemon-test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def _unknown_callback_quarantine_receipt() -> dict[str, object]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-neutral-quarantine@1"
        ),
        "operation": "database_portal_neutral_failure_quarantine",
        "failure_kind": "provider_callback_outcome_unknown",
        "retry_suppressed": True,
        "root_cause_required": True,
        "provider_effect_state": "unknown_may_have_started",
        "unknown_callback_reopen_count": 0,
    }


def test_unknown_callback_without_landed_outputs_reopens(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        result = daemon.run_once()
        reopened = result["unknown_callback_reopens"]
        assert reopened
        assert reopened[0]["reopened"] is True
        assert reopened[0]["task_cid"] == "task:cid:001"
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status != "quarantined"
    finally:
        daemon.close()


def test_unknown_callback_without_declared_outputs_stays_quarantined(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        daemon.materialize_population(_population(1))
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        result = daemon.run_once()
        assert result["unknown_callback_reopens"] == []
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status == "quarantined"
        assert result["selection_idle_reason"] == "no_ready_tasks"
    finally:
        daemon.close()


def test_unknown_callback_reopen_count_survives_later_claim_receipt(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=_unknown_callback_quarantine_receipt(),
        )
        first = daemon.run_once()
        assert first["unknown_callback_reopens"]
        assert first["unknown_callback_reopens"][0]["unknown_callback_reopen_count"] == 1
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "in_progress",
            receipt={
                "operation": "database_claim",
                "claim_id": "claim:fresh",
                "attempt_id": "attempt:fresh",
            },
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        claim_receipt = task.body.get("completion_receipt")
        assert isinstance(claim_receipt, dict)
        assert claim_receipt.get("unknown_callback_reopen_count") == 1
        receipt = _unknown_callback_quarantine_receipt()
        receipt.pop("unknown_callback_reopen_count", None)
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=receipt,
        )
        second = daemon.run_once()
        assert second["unknown_callback_reopens"]
        assert second["unknown_callback_reopens"][0]["unknown_callback_reopen_count"] == 2
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        current_receipt = current.body.get("completion_receipt")
        assert isinstance(current_receipt, dict)
        assert current_receipt.get("unknown_callback_reopen_count") == 2
    finally:
        daemon.close()


def test_unknown_callback_quarantine_receipt_count_does_not_block_reopen(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        receipt = _unknown_callback_quarantine_receipt()
        receipt["unknown_callback_reopen_count"] = 4
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=receipt,
        )
        result = daemon.run_once()
        reopened = result["unknown_callback_reopens"]
        assert reopened
        assert reopened[0]["reopened"] is True
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status != "quarantined"
    finally:
        daemon.close()


def test_unaccepted_unknown_callback_is_retired_not_quarantined(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    now = {"ms": 1_000}
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"
    repo = _git_repo(tmp_path)

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        raise SimulatedProcessCrash("injected unaccepted-claim crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:unaccepted-unknown",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        repo_root=repo,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        first.materialize_population(population)
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(SimulatedProcessCrash):
            first._resume_attempt_without_process_crash(attempt)
        now["ms"] = 10_000
        claim = first.coordinator.get_task_claim(attempt.claim_id)
        assert claim is not None
        first.coordinator.expire_task_claim(claim, now_ms=now["ms"])
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "in_progress",
            receipt={
                "operation": "database_claim",
                "claim_id": "claim:other-owner",
                "attempt_id": "attempt:other-owner",
            },
        )
    finally:
        first.close()

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:unaccepted-unknown",
        provider_fn=lambda attempt: {"status": "ok", "task_cid": attempt.task_cid},
        strict_task_sharding=True,
        repo_root=repo,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        replay = restarted.run_once()
        current = restarted.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        retired = [
            item
            for item in replay["expired_attempt_reconciliations"]
            if item.get("attempt_id") == attempt.attempt_id
        ]
        assert retired
        assert retired[0]["reason"] != "portal_neutral_failure"
    finally:
        restarted.close()


def test_portal_setup_error_requeues_instead_of_unknown_callback_quarantine(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=repo,
        provider_fn=lambda attempt: (_ for _ in ()).throw(
            DatabasePortalBridgeError(
                "external_protected_checkout_recovery_required"
            )
        ),
        strict_task_sharding=True,
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result["retryable"] is True
        assert "external_protected_checkout_recovery_required" in str(
            result.get("reason") or ""
        )
        current = daemon.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        stale = daemon.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status != "running"
    finally:
        daemon.close()


def _write_supervisor_protected_recovery_journal(repo: Path) -> Path:
    lock_path = checkout_mutation_lock_path(repo)
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="APMC-013",
        attempt=1,
        extra={"operation": "generated_board_update"},
    )
    metadata.update(
        {
            "pid": 999999999,
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_supervisor",
            "protected_paths": ["docs/generated.todo.md"],
        }
    )
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    return lock_path


def test_protected_checkout_setup_block_classifier() -> None:
    assert is_protected_checkout_setup_block(
        "external_protected_checkout_recovery_required"
    )
    assert is_protected_checkout_setup_block(
        "DatabasePortalBridgeError: external protected checkout recovery required"
    )
    assert not is_protected_checkout_setup_block("portal_consumed_no_progress")


def test_supervisor_recovery_journal_defers_before_callback_intent(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    lock_path = _write_supervisor_protected_recovery_journal(repo)
    provider_calls: list[str] = []

    def provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.task_cid)
        return {"status": "ok", "task_cid": attempt.task_cid}

    daemon = _open_daemon(
        tmp_path / "lane",
        repo_root=repo,
        provider_fn=provider,
        strict_task_sharding=True,
    )
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        attempt = daemon.claim_next()
        assert attempt is not None
        result = daemon._resume_attempt_without_process_crash(attempt)
        assert result["retryable"] is True
        assert "external_protected_checkout_recovery_required" in str(
            result.get("reason") or ""
        )
        current = daemon.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        assert current.status == "todo"
        assert provider_calls == []
        recorded = daemon.provider_invocation_recorded(
            attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        assert recorded is None
        assert lock_path.is_file()
        stale = daemon.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status != "running"
    finally:
        daemon.close()


def test_unknown_callback_reopen_continues_while_outputs_are_missing(
    tmp_path: Path,
) -> None:
    repo = _git_repo(tmp_path)
    daemon = _open_daemon(tmp_path / "lane", repo_root=repo)
    try:
        population = _population(1)
        tasks = population["tasks"]
        assert isinstance(tasks, list)
        tasks[0]["outputs"] = [{"path": "missing.py"}]
        daemon.materialize_population(population)
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "todo",
            receipt={
                "operation": "reopen_unimplemented_unknown_callback_quarantine",
                "unknown_callback_reopen_count": 4,
            },
        )
        task = daemon.task_source.get("task:cid:001")
        assert task is not None
        receipt = _unknown_callback_quarantine_receipt()
        daemon.task_source.compare_and_set_status(
            "task:cid:001",
            int(task.revision),
            "quarantined",
            receipt=receipt,
        )
        result = daemon.run_once()
        reopened = result["unknown_callback_reopens"]
        assert reopened
        assert reopened[0]["reopened"] is True
        current = daemon.task_source.get("task:cid:001")
        assert current is not None
        assert current.status != "quarantined"
    finally:
        daemon.close()


def test_reopened_task_retires_stale_running_unknown_callback(
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
        raise SimulatedProcessCrash("injected leftover running crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:stale-running-reopen",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(
            SimulatedProcessCrash,
            match="injected leftover running crash",
        ):
            first._resume_attempt_without_process_crash(attempt)
        running = first.get_attempt(attempt.attempt_id)
        assert running is not None and running.status == "running"
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "todo",
            receipt={
                "operation": "reopen_unimplemented_unknown_callback_quarantine",
                "unknown_callback_reopen_count": 1,
            },
        )
    finally:
        first.close()

    def success_provider(attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append(attempt.attempt_id)
        return {"status": "ok", "task_cid": attempt.task_cid}

    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:stale-running-reopen",
        provider_fn=success_provider,
        strict_task_sharding=True,
    )
    try:
        replay = restarted.run_once()
        retired = [
            item
            for item in replay["expired_attempt_reconciliations"]
            if item.get("reason") == "control_task_left_in_progress"
        ]
        assert retired
        assert retired[0]["attempt_id"] == attempt.attempt_id
        stale = restarted.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status == "failed"
        current = restarted.task_source.get(attempt.task_cid)
        assert current is not None
        assert current.status != "quarantined"
        assert attempt.attempt_id not in provider_calls[1:]
    finally:
        restarted.close()


def test_expired_unknown_callback_with_rebound_in_progress_is_retired(
    tmp_path: Path,
) -> None:
    class SimulatedProcessCrash(BaseException):
        pass

    now = {"ms": 1_000}
    control_path = tmp_path / "control.duckdb"
    lane_path = tmp_path / "lane"

    def crash_after_callback_started(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        raise SimulatedProcessCrash("injected rebound in-progress crash")

    first = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-in-progress",
        provider_fn=crash_after_callback_started,
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        first.materialize_population(_population(1))
        attempt = first.claim_next()
        assert attempt is not None
        with pytest.raises(SimulatedProcessCrash):
            first._resume_attempt_without_process_crash(attempt)
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "todo",
            receipt={"operation": "reopen_unimplemented_unknown_callback_quarantine"},
        )
        task = first.task_source.get(attempt.task_cid)
        assert task is not None
        first.task_source.compare_and_set_status(
            attempt.task_cid,
            int(task.revision),
            "in_progress",
            receipt={
                "operation": "database_claim",
                "claim_id": "claim:rebound-owner",
                "attempt_id": "attempt:rebound-owner",
            },
        )
    finally:
        first.close()

    now["ms"] = 7_000
    restarted = _open_daemon(
        lane_path,
        control_path=control_path,
        session="session:rebound-in-progress",
        provider_fn=lambda attempt: {"status": "ok", "task_cid": attempt.task_cid},
        strict_task_sharding=True,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        replay = restarted.run_once()
        retired = [
            item
            for item in replay["expired_attempt_reconciliations"]
            if item.get("reason") == "control_task_left_in_progress"
        ]
        assert retired
        assert retired[0]["attempt_id"] == attempt.attempt_id
        stale = restarted.get_attempt(attempt.attempt_id)
        assert stale is not None
        assert stale.status == "failed"
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


def test_unusable_candidate_retries_instead_of_blocking(tmp_path: Path) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalCandidateRetry("no_change_completion_not_allowed")

    daemon = _open_daemon(
        tmp_path,
        session="session:candidate-retry",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        result = daemon.run_once()
        implementation = result["implementation_result"]
        assert implementation["portal_retryable_failure"] is True
        assert implementation["portal_terminal_failure"] is False
        assert implementation["attempt_consumed"] is True
        assert implementation["provider_dispatched"] is True
        assert implementation["retry_state"]["status"] == "retrying"
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt is not None
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        failed = daemon.phase_history(attempt.attempt_id)[-1]["body"]
        assert failed["reason"] == "no_change_completion_not_allowed"
        assert failed["portal_retryable_failure"] is True
    finally:
        daemon.close()


def test_reconcile_rearms_blocked_portal_provider_failed(tmp_path: Path) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError("portal_provider_failed")

    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-portal-provider-failed",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["changed"] is True
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert daemon.reconcile_terminal_portal_failures() == []
    finally:
        daemon.close()


def test_reconcile_rearms_blocked_checkout_contention(tmp_path: Path) -> None:
    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        raise DatabasePortalBridgeError(
            "external_protected_checkout_recovery_required"
        )

    daemon = _open_daemon(
        tmp_path,
        session="session:rearm-checkout-contention",
        provider_fn=provider,
        max_task_attempts=4,
    )
    try:
        daemon.materialize_population(_population(1))
        failed_result = daemon.run_once()
        attempt = daemon.get_attempt(failed_result["attempt_id"])
        assert attempt is not None
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"

        outcomes = daemon.reconcile_terminal_portal_failures()
        assert len(outcomes) == 1
        assert outcomes[0]["status"] == "retrying"
        assert outcomes[0]["changed"] is True
        assert outcomes[0]["evidence_source"] == (
            "portal_checkout_contention_reclassified"
        )
        task = daemon.task_source.get(attempt.task_cid)
        assert task is not None
        assert task.status == "retrying"
        assert daemon.reconcile_terminal_portal_failures() == []
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
        recent = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime(
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


@pytest.mark.parametrize(
    ("mutation", "error_type"),
    [
        ("foreign_operation", DatabaseImplementationConflictError),
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
        if mutation == "foreign_operation":
            receipt["operation"] = "database_portal_retry"
        elif mutation == "missing_seed":
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


def test_current_control_recheck_rejects_task_that_became_manual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path, session="session:manual-race")
    try:
        daemon.materialize_population(_population(1))
        source = daemon.task_source
        original_get = source.get
        mutated = False

        def become_manual(task_cid: str) -> object:
            nonlocal mutated
            current = original_get(task_cid)
            if current is not None and not mutated:
                source._intent.upsert_task(
                    task_cid=current.task_cid,
                    task_alias=current.task_alias,
                    goal_cid=current.goal_cid,
                    ordinal=current.ordinal,
                    status=current.status,
                    priority=current.priority,
                    plan_cid=current.plan_cid,
                    objective_id=current.objective_id,
                    body={**dict(current.body), "completion": "manual"},
                    identity={"task_cid": current.task_cid},
                    expected_revision=current.revision,
                    dependencies=current.dependencies,
                    outputs=current.outputs,
                    acceptance=current.acceptance,
                    validations=current.validations,
                )
                mutated = True
            return original_get(task_cid)

        monkeypatch.setattr(source, "get", become_manual)
        assert daemon.claim_next() is None
        assert mutated is True
        assert daemon.list_running_attempts() == []
        projection = daemon.coordinator.coordination_registry_projection()
        assert projection["counts"]["active_task_claims"] == 0
        current = original_get("task:cid:001")
        assert current is not None
        assert current.status == "ready"
        assert current.body["completion"] == "manual"
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


def test_quack_runner_builders_use_lane_private_sidecars(tmp_path: Path) -> None:
    control = tmp_path / "control.duckdb"

    def lane_args(index: int):
        return parse_args(
            [
                "--task-source-kind",
                "duckdb",
                "--authority-mode",
                "quack",
                "--database-path",
                str(control),
                "--endpoint-secret-handle",
                "handle:test-quack",
                "--quack-endpoint",
                "quack:127.0.0.1:45671",
                "--state-store-id",
                "state/control.duckdb",
                "--state-store-generation",
                "test-generation",
                "--state-schema-revision",
                "datasets-authoritative-operational-v1",
                "--todo-path",
                str(tmp_path / "unused.md"),
                "--state-dir",
                str(tmp_path / f"lane-{index}"),
                "--state-prefix",
                f"pcpc_lane_{index}",
                "--task-shard-count",
                "4",
                "--task-shard-index",
                str(index),
                "--strict-task-sharding",
                "--once",
            ]
        )

    first, _first_context = build_portal_implementation_daemon_from_args(
        lane_args(0),
        repo_root=tmp_path,
    )
    second, _second_context = build_portal_implementation_daemon_from_args(
        lane_args(1),
        repo_root=tmp_path,
    )
    third = build_database_implementation_daemon_from_args(
        lane_args(2),
        database_path=control,
    )
    try:
        assert isinstance(first, DatabaseImplementationDaemon)
        assert isinstance(second, DatabaseImplementationDaemon)
        assert isinstance(third, DatabaseImplementationDaemon)
        assert first.database_path == second.database_path == third.database_path == control
        assert first.execution_path != second.execution_path
        assert first.coordination_path != second.coordination_path
        assert third.execution_path not in {first.execution_path, second.execution_path}
        assert third.coordination_path not in {
            first.coordination_path,
            second.coordination_path,
        }
        assert first.execution_path.parent == tmp_path / "lane-0"
        assert second.execution_path.parent == tmp_path / "lane-1"
        assert third.execution_path.parent == tmp_path / "lane-2"
        assert first.coordination_path.parent == tmp_path / "lane-0"
        assert second.coordination_path.parent == tmp_path / "lane-1"
        assert third.coordination_path.parent == tmp_path / "lane-2"
        assert first.strict_task_sharding is True
        assert second.strict_task_sharding is True
        assert third.strict_task_sharding is True
        assert first.task_shard_index == 0
        assert second.task_shard_index == 1
        assert third.task_shard_index == 2
    finally:
        first.close()
        second.close()
        third.close()


def test_database_lanes_register_disjoint_hash_shards(tmp_path: Path) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(_population(8), repository_tree_id="tree:dqp-018")
    daemons = [
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / f"lane-{index}" / "coordination.duckdb",
            execution_path=tmp_path / f"lane-{index}" / "execution.duckdb",
            owner_session_id=f"session:lane:{index}",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_source=source,
            task_shard_count=2,
            task_shard_index=index,
            strict_task_sharding=True,
        )
        for index in range(2)
    ]
    try:
        observed = [set(daemon.sync_ready_tasks_into_coordination()) for daemon in daemons]
        expected = [set(), set()]
        for index in range(1, 9):
            alias = f"DQP-T{index:03d}"
            task_cid = f"task:cid:{index:03d}"
            digest = hashlib.sha256(alias.encode("utf-8")).hexdigest()
            expected[int(digest[:8], 16) % 2].add(task_cid)
        assert observed == expected
        assert observed[0].isdisjoint(observed[1])
        assert observed[0] | observed[1] == {
            f"task:cid:{index:03d}" for index in range(1, 9)
        }
    finally:
        for daemon in daemons:
            daemon.close()
        source.close()


def test_already_in_progress_control_task_creates_no_execution_attempt(
    tmp_path: Path,
) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(_population(1), repository_tree_id="tree:dqp-018")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane" / "coordination.duckdb",
        execution_path=tmp_path / "lane" / "execution.duckdb",
        owner_session_id="session:stale-local-ready",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        require_real_execution=True,
    )
    try:
        daemon.open()
        task = source.get_task("task:cid:001")
        assert task is not None
        daemon.coordinator.register_task(
            task_cid=task.task_cid,
            task_id=task.task_alias,
        )
        source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "in_progress",
            {"operation": "competing_lane_claim"},
        )
        assert daemon.claim_next() is None
        assert daemon.list_running_attempts() == []
        current = source.get_task(task.task_cid)
        assert current is not None
        assert current.status == "in_progress"
    finally:
        daemon.close()
        source.close()


def test_authoritative_cas_race_creates_no_execution_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    source.materialize(_population(1), repository_tree_id="tree:dqp-018")
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "lane" / "coordination.duckdb",
        execution_path=tmp_path / "lane" / "execution.duckdb",
        owner_session_id="session:cas-race",
        authority_mode="embedded",
        task_source_kind="duckdb",
        task_source=source,
        require_real_execution=True,
    )
    try:
        daemon.open()

        def reject_cas(*_args, **_kwargs):
            raise DatabaseTaskSourceConflictError("competing control CAS won")

        monkeypatch.setattr(source, "compare_and_set_status", reject_cas)
        assert daemon.claim_next() is None
        assert daemon.list_running_attempts() == []
        task = source.get_task("task:cid:001")
        assert task is not None
        assert task.status == "ready"
    finally:
        daemon.close()
        source.close()


def test_portal_bridge_failure_requeues_exact_claim_for_later_reclaim(
    tmp_path: Path,
) -> None:
    now = {"ms": 1_000}
    provider_attempts: list[DatabaseTaskAttempt] = []

    def fail_first_portal_pass(
        attempt: DatabaseTaskAttempt,
    ) -> dict[str, object]:
        provider_attempts.append(attempt)
        if len(provider_attempts) == 1:
            raise DatabasePortalBridgeDeferred(
                "launch_redteam_forced_retryable_portal_bridge_failure",
                backoff_seconds=1,
            )
        return {
            "status": "ok",
            "accepted": True,
            "task_cid": attempt.task_cid,
        }

    daemon = _open_daemon(
        tmp_path,
        session="session:portal-requeue",
        provider_fn=fail_first_portal_pass,
        lease_ms=5_000,
        clock_ms=lambda: now["ms"],
    )
    try:
        daemon.materialize_population(_population(1))

        first_pass = daemon.run_once()
        first_result = first_pass["implementation_result"]
        assert first_result["portal_retryable_failure"] is True
        assert first_result["status"] == "failed"
        assert first_result["deferred"] is True
        assert first_result["retry_state"]["changed"] is True
        assert first_result["retry_state"]["status"] == "retrying"
        assert first_result["retry_state"]["backoff_seconds"] == 1

        first_attempt_id = str(first_result["attempt_id"])
        first_attempt = daemon.get_attempt(first_attempt_id)
        assert first_attempt is not None
        assert first_attempt.status == "failed"
        assert first_attempt.committed_phase == "failed"
        failure_phases = [
            phase
            for phase in daemon.phase_history(first_attempt_id)
            if phase["phase"] == "failed"
        ]
        assert len(failure_phases) == 1
        failure_receipt = failure_phases[0]["body"]
        assert failure_receipt["portal_retryable_failure"] is True
        assert failure_receipt["portal_terminal_failure"] is False
        assert failure_receipt["deferred"] is True
        assert failure_receipt["attempt_consumed"] is False
        assert failure_receipt["provider_dispatched"] is False
        assert failure_receipt["typed_deferral_slot_consumed"] is True
        assert failure_receipt["backoff_seconds"] == 1
        assert failure_receipt["typed_deferral"]["attempt_id"] == first_attempt_id

        initial_claim = daemon.coordinator.get_task_claim(first_attempt.claim_id)
        initial_coordination_attempt = daemon.coordinator.get_task_attempt(
            first_attempt_id
        )
        assert initial_claim is not None
        assert initial_coordination_attempt is not None
        assert initial_claim.state.value == "accepted"
        assert initial_coordination_attempt.status.value == "running"
        control_after_failure = daemon.task_source.get(first_attempt.task_cid)
        assert control_after_failure is not None
        assert control_after_failure.status == "retrying"
        assert daemon.list_running_attempts() == []

        now["ms"] = 6_001
        second_pass = daemon.run_once()
        second_result = second_pass["implementation_result"]
        assert second_result["status"] == "succeeded"
        second_attempt = second_result["attempt"]
        assert second_attempt["attempt_id"] != first_attempt_id
        assert second_attempt["attempt_number"] == 2
        assert second_attempt["fencing_token"] > first_attempt.fencing_token
        assert len(provider_attempts) == 2
        expired_claim = daemon.coordinator.get_task_claim(first_attempt.claim_id)
        assert expired_claim is not None
        assert expired_claim.state.value == "expired"
        final_control = daemon.task_source.get(first_attempt.task_cid)
        assert final_control is not None
        assert final_control.status == "completed"
    finally:
        daemon.close()


def test_database_daemon_rejects_idle_lane_work_stealing(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="does not support idle-lane work stealing",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            authority_mode="embedded",
            task_source_kind="duckdb",
            task_shard_count=2,
            task_shard_index=0,
            strict_task_sharding=True,
            idle_lane_work_stealing="virgin-transfer",
        )
