"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
import hashlib
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DatabaseTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA,
    DatabasePortalBridgeDeferred,
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    database_portal_bridge as database_portal_bridge_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationConflictError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    PortalTaskState,
    parse_args,
    parse_task_file,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    bind_database_portal_execution_from_args,
    build_portal_implementation_daemon_from_args,
)


def _attempt() -> DatabaseTaskAttempt:
    return DatabaseTaskAttempt(
        attempt_id="attempt:001",
        claim_id="claim:001",
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        attempt_number=1,
        owner_session_id="session:bridge",
        fencing_token=7,
        fence_epoch=3,
        lease_id="lease:001",
        committed_phase="claimed",
        status="running",
        started_at_ms=1,
    )


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        task_cid="task:cid:004",
        task_alias="LGSWF-004",
        goal_cid="goal:inventory",
        plan_cid="plan:lgswf:1",
        revision=11,
        priority="P0",
        dependencies=("task:cid:003",),
        outputs=({"path": "inventory/result.json"},),
        validations=({"argv": ["python3", "-m", "pytest", "focused.py"]},),
        acceptance=({"criterion": "Focused validation passes"},),
        body={
            "objective": "Produce the current authority inventory",
            "completion": "auto",
            "track": "analysis",
            "read_scope": ["ipfs_accelerate_py/agent_supervisor"],
            "write_scope": ["inventory/result.json"],
            "completion_contract": "Focused validation passes",
        },
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _reconciliation_receipt(paths: object, receipt_id: str) -> dict[str, object]:
    receipt_path = Path(paths.reconciliation) / (
        receipt_id.removeprefix("sha256:") + ".json"
    )
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def _seed_interrupted_database_portal_attempt(
    tmp_path: Path,
    *,
    owner_session_id: str = "",
    seed_nested_state: bool = True,
) -> tuple[
    Path,
    DatabaseImplementationDaemon,
    DatabasePortalExecutionBridge,
    DatabaseTaskAttempt,
    object,
]:
    from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
        ProcessBirthIdentity,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    state_dir = repo / "state"
    worktree_root = repo / "worktrees"
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(repo / "control.duckdb"),
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "pctdd",
            "--task-prefix",
            "## PCTDD-",
            "--worktree-root",
            str(worktree_root),
            "--merge-target-branch",
            "main",
            "--implement",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon = DatabaseImplementationDaemon(
        database_path=repo / "control.duckdb",
        max_task_attempts=3,
        owner_session_id=owner_session_id,
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        task_prefix="PCTDD-",
        require_real_execution=True,
    )
    bridge = bind_database_portal_execution_from_args(
        daemon,
        args,
        repo_root=repo,
        portal_daemon_class=PortalImplementationDaemon,
    )
    assert isinstance(bridge, DatabasePortalExecutionBridge)
    daemon.materialize_population(
        {
            "repository_tree_id": "tree:shutdown-reconciliation",
            "tasks": [
                {
                    "task_cid": "task:cid:pctdd-001",
                    "task_id": "PCTDD-001",
                    "goal_cid": "goal:pctdd",
                    "status": "ready",
                    "validation_commands": ["python -m pytest focused.py"],
                }
            ],
        }
    )
    attempt = daemon.claim_next()
    assert attempt is not None
    record = daemon.task_source.get_task(attempt.task_cid)
    assert record is not None
    paths, _binding = bridge._ensure_attempt_projection(attempt, record)
    if not seed_nested_state:
        return repo, daemon, bridge, attempt, paths
    task = parse_task_file(
        paths.task_projection,
        task_header_prefix="## PCTDD-001",
    )[0]
    branch = "implementation/pctdd-001-stale-attempt-1"
    worktree = worktree_root / "pctdd-001-stale"
    _git(repo, "worktree", "add", "-b", branch, str(worktree), "HEAD")
    portal = PortalImplementationDaemon(
        todo_path=paths.task_projection,
        state_path=paths.state,
        strategy_path=paths.strategy,
        events_path=paths.events,
        repo_root=repo,
        task_header_prefix="## PCTDD-",
        implement=False,
        worktree_root=worktree_root,
        merge_target_branch="main",
        execution_slice_task_ids=("PCTDD-001",),
    )
    identity = portal._identity_for_task(task)
    lifecycle = portal.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=1,
        lane_id="terminated-database-lane",
        workspace_path=worktree,
        branch=branch,
        merge_target="main",
        state_dir=str(paths.state.parent.resolve()),
        owner=ProcessBirthIdentity(
            pid=2**30 - 73,
            start_time_ticks=1,
            boot_id="dead-database-portal-owner",
        ),
    )
    lifecycle = portal.worktree_lifecycle.mark_active(
        worktree,
        lease_id=lifecycle.lease_id,
        expected_fence=lifecycle.fence,
    )
    PortalTaskState(
        task_statuses={task.task_id: "in_progress"},
        task_identities={
            task.task_id: {
                "canonical_task_key": task.canonical_task_key,
                "canonical_task_cid": task.canonical_task_cid,
                "board_namespace": task.board_namespace,
            }
        },
        implementation_attempts={task.task_id: 1},
        implementation_attempts_by_cid={task.canonical_task_cid: 1},
    ).save(paths.state)
    portal.close_event_runtime()
    return repo, daemon, bridge, attempt, paths


def _activate_nested_portal_state(
    repo: Path,
    paths: object,
    *,
    active_provider_runner: dict[str, object] | None = None,
) -> PortalTaskState:
    task = parse_task_file(
        paths.task_projection,
        task_header_prefix="## PCTDD-001",
    )[0]
    state = PortalTaskState.load(paths.state)
    state.active_task_id = task.task_id
    state.active_task_key = task.canonical_task_key
    state.active_task_cid = task.canonical_task_cid
    state.active_attempt = 1
    state.active_phase = "implementing"
    state.active_worktree_path = str(repo / "worktrees" / "pctdd-001-stale")
    state.active_branch = "implementation/pctdd-001-stale-attempt-1"
    state.implementation_in_progress = True
    state.active_provider_runner = dict(active_provider_runner or {})
    assert state.save(paths.state) is True
    return state


def _database_portal_successor(
    repo: Path,
    *,
    owner_session_id: str = "",
    task_prefix: str = "PCTDD-",
    task_shard_count: int = 1,
    task_shard_index: int = 0,
    strict_task_sharding: bool = False,
) -> DatabaseImplementationDaemon:
    daemon_args = [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(repo / "control.duckdb"),
            "--state-dir",
            str(repo / "state"),
            "--state-prefix",
            "pctdd",
            "--task-prefix",
            f"## {task_prefix}",
            "--worktree-root",
            str(repo / "worktrees"),
            "--merge-target-branch",
            "main",
            "--implement",
            "--max-task-attempts",
            "3",
            "--once",
            "--task-shard-count",
            str(task_shard_count),
            "--task-shard-index",
            str(task_shard_index),
        ]
    if strict_task_sharding:
        daemon_args.append("--strict-task-sharding")
    args = parse_args(daemon_args)
    successor = DatabaseImplementationDaemon(
        database_path=repo / "control.duckdb",
        max_task_attempts=3,
        owner_session_id=owner_session_id,
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        task_prefix=task_prefix,
        task_shard_count=task_shard_count,
        task_shard_index=task_shard_index,
        strict_task_sharding=strict_task_sharding,
        require_real_execution=True,
    )
    bind_database_portal_execution_from_args(
        successor,
        args,
        repo_root=repo,
        portal_daemon_class=PortalImplementationDaemon,
    )
    return successor


def _seed_terminal_repair_history(
    daemon: DatabaseImplementationDaemon,
    *,
    count: int,
    tied_started_at: bool = False,
) -> dict[str, dict[str, dict[str, object]]]:
    """Insert closed historical terminal sagas for pagination tests."""

    receipts: dict[str, dict[str, dict[str, object]]] = {}
    connection = daemon._require_connection()
    for index in range(count):
        attempt = DatabaseTaskAttempt(
            attempt_id=f"attempt:terminal-history:{index:04d}",
            claim_id=f"claim:terminal-history:{index:04d}",
            task_cid=f"task:terminal-history:{index:04d}",
            task_alias=f"PCTDD-HISTORY-{index:04d}",
            attempt_number=1,
            owner_session_id=daemon.owner_session_id,
            fencing_token=index + 1,
            fence_epoch=1,
            lease_id=f"lease:terminal-history:{index:04d}",
            committed_phase="failed",
            status="failed",
            started_at_ms=1 if tied_started_at else index + 1,
            finished_at_ms=index + 2,
            revision=2,
        )
        prepared_id = "sha256:" + hashlib.sha256(
            f"prepared:{index}".encode()
        ).hexdigest()
        barrier_id = "sha256:" + hashlib.sha256(
            f"barrier:{index}".encode()
        ).hexdigest()
        terminal_id = "sha256:" + hashlib.sha256(
            f"terminal:{index}".encode()
        ).hexdigest()
        disposition = "terminalized_for_retry"
        link: dict[str, object] = {
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
            "binding_id": f"binding:history:{index:04d}",
            "nested_state_digest": f"sha256:{index:064x}",
            "nested_reason": "historical_terminal_repair_test",
            "nested_reconciled": True,
            "trigger": "restart",
            "intended_database_disposition": disposition,
            "prepared_reconciliation_receipt_id": prepared_id,
            "commit_barrier_receipt_id": barrier_id,
        }
        link["evidence_id"] = content_identity(link)
        saga: dict[str, object] = {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "database-portal-terminal-reconciliation-saga@1"
            ),
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "task_cid": attempt.task_cid,
            "attempt_number": attempt.attempt_number,
            "owner_session_id": attempt.owner_session_id,
            "lease_id": attempt.lease_id,
            "fencing_token": attempt.fencing_token,
            "fence_epoch": attempt.fence_epoch,
            "intended_database_disposition": disposition,
            "evidence_id": link["evidence_id"],
            "prepared_reconciliation_receipt_id": prepared_id,
            "commit_barrier_receipt_id": barrier_id,
            "stage": "terminal",
            "receipt_id": terminal_id,
        }
        saga["record_id"] = content_identity(saga)
        connection.execute(
            """
            INSERT INTO database_task_attempts(
                attempt_id, claim_id, task_cid, task_alias, attempt_number,
                owner_session_id, fencing_token, fence_epoch, lease_id,
                committed_phase, status, started_at_ms, finished_at_ms,
                revision, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                attempt.attempt_id,
                attempt.claim_id,
                attempt.task_cid,
                attempt.task_alias,
                attempt.attempt_number,
                attempt.owner_session_id,
                attempt.fencing_token,
                attempt.fence_epoch,
                attempt.lease_id,
                attempt.committed_phase,
                attempt.status,
                attempt.started_at_ms,
                attempt.finished_at_ms,
                attempt.revision,
                "{}",
            ],
        )
        connection.execute(
            """
            INSERT INTO attempt_phases(
                attempt_id, phase, committed_at_ms, fencing_token,
                fence_epoch, revision, body_json
            ) VALUES (?, 'failed', ?, ?, ?, ?, ?)
            """,
            [
                attempt.attempt_id,
                attempt.finished_at_ms,
                attempt.fencing_token,
                attempt.fence_epoch,
                attempt.revision,
                json.dumps(
                    {
                        "database_disposition": disposition,
                        "terminal_reconciliation": link,
                    },
                    sort_keys=True,
                ),
            ],
        )
        connection.execute(
            """
            INSERT INTO database_portal_terminal_reconciliations(
                attempt_id, task_cid, claim_id, attempt_number,
                owner_session_id, lease_id, fencing_token, fence_epoch,
                intended_database_disposition, evidence_id,
                prepared_reconciliation_receipt_id,
                commit_barrier_receipt_id, stage, receipt_id, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                attempt.attempt_id,
                attempt.task_cid,
                attempt.claim_id,
                attempt.attempt_number,
                attempt.owner_session_id,
                attempt.lease_id,
                attempt.fencing_token,
                attempt.fence_epoch,
                disposition,
                link["evidence_id"],
                prepared_id,
                barrier_id,
                "terminal",
                terminal_id,
                json.dumps(saga, sort_keys=True),
            ],
        )
        receipts[attempt.attempt_id] = {
            "prepared": {
                "receipt_id": prepared_id,
                "reconciled": True,
                "blocked": False,
            },
            "commit_barrier": {
                "receipt_id": barrier_id,
                "prepared_reconciliation_receipt_id": prepared_id,
                "intended_database_disposition": disposition,
            },
            "terminal": {
                "receipt_id": terminal_id,
                "terminal_reconciliation_evidence_id": link["evidence_id"],
                "prepared_reconciliation_receipt_id": prepared_id,
                "database_disposition": disposition,
                "database_attempt_status": "failed",
                "database_attempt_phase": "failed",
            },
        }
    return receipts


class _TerminalRepairReceiptAuthority:
    def __init__(
        self,
        receipts: dict[str, dict[str, dict[str, object]]],
    ) -> None:
        self.receipts = receipts

    def load_reconciliation_receipt(
        self,
        attempt: DatabaseTaskAttempt,
        receipt_id: str,
        *,
        required_stage: str,
    ) -> dict[str, object]:
        receipt = dict(self.receipts[attempt.attempt_id][required_stage])
        assert receipt["receipt_id"] == receipt_id
        return receipt

    def persist_reconciliation_receipt(
        self,
        attempt: DatabaseTaskAttempt,
        payload: dict[str, object],
    ) -> dict[str, object]:
        assert payload["stage"] == "terminal"
        receipt_id = "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
        receipt = {**payload, "receipt_id": receipt_id}
        self.receipts[attempt.attempt_id]["terminal"] = dict(receipt)
        return receipt


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_repair_cursor_survives_fresh_once_daemon_per_page(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "control.duckdb"
    coordination_path = tmp_path / "coordination.duckdb"
    execution_path = tmp_path / "execution.duckdb"

    def open_daemon() -> DatabaseImplementationDaemon:
        return DatabaseImplementationDaemon(
            database_path=database_path,
            coordination_path=coordination_path,
            execution_path=execution_path,
            owner_session_id="session:terminal-repair-pagination",
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            require_real_execution=False,
        )

    seed = open_daemon()
    try:
        receipts = _seed_terminal_repair_history(
            seed,
            count=201,
            tied_started_at=True,
        )
    finally:
        seed.close()
    authority = _TerminalRepairReceiptAuthority(receipts)

    first = open_daemon()
    try:
        first_page = first._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="fresh_once_process_1",
        )
        assert sum(
            item.get("reason")
            == "terminal_reconciliation_receipt_verified"
            for item in first_page
        ) == 100
        assert first_page[-1]["reason"] == (
            "terminal_reconciliation_repair_batch_pending"
        )
        assert first._database_portal_terminal_repair_cursor() is not None
    finally:
        first.close()

    second = open_daemon()
    try:
        second_page = second._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="fresh_once_process_2",
        )
        verified = [
            item
            for item in second_page
            if item.get("reason")
            == "terminal_reconciliation_receipt_verified"
        ]
        assert len(verified) == 100
        assert verified[-1]["attempt_id"] == "attempt:terminal-history:0199"
        assert second_page[-1]["has_more"] is True
        assert second._database_portal_terminal_repair_cursor() == (
            1,
            "attempt:terminal-history:0199",
        )
    finally:
        second.close()

    third = open_daemon()
    try:
        third_page = third._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="fresh_once_process_3",
        )
        assert len(third_page) == 1
        assert third_page[0]["attempt_id"] == (
            "attempt:terminal-history:0200"
        )
        assert third._database_portal_terminal_repair_cursor() == (
            1,
            "attempt:terminal-history:0200",
        )
    finally:
        third.close()

    fourth = open_daemon()
    try:
        maintenance_page = fourth._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="fresh_once_process_4",
        )
        assert len(maintenance_page) == 100
        assert all(
            item.get("reason")
            == "terminal_reconciliation_receipt_verified"
            for item in maintenance_page
        )
        assert fourth._database_portal_terminal_repair_cursor() == (
            1,
            "attempt:terminal-history:0200",
        )
        assert fourth.run_once()["selection_idle_reason"] == "no_ready_tasks"
    finally:
        fourth.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_repair_pending_run_once_never_claims_or_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-repair-no-dispatch",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=101)
        daemon._database_portal_bridge = _TerminalRepairReceiptAuthority(
            receipts
        )
        monkeypatch.setattr(
            daemon,
            "reconcile_orphaned_canonical_claims",
            lambda: pytest.fail("pending audit reached orphan reconciliation"),
        )
        monkeypatch.setattr(
            daemon,
            "claim_next",
            lambda: pytest.fail("pending audit claimed new work"),
        )

        result = daemon.run_once()

        assert result["selection_idle_reason"] == (
            "database_portal_reconciliation_pending"
        )
        reconciliation = result["database_portal_reconciliation"]
        assert reconciliation["repair_batch_pending"] is True
        assert reconciliation["reconciled"] is False
        assert reconciliation["quiesced"] is False
        assert reconciliation["safe_to_restart"] is False
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_commit_barrier_behind_high_water_is_still_repaired(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-repair-behind-cursor",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=1)
        attempt = daemon.get_attempt("attempt:terminal-history:0000")
        assert attempt is not None
        saga = dict(
            daemon._database_portal_terminal_reconciliation_saga(attempt)
            or {}
        )
        saga["stage"] = "commit_barrier"
        saga["receipt_id"] = ""
        saga.pop("record_id", None)
        saga["record_id"] = content_identity(saga)
        daemon._require_connection().execute(
            """
            UPDATE database_portal_terminal_reconciliations
            SET stage = 'commit_barrier', receipt_id = '', record_json = ?
            WHERE attempt_id = ?
            """,
            [json.dumps(saga, sort_keys=True), attempt.attempt_id],
        )
        daemon._set_database_portal_terminal_repair_cursor(
            (10_000, "attempt:terminal-history:high-water")
        )
    finally:
        daemon.close()

    successor = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-repair-behind-cursor",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    authority = _TerminalRepairReceiptAuthority(receipts)
    try:
        outcomes = successor._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="behind_cursor_pending_saga",
        )
        assert len(outcomes) == 1
        assert outcomes[0]["reason"] == (
            "terminal_reconciliation_receipt_repaired"
        )
        repaired = successor.get_attempt("attempt:terminal-history:0000")
        assert repaired is not None
        repaired_saga = successor._database_portal_terminal_reconciliation_saga(
            repaired
        )
        assert repaired_saga is not None and repaired_saga["stage"] == "terminal"
        assert successor._database_portal_terminal_repair_cursor() == (
            10_000,
            "attempt:terminal-history:high-water",
        )
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_far_ahead_commit_barrier_cannot_skip_contiguous_terminal_audit(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-repair-contiguous",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=151)
        far_attempt = daemon.get_attempt("attempt:terminal-history:0150")
        assert far_attempt is not None
        saga = dict(
            daemon._database_portal_terminal_reconciliation_saga(far_attempt)
            or {}
        )
        saga["stage"] = "commit_barrier"
        saga["receipt_id"] = ""
        saga.pop("record_id", None)
        saga["record_id"] = content_identity(saga)
        daemon._require_connection().execute(
            """
            UPDATE database_task_attempts SET started_at_ms = 1000000
            WHERE attempt_id = ?
            """,
            [far_attempt.attempt_id],
        )
        daemon._require_connection().execute(
            """
            UPDATE database_portal_terminal_reconciliations
            SET stage = 'commit_barrier', receipt_id = '', record_json = ?
            WHERE attempt_id = ?
            """,
            [json.dumps(saga, sort_keys=True), far_attempt.attempt_id],
        )
        authority = _TerminalRepairReceiptAuthority(receipts)

        first = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="far_ahead_dirty_page",
        )
        assert first[-1]["has_more"] is True
        assert any(
            item.get("attempt_id") == far_attempt.attempt_id
            and item.get("reason")
            == "terminal_reconciliation_receipt_repaired"
            for item in first
        )
        # The far-ahead dirty saga consumed one page slot but did not move the
        # contiguous audit cursor past any of the 52 still-unverified rows.
        assert daemon._database_portal_terminal_repair_cursor() == (
            99,
            "attempt:terminal-history:0098",
        )

        second = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="contiguous_audit_tail",
        )
        observed = {
            str(item.get("attempt_id") or "")
            for item in second
            if item.get("reason")
            == "terminal_reconciliation_receipt_verified"
        }
        assert {
            f"attempt:terminal-history:{index:04d}"
            for index in range(99, 151)
        } <= observed
        assert not any(item.get("has_more") is True for item in second)
        assert daemon._database_portal_terminal_repair_cursor() == (
            1_000_000,
            far_attempt.attempt_id,
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("mismatch", ("store_id", "tampered"))
def test_terminal_repair_cursor_record_mismatch_fails_closed(
    tmp_path: Path,
    mismatch: str,
) -> None:
    paths = {
        "database_path": tmp_path / "control.duckdb",
        "coordination_path": tmp_path / "coordination.duckdb",
        "execution_path": tmp_path / "execution.duckdb",
    }
    seed = DatabaseImplementationDaemon(
        **paths,
        owner_session_id="session:terminal-repair-scope",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        control_store_id="store:accepted",
        control_store_generation="generation:1",
        require_real_execution=False,
    )
    try:
        seed._set_database_portal_terminal_repair_cursor((7, "attempt:cursor"))
        metadata_key = (
            seed._database_portal_terminal_repair_cursor_metadata_key()
        )
        row = seed._require_connection().execute(
            "SELECT value FROM daemon_execution_metadata WHERE key = ?",
            [metadata_key],
        ).fetchone()
        assert row is not None
        record = json.loads(row[0])
        if mismatch == "tampered":
            record["unreviewed"] = True
        else:
            record["control_store_id"] = "store:substituted"
            record.pop("record_id")
            record["record_id"] = content_identity(record)
        seed._require_connection().execute(
            """
            UPDATE daemon_execution_metadata SET value = ? WHERE key = ?
            """,
            [json.dumps(record, sort_keys=True), metadata_key],
        )
    finally:
        seed.close()

    successor = DatabaseImplementationDaemon(
        **paths,
        owner_session_id="session:terminal-repair-scope",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        control_store_id="store:accepted",
        control_store_generation="generation:1",
        require_real_execution=False,
    )
    try:
        with pytest.raises(
            DatabaseImplementationConflictError,
            match="cursor (?:changed store authority|is not closed)",
        ):
            successor._database_portal_terminal_repair_cursor()
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_repair_cursor_generation_rotation_is_independent(
    tmp_path: Path,
) -> None:
    paths = {
        "database_path": tmp_path / "control.duckdb",
        "coordination_path": tmp_path / "coordination.duckdb",
        "execution_path": tmp_path / "execution.duckdb",
    }
    first = DatabaseImplementationDaemon(
        **paths,
        owner_session_id="session:terminal-repair-generation",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        control_store_id="store:accepted",
        control_store_generation="generation:1",
        require_real_execution=False,
    )
    try:
        first._set_database_portal_terminal_repair_cursor(
            (7, "attempt:generation:1")
        )
        first_key = (
            first._database_portal_terminal_repair_cursor_metadata_key()
        )
    finally:
        first.close()

    second = DatabaseImplementationDaemon(
        **paths,
        owner_session_id="session:terminal-repair-generation",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        control_store_id="store:accepted",
        control_store_generation="generation:2",
        require_real_execution=False,
    )
    try:
        second_key = (
            second._database_portal_terminal_repair_cursor_metadata_key()
        )
        assert second_key != first_key
        assert second._database_portal_terminal_repair_cursor() is None
        second._set_database_portal_terminal_repair_cursor(
            (9, "attempt:generation:2")
        )
    finally:
        second.close()

    restored = DatabaseImplementationDaemon(
        **paths,
        owner_session_id="session:terminal-repair-generation",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        control_store_id="store:accepted",
        control_store_generation="generation:1",
        require_real_execution=False,
    )
    try:
        assert restored._database_portal_terminal_repair_cursor() == (
            7,
            "attempt:generation:1",
        )
    finally:
        restored.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_round_robin_audit_detects_post_high_water_corruption(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-round-robin-corruption",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=1)
        authority = _TerminalRepairReceiptAuthority(receipts)
        initial = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="initial_high_water",
        )
        assert initial[0]["reconciled"] is True
        assert daemon._database_portal_terminal_repair_cursor() == (
            1,
            "attempt:terminal-history:0000",
        )

        audit = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="round_robin_audit",
        )
        assert audit[0]["reason"] == (
            "terminal_reconciliation_receipt_verified"
        )
        receipts["attempt:terminal-history:0000"]["terminal"][
            "database_attempt_status"
        ] = "tampered"
        # First bounded maintenance pass wraps the cursor.  It never gates
        # healthy ordinary work merely because another audit epoch exists.
        assert daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="round_robin_wrap",
        ) == []
        corrupt = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="round_robin_corruption",
        )
        assert any(item.get("blocked") is True for item in corrupt)
        assert daemon._database_portal_terminal_audit_cursor() == (
            -1,
            "",
            1,
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("corruption", ("delete_saga", "corrupt_stage"))
def test_round_robin_audit_detects_post_high_water_saga_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-saga-corruption",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=1)
        authority = _TerminalRepairReceiptAuthority(receipts)
        initial = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="saga_corruption_initial_high_water",
        )
        assert initial[0]["reconciled"] is True
        if corruption == "delete_saga":
            daemon._require_connection().execute(
                """
                DELETE FROM database_portal_terminal_reconciliations
                WHERE attempt_id = ?
                """,
                ["attempt:terminal-history:0000"],
            )
        else:
            daemon._require_connection().execute(
                """
                UPDATE database_portal_terminal_reconciliations
                SET stage = 'corrupt-stage'
                WHERE attempt_id = ?
                """,
                ["attempt:terminal-history:0000"],
            )

        audit = daemon._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="saga_corruption_round_robin",
        )

        assert len(audit) == 1
        assert audit[0]["blocked"] is True
        assert audit[0]["reason"] == (
            "terminal_reconciliation_receipt_repair_failed"
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_exact_pre_provider_terminal_link_restores_missing_saga_barrier(
    tmp_path: Path,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    successor: DatabaseImplementationDaemon | None = None
    try:
        assert not attempt.phase_committed("provider")
        reconciled = predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="pre_provider_setup_failure",
            force=True,
        )
        assert reconciled["reconciled"] is True, reconciled
        terminal = predecessor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        initial_saga = (
            predecessor._database_portal_terminal_reconciliation_saga(terminal)
        )
        assert initial_saga is not None and initial_saga["stage"] == "terminal"
        evidence_id = initial_saga["evidence_id"]
        receipt_id = initial_saga["receipt_id"]
        predecessor._require_connection().execute(
            """
            DELETE FROM database_portal_terminal_reconciliations
            WHERE attempt_id = ?
            """,
            [terminal.attempt_id],
        )
        assert (
            predecessor._database_portal_terminal_reconciliation_saga(terminal)
            is None
        )
        predecessor.close()

        successor = _database_portal_successor(repo)
        current = successor.get_attempt(terminal.attempt_id)
        assert current is not None and current.status == "failed"
        repairs = successor._repair_database_portal_terminal_receipts(
            bridge=successor._database_portal_bridge,
            trigger="restart_missing_saga_barrier",
            exact_attempt=current,
        )

        assert len(repairs) == 1, repairs
        assert repairs[0]["reconciled"] is True
        assert repairs[0]["reason"] == (
            "terminal_reconciliation_receipt_repaired"
        )
        restored = successor._database_portal_terminal_reconciliation_saga(
            current
        )
        assert restored is not None and restored["stage"] == "terminal"
        assert restored["evidence_id"] == evidence_id
        assert restored["receipt_id"] == receipt_id
        assert successor.get_attempt(current.attempt_id) == current
    finally:
        if successor is not None:
            successor.close()
        else:
            predecessor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_round_robin_audit_nominates_terminal_row_inserted_behind_high_water(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-old-key-insert",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        daemon._set_database_portal_terminal_repair_cursor(
            (10_000, "attempt:terminal-history:high-water")
        )
        receipts = _seed_terminal_repair_history(daemon, count=1)
        outcomes = daemon._repair_database_portal_terminal_receipts(
            bridge=_TerminalRepairReceiptAuthority(receipts),
            trigger="old_key_terminal_insert",
        )
        assert [item.get("attempt_id") for item in outcomes] == [
            "attempt:terminal-history:0000"
        ]
        assert outcomes[0]["reason"] == (
            "terminal_reconciliation_receipt_verified"
        )
        assert daemon._database_portal_terminal_repair_cursor() == (
            10_000,
            "attempt:terminal-history:high-water",
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_exact_terminal_repair_rejects_mutated_attempt_row_identity(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-exact-row",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=1)
        exact = daemon.get_attempt("attempt:terminal-history:0000")
        assert exact is not None
        daemon._require_connection().execute(
            """
            UPDATE database_task_attempts SET revision = revision + 1
            WHERE attempt_id = ?
            """,
            [exact.attempt_id],
        )
        outcomes = daemon._repair_database_portal_terminal_receipts(
            bridge=_TerminalRepairReceiptAuthority(receipts),
            trigger="exact_attempt_mutation",
            exact_attempt=exact,
        )
        assert len(outcomes) == 1
        assert outcomes[0]["blocked"] is True
        assert "exact terminal reconciliation attempt authority changed" in (
            outcomes[0]["error"]
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_repair_cursor_write_failure_blocks_page_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:terminal-repair-cursor-failure",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=False,
    )
    try:
        receipts = _seed_terminal_repair_history(daemon, count=101)
        monkeypatch.setattr(
            daemon,
            "_set_database_portal_terminal_repair_cursor",
            lambda _cursor: (_ for _ in ()).throw(
                RuntimeError("injected cursor CAS response loss")
            ),
        )
        outcomes = daemon._repair_database_portal_terminal_receipts(
            bridge=_TerminalRepairReceiptAuthority(receipts),
            trigger="cursor_write_failure",
        )
        assert any(
            item.get("reason")
            == "terminal_reconciliation_cursor_commit_failed"
            and item.get("blocked") is True
            for item in outcomes
        )
        assert outcomes[-1]["reason"] == (
            "terminal_reconciliation_repair_batch_pending"
        )
        assert outcomes[-1]["blocked"] is True
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_terminal_repair_page_two_corruption_does_not_advance_high_water(
    tmp_path: Path,
) -> None:
    paths = {
        "database_path": tmp_path / "control.duckdb",
        "coordination_path": tmp_path / "coordination.duckdb",
        "execution_path": tmp_path / "execution.duckdb",
    }

    def open_daemon() -> DatabaseImplementationDaemon:
        return DatabaseImplementationDaemon(
            **paths,
            owner_session_id="session:terminal-repair-page-two-corruption",
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            require_real_execution=False,
        )

    first = open_daemon()
    try:
        receipts = _seed_terminal_repair_history(first, count=201)
        authority = _TerminalRepairReceiptAuthority(receipts)
        first_page = first._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="page_one",
        )
        assert first_page[-1]["has_more"] is True
        first_cursor = first._database_portal_terminal_repair_cursor()
        assert first_cursor == (100, "attempt:terminal-history:0099")
    finally:
        first.close()

    receipts["attempt:terminal-history:0150"]["terminal"][
        "database_attempt_status"
    ] = "tampered"
    second = open_daemon()
    try:
        second_page = second._repair_database_portal_terminal_receipts(
            bridge=authority,
            trigger="page_two",
        )
        assert any(item.get("blocked") is True for item in second_page)
        assert second._database_portal_terminal_repair_cursor() == first_cursor
    finally:
        second.close()


def test_datasets_authority_marker_reaches_provider_without_state_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION",
        DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    )
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        '{"credential":"must-not-propagate"}',
    )
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "secret-token")
    portal = SimpleNamespace(_canonical_ref=lambda task: "task:cid:004")
    task = SimpleNamespace(task_id="LGSWF-004")

    environment = PortalImplementationDaemon._implementation_process_environment(
        portal,
        task,
        attempt=2,
        checkpoint_dir=tmp_path / "checkpoint",
    )

    assert environment[SEMANTIC_TRUTH_AUTHORITY_ENV] == "ipfs_datasets_py"
    assert environment[SEMANTIC_WRITER_POLICY_ENV] == "reference_only"
    assert "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION" not in environment
    assert "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON" not in environment
    assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in environment

    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION", "schema-v1")
    ordinary_environment = (
        PortalImplementationDaemon._implementation_process_environment(
            portal,
            task,
            attempt=3,
            checkpoint_dir=tmp_path / "ordinary-checkpoint",
        )
    )
    assert SEMANTIC_TRUTH_AUTHORITY_ENV not in ordinary_environment
    assert SEMANTIC_WRITER_POLICY_ENV not in ordinary_environment


class _TaskSource:
    def __init__(self, record: object) -> None:
        self.record = record

    def get_task(self, task_cid: str) -> object | None:
        return self.record if task_cid == "task:cid:004" else None


class _CompletingPortal:
    def __init__(self, paths: object, task_alias: str) -> None:
        self.paths = paths
        self.task_alias = task_alias
        self.closed = False

    def run_once(self) -> dict[str, object]:
        text = self.paths.task_projection.read_text(encoding="utf-8")
        self.paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        self.paths.state.write_text(
            json.dumps(
                {
                    "last_implementation_commit": "a" * 40,
                    "last_merge_returncode": 0,
                }
            ),
            encoding="utf-8",
        )
        task = parse_task_file(
            self.paths.task_projection,
            task_header_prefix=f"## {self.task_alias}",
        )[0]
        self.paths.events.write_text(
            json.dumps(
                {
                    "type": "task_completed",
                    "task_id": self.task_alias,
                    "canonical_task_key": task.canonical_task_key,
                    "canonical_task_cid": task.canonical_task_cid,
                    "board_namespace": task.board_namespace,
                    "event_id": "event:complete",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "task_count": 1,
            "completed_count": 1,
            "active_task_id": self.task_alias,
            "implementation_result": {
                "task_id": self.task_alias,
                "returncode": 0,
                "implementation_commit": "a" * 40,
                # Raw model output must not enter the database receipt.
                "model_response": "private provider payload",
            },
            "merge_reconciliation": [
                {
                    "task_id": self.task_alias,
                    "returncode": 0,
                    "merge_commit": "b" * 40,
                    "provider_payload": "private",
                }
            ],
        }

    def close_event_runtime(self) -> None:
        self.closed = True


def test_bridge_uses_only_attempt_local_projection_and_seals_receipt(
    tmp_path: Path,
) -> None:
    canonical_board = tmp_path / "canonical-board.md"
    canonical_board.write_text(
        "# Canonical\n\n## LGSWF-004 Authority\n\n- Status: ready\n",
        encoding="utf-8",
    )
    original = canonical_board.read_bytes()
    portals: list[_CompletingPortal] = []

    def factory(paths: object, alias: str) -> _CompletingPortal:
        portal = _CompletingPortal(paths, alias)
        portals.append(portal)
        return portal

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=factory,
    )
    provider = bridge.run_provider(_attempt())
    effect = bridge.apply_effect(_attempt(), provider)
    validation = bridge.validate_effect(_attempt(), effect)

    assert provider["schema"] == DATABASE_PORTAL_EXECUTION_RECEIPT_SCHEMA
    assert provider["accepted"] is True
    assert provider["provider"] == "PortalImplementationDaemon"
    assert provider["completion_authority"] == "DatabaseImplementationDaemon"
    assert provider["evidence_digest"].startswith("sha256:")
    assert "private provider payload" not in json.dumps(provider)
    assert "provider_payload" not in json.dumps(provider)
    assert effect["status"] == "applied"
    assert validation["outcome"] == "passed"
    assert validation["evidence_digest"] == provider["evidence_digest"]
    assert canonical_board.read_bytes() == original
    assert portals and portals[0].closed is True
    attempt_boards = list((tmp_path / "attempts").glob("*/task-projection.md"))
    assert len(attempt_boards) == 1
    assert "Projection authority: false" in attempt_boards[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("canonical_cid", [None, "cid:wrong-projection"])
def test_recovery_rejects_alias_only_or_wrong_canonical_completion_event(
    tmp_path: Path,
    canonical_cid: str | None,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), _record())
    text = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        text.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    task = parse_task_file(
        paths.task_projection,
        task_header_prefix="## LGSWF-004",
    )[0]
    event = {
        "type": "task_completed",
        "task_id": task.task_id,
        "canonical_task_key": task.canonical_task_key,
        "board_namespace": task.board_namespace,
    }
    if canonical_cid is not None:
        event["canonical_task_cid"] = canonical_cid
    paths.events.write_text(json.dumps(event) + "\n", encoding="utf-8")

    assert bridge.recover_provider_result(_attempt()) is None


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_materialize_round_trip_preserves_shell_text_and_argv(
    tmp_path: Path,
) -> None:
    shell_text = (
        "python3 -m pytest focused.py -q && git diff --check"
    )
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        source.materialize(
            {
                "repository_tree_id": "tree:bridge-validation",
                "objectives": [
                    {
                        "goal_cid": "goal:inventory",
                        "goal_id": "PCTDD-G011",
                        "title": "Inventory",
                    }
                ],
                "tasks": [
                    {
                        "task_cid": "task:cid:004",
                        "task_id": "LGSWF-004",
                        "goal_cid": "goal:inventory",
                        "objective": "Preserve validation forms",
                        "outputs": [{"path": "inventory/result.json"}],
                        "validation_commands": [
                            shell_text,
                            {
                                "argv": [
                                    "python3",
                                    "-m",
                                    "pytest",
                                    "focused path.py",
                                ]
                            },
                        ],
                        "acceptance": "Focused validation passes",
                    }
                ],
            }
        )
        record = source.get_task("task:cid:004")
        assert record is not None
        assert record.validations[0]["argv"] == [shell_text]
        assert record.validations[0]["policy"]["representation"] == (
            "shell_text"
        )
        assert record.validations[1]["policy"]["representation"] == "argv"

        bridge = DatabasePortalExecutionBridge(
            task_source=source,
            attempt_root=tmp_path / "attempts",
            portal_factory=lambda _paths, _alias: None,
        )
        paths, binding = bridge._ensure_attempt_projection(_attempt(), record)
        projection = paths.task_projection.read_text(encoding="utf-8")

    assert f"- Validation: {shell_text} ; " in projection
    assert "python3 -m pytest 'focused path.py'" in projection
    assert f"'{shell_text}'" not in projection
    assert binding["task_cid"] == "task:cid:004"


def test_bridge_rejects_malformed_typed_shell_text_validation(
    tmp_path: Path,
) -> None:
    record = _record()
    record.validations = (
        {
            "argv": ["pytest focused.py", "git diff --check"],
            "policy": {"representation": "shell_text"},
        },
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="shell_text validation must contain exactly one command",
    ):
        bridge._render_projection(_attempt(), record)


def test_bridge_rejects_projection_contract_tampering(tmp_path: Path) -> None:
    class TamperingPortal(_CompletingPortal):
        def run_once(self) -> dict[str, object]:
            text = self.paths.task_projection.read_text(encoding="utf-8")
            self.paths.task_projection.write_text(
                text.replace(
                    "- Acceptance: Focused validation passes",
                    "- Acceptance: no validation required",
                ),
                encoding="utf-8",
            )
            return {"implementation_result": {"returncode": 0}}

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda paths, alias: TamperingPortal(paths, alias),
    )
    with pytest.raises(DatabasePortalBridgeError, match="outside its mutable status"):
        bridge.run_provider(_attempt())


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_production_database_daemon_cannot_complete_with_default_noops(
    tmp_path: Path,
) -> None:
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:fail-closed",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        require_real_execution=True,
    )
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:bridge",
                "tasks": [
                    {
                        "task_cid": "task:cid:004",
                        "task_id": "LGSWF-004",
                        "goal_cid": "goal:inventory",
                        "status": "ready",
                        "priority": "P0",
                        "ordinal": 4,
                        "title": "Inventory",
                    }
                ],
            }
        )
        result = daemon.run_once()
        assert result["implementation_result"]["callback_failure"] is True
        assert "no provider executor" in result["implementation_result"]["reason"]
        task = daemon.task_source.get_task("task:cid:004")
        assert task is not None
        assert task.status != "completed"
        assert (
            daemon.provider_invocation_recorded(
            result["attempt_id"],
            idempotency_key=f"provider:{result['attempt_id']}",
            )
            is None
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_real_database_portal_bridge_blocks_raised_result_without_redispatch(
    tmp_path: Path,
) -> None:
    provider_attempts: list[str] = []
    inner_caps: list[int] = []
    portal_roots: list[Path] = []

    class FailingPortal:
        def __init__(self, **kwargs: object) -> None:
            inner_caps.append(int(kwargs["max_task_attempts"]))
            state_path = Path(str(kwargs["state_path"]))
            portal_roots.append(state_path.parent)

        def run_once(self) -> dict[str, object]:
            provider_attempts.append(str(portal_roots[-1]))
            return {
                "implementation_result": {
                    "task_id": "PCTDD-001",
                    "returncode": 1,
                    "reason": "declared_validation_failed",
                }
            }

        def close_event_runtime(self) -> None:
            return None

    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "pctdd",
            "--implement",
            "--max-task-attempts",
            "2",
            "--once",
        ]
    )
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:real-bridge-cap",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        max_task_attempts=2,
        require_real_execution=True,
    )
    try:
        bind_database_portal_execution_from_args(
            daemon,
            args,
            repo_root=tmp_path,
            portal_daemon_class=FailingPortal,
        )
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:real-bridge-cap",
                "tasks": [
                    {
                        "task_cid": "task:cid:pctdd-001",
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "status": "ready",
                        "validation_commands": ["python -m pytest focused.py"],
                    }
                ],
            }
        )
        first = daemon.run_once()
        second = daemon.run_once()
        idle = daemon.run_once()
        assert first["implementation_result"][
            "provider_reconciliation_pending"
        ] is True
        assert second["implementation_result"]["retry_exhausted"] is True
        assert idle["implementation_result"] is None
        assert len(provider_attempts) == 1
        assert inner_caps == [1]
        assert len(set(portal_roots)) == 1
        attempt = daemon.get_attempt(str(first["attempt_id"]))
        assert attempt is not None and attempt.status == "failed"
        task = daemon.task_source.get_task("task:cid:pctdd-001")
        assert task is not None and task.status == "blocked"
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_lost_portal_provider_return_recovers_without_reimplementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    portal_calls: list[str] = []

    class CompletingPortal:
        def __init__(self, **kwargs: object) -> None:
            self.projection = Path(str(kwargs["todo_path"]))
            self.state = Path(str(kwargs["state_path"]))
            self.events = Path(str(kwargs["events_path"]))

        def run_once(self) -> dict[str, object]:
            portal_calls.append("run")
            text = self.projection.read_text(encoding="utf-8")
            self.projection.write_text(
                text.replace("- Status: ready", "- Status: completed"),
                encoding="utf-8",
            )
            self.state.write_text('{"accepted":true}\n', encoding="utf-8")
            task = parse_task_file(
                self.projection,
                task_header_prefix="## PCTDD-001",
            )[0]
            self.events.write_text(
                json.dumps(
                    {
                        "type": "task_completed",
                        "task_id": "PCTDD-001",
                        "canonical_task_key": task.canonical_task_key,
                        "canonical_task_cid": task.canonical_task_cid,
                        "board_namespace": task.board_namespace,
                        "event_id": "event:portal-complete",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return {
                "implementation_result": {
                    "task_id": "PCTDD-001",
                    "returncode": 0,
                    "implementation_commit": "a" * 40,
                }
            }

        def close_event_runtime(self) -> None:
            return None

    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "pctdd",
            "--implement",
            "--max-task-attempts",
            "2",
            "--once",
        ]
    )
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:portal-return-recovery",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        max_task_attempts=2,
        require_real_execution=True,
    )
    try:
        bind_database_portal_execution_from_args(
            daemon,
            args,
            repo_root=tmp_path,
            portal_daemon_class=CompletingPortal,
        )
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:portal-return-recovery",
                "tasks": [
                    {
                        "task_cid": "task:cid:pctdd-001",
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "status": "ready",
                        "validation_commands": ["python -m pytest focused.py"],
                    }
                ],
            }
        )
        original_record = daemon._record_callback_dispatch_outcome
        injected = {"done": False}

        def lose_return(*call_args: object, **call_kwargs: object) -> None:
            if (
                not injected["done"]
                and call_kwargs.get("dispatch_kind") == "provider"
                and call_kwargs.get("outcome") == "returned"
            ):
                injected["done"] = True
                raise RuntimeError("lost Portal provider return")
            original_record(*call_args, **call_kwargs)

        monkeypatch.setattr(daemon, "_record_callback_dispatch_outcome", lose_return)
        pending = daemon.run_once()["implementation_result"]
        assert pending["status"] == "provider_reconciliation_pending"
        assert pending["retry_budget_consumed"] is False

        monkeypatch.setattr(
            daemon, "_record_callback_dispatch_outcome", original_record
        )
        recovered = daemon.run_once()["implementation_result"]
        assert recovered["status"] == "succeeded"
        assert recovered["attempt"]["attempt_id"] == pending["attempt_id"]
        assert recovered["attempt"]["attempt_number"] == 1
        assert recovered["provider_result"]["accepted"] is True
        assert portal_calls == ["run"]
        task = daemon.task_source.get_task("task:cid:pctdd-001")
        assert task is not None and task.status == "completed"
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("recovery_case", ["absent", "corrupt", "unaccepted"])
def test_unknown_portal_dispatch_without_terminal_evidence_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_case: str,
) -> None:
    class PortalMustNotRun:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("unknown Portal dispatch must not be repeated")

    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "pctdd",
            "--implement",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:portal-unknown-block",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        max_task_attempts=3,
        require_real_execution=True,
    )
    try:
        bridge = bind_database_portal_execution_from_args(
            daemon,
            args,
            repo_root=tmp_path,
            portal_daemon_class=PortalMustNotRun,
        )
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:portal-unknown-block",
                "tasks": [
                    {
                        "task_cid": "task:cid:pctdd-001",
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "status": "ready",
                        "validation_commands": ["python -m pytest focused.py"],
                    }
                ],
            }
        )
        attempt = daemon.claim_next()
        assert attempt is not None
        assert isinstance(bridge, DatabasePortalExecutionBridge)
        if recovery_case == "corrupt":
            record = daemon.task_source.get_task(attempt.task_cid)
            assert record is not None
            paths, _binding = bridge._ensure_attempt_projection(attempt, record)
            corrupt = json.loads(paths.binding.read_text(encoding="utf-8"))
            corrupt["fencing_token"] = int(corrupt["fencing_token"]) + 1
            paths.binding.write_text(
                json.dumps(corrupt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        elif recovery_case == "unaccepted":
            monkeypatch.setattr(
                bridge,
                "recover_provider_result",
                lambda _attempt: {"status": "succeeded", "accepted": False},
            )
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )

        result = daemon.run_once()["implementation_result"]
        assert result["status"] == "retry_exhausted"
        assert result["retry_exhausted"] is True
        assert result["reason"] in {
            "provider dispatch outcome is unknown and has no exact durable terminal evidence",
            "provider recovery rejected corrupt or mismatched durable evidence",
            "provider recovery returned unaccepted terminal evidence",
        }
        task = daemon.task_source.get_task("task:cid:pctdd-001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["forced_block"] is True
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert daemon.run_once()["implementation_result"] is None
        daemon.close()
        daemon = DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            owner_session_id="session:portal-unknown-block",
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            max_task_attempts=3,
            require_real_execution=True,
        )
        bind_database_portal_execution_from_args(
            daemon,
            args,
            repo_root=tmp_path,
            portal_daemon_class=PortalMustNotRun,
        )
        assert daemon.run_once()["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            ["task:cid:pctdd-001"],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_effectful_portal_exception_blocks_without_fresh_attempt(
    tmp_path: Path,
) -> None:
    side_effect = tmp_path / "portal-side-effect"
    portal_calls: list[str] = []

    class ExplodingPortal:
        def __init__(self, **_kwargs: object) -> None:
            return None

        def run_once(self) -> dict[str, object]:
            portal_calls.append("run")
            side_effect.write_text("landed\n", encoding="utf-8")
            raise RuntimeError("Portal return lost after external side effect")

        def close_event_runtime(self) -> None:
            return None

    args = parse_args(
        [
            "--task-source-kind", "duckdb",
            "--authority-mode", "embedded_exclusive",
            "--database-path", str(tmp_path / "control.duckdb"),
            "--state-dir", str(tmp_path / "state"),
            "--state-prefix", "pctdd",
            "--implement",
            "--max-task-attempts", "3",
            "--once",
        ]
    )
    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:portal-effectful-exception",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        max_task_attempts=3,
        require_real_execution=True,
    )
    try:
        bind_database_portal_execution_from_args(
            daemon, args, repo_root=tmp_path, portal_daemon_class=ExplodingPortal
        )
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:portal-effectful-exception",
                "tasks": [{
                    "task_cid": "task:cid:pctdd-001",
                    "task_id": "PCTDD-001",
                    "goal_cid": "goal:pctdd",
                    "status": "ready",
                    "validation_commands": ["python -m pytest focused.py"],
                }],
            }
        )
        failed = daemon.run_once()["implementation_result"]
        assert failed["status"] == "retry_exhausted"
        assert failed["retry_exhausted"] is True
        assert portal_calls == ["run"]
        assert side_effect.read_text(encoding="utf-8") == "landed\n"
        task = daemon.task_source.get_task("task:cid:pctdd-001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["forced_block"] is True
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert daemon.run_once()["implementation_result"] is None
        assert portal_calls == ["run"]
        daemon.close()

        class PortalMustNotRun:
            def __init__(self, **_kwargs: object) -> None:
                raise AssertionError("raised Portal callback was redispatched")

        daemon = DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            owner_session_id="session:portal-effectful-exception",
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            max_task_attempts=3,
            require_real_execution=True,
        )
        bind_database_portal_execution_from_args(
            daemon,
            args,
            repo_root=tmp_path,
            portal_daemon_class=PortalMustNotRun,
        )
        assert daemon.run_once()["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            ["task:cid:pctdd-001"],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert portal_calls == ["run"]
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_live_effect_exception_is_permanently_blocked_across_restart(
    tmp_path: Path,
) -> None:
    callback_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        callback_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    def effect(
        _attempt: DatabaseTaskAttempt,
        _provider_result: Mapping[str, object],
    ) -> dict[str, object]:
        callback_calls.append("effect")
        raise RuntimeError("effect outcome was lost after callback entry")

    def validation(
        _attempt: DatabaseTaskAttempt,
        _provider_result: Mapping[str, object],
        _effect_result: Mapping[str, object],
    ) -> dict[str, object]:
        callback_calls.append("validation")
        return {"outcome": "passed"}

    daemon = DatabaseImplementationDaemon(
        database_path=tmp_path / "control.duckdb",
        coordination_path=tmp_path / "coordination.duckdb",
        execution_path=tmp_path / "execution.duckdb",
        owner_session_id="session:live-effect-exception",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        max_task_attempts=3,
        require_real_execution=True,
    )
    daemon.bind_execution_callbacks(
        provider_fn=provider,
        effect_fn=effect,
        validation_fn=validation,
    )
    try:
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:live-effect-exception",
                "tasks": [
                    {
                        "task_cid": "task:cid:pctdd-effect-001",
                        "task_id": "PCTDD-EFFECT-001",
                        "goal_cid": "goal:pctdd",
                        "status": "ready",
                        "validation_commands": ["python -m pytest focused.py"],
                    }
                ],
            }
        )
        failed = daemon.run_once()["implementation_result"]
        assert failed["status"] == "retry_exhausted"
        task = daemon.task_source.get_task("task:cid:pctdd-effect-001")
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert callback_calls == ["provider", "effect"]
        daemon.close()

        daemon = DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb",
            owner_session_id="session:live-effect-exception",
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            max_task_attempts=3,
            require_real_execution=True,
        )

        def forbidden(*_args: object, **_kwargs: object) -> Mapping[str, object]:
            raise AssertionError("effectful callback was redispatched")

        daemon.bind_execution_callbacks(
            provider_fn=forbidden,
            effect_fn=forbidden,
            validation_fn=forbidden,
        )
        assert daemon.run_once()["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            ["task:cid:pctdd-effect-001"],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_calls == ["provider", "effect"]
    finally:
        daemon.close()


def test_quack_mode_refuses_direct_duckdb_execution(tmp_path: Path) -> None:
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="loopback quack:",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "control.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
        )


def test_quack_mode_requires_exact_control_store_cursor_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", raising=False)
    monkeypatch.delenv(
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", raising=False
    )
    with pytest.raises(
        DatabaseImplementationAuthorityError,
        match="exact control-store id and generation",
    ):
        DatabaseImplementationDaemon(
            database_path=tmp_path / "lane-control.duckdb",
            coordination_path=tmp_path / "lane-coordination.duckdb",
            execution_path=tmp_path / "lane-execution.duckdb",
            authority_mode="quack",
            task_source_kind="duckdb",
            quack_uri="quack:127.0.0.1:45671",
            install_schema=False,
        )


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_configured_production_runner_binds_real_portal_bridge(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(tmp_path / "control.duckdb"),
            "--todo-path",
            str(tmp_path / "canonical-board.md"),
            "--state-dir",
            str(tmp_path / "state"),
            "--state-prefix",
            "lgswf",
            "--worktree-root",
            ".worktrees",
            "--implement",
            "--max-task-attempts",
            "2",
            "--once",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        args,
        repo_root=tmp_path,
    )
    try:
        assert isinstance(daemon, DatabaseImplementationDaemon)
        assert daemon.require_real_execution is True
        assert daemon.max_task_attempts == 2
        assert daemon.execution_callbacks_bound is True
        assert daemon.markdown_path is None
        assert daemon.markdown_status_write_count == 0
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_shutdown_reconciles_exact_nested_attempt_and_lifecycle(
    tmp_path: Path,
) -> None:
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    try:
        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )

        assert result["reconciled"] is True, result
        assert result["blocked"] is False
        assert result["active_attempt_count"] == 1
        item = result["attempts"][0]
        assert item["attempt_id"] == attempt.attempt_id
        assert item["claim_id"] == attempt.claim_id
        assert item["task_cid"] == attempt.task_cid
        assert item["nested_state"]["present"] is True
        assert item["nested_state"]["active"] is False
        assert item["portal_reconciliation"]["reconciled"] is True
        # A one-way predecessor admission is conservatively marked entered.
        # Without an exact dispatch/result row, shutdown can prove current
        # quiescence but cannot prove that no prior callback effect occurred.
        assert item["database_disposition"] == "blocked_unknown_outcome"
        assert item["database_attempt_status"] == "failed"
        assert item["database_task_status"] == "blocked"
        assert item["terminal_reconciliation_evidence_id"]
        nested_after = PortalTaskState.load(paths.state)
        assert nested_after.active_task_id == ""
        assert nested_after.active_attempt == 0
        assert nested_after.implementation_in_progress is False
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None
        terminal_link = task.body["completion_receipt"][
            "terminal_reconciliation"
        ]
        assert terminal_link["attempt_id"] == attempt.attempt_id
        assert terminal_link["claim_id"] == attempt.claim_id
        persisted = _reconciliation_receipt(
            paths, item["reconciliation_receipt_id"]
        )
        assert persisted["attempt_id"] == attempt.attempt_id
        assert persisted["task_cid"] == attempt.task_cid
        assert persisted["database_disposition"] == (
            "blocked_unknown_outcome"
        )
        assert persisted["receipt_id"] == item["reconciliation_receipt_id"]
        assert _git(repo, "status", "--porcelain", "--", "README.md") == ""
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        daemon.close()
        daemon = _database_portal_successor(repo)
        callbacks: list[str] = []

        def forbidden(*_args: object, **_kwargs: object) -> object:
            callbacks.append("callback")
            raise AssertionError("unknown predecessor callback was retried")

        daemon._provider_fn = forbidden
        daemon._effect_fn = forbidden
        daemon._validation_fn = forbidden
        restarted = daemon.run_once()
        assert restarted["implementation_result"] is None
        count = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert count is not None and int(count[0]) == 1
        assert callbacks == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("selection_change", ("prefix", "shard"))
def test_restart_gate_reconciles_all_owner_attempts_outside_current_selection(
    tmp_path: Path,
    selection_change: str,
) -> None:
    repo, seed, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    seed.close()
    if selection_change == "prefix":
        successor = _database_portal_successor(repo, task_prefix="OTHER-")
        trigger = "database_daemon_startup"
    else:
        home = int(
            hashlib.sha256(b"PCTDD-001").hexdigest()[:8], 16
        ) % 2
        successor = _database_portal_successor(
            repo,
            task_shard_count=2,
            task_shard_index=1 - home,
            strict_task_sharding=True,
        )
        trigger = "supervisor_signal_shutdown"
    try:
        assert successor.list_running_attempts() == []
        assert [
            item.attempt_id
            for item in successor.list_running_attempts(
                apply_selection=False
            )
        ] == [attempt.attempt_id]

        reconciliation = (
            successor.reconcile_quiesced_database_portal_attempts(
                trigger=trigger,
                force=True,
            )
        )

        assert reconciliation["blocked"] is False, reconciliation
        assert reconciliation["active_attempt_count"] == 1
        assert reconciliation["reason"] != "database_portal_already_quiesced"
        assert reconciliation["attempts"][0]["attempt_id"] == (
            attempt.attempt_id
        )
        assert successor.get_attempt(attempt.attempt_id).status == "failed"
        assert successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts"
        ).fetchone()[0] == 1
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("selection_change", ("prefix", "shard"))
def test_preserved_owner_attempt_outside_selection_remains_a_persistent_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_change: str,
) -> None:
    repo, predecessor, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    text = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        text.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    projected_task = parse_task_file(
        paths.task_projection,
        task_header_prefix="## PCTDD-001",
    )[0]
    paths.events.write_text(
        json.dumps(
            {
                "type": "task_completed",
                "task_id": projected_task.task_id,
                "canonical_task_key": projected_task.canonical_task_key,
                "canonical_task_cid": projected_task.canonical_task_cid,
                "board_namespace": projected_task.board_namespace,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    predecessor.close()

    if selection_change == "prefix":
        successor = _database_portal_successor(repo, task_prefix="OTHER-")
    else:
        home = int(hashlib.sha256(b"PCTDD-001").hexdigest()[:8], 16) % 2
        successor = _database_portal_successor(
            repo,
            task_shard_count=2,
            task_shard_index=1 - home,
            strict_task_sharding=True,
        )
    callbacks: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callbacks.append("called")
        raise AssertionError("out-of-selection predecessor dispatched work")

    monkeypatch.setattr(successor, "claim_next", forbidden)
    monkeypatch.setattr(
        successor,
        "reconcile_blocked_unknown_outcome_tasks",
        forbidden,
    )
    bound_bridge = getattr(successor._provider_fn, "__self__", None)
    assert isinstance(bound_bridge, DatabasePortalExecutionBridge)
    original_portal_factory = bound_bridge.portal_factory

    def reconciliation_only_portal_factory(
        *args: object,
        **kwargs: object,
    ) -> object:
        portal = original_portal_factory(*args, **kwargs)
        # Startup is allowed to call the existing quiesced-attempt
        # reconciler.  Any later attempt to enter the ordinary Portal pass is
        # a provider redispatch and must fail this test.
        portal.run_once = forbidden
        return portal

    bound_bridge.portal_factory = reconciliation_only_portal_factory
    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_portal_exact_phase_evidence_projected"
        ), first
        first_item = first["database_portal_reconciliation"]["attempts"][0]
        assert first_item["database_disposition"] == (
            "preserved_for_exact_phase_resume"
        )
        assert successor._database_portal_reconciliation_checked is True
        projected = successor.get_attempt(attempt.attempt_id)
        assert projected is not None and projected.phase_committed("provider")

        for _pass in range(2):
            blocked = successor.run_once()
            assert blocked["selection_idle_reason"] == (
                "database_portal_owner_attempt_outside_selection"
            )
            gate = blocked["database_portal_reconciliation"]
            assert gate["blocked"] is True
            assert gate["safe_to_restart"] is False
            assert gate["attempts"][0]["attempt_id"] == attempt.attempt_id
            current = successor.get_attempt(attempt.attempt_id)
            assert current is not None and current.status == "running"
        assert callbacks == []
        count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts"
        ).fetchone()
        assert count is not None and int(count[0]) == 1
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_expired_prepared_completion_reconciles_before_outside_selection_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    text = paths.task_projection.read_text(encoding="utf-8")
    paths.task_projection.write_text(
        text.replace("- Status: ready", "- Status: completed"),
        encoding="utf-8",
    )
    projected_task = parse_task_file(
        paths.task_projection,
        task_header_prefix="## PCTDD-001",
    )[0]
    paths.events.write_text(
        json.dumps(
            {
                "type": "task_completed",
                "task_id": projected_task.task_id,
                "canonical_task_key": projected_task.canonical_task_key,
                "canonical_task_cid": projected_task.canonical_task_cid,
                "board_namespace": projected_task.board_namespace,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    claim = predecessor.coordinator.get_task_claim(attempt.claim_id)
    task = predecessor.task_source.get_task(attempt.task_cid)
    assert claim is not None and task is not None
    prepared = predecessor.coordinator.prepare_task_completion(
        claim,
        control_expected_revision=int(task.revision),
        control_expected_status=str(task.status),
        evidence_digest="sha256:" + "a" * 64,
        body={"validation": {"outcome": "passed"}},
        now_ms=predecessor._now_ms(),
    )
    assert prepared["status"] == "prepared"
    predecessor.close()

    successor = _database_portal_successor(repo, task_prefix="OTHER-")
    monkeypatch.setattr(successor, "_now_ms", lambda: 2_000_000_000_000)
    callbacks: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callbacks.append("called")
        raise AssertionError("prepared predecessor dispatched work")

    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    bound_bridge = getattr(successor._provider_fn, "__self__", None)
    assert isinstance(bound_bridge, DatabasePortalExecutionBridge)
    original_portal_factory = bound_bridge.portal_factory

    def reconciliation_only_portal_factory(
        *args: object,
        **kwargs: object,
    ) -> object:
        portal = original_portal_factory(*args, **kwargs)
        portal.run_once = forbidden
        return portal

    bound_bridge.portal_factory = reconciliation_only_portal_factory
    try:
        result = successor.run_once()
        assert result["selection_idle_reason"] == (
            "database_prepared_completions_reconciled"
        ), result
        completions = result["completion_reconciliations"]
        assert len(completions) == 1
        assert completions[0]["status"] == "aborted"
        assert completions[0]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert completions[0]["retry_required"] is False
        old_attempt = successor.get_attempt(attempt.attempt_id)
        assert old_attempt is not None and old_attempt.status == "failed"
        control_task = successor.task_source.get_task(attempt.task_cid)
        assert control_task is not None and control_task.status == "blocked"
        assert control_task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        assert callbacks == []
        later = successor.run_once()
        assert later["implementation_result"] is None
        assert callbacks == []
    finally:
        successor.close()

    # A fresh process must not reinterpret the aborted completion barrier as
    # authority to create attempt 2 and repeat a callback that already crossed
    # the durable dispatch boundary.
    successor = _database_portal_successor(repo)
    successor._provider_fn = forbidden
    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    try:
        for _pass in range(2):
            restarted = successor.run_once()
            assert restarted["implementation_result"] is None
        count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert count is not None and int(count[0]) == 1
        assert callbacks == []
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_expired_prepared_completion_with_missing_attempt_blocks_before_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    claim = predecessor.coordinator.get_task_claim(attempt.claim_id)
    task = predecessor.task_source.get_task(attempt.task_cid)
    assert claim is not None and task is not None
    prepared = predecessor.coordinator.prepare_task_completion(
        claim,
        control_expected_revision=int(task.revision),
        control_expected_status=str(task.status),
        evidence_digest="sha256:" + "b" * 64,
        body={"validation": {"outcome": "passed"}},
        now_ms=predecessor._now_ms(),
    )
    assert prepared["status"] == "prepared"
    connection = predecessor._require_connection()
    connection.execute(
        "DELETE FROM attempt_phases WHERE attempt_id = ?",
        [attempt.attempt_id],
    )
    connection.execute(
        "DELETE FROM database_task_attempts WHERE attempt_id = ?",
        [attempt.attempt_id],
    )
    predecessor.close()

    def open_daemon() -> DatabaseImplementationDaemon:
        daemon = DatabaseImplementationDaemon(
            database_path=repo / "control.duckdb",
            max_task_attempts=3,
            owner_session_id="",
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            task_prefix="PCTDD-",
            require_real_execution=True,
        )
        monkeypatch.setattr(daemon, "_now_ms", lambda: 2_000_000_000_000)
        return daemon

    callbacks: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callbacks.append("callback")
        raise AssertionError("missing completion projection was redispatched")

    successor = open_daemon()
    successor.bind_execution_callbacks(
        provider_fn=forbidden,
        effect_fn=forbidden,
        validation_fn=forbidden,
    )
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_prepared_completions_reconciled"
        ), first
        assert len(first["completion_reconciliations"]) == 1
        outcome = first["completion_reconciliations"][0]
        assert outcome["status"] == "blocked"
        assert outcome["reason"] == "callback_authority_incomplete_blocked"
        assert outcome["attempt_projection_missing"] is True
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        receipt = task.body["completion_receipt"]
        assert receipt["prepared_completion_projection_missing"] is True
        assert receipt["coordination_preparation_digest"] == (
            prepared["preparation_digest"]
        )
        assert successor.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is not None
        successor.materialize_population(
            {
                "repository_tree_id": "tree:projection-loss-unrelated-work",
                "tasks": [
                    {
                        "task_cid": "task:cid:pctdd-002",
                        "task_id": "PCTDD-002",
                        "goal_cid": "goal:pctdd",
                        "status": "ready",
                        "validation_commands": [
                            "python -m pytest unrelated.py"
                        ],
                    }
                ],
            }
        )
    finally:
        successor.close()

    # The durable PREPARED barrier and its exact control block survive fresh
    # --once processes without becoming a global startup latch.  The blocked
    # task stays unready while an unrelated task may still make progress.
    successor = open_daemon()

    progressed: list[str] = []

    def provider(selected: DatabaseTaskAttempt) -> Mapping[str, object]:
        progressed.append(f"provider:{selected.task_cid}")
        return {"status": "accepted", "accepted": True}

    def effect(
        selected: DatabaseTaskAttempt,
        _provider_result: Mapping[str, object],
    ) -> Mapping[str, object]:
        progressed.append(f"effect:{selected.task_cid}")
        return {"status": "applied"}

    def validation(
        selected: DatabaseTaskAttempt,
        _effect_result: Mapping[str, object],
    ) -> Mapping[str, object]:
        progressed.append(f"validation:{selected.task_cid}")
        return {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "c" * 64,
        }

    successor.bind_execution_callbacks(
        provider_fn=provider,
        effect_fn=effect,
        validation_fn=validation,
    )
    try:
        replay = successor.run_once()
        assert replay["implementation_result"] is not None
        assert replay["claimed_task_cid"] == "task:cid:pctdd-002"
        blocked_count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert blocked_count is not None and int(blocked_count[0]) == 0
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert successor.coordinator.get_prepared_task_completion(
            attempt.task_cid
        ) is not None
        assert progressed == [
            "provider:task:cid:pctdd-002",
            "effect:task:cid:pctdd-002",
            "validation:task:cid:pctdd-002",
        ]
        assert callbacks == []
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_supervisor_sigterm_reconciles_database_portal_before_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_session_id = "session:pctdd-named-lane"
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            owner_session_id=owner_session_id,
        )
    )
    daemon.close()
    state_dir = repo / "state"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "non-authoritative.md",
            state_path=state_dir / "pctdd_task_state.json",
            strategy_path=state_dir / "pctdd_strategy.json",
            events_path=state_dir / "pctdd_supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            state_prefix="pctdd",
            task_prefix="## PCTDD-",
            max_task_attempts=3,
            database_owner_session_id=owner_session_id,
            implement=True,
            worktree_root=repo / "worktrees",
            merge_target_branch="main",
            database_program=DatabaseProgramConfig(
                authority_mode="embedded_exclusive",
                task_source_kind="duckdb",
                store_id="control.duckdb",
            ),
        )
    )
    monkeypatch.setattr(
        supervisor,
        "_run_forever_loop",
        lambda: signal.raise_signal(signal.SIGTERM),
    )
    command = supervisor._build_daemon_command()
    owner_indexes = [
        index
        for index, value in enumerate(command[:-1])
        if value == "--owner-session-id"
    ]
    assert len(owner_indexes) == 1
    assert command[owner_indexes[0] + 1] == owner_session_id
    from threading import Event, Thread

    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        serialized_lock_update,
    )

    competing_launch_attempted = Event()
    competing_launch_acquired = Event()
    competing_thread: list[Thread] = []

    def competing_launch() -> None:
        competing_launch_attempted.set()
        with serialized_lock_update(
            supervisor._managed_daemon_launch_lock_path()
        ):
            competing_launch_acquired.set()

    def terminate_under_launch_lock(**_kwargs: object) -> dict[str, object]:
        thread = Thread(target=competing_launch, daemon=True)
        competing_thread.append(thread)
        thread.start()
        assert competing_launch_attempted.wait(timeout=1.0)
        assert not competing_launch_acquired.is_set()
        return {
            "pid": 4321,
            "terminated": True,
            "quiesced": True,
            "remaining_pid": None,
            "markers_removed": True,
            "daemon_fence": {"safe_to_restart": True},
            "provider_runner_fence": {"safe_to_restart": True},
        }

    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        terminate_under_launch_lock,
    )
    original_reconcile = (
        supervisor._reconcile_interrupted_database_portal_attempts
    )

    def reconcile_while_launch_inhibited() -> dict[str, object]:
        assert not competing_launch_acquired.is_set()
        result = original_reconcile()
        assert not competing_launch_acquired.is_set()
        return result

    monkeypatch.setattr(
        supervisor,
        "_reconcile_interrupted_database_portal_attempts",
        reconcile_while_launch_inhibited,
    )

    with pytest.raises(SystemExit) as stopped:
        supervisor.run_forever()

    assert stopped.value.code == 128 + signal.SIGTERM
    assert competing_thread
    competing_thread[0].join(timeout=2.0)
    assert competing_launch_acquired.is_set()
    status = json.loads(
        (state_dir / "pctdd_supervisor_status.json").read_text(
            encoding="utf-8"
        )
    )
    reconciliation = status["interrupted_implementation_reconciliation"]
    assert reconciliation["reconciled"] is True
    assert reconciliation["active_attempt_count"] == 1
    assert reconciliation["attempts"][0]["attempt_id"] == attempt.attempt_id
    assert reconciliation["attempts"][0]["task_cid"] == attempt.task_cid
    assert reconciliation["attempts"][0]["database_disposition"] == (
        "blocked_unknown_outcome"
    )
    receipt_id = reconciliation["attempts"][0]["reconciliation_receipt_id"]
    receipt = _reconciliation_receipt(paths, receipt_id)
    assert receipt["trigger"] == "supervisor_signal_shutdown"
    assert receipt["attempt_id"] == attempt.attempt_id
    assert receipt["attempt_number"] == attempt.attempt_number
    assert receipt["owner_session_id"] == owner_session_id
    successor = _database_portal_successor(
        repo,
        owner_session_id=owner_session_id,
    )
    callbacks: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callbacks.append("callback")
        raise AssertionError("SIGTERM unknown callback was retried")

    successor._provider_fn = forbidden
    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    try:
        restarted = successor.run_once()
        assert restarted["implementation_result"] is None
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert count is not None and int(count[0]) == 1
        assert callbacks == []
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_supervisor_shutdown_does_not_open_or_mutate_db_when_not_quiesced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    daemon.close()
    state_dir = repo / "state"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "non-authoritative.md",
            state_path=state_dir / "pctdd_task_state.json",
            strategy_path=state_dir / "pctdd_strategy.json",
            events_path=state_dir / "pctdd_supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            state_prefix="pctdd",
            task_prefix="## PCTDD-",
            max_task_attempts=3,
            implement=True,
            worktree_root=repo / "worktrees",
            merge_target_branch="main",
            database_program=DatabaseProgramConfig(
                authority_mode="embedded_exclusive",
                task_source_kind="duckdb",
                store_id="control.duckdb",
            ),
        )
    )
    monkeypatch.setattr(
        supervisor,
        "_run_forever_loop",
        lambda: signal.raise_signal(signal.SIGTERM),
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_managed_daemon_tree",
        lambda **_kwargs: {
            "pid": 4321,
            "terminated": False,
            "quiesced": False,
            "remaining_pid": 4321,
        },
    )

    def forbidden_database_reconciliation() -> dict[str, object]:
        raise AssertionError("database must not open before daemon quiescence")

    monkeypatch.setattr(
        supervisor,
        "_reconcile_interrupted_database_portal_attempts",
        forbidden_database_reconciliation,
    )

    with pytest.raises(SystemExit):
        supervisor.run_forever()

    status = json.loads(
        (state_dir / "pctdd_supervisor_status.json").read_text(
            encoding="utf-8"
        )
    )
    reconciliation = status["interrupted_implementation_reconciliation"]
    assert reconciliation["reconciled"] is False
    assert reconciliation["blocked"] is True
    assert reconciliation["reason"] == "managed_database_daemon_not_quiesced"
    nested = PortalTaskState.load(paths.state)
    assert nested.task_statuses["PCTDD-001"] == "in_progress"

    verifier = DatabaseImplementationDaemon(
        database_path=repo / "control.duckdb",
        max_task_attempts=3,
        owner_session_id="",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        task_prefix="PCTDD-",
        require_real_execution=True,
    )
    try:
        current = verifier.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        task = verifier.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
    finally:
        verifier.close()


def test_quack_shutdown_reconciliation_scopes_exact_program_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        database_portal_bridge as bridge_module,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )

    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    program = DatabaseProgramConfig(
        authority_mode="quack",
        task_source_kind="duckdb",
        endpoint_secret_handle="env://SELECTED_QUACK_TOKEN",
        quack_endpoint="quack://127.0.0.1:41307",
        store_id="state/control.duckdb",
        store_generation="pctdd-v1-g6",
        schema_revision="3",
    )
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "non-authoritative.md",
            state_path=state_dir / "pctdd_task_state.json",
            strategy_path=state_dir / "pctdd_strategy.json",
            events_path=state_dir / "pctdd_supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            state_prefix="pctdd",
            task_prefix="## PCTDD-",
            database_program=program,
        )
    )

    conflicting = {
        "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE": "handle:wrong",
        "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT": "quack://127.0.0.1:49999",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_ID": "wrong/control.duckdb",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION": "99",
        "IPFS_ACCELERATE_AGENT_STATE_SCHEMA_REVISION": "98",
        "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION": "97",
        "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION": "96",
        "IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT": str(tmp_path / "wrong"),
        "IPFS_ACCELERATE_AGENT_EVENT_STORE_PATH": "wrong-events",
        "IPFS_ACCELERATE_AGENT_QUACK_TOKEN": "raw-token-must-not-win",
        "QUACK_TOKEN": "second-raw-token-must-not-win",
        "SELECTED_QUACK_TOKEN": "handle-target-must-not-win",
    }
    for name, value in conflicting.items():
        monkeypatch.setenv(name, value)
    before = {name: os.environ.get(name) for name in conflicting}
    selected = program.environment()
    observed: dict[str, object] = {}

    def assert_selected_environment() -> None:
        for name, value in selected.items():
            assert os.environ.get(name) == value
        assert os.environ["IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT"] == str(
            repo.resolve()
        )
        assert (
            "IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION"
            not in os.environ
        )
        assert (
            "IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION"
            not in os.environ
        )
        assert "IPFS_ACCELERATE_AGENT_EVENT_STORE_PATH" not in os.environ
        assert "IPFS_ACCELERATE_AGENT_QUACK_TOKEN" not in os.environ
        assert "QUACK_TOKEN" not in os.environ
        assert "SELECTED_QUACK_TOKEN" not in os.environ

    class FakeDatabaseImplementationDaemon:
        def __init__(self, **kwargs: object) -> None:
            assert_selected_environment()
            assert kwargs["authority_mode"] == "quack"
            assert kwargs["task_source_kind"] == "duckdb"
            assert kwargs["quack_uri"] == program.quack_endpoint
            observed["daemon_kwargs"] = dict(kwargs)
            self.task_source = object()

        def bind_execution_callbacks(self, **kwargs: object) -> None:
            observed["callbacks"] = tuple(sorted(kwargs))

        def bind_database_portal_bridge(self, bridge: object) -> None:
            observed["bridge"] = bridge

        def reconcile_quiesced_database_portal_attempts(
            self,
            **kwargs: object,
        ) -> dict[str, object]:
            assert_selected_environment()
            assert kwargs == {
                "trigger": "supervisor_signal_shutdown",
                "force": True,
            }
            return {"reconciled": True, "blocked": False}

        def close(self) -> None:
            assert_selected_environment()
            observed["closed"] = True

    class FakeDatabasePortalExecutionBridge:
        def __init__(self, **kwargs: object) -> None:
            observed["bridge_kwargs"] = dict(kwargs)

        def run_provider(self, *_args: object) -> dict[str, object]:
            return {}

        def apply_effect(self, *_args: object) -> dict[str, object]:
            return {}

        def validate_effect(self, *_args: object) -> dict[str, object]:
            return {}

    monkeypatch.setattr(
        daemon_module,
        "DatabaseImplementationDaemon",
        FakeDatabaseImplementationDaemon,
    )
    monkeypatch.setattr(
        bridge_module,
        "DatabasePortalExecutionBridge",
        FakeDatabasePortalExecutionBridge,
    )

    result = supervisor._reconcile_interrupted_database_portal_attempts()

    assert result == {"reconciled": True, "blocked": False}
    assert observed["closed"] is True
    assert observed["callbacks"] == (
        "effect_fn",
        "provider_fn",
        "validation_fn",
    )
    assert {name: os.environ.get(name) for name in conflicting} == before
    for name in selected:
        if name not in conflicting:
            assert name not in os.environ


def test_managed_database_owner_mismatch_blocks_before_direct_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_supervisor as supervisor_module,
    )

    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    owner_session_id = "session:selected-database-owner"
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "non-authoritative.md",
            state_path=state_dir / "pctdd_task_state.json",
            strategy_path=state_dir / "pctdd_strategy.json",
            events_path=state_dir / "pctdd_supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            state_prefix="pctdd",
            task_prefix="## PCTDD-",
            database_program=DatabaseProgramConfig(
                authority_mode="embedded_exclusive",
                task_source_kind="duckdb",
                store_id="control.duckdb",
            ),
            database_owner_session_id=owner_session_id,
        )
    )
    selected_command = supervisor._build_daemon_command()
    owner_index = selected_command.index("--owner-session-id")
    assert selected_command[owner_index + 1] == owner_session_id
    assert supervisor._managed_daemon_command_belongs_to_scope(
        selected_command
    )
    foreign_command = list(selected_command)
    foreign_command[owner_index + 1] = "session:foreign-database-owner"
    foreign_scope = supervisor._managed_daemon_owner_scope()
    foreign_scope["database_owner_session_id"] = (
        "session:foreign-database-owner"
    )
    identity = SimpleNamespace(
        process_birth=SimpleNamespace(pid=4321),
        owner_scope=foreign_scope,
        command=tuple(foreign_command),
    )
    monkeypatch.setattr(
        supervisor_module,
        "load_supervised_child_identity",
        lambda _path: identity,
    )

    fence = supervisor._fence_recorded_managed_daemon(pid=4321)

    assert fence == {
        "fenced": False,
        "reason": "managed_daemon_ownership_scope_mismatch",
    }
    assert not supervisor._managed_daemon_command_belongs_to_scope(
        foreign_command
    )

    def forbidden_database_open() -> dict[str, object]:
        raise AssertionError("owner mismatch must block before direct DB open")

    monkeypatch.setattr(
        supervisor,
        "_reconcile_interrupted_database_portal_attempts",
        forbidden_database_open,
    )
    reconciliation = (
        supervisor._reconcile_interrupted_implementation_after_shutdown(
            cleanup={
                "quiesced": False,
                "remaining_pid": 4321,
                "markers_removed": False,
                "daemon_fence": {
                    **fence,
                    "safe_to_restart": False,
                },
                "provider_runner_fence": {"safe_to_restart": True},
            }
        )
    )
    assert reconciliation["blocked"] is True
    assert reconciliation["reason"] == "managed_database_daemon_not_quiesced"


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_restart_keeps_incomplete_callback_authority_permanently_blocked(
    tmp_path: Path,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    predecessor._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
        outcome="deferred",
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    task_before = predecessor.task_source.get_task(attempt.task_cid)
    assert task_before is not None
    predecessor_process = task_before.body["completion_receipt"][
        "process_instance_id"
    ]
    predecessor.close()

    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded_exclusive",
            "--database-path",
            str(repo / "control.duckdb"),
            "--state-dir",
            str(repo / "state"),
            "--state-prefix",
            "pctdd",
            "--task-prefix",
            "## PCTDD-",
            "--worktree-root",
            str(repo / "worktrees"),
            "--merge-target-branch",
            "main",
            "--implement",
            "--max-task-attempts",
            "3",
            "--once",
        ]
    )
    successor = DatabaseImplementationDaemon(
        database_path=repo / "control.duckdb",
        max_task_attempts=3,
        owner_session_id="",
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        task_prefix="PCTDD-",
        require_real_execution=True,
    )
    callback_attempts: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("incomplete callback authority was redispatched")

    try:
        bind_database_portal_execution_from_args(
            successor,
            args,
            repo_root=repo,
            portal_daemon_class=PortalImplementationDaemon,
        )
        successor._provider_fn = forbidden
        successor._effect_fn = forbidden
        successor._validation_fn = forbidden

        first_pass = successor.run_once()

        assert first_pass["selection_idle_reason"] == (
            "database_portal_reconciliation_completed"
        )
        assert first_pass["implementation_result"] is None
        assert first_pass["unknown_outcome_rearms"] == []
        reconciliation = first_pass["database_portal_reconciliation"]
        assert reconciliation["active_attempt_count"] == 1
        assert reconciliation["attempts"][0]["database_disposition"] == (
            "blocked_unknown_outcome"
        )
        assert successor.list_running_attempts() == []
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        receipt = task.body["completion_receipt"]
        assert receipt["reason"] == "callback_authority_incomplete_blocked"
        assert receipt["process_instance_id"] == predecessor_process
        assert receipt["reconciled_by_process_instance_id"] == (
            successor.process_instance_id
        )
        assert receipt["terminal_reconciliation"]["attempt_id"] == (
            attempt.attempt_id
        )
        successor.close()
        successor = _database_portal_successor(repo)
        successor._provider_fn = forbidden
        successor._effect_fn = forbidden
        successor._validation_fn = forbidden
        second_pass = successor.run_once()
        assert second_pass["implementation_result"] is None
        attempt_count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempt_count is not None and int(attempt_count[0]) == 1
        assert callback_attempts == []
    finally:
        successor.close()


@pytest.mark.parametrize(
    "mutation",
    ("duplicate_key", "extra_field", "noncanonical_number"),
)
def test_shutdown_binding_parser_rejects_noncanonical_or_widened_records(
    tmp_path: Path,
    mutation: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), _record())
    raw = paths.binding.read_text(encoding="utf-8")
    if mutation == "duplicate_key":
        raw = raw.replace(
            '  "attempt_id": "attempt:001",',
            '  "attempt_id": "attempt:other",\n'
            '  "attempt_id": "attempt:001",',
        )
    else:
        payload = json.loads(raw)
        if mutation == "extra_field":
            payload["new_authority"] = "silently-widened"
        else:
            payload["task_revision"] = 11.0
        unsigned = dict(payload)
        unsigned.pop("binding_id", None)
        import hashlib

        payload["binding_id"] = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
        raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    paths.binding.write_text(raw, encoding="utf-8")

    with pytest.raises(DatabasePortalBridgeError):
        bridge.reconcile_quiesced_attempt(_attempt())


@pytest.mark.parametrize(
    "raw_state",
    (
        '{"active_task_id":"other","active_task_id":"LGSWF-004"}\n',
        '{"active_attempt":NaN}\n',
        '{"unreviewed_state_authority":true}\n',
        '{"active_attempt":true}\n',
    ),
)
def test_shutdown_state_parser_rejects_malformed_or_widened_records(
    tmp_path: Path,
    raw_state: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), _record())
    paths.state.write_text(raw_state, encoding="utf-8")

    with pytest.raises(DatabasePortalBridgeError):
        bridge.reconcile_quiesced_attempt(_attempt())


@pytest.mark.parametrize("artifact_case", ("partial", "state_symlink"))
def test_shutdown_rejects_partial_or_symlinked_nested_attempt_artifacts(
    tmp_path: Path,
    artifact_case: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths, _binding = bridge._ensure_attempt_projection(_attempt(), _record())
    if artifact_case == "partial":
        paths.task_projection.unlink()
    else:
        target = tmp_path / "untrusted-state.json"
        target.write_text("{}\n", encoding="utf-8")
        paths.state.symlink_to(target)

    with pytest.raises(DatabasePortalBridgeError):
        bridge.reconcile_quiesced_attempt(_attempt())


def test_shutdown_rejects_symlinked_attempt_authority_before_state_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path / "outside-attempt-authority"
    outside.mkdir()
    attempt_root = tmp_path / "attempts"
    attempt_root.symlink_to(outside, target_is_directory=True)
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=attempt_root,
        portal_factory=lambda _paths, _alias: None,
    )
    monkeypatch.setattr(
        bridge,
        "_strict_state_record",
        lambda _path: pytest.fail("symlinked authority nominated state"),
    )

    with pytest.raises(DatabasePortalBridgeError, match="escape"):
        bridge.reconcile_quiesced_attempt(_attempt())


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_reconciliation_evidence_write_failure_precedes_database_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    original_persist = bridge.persist_reconciliation_receipt

    def fail_prepared_receipt(
        selected_attempt: object,
        payload: object,
    ) -> dict[str, object]:
        assert isinstance(payload, dict)
        if payload.get("stage") == "prepared":
            raise OSError("injected immutable evidence write failure")
        return original_persist(selected_attempt, payload)

    monkeypatch.setattr(
        bridge,
        "persist_reconciliation_receipt",
        fail_prepared_receipt,
    )
    try:
        with pytest.raises(
            OSError, match="injected immutable evidence write failure"
        ):
            daemon.reconcile_quiesced_database_portal_attempts(
                trigger="supervisor_signal_shutdown",
                force=True,
            )

        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("fsync_target", ("receipt_directory", "attempt_root"))
def test_reconciliation_directory_fsync_failure_precedes_database_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fsync_target: str,
) -> None:
    import ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge as bridge_module

    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    original_open = bridge_module.os.open
    rejected_directory = (
        paths.reconciliation
        if fsync_target == "receipt_directory"
        else paths.reconciliation.parent
    )

    def fail_reconciliation_directory_open(
        selected_path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        if Path(selected_path) == rejected_directory:
            raise OSError("injected reconciliation directory fsync failure")
        return original_open(selected_path, flags, *args, **kwargs)

    monkeypatch.setattr(
        bridge_module.os,
        "open",
        fail_reconciliation_directory_open,
    )
    try:
        with pytest.raises(
            OSError, match="injected reconciliation directory fsync failure"
        ):
            daemon.reconcile_quiesced_database_portal_attempts(
                trigger="supervisor_signal_shutdown",
                force=True,
            )

        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "in_progress"
    finally:
        daemon.close()


def test_reconciliation_receipt_loader_repairs_exact_link_publication_crash(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    receipt = bridge.persist_reconciliation_receipt(
        attempt,
        {
            "stage": "blocked",
            "trigger": "test_link_publication_crash",
            "reconciled_at": "2026-08-30T00:00:00+00:00",
            "reconciled": False,
            "blocked": True,
            "reason": "test_only",
        },
    )
    final = Path(str(receipt["receipt_path"]))
    temporary = final.parent / f".{final.name}.crash-window.tmp"
    os.link(final, temporary)
    assert final.stat().st_nlink == 2

    loaded = bridge.load_reconciliation_receipt(
        attempt,
        str(receipt["receipt_id"]),
        required_stage="blocked",
    )

    assert loaded["receipt_id"] == receipt["receipt_id"]
    assert final.stat().st_nlink == 1
    assert not temporary.exists()


def test_reconciliation_receipt_recovery_fsyncs_final_only_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    receipt = bridge.persist_reconciliation_receipt(
        _attempt(),
        {
            "stage": "blocked",
            "trigger": "test_final_only_directory_durability",
            "reconciled_at": "2026-08-30T00:00:00+00:00",
            "reconciled": False,
            "blocked": True,
            "reason": "test_only",
        },
    )
    final = Path(str(receipt["receipt_path"]))
    original_fsync = database_portal_bridge_module._fsync_directory
    fsynced: list[Path] = []

    def observe_fsync(path: Path) -> None:
        fsynced.append(path)
        original_fsync(path)

    monkeypatch.setattr(
        database_portal_bridge_module,
        "_fsync_directory",
        observe_fsync,
    )

    database_portal_bridge_module._recover_immutable_link_publication(final)

    assert fsynced == [final.parent, final.parent.parent]
    assert final.stat().st_nlink == 1


@pytest.mark.parametrize(
    "crash_shape",
    (
        "link_before_unlink",
        "prelink_final_absent",
        "prelink_final_present",
        "stage_link_before_unlink",
    ),
)
def test_reconciliation_receipt_persist_repairs_exact_publication_prefix(
    tmp_path: Path,
    crash_shape: str,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    payload = {
        "stage": "blocked",
        "trigger": "test_persist_publication_crash",
        "reconciled_at": "2026-08-30T00:00:00+00:00",
        "reconciled": False,
        "blocked": True,
        "reason": "test_only",
    }
    receipt = bridge.persist_reconciliation_receipt(attempt, payload)
    final = Path(str(receipt["receipt_path"]))
    encoded = final.read_bytes()
    temporary = final.parent / f".{final.name}.{crash_shape}.tmp"
    stage = final.parent / f".{final.name}.{crash_shape}.stage"
    if crash_shape == "link_before_unlink":
        os.link(final, temporary)
        assert final.stat().st_nlink == 2
    elif crash_shape == "stage_link_before_unlink":
        final.unlink()
        stage.write_bytes(encoded)
        os.link(stage, temporary)
        assert stage.stat().st_nlink == 2
        assert temporary.stat().st_ino == stage.stat().st_ino
    else:
        temporary.write_bytes(encoded)
        if crash_shape == "prelink_final_absent":
            final.unlink()
        else:
            assert final.stat().st_ino != temporary.stat().st_ino

    replayed = bridge.persist_reconciliation_receipt(attempt, payload)

    assert replayed["receipt_id"] == receipt["receipt_id"]
    assert final.read_bytes() == encoded
    assert final.stat().st_nlink == 1
    assert not temporary.exists()
    assert not stage.exists()


def test_reconciliation_receipt_persist_rejects_mismatched_prelink_temporary(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    payload = {
        "stage": "blocked",
        "trigger": "test_mismatched_publication_crash",
        "reconciled_at": "2026-08-30T00:00:00+00:00",
        "reconciled": False,
        "blocked": True,
        "reason": "test_only",
    }
    receipt = bridge.persist_reconciliation_receipt(attempt, payload)
    final = Path(str(receipt["receipt_path"]))
    temporary = final.parent / f".{final.name}.mismatch.tmp"
    temporary.write_bytes(b"not the immutable receipt\n")

    with pytest.raises(DatabasePortalBridgeError, match="not exact"):
        bridge.persist_reconciliation_receipt(attempt, payload)

    assert final.exists()
    assert temporary.exists()


def test_immutable_publication_revalidates_concurrent_visible_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serialized competing payload is never mistaken for our success."""

    final = tmp_path / "evidence.json"
    requested_payload = b'{"writer":"requested"}\n'
    competing_payload = b'{"writer":"competing"}\n'
    original_mkstemp = database_portal_bridge_module.tempfile.mkstemp
    competing_stage_visible = threading.Event()
    release_competitor = threading.Event()

    def pause_competitor_before_stage_write(
        *args: object,
        **kwargs: object,
    ) -> tuple[int, str]:
        descriptor, name = original_mkstemp(*args, **kwargs)
        if threading.current_thread().name == "competing-writer":
            competing_stage_visible.set()
            assert release_competitor.wait(timeout=10.0)
        return descriptor, name

    monkeypatch.setattr(
        database_portal_bridge_module.tempfile,
        "mkstemp",
        pause_competitor_before_stage_write,
    )
    outcomes: dict[str, BaseException | None] = {}

    def publish(name: str, payload: bytes) -> None:
        try:
            database_portal_bridge_module._atomic_write_if_absent(final, payload)
        except BaseException as exc:  # assertion evidence from worker thread
            outcomes[name] = exc
        else:
            outcomes[name] = None

    competing = threading.Thread(
        target=publish,
        args=("competing", competing_payload),
        name="competing-writer",
    )
    competing.start()
    assert competing_stage_visible.wait(timeout=10.0)
    requested = threading.Thread(
        target=publish,
        args=("requested", requested_payload),
        name="requested-writer",
    )
    requested.start()
    release_competitor.set()
    competing.join(timeout=15.0)
    requested.join(timeout=15.0)

    assert not competing.is_alive()
    assert not requested.is_alive()
    assert outcomes["competing"] is None
    assert isinstance(outcomes["requested"], DatabasePortalBridgeError)
    assert "not exact" in str(outcomes["requested"])
    assert final.read_bytes() == competing_payload
    assert final.stat().st_nlink == 1
    assert list(tmp_path.glob(f".{final.name}.*.tmp")) == []
    assert list(tmp_path.glob(f".{final.name}.*.stage")) == []


def test_immutable_publication_never_overwrites_ready_name_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "evidence.json"
    prior_ready_payload = b"prior recovery evidence\n"
    original_link = database_portal_bridge_module.os.link
    collision_paths: list[Path] = []

    def publish_collision_inside_link(
        source: object,
        destination: object,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.endswith(".stage"):
            destination_path.write_bytes(prior_ready_payload)
            collision_paths.append(destination_path)
        original_link(source, destination)

    monkeypatch.setattr(
        database_portal_bridge_module.os,
        "link",
        publish_collision_inside_link,
    )

    with pytest.raises(DatabasePortalBridgeError, match="ready name already exists"):
        database_portal_bridge_module._atomic_write_if_absent(
            final,
            b"new requested evidence\n",
        )

    assert not final.exists()
    assert len(collision_paths) == 1
    assert collision_paths[0].read_bytes() == prior_ready_payload
    assert list(tmp_path.glob(f".{final.name}.*.stage")) == []


def test_concurrent_identical_reconciliation_receipt_publication_converges(
    tmp_path: Path,
) -> None:
    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    payload = {
        "stage": "blocked",
        "trigger": "test_concurrent_publication",
        "reconciled_at": "2026-08-30T00:00:11+00:00",
        "reconciled": False,
        "blocked": True,
        "reason": "test_only",
    }
    start_barrier = threading.Barrier(2, timeout=10.0)
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def publish() -> None:
        try:
            start_barrier.wait()
            results.append(
                bridge.persist_reconciliation_receipt(attempt, payload)
            )
        except BaseException as exc:  # assertion evidence from worker thread
            failures.append(exc)

    workers = [threading.Thread(target=publish) for _index in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15.0)

    assert all(not worker.is_alive() for worker in workers)
    assert failures == []
    assert len(results) == 2
    assert results[0]["receipt_id"] == results[1]["receipt_id"]
    final = Path(str(results[0]["receipt_path"]))
    assert final.is_file()
    assert final.stat().st_nlink == 1
    assert list(final.parent.glob(f".{final.name}.*.tmp")) == []
    assert list(final.parent.glob(f".{final.name}.*.stage")) == []


def test_concurrent_recovery_can_publish_and_unlink_delayed_writer_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live zero-byte stage is invisible to serialized recovery."""

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    payload = {
        "stage": "blocked",
        "trigger": "test_concurrent_prelink_recovery",
        "reconciled_at": "2026-08-30T00:00:13+00:00",
        "reconciled": False,
        "blocked": True,
        "reason": "test_only",
    }
    original_mkstemp = database_portal_bridge_module.tempfile.mkstemp
    live_stage_visible = threading.Event()
    release_writer = threading.Event()

    def pause_writer_before_stage_write(
        *args: object,
        **kwargs: object,
    ) -> tuple[int, str]:
        descriptor, name = original_mkstemp(*args, **kwargs)
        if threading.current_thread().name == "delayed-writer":
            live_stage_visible.set()
            assert release_writer.wait(timeout=10.0)
        return descriptor, name

    monkeypatch.setattr(
        database_portal_bridge_module.tempfile,
        "mkstemp",
        pause_writer_before_stage_write,
    )
    results: list[dict[str, object]] = []
    failures: list[BaseException] = []

    def publish() -> None:
        try:
            results.append(
                bridge.persist_reconciliation_receipt(attempt, payload)
            )
        except BaseException as exc:  # assertion evidence from worker thread
            failures.append(exc)

    delayed = threading.Thread(
        target=publish,
        name="delayed-writer",
    )
    delayed.start()
    assert live_stage_visible.wait(timeout=10.0)
    recovering = threading.Thread(
        target=publish,
        name="recovering-writer",
    )
    recovering.start()
    recovering.join(timeout=0.1)
    assert recovering.is_alive()
    release_writer.set()
    delayed.join(timeout=15.0)
    recovering.join(timeout=15.0)

    assert not recovering.is_alive()
    assert not delayed.is_alive()
    assert failures == []
    assert len(results) == 2
    assert results[0]["receipt_id"] == results[1]["receipt_id"]
    final = Path(str(results[0]["receipt_path"]))
    assert final.is_file()
    assert final.stat().st_nlink == 1
    assert list(final.parent.glob(f".{final.name}.*.tmp")) == []
    assert list(final.parent.glob(f".{final.name}.*.stage")) == []


def test_immutable_receipt_publication_rejects_enoent_without_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vanished source is not success unless another writer published final."""

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    payload = {
        "stage": "blocked",
        "trigger": "test_missing_source_and_final",
        "reconciled_at": "2026-08-30T00:00:17+00:00",
        "reconciled": False,
        "blocked": True,
        "reason": "test_only",
    }
    original_link = database_portal_bridge_module.os.link

    def lose_source_without_publication(
        source: object,
        destination: object,
    ) -> None:
        if Path(destination).name.endswith(".tmp"):
            original_link(source, destination)
            return
        os.unlink(source)
        raise FileNotFoundError(str(source))

    monkeypatch.setattr(
        database_portal_bridge_module.os,
        "link",
        lose_source_without_publication,
    )

    with pytest.raises(
        DatabasePortalBridgeError,
        match="did not produce a settled final object",
    ):
        bridge.persist_reconciliation_receipt(attempt, payload)

    reconciliation = bridge._paths(attempt).reconciliation
    assert list(reconciliation.glob("*.json")) == []
    assert list(reconciliation.glob(".*.tmp")) == []


@pytest.mark.parametrize(
    "disappearance_window",
    ("before_exact_regular", "during_cleanup"),
)
def test_immutable_receipt_recovery_requires_convergence_after_temp_disappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disappearance_window: str,
) -> None:
    """A concurrent unlink is benign only after exact final convergence."""

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(_record()),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    attempt = _attempt()
    payload = {
        "stage": "blocked",
        "trigger": f"test_temp_disappears_{disappearance_window}",
        "reconciled_at": "2026-08-30T00:00:19+00:00",
        "reconciled": False,
        "blocked": True,
        "reason": "test_only",
    }
    receipt = bridge.persist_reconciliation_receipt(attempt, payload)
    final = Path(str(receipt["receipt_path"]))
    encoded = final.read_bytes()
    temporary = final.parent / (
        f".{final.name}.{disappearance_window}.tmp"
    )
    temporary.write_bytes(encoded)
    triggered = {"done": False}

    if disappearance_window == "before_exact_regular":
        original_lstat = Path.lstat

        def disappear_before_lstat(
            selected: Path,
            *args: object,
            **kwargs: object,
        ) -> os.stat_result:
            if selected == temporary and not triggered["done"]:
                triggered["done"] = True
                temporary.unlink()
            return original_lstat(selected, *args, **kwargs)

        monkeypatch.setattr(Path, "lstat", disappear_before_lstat)
    else:
        original_unlink = Path.unlink

        def disappear_during_unlink(
            selected: Path,
            *args: object,
            **kwargs: object,
        ) -> None:
            if selected == temporary and not triggered["done"]:
                triggered["done"] = True
                original_unlink(selected, *args, **kwargs)
                raise FileNotFoundError(str(selected))
            original_unlink(selected, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", disappear_during_unlink)

    replayed = bridge.persist_reconciliation_receipt(attempt, payload)

    assert triggered["done"] is True
    assert replayed["receipt_id"] == receipt["receipt_id"]
    assert final.read_bytes() == encoded
    assert final.stat().st_nlink == 1
    assert list(final.parent.glob(f".{final.name}.*.tmp")) == []


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("temp_content", ("exact", "mismatched"))
def test_fresh_reconciliation_recovers_only_exact_final_absent_receipt_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    temp_content: str,
) -> None:
    repo, predecessor, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    receipt = bridge.persist_reconciliation_receipt(
        attempt,
        {
            "stage": "blocked",
            "trigger": "test_prelink_process_crash",
            # Recovery must consume the persisted immutable bytes; it cannot
            # reconstruct a timestamp-varying payload in the next process.
            "reconciled_at": "2026-08-30T00:00:17+00:00",
            "reconciled": False,
            "blocked": True,
            "reason": "injected_after_temp_fsync_before_link",
        },
    )
    final = Path(str(receipt["receipt_path"]))
    encoded = final.read_bytes()
    temporary = final.parent / f".{final.name}.prelink-crash.tmp"
    with temporary.open("wb") as handle:
        handle.write(
            encoded
            if temp_content == "exact"
            else b"not the admitted immutable receipt\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    final.unlink()
    directory_fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    predecessor.close()

    successor = _database_portal_successor(repo)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("receipt crash recovery reached dispatch")

    monkeypatch.setattr(successor, "claim_next", forbidden)
    successor._provider_fn = forbidden
    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    try:
        result = successor.run_once()
        if temp_content == "exact":
            assert result["selection_idle_reason"] == (
                "database_portal_reconciliation_completed"
            ), result
            assert final.read_bytes() == encoded
            assert final.stat().st_nlink == 1
            assert not temporary.exists()
            current = successor.get_attempt(attempt.attempt_id)
            assert current is not None and current.status == "failed"
        else:
            reconciliation = result["database_portal_reconciliation"]
            assert reconciliation["blocked"] is True
            assert reconciliation["reason"] == (
                "database_portal_attempt_reconciliation_blocked"
            )
            current = successor.get_attempt(attempt.attempt_id)
            assert current is not None and current.status == "running"
            assert not final.exists()
            assert temporary.exists()
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_nested_state_replacement_before_fence_never_nominates_a_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor as supervisor_runtime

    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    original_verify = bridge._verify_nested_state_identity

    def replace_after_identity(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_verify(*args, **kwargs)
        replacement = PortalTaskState.load(paths.state)
        replacement.heartbeat_at = "replacement-between-validation-and-fence"
        assert replacement.save(paths.state) is True
        return result

    monkeypatch.setattr(
        bridge,
        "_verify_nested_state_identity",
        replace_after_identity,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "fence_ordinary_provider_runner",
        lambda *_args, **_kwargs: pytest.fail(
            "drifted nested state nominated a provider signal"
        ),
    )
    try:
        result = bridge.reconcile_quiesced_attempt(attempt)

        assert result["reconciled"] is False
        assert result["blocked"] is True
        assert result["reason"] == "nested_state_changed_before_provider_fence"
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_shutdown_preserves_exact_terminal_portal_success_for_phase_resume(
    tmp_path: Path,
) -> None:
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    successor: DatabaseImplementationDaemon | None = None
    try:
        text = paths.task_projection.read_text(encoding="utf-8")
        paths.task_projection.write_text(
            text.replace("- Status: ready", "- Status: completed"),
            encoding="utf-8",
        )
        task = parse_task_file(
            paths.task_projection,
            task_header_prefix="## PCTDD-001",
        )[0]
        paths.events.write_text(
            json.dumps(
                {
                    "type": "task_completed",
                    "task_id": task.task_id,
                    "canonical_task_key": task.canonical_task_key,
                    "canonical_task_cid": task.canonical_task_cid,
                    "board_namespace": task.board_namespace,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )

        assert result["reconciled"] is True, result
        item = result["attempts"][0]
        assert item["terminal_provider_evidence"] is True
        assert item["database_disposition"] == (
            "preserved_for_exact_phase_resume"
        )
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        assert current.committed_phase == "claimed"
        database_task = daemon.task_source.get_task(attempt.task_cid)
        assert database_task is not None
        assert database_task.status == "in_progress"
        indexed = daemon._require_connection().execute(
            """
            SELECT COUNT(*) FROM database_portal_terminal_reconciliations
            WHERE attempt_id = ?
            """,
            [attempt.attempt_id],
        ).fetchone()
        assert indexed is not None and indexed[0] == 0

        daemon.close()
        successor = _database_portal_successor(repo)
        restart = successor.run_once()
        assert restart["selection_idle_reason"] == (
            "database_portal_exact_phase_evidence_projected"
        )
        restart_item = restart["database_portal_reconciliation"]["attempts"][0]
        assert restart_item["database_disposition"] == (
            "preserved_for_exact_phase_resume"
        )
        resumed = successor.get_attempt(attempt.attempt_id)
        assert resumed is not None and resumed.status == "running"
        assert resumed.phase_committed("provider")
        invocation_count = successor._require_connection().execute(
            "SELECT COUNT(*) FROM provider_invocations WHERE attempt_id = ?",
            [attempt.attempt_id],
        ).fetchone()
        assert invocation_count is not None and int(invocation_count[0]) == 1

        # A fresh --once child now sees the projected durable provider phase
        # and continues from the later deterministic effect/validation path;
        # it does not preserve/recover the provider forever.
        successor.close()
        successor = _database_portal_successor(repo)
        completed = successor.run_once()
        assert completed["implementation_result"]["status"] == "succeeded"
        terminal = successor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "succeeded"
    finally:
        if successor is not None:
            successor.close()
        else:
            daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_shutdown_fences_exact_live_nested_ordinary_runner(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor import (
        _ordinary_provider_runner_observation,
        _process_start_ticks,
    )

    repo, daemon, _bridge, _attempt_record, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        # Popen returns after fork but the child can still be crossing exec.
        # Require two identical post-exec procfs observations so the stored
        # signal authority never captures the parent pytest argv transiently.
        previous: tuple[object, ...] | None = None
        deadline = time.monotonic() + 2.0
        while True:
            observed = _ordinary_provider_runner_observation(process.pid)
            if (
                observed[1] is not None
                and observed[2]
                and observed == previous
            ):
                boot_id, birth, argv_sha256 = observed
                break
            if time.monotonic() >= deadline:
                pytest.fail("ordinary provider child did not reach stable exec")
            previous = observed
            time.sleep(0.01)
        assert birth is not None
        parent_pid, process_group, session_id, start_ticks = birth
        owner_start_ticks = _process_start_ticks(os.getpid())
        assert isinstance(owner_start_ticks, int) and owner_start_ticks > 0
        state = _activate_nested_portal_state(repo, paths)
        receipt_body = {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "ordinary-provider-runner-birth@1"
            ),
            "task_id": state.active_task_id,
            "attempt": state.active_attempt,
            "task_revision_cid": state.active_task_cid,
            "workspace_path": state.active_worktree_path,
            "owner_pid": parent_pid,
            "owner_start_ticks": owner_start_ticks,
            "pid": process.pid,
            "start_time_ticks": start_ticks,
            "boot_id": boot_id,
            "process_group_id": process_group,
            "session_id": session_id,
            "argv_sha256": argv_sha256,
        }
        state.active_provider_runner = {
            **receipt_body,
            "receipt_id": content_identity(receipt_body),
        }
        assert state.save(paths.state) is True

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )

        assert result["reconciled"] is True, result
        item = result["attempts"][0]
        assert item["provider_runner_fence"]["safe_to_restart"] is True
        assert item["provider_runner_fence"]["fenced"] is True
        process.wait(timeout=3)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize(
    ("fence_case", "safe"),
    (("malformed", False), ("pid_reused", True), ("fence_failed", False)),
)
def test_database_shutdown_honors_nested_ordinary_fence_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fence_case: str,
    safe: bool,
) -> None:
    import ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor as supervisor_runtime

    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    state = _activate_nested_portal_state(repo, paths)
    if fence_case == "malformed":
        state.active_provider_runner = {
            "schema": "unknown-provider-runner-birth@99"
        }
    else:
        state.active_provider_runner = {"schema": "synthetic-for-fence-test"}
        outcome = {
            "applicable": True,
            "safe_to_restart": safe,
            "fenced": False,
            "pid": 998877,
            "reason": (
                "ordinary_provider_runner_recorded_birth_dead"
                if safe
                else "ordinary_provider_runner_exact_birth_fence_failed"
            ),
        }
        if safe:
            outcome["pid_reused"] = True
        monkeypatch.setattr(
            supervisor_runtime,
            "fence_ordinary_provider_runner",
            lambda _status, **_kwargs: dict(outcome),
        )
    assert state.save(paths.state) is True

    try:
        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )

        item = result["attempts"][0]
        assert item["provider_runner_fence"]["safe_to_restart"] is safe
        current = daemon.get_attempt(attempt.attempt_id)
        task = daemon.task_source.get_task(attempt.task_cid)
        if safe:
            assert result["reconciled"] is True
            assert current is not None and current.status == "failed"
            # A dead/reused runner proves current quiescence, not that an
            # entered predecessor never produced an external effect before
            # its dispatch journal was lost.  Preserve the stricter permanent
            # unknown-outcome boundary across process restart.
            assert task is not None and task.status == "blocked"
            assert item["database_disposition"] == (
                "blocked_unknown_outcome"
            )
            assert task.body["completion_receipt"]["reason"] == (
                "callback_authority_incomplete_blocked"
            )
            daemon.close()
            daemon = _database_portal_successor(repo)
            callbacks: list[str] = []

            def forbidden(*_args: object, **_kwargs: object) -> object:
                callbacks.append("callback")
                raise AssertionError("reused runner callback was retried")

            daemon._provider_fn = forbidden
            daemon._effect_fn = forbidden
            daemon._validation_fn = forbidden
            restarted = daemon.run_once()
            assert restarted["implementation_result"] is None
            count = daemon._require_connection().execute(
                "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
                [attempt.task_cid],
            ).fetchone()
            assert count is not None and int(count[0]) == 1
            assert callbacks == []
        else:
            assert result["reconciled"] is False
            assert result["blocked"] is True
            assert item["reason"] == "nested_provider_runner_fence_unproven"
            assert current is not None and current.status == "running"
            assert task is not None and task.status == "in_progress"
            nested = PortalTaskState.load(paths.state)
            assert nested.active_task_id == "PCTDD-001"
            assert nested.implementation_in_progress is True
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_startup_replays_blocked_nested_fence_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    _activate_nested_portal_state(repo, paths)
    predecessor.close()
    successor = _database_portal_successor(repo)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("blocked nested fence reached dispatch/rearm")

    monkeypatch.setattr(
        successor, "_resume_attempt_without_process_crash", forbidden
    )
    monkeypatch.setattr(
        successor, "reconcile_blocked_unknown_outcome_tasks", forbidden
    )
    monkeypatch.setattr(successor, "claim_next", forbidden)
    try:
        first = successor.run_once()
        second = successor.run_once()

        for result in (first, second):
            assert result["selection_idle_reason"] == (
                "database_portal_reconciliation_blocked"
            )
            assert result["implementation_result"] is None
            reconciliation = result["database_portal_reconciliation"]
            assert reconciliation["blocked"] is True
            assert reconciliation["attempts"][0]["reason"] == (
                "nested_active_provider_runner_fence_missing"
            )
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        state = PortalTaskState.load(paths.state)
        assert state.active_phase == "implementing"
        assert state.active_provider_runner == {}
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_database_startup_retries_transient_fence_then_reconciles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    _activate_nested_portal_state(
        repo,
        paths,
        active_provider_runner={"schema": "unknown-runner-birth@99"},
    )
    predecessor.close()
    successor = _database_portal_successor(repo)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("reconciliation pass reached dispatch/rearm")

    monkeypatch.setattr(
        successor, "_resume_attempt_without_process_crash", forbidden
    )
    monkeypatch.setattr(
        successor, "reconcile_blocked_unknown_outcome_tasks", forbidden
    )
    monkeypatch.setattr(successor, "claim_next", forbidden)
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_portal_reconciliation_blocked"
        )

        state = PortalTaskState.load(paths.state)
        state.active_phase = "validating"
        state.active_phase_detail = "provider completed before shutdown"
        state.active_provider_runner = {}
        assert state.save(paths.state) is True

        second = successor.run_once()
        assert second["selection_idle_reason"] == (
            "database_portal_reconciliation_completed"
        )
        reconciliation = second["database_portal_reconciliation"]
        assert reconciliation["blocked"] is False
        assert reconciliation["reconciled_attempt_count"] == 1
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "failed"
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_repeated_identical_blocked_reconciliation_is_content_idempotent(
    tmp_path: Path,
) -> None:
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    _activate_nested_portal_state(
        repo,
        paths,
        active_provider_runner={"schema": "unknown-runner-birth@99"},
    )
    try:
        receipt_ids: set[str] = set()
        for _index in range(140):
            result = daemon.reconcile_quiesced_database_portal_attempts(
                trigger="supervisor_signal_shutdown",
                force=True,
            )
            assert result["blocked"] is True
            receipt_ids.add(
                str(result["attempts"][0]["reconciliation_receipt_id"])
            )
        assert len(receipt_ids) == 1
        first_receipt_id = next(iter(receipt_ids))
        first_receipt_path = Path(paths.reconciliation) / (
            first_receipt_id.removeprefix("sha256:") + ".json"
        )
        assert first_receipt_path.is_file()

        # A genuinely different blocked observation remains distinct rather
        # than being folded into the earlier immutable evidence.
        state = PortalTaskState.load(paths.state)
        state.active_provider_runner = {}
        assert state.save(paths.state) is True
        distinct = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
        assert distinct["blocked"] is True
        distinct_receipt_id = str(
            distinct["attempts"][0]["reconciliation_receipt_id"]
        )
        assert distinct_receipt_id != first_receipt_id
        assert first_receipt_path.is_file()

        state = PortalTaskState.load(paths.state)
        state.active_phase = "validating"
        state.active_phase_detail = "provider completed before shutdown"
        state.active_provider_runner = {}
        assert state.save(paths.state) is True
        recovered = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
        assert recovered["blocked"] is False
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        assert first_receipt_path.is_file()
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_active_sealed_runner_schema_delegates_to_portal_authority(
    tmp_path: Path,
) -> None:
    repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    _activate_nested_portal_state(
        repo,
        paths,
        active_provider_runner={
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "provider-runner-birth@1"
            )
        },
    )
    portal_calls: list[str] = []

    class ExistingSealedAuthority:
        def reconcile_quiesced_active_attempt(self) -> dict[str, object]:
            portal_calls.append("reconcile")
            return {
                "reconciled": True,
                "blocked": False,
                "reason": "existing_sealed_authority_reconciled",
            }

        def close_event_runtime(self) -> None:
            return None

    bridge.portal_factory = lambda _paths, _alias: ExistingSealedAuthority()
    try:
        result = bridge.reconcile_quiesced_attempt(attempt)

        assert result["reconciled"] is True
        assert result["provider_runner_reconciliation_authority"] == (
            "delegated_to_portal_sealed_authority"
        )
        assert result["provider_runner_fence"]["fenced"] is False
        assert portal_calls == ["reconcile"]
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize(
    "fault_boundary",
    ("after_task_cas", "after_claim_release", "after_local_terminal"),
)
@pytest.mark.parametrize("lease_elapsed", (False, True))
def test_terminal_reconciliation_post_cas_replay_converges_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_boundary: str,
    lease_elapsed: bool,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    if fault_boundary == "after_task_cas":
        original_release = predecessor.coordinator.release

        def fail_release(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("injected after task CAS")

        monkeypatch.setattr(predecessor.coordinator, "release", fail_release)
    elif fault_boundary == "after_claim_release":
        original_commit = predecessor.commit_phase

        def fail_failed_phase(
            selected: DatabaseTaskAttempt,
            phase: str,
            **kwargs: object,
        ) -> DatabaseTaskAttempt:
            if phase == "failed":
                raise RuntimeError("injected after claim release")
            return original_commit(selected, phase, **kwargs)

        monkeypatch.setattr(predecessor, "commit_phase", fail_failed_phase)
    else:
        original_persist = predecessor._database_portal_bridge.persist_reconciliation_receipt

        def fail_terminal_receipt(
            selected: DatabaseTaskAttempt,
            payload: object,
        ) -> dict[str, object]:
            assert isinstance(payload, dict)
            if payload.get("stage") == "terminal":
                raise RuntimeError("injected after local terminal")
            return original_persist(selected, payload)

        monkeypatch.setattr(
            predecessor._database_portal_bridge,
            "persist_reconciliation_receipt",
            fail_terminal_receipt,
        )
    with pytest.raises(RuntimeError, match="injected after"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
    task = predecessor.task_source.get_task(attempt.task_cid)
    assert task is not None and task.status in {"retrying", "blocked"}
    link = task.body["completion_receipt"]["terminal_reconciliation"]
    assert link["commit_barrier_receipt_id"]
    if fault_boundary == "after_task_cas":
        monkeypatch.setattr(predecessor.coordinator, "release", original_release)
    predecessor.close()

    successor = _database_portal_successor(repo)
    if lease_elapsed:
        monkeypatch.setattr(successor, "_now_ms", lambda: 10**15)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-CAS replay reached a callback or dispatch")

    monkeypatch.setattr(successor, "claim_next", forbidden)
    monkeypatch.setattr(successor, "reconcile_blocked_unknown_outcome_tasks", forbidden)
    try:
        result = successor.run_once()
        assert result["selection_idle_reason"] == (
            "database_portal_reconciliation_completed"
        ), result
        reconciliation = result["database_portal_reconciliation"]
        assert reconciliation["reason"] == (
            "database_portal_post_cas_transitions_reconciled"
        )
        assert reconciliation["continuation_required"] is True
        expected_reason = (
            "terminal_reconciliation_receipt_repaired"
            if fault_boundary == "after_local_terminal"
            else "terminal_reconciliation_saga_replayed"
        )
        assert reconciliation["attempts"][0]["reason"] == expected_reason
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "failed"
        assert successor.list_running_attempts() == []
        if fault_boundary == "after_local_terminal":
            # The receipt/index repair is the only action in this pass.  A
            # later pass may reassess ordinary work, but it cannot be folded
            # into the same durable transition.
            reassessed: list[str] = []

            def observe_rearms() -> list[dict[str, object]]:
                reassessed.append("rearm")
                return []

            def observe_claim() -> None:
                reassessed.append("claim")
                return None

            monkeypatch.setattr(
                successor,
                "reconcile_blocked_unknown_outcome_tasks",
                observe_rearms,
            )
            monkeypatch.setattr(successor, "claim_next", observe_claim)
            successor.run_once()
            assert reassessed == ["rearm", "claim"]
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_post_cas_failure_exact_repair_is_not_hidden_by_older_global_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    original_release = predecessor.coordinator.release

    def fail_after_task_cas(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected after task CAS before exact repair")

    monkeypatch.setattr(
        predecessor.coordinator,
        "release",
        fail_after_task_cas,
    )
    with pytest.raises(RuntimeError, match="before exact repair"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
    monkeypatch.setattr(predecessor.coordinator, "release", original_release)
    historical_receipts = _seed_terminal_repair_history(
        predecessor,
        count=101,
    )
    predecessor.close()
    historical_authority = _TerminalRepairReceiptAuthority(
        historical_receipts
    )

    def patch_historical_receipts(
        daemon: DatabaseImplementationDaemon,
    ) -> None:
        bridge = daemon._database_portal_bridge
        assert bridge is not None
        original_load = bridge.load_reconciliation_receipt

        def load(
            selected: DatabaseTaskAttempt,
            receipt_id: str,
            *,
            required_stage: str = "",
        ) -> dict[str, object]:
            if selected.attempt_id in historical_receipts:
                return historical_authority.load_reconciliation_receipt(
                    selected,
                    receipt_id,
                    required_stage=required_stage,
                )
            return original_load(
                selected,
                receipt_id,
                required_stage=required_stage,
            )

        monkeypatch.setattr(bridge, "load_reconciliation_receipt", load)

    first = _database_portal_successor(repo)
    patch_historical_receipts(first)
    try:
        first_page = first.run_once()
        first_reconciliation = first_page["database_portal_reconciliation"]
        assert first_reconciliation["repair_batch_pending"] is True
        assert first_reconciliation["blocked"] is False
        exact = [
            item
            for item in first_reconciliation["attempts"]
            if item.get("attempt_id") == attempt.attempt_id
        ]
        assert len(exact) == 1
        assert exact[0]["reason"] == "terminal_reconciliation_saga_replayed"
        current = first.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "failed"
        saga = first._database_portal_terminal_reconciliation_saga(current)
        assert saga is not None and saga["stage"] == "terminal"
    finally:
        first.close()

    second = _database_portal_successor(repo)
    patch_historical_receipts(second)
    try:
        tail = second._repair_database_portal_terminal_receipts(
            bridge=second._database_portal_bridge,
            trigger="fresh_tail_audit",
        )
        historical_tail = [
            item
            for item in tail
            if item.get("reason")
            == "terminal_reconciliation_receipt_verified"
            and item.get("attempt_id") in historical_receipts
        ]
        assert len(historical_tail) == 1
        assert historical_tail[0]["attempt_id"] == (
            "attempt:terminal-history:0100"
        )
    finally:
        second.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_corrupt_terminal_audit_blocks_before_post_cas_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    original_release = predecessor.coordinator.release

    def fail_after_task_cas(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected post-CAS running projection")

    monkeypatch.setattr(
        predecessor.coordinator,
        "release",
        fail_after_task_cas,
    )
    with pytest.raises(RuntimeError, match="post-CAS running projection"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
    monkeypatch.setattr(predecessor.coordinator, "release", original_release)
    historical_receipts = _seed_terminal_repair_history(
        predecessor,
        count=1,
    )
    historical_attempt_id = "attempt:terminal-history:0000"
    historical_receipts[historical_attempt_id]["terminal"][
        "database_disposition"
    ] = "tampered-disposition"
    predecessor.close()

    successor = _database_portal_successor(repo)
    bridge = successor._database_portal_bridge
    assert bridge is not None
    original_load = bridge.load_reconciliation_receipt
    authority = _TerminalRepairReceiptAuthority(historical_receipts)

    def load(
        selected: DatabaseTaskAttempt,
        receipt_id: str,
        *,
        required_stage: str = "",
    ) -> dict[str, object]:
        if selected.attempt_id == historical_attempt_id:
            return authority.load_reconciliation_receipt(
                selected,
                receipt_id,
                required_stage=required_stage,
            )
        return original_load(
            selected,
            receipt_id,
            required_stage=required_stage,
        )

    monkeypatch.setattr(bridge, "load_reconciliation_receipt", load)
    try:
        result = successor.run_once()
        assert result["selection_idle_reason"] == (
            "database_portal_reconciliation_blocked"
        )
        reconciliation = result["database_portal_reconciliation"]
        assert reconciliation["reason"] == (
            "database_portal_terminal_repair_batch_blocked"
        )
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
        saga = successor._database_portal_terminal_reconciliation_saga(current)
        assert saga is not None and saga["stage"] == "commit_barrier"
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("canonical_change", ("same", "replacement", "deleted"))
def test_pre_cas_terminal_saga_replays_without_regeneration_or_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_change: str,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    predecessor._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
        outcome="deferred",
    )

    def fail_after_saga(*_args: object, **_kwargs: object) -> object:
        saga = predecessor._database_portal_terminal_reconciliation_saga(
            attempt
        )
        assert saga is not None and saga["stage"] == "commit_barrier"
        raise RuntimeError("injected after terminal saga barrier")

    monkeypatch.setattr(predecessor, "_finalize_failed_attempt", fail_after_saga)
    with pytest.raises(RuntimeError, match="terminal saga barrier"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
    predecessor.close()

    successor = _database_portal_successor(repo)
    canonical_get = successor.task_source.get
    if canonical_change == "replacement":
        successor.materialize_population(
            {
                "repository_tree_id": "tree:pre-cas-saga-replacement",
                "tasks": [
                    {
                        "task_cid": attempt.task_cid,
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "title": "Replacement after terminal saga barrier",
                        "status": "ready",
                        "validation_commands": [
                            "python -m pytest replacement.py"
                        ],
                    }
                ],
            }
        )
    before = canonical_get(attempt.task_cid)
    before_bytes = (
        json.dumps(before.to_dict(), sort_keys=True)
        if before is not None
        else ""
    )
    if canonical_change == "deleted":
        monkeypatch.setattr(successor.task_source, "get", lambda _cid: None)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("pre-CAS saga replay reached dispatch")

    monkeypatch.setattr(successor, "claim_next", forbidden)
    try:
        result = successor.run_once()
        assert result["selection_idle_reason"] == (
            "database_portal_reconciliation_completed"
        )
        item = result["database_portal_reconciliation"]["attempts"][0]
        assert item["reason"] == "terminal_reconciliation_saga_replayed"
        expected_disposition = (
            "blocked_unknown_outcome"
            if canonical_change == "same"
            else "superseded_attempt_revoked"
        )
        assert item["database_disposition"] == expected_disposition
        terminal = successor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        if canonical_change == "replacement":
            after = canonical_get(attempt.task_cid)
            assert after is not None
            assert json.dumps(after.to_dict(), sort_keys=True) == before_bytes
    finally:
        successor.close()

    if canonical_change == "same":
        callbacks: list[str] = []

        def forbidden_callback(*_args: object, **_kwargs: object) -> object:
            callbacks.append("callback")
            raise AssertionError("blocked terminal saga callback was retried")

        successor = _database_portal_successor(repo)
        successor._provider_fn = forbidden_callback
        successor._effect_fn = forbidden_callback
        successor._validation_fn = forbidden_callback
        try:
            for _pass in range(2):
                restarted = successor.run_once()
                assert restarted["implementation_result"] is None
            task = successor.task_source.get_task(attempt.task_cid)
            assert task is not None and task.status == "blocked"
            assert task.body["completion_receipt"]["reason"] == (
                "callback_authority_incomplete_blocked"
            )
            count = successor._require_connection().execute(
                "SELECT COUNT(*) FROM database_task_attempts "
                "WHERE task_cid = ?",
                [attempt.task_cid],
            ).fetchone()
            assert count is not None and int(count[0]) == 1
            assert callbacks == []
        finally:
            successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_superseded_pre_cas_saga_blocks_if_exact_old_epoch_reappears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    original = predecessor.task_source.get(attempt.task_cid)
    assert original is not None
    _attempt_paths, binding = bridge._ensure_attempt_projection(
        attempt,
        original,
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    predecessor._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
        outcome="deferred",
    )
    predecessor._record_database_portal_attempt_binding(
        attempt,
        binding,
        "portal_entered",
    )
    predecessor.materialize_population(
        {
            "repository_tree_id": "tree:superseded-saga",
            "tasks": [
                {
                    "task_cid": attempt.task_cid,
                    "task_id": "PCTDD-001",
                    "goal_cid": "goal:pctdd",
                    "title": "Replacement before saga barrier",
                    "status": "ready",
                    "validation_commands": ["python -m pytest replacement.py"],
                }
            ],
        }
    )

    def fail_after_saga(*_args: object, **_kwargs: object) -> object:
        saga = predecessor._database_portal_terminal_reconciliation_saga(
            attempt
        )
        assert saga is not None
        assert saga["intended_database_disposition"] == (
            "superseded_attempt_revoked"
        )
        raise RuntimeError("injected after superseded saga barrier")

    monkeypatch.setattr(predecessor, "_finalize_failed_attempt", fail_after_saga)
    with pytest.raises(RuntimeError, match="superseded saga barrier"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup",
            force=True,
        )
    predecessor.close()

    successor = _database_portal_successor(repo)
    monkeypatch.setattr(
        successor.task_source,
        "get",
        lambda _task_cid: original,
    )
    try:
        first = successor.run_once()
        second = successor.run_once()
        for result in (first, second):
            assert result["selection_idle_reason"] == (
                "database_portal_reconciliation_blocked"
            )
            item = result["database_portal_reconciliation"]["attempts"][0]
            assert item["reason"] == (
                "database_portal_pre_cas_saga_replay_invalid"
            )
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("lease_elapsed", (False, True))
def test_completed_task_post_cas_replay_precedes_portal_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lease_elapsed: bool,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = predecessor.commit_phase(attempt, "context", body={})
    current = predecessor.commit_phase(
        current,
        "provider",
        body={"idempotency_key": f"provider:{attempt.attempt_id}"},
    )
    current = predecessor.commit_phase(
        current,
        "effect",
        body={"idempotency_key": f"effect:{attempt.attempt_id}"},
    )
    validation = {
        "outcome": "passed",
        "evidence_digest": "sha256:" + "ab" * 32,
        "argv": ["python", "-m", "pytest", "focused.py"],
    }
    current = predecessor.commit_phase(current, "validation", body=validation)
    original_complete_claim = predecessor.coordinator.complete_task_claim

    def fail_after_control_cas(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected after completed task CAS")

    monkeypatch.setattr(
        predecessor.coordinator,
        "complete_task_claim",
        fail_after_control_cas,
    )
    with pytest.raises(RuntimeError, match="completed task CAS"):
        predecessor.complete_attempt(current, validation_result=validation)
    task = predecessor.task_source.get_task(attempt.task_cid)
    assert task is not None and task.status == "completed"
    monkeypatch.setattr(
        predecessor.coordinator,
        "complete_task_claim",
        original_complete_claim,
    )
    predecessor.close()

    successor = _database_portal_successor(repo)
    if lease_elapsed:
        monkeypatch.setattr(successor, "_now_ms", lambda: 10**15)
    forbidden_calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        forbidden_calls.append("callback")
        raise AssertionError("completed post-CAS replay reached Portal")

    successor._provider_fn = forbidden
    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    monkeypatch.setattr(
        successor,
        "reconcile_prepared_task_completions",
        lambda: pytest.fail(
            "exact post-CAS success replay scanned the bounded global page"
        ),
    )
    try:
        result = successor.run_once()
        assert result["selection_idle_reason"] == (
            "database_portal_reconciliation_completed"
        )
        item = result["database_portal_reconciliation"]["attempts"][0]
        assert item["database_disposition"] == "completed_post_cas_replay"
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "succeeded"
        assert forbidden_calls == []
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_completed_task_post_cas_replay_rejects_tampered_validation_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = predecessor.commit_phase(attempt, "context", body={})
    current = predecessor.commit_phase(
        current,
        "provider",
        body={"idempotency_key": f"provider:{attempt.attempt_id}"},
    )
    current = predecessor.commit_phase(
        current,
        "effect",
        body={"idempotency_key": f"effect:{attempt.attempt_id}"},
    )
    validation = {
        "outcome": "passed",
        "evidence_digest": "sha256:" + "cd" * 32,
        "argv": ["python", "-m", "pytest", "focused.py"],
    }
    current = predecessor.commit_phase(current, "validation", body=validation)

    def fail_after_control_cas(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected after completed task CAS")

    monkeypatch.setattr(
        predecessor.coordinator,
        "complete_task_claim",
        fail_after_control_cas,
    )
    with pytest.raises(RuntimeError, match="completed task CAS"):
        predecessor.complete_attempt(current, validation_result=validation)
    predecessor._require_connection().execute(
        """
        UPDATE attempt_phases SET body_json = ?
        WHERE attempt_id = ? AND phase = 'validation'
        """,
        [
            json.dumps(
                {
                    **validation,
                    "evidence_digest": "sha256:" + "ef" * 32,
                },
                sort_keys=True,
            ),
            attempt.attempt_id,
        ],
    )
    predecessor.close()

    successor = _database_portal_successor(repo)
    try:
        first = successor.run_once()
        second = successor.run_once()
        for result in (first, second):
            assert result["selection_idle_reason"] == (
                "database_portal_reconciliation_blocked"
            )
            item = result["database_portal_reconciliation"]["attempts"][0]
            assert item["reason"] == "database_portal_post_cas_replay_invalid"
            assert "validation authority" in item["error"]
        durable = successor.get_attempt(attempt.attempt_id)
        assert durable is not None and durable.status == "running"
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_post_cas_replay_does_not_bypass_second_running_nested_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, bridge, first_attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        first_attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{first_attempt.attempt_id}",
    )

    def fail_release(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected after task CAS")

    original_release = predecessor.coordinator.release
    monkeypatch.setattr(predecessor.coordinator, "release", fail_release)
    with pytest.raises(RuntimeError, match="task CAS"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
    monkeypatch.setattr(predecessor.coordinator, "release", original_release)
    predecessor.materialize_population(
        {
            "repository_tree_id": "tree:second-running-attempt",
            "tasks": [
                {
                    "task_cid": "task:cid:pctdd-002",
                    "task_id": "PCTDD-002",
                    "goal_cid": "goal:pctdd",
                    "status": "ready",
                    "validation_commands": ["python -m pytest second.py"],
                }
            ],
        }
    )
    second_attempt = predecessor.claim_next()
    assert second_attempt is not None
    assert second_attempt.attempt_id != first_attempt.attempt_id
    predecessor._begin_callback_dispatch(
        second_attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{second_attempt.attempt_id}",
    )
    record = predecessor.task_source.get_task(second_attempt.task_cid)
    assert record is not None
    second_paths, _binding = bridge._ensure_attempt_projection(
        second_attempt,
        record,
    )
    second_task = parse_task_file(
        second_paths.task_projection,
        task_header_prefix="## PCTDD-002",
    )[0]
    nested = PortalTaskState(
        task_statuses={second_task.task_id: "in_progress"},
        task_identities={
            second_task.task_id: {
                "canonical_task_key": second_task.canonical_task_key,
                "canonical_task_cid": second_task.canonical_task_cid,
                "board_namespace": second_task.board_namespace,
            }
        },
        implementation_attempts={second_task.task_id: 1},
        implementation_attempts_by_cid={second_task.canonical_task_cid: 1},
    )
    nested.active_task_id = second_task.task_id
    nested.active_task_key = second_task.canonical_task_key
    nested.active_task_cid = second_task.canonical_task_cid
    nested.active_attempt = 1
    nested.active_phase = "implementing"
    nested.implementation_in_progress = True
    nested.active_provider_runner = {}
    assert nested.save(second_paths.state) is True
    predecessor.close()

    successor = _database_portal_successor(repo)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("blocked multi-attempt gate reached dispatch")

    monkeypatch.setattr(successor, "claim_next", forbidden)
    monkeypatch.setattr(successor, "reconcile_blocked_unknown_outcome_tasks", forbidden)
    try:
        first = successor.run_once()
        second = successor.run_once()
        for result in (first, second):
            assert result["selection_idle_reason"] == (
                "database_portal_reconciliation_blocked"
            )
            reconciliation = result["database_portal_reconciliation"]
            assert reconciliation["blocked"] is True
            assert any(
                item.get("attempt_id") == second_attempt.attempt_id
                and item.get("reason") == "nested_active_provider_runner_fence_missing"
                for item in reconciliation["attempts"]
            )
        terminal = successor.get_attempt(first_attempt.attempt_id)
        blocked_nested = successor.get_attempt(second_attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        assert blocked_nested is not None and blocked_nested.status == "running"
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("dispatch_kind", ("provider", "effect"))
def test_durable_raised_callback_outcome_remains_unknown_and_blocked(
    tmp_path: Path,
    dispatch_kind: str,
) -> None:
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    try:
        current = attempt
        if dispatch_kind == "effect":
            current = daemon.commit_phase(current, "context", body={})
            daemon._begin_callback_dispatch(
                current,
                dispatch_kind="provider",
                idempotency_key=f"provider:{attempt.attempt_id}",
            )
            daemon._record_database_portal_attempt_binding(
                current,
                json.loads(paths.binding.read_text(encoding="utf-8")),
                "portal_entered",
            )
            daemon._record_callback_dispatch_outcome(
                current,
                dispatch_kind="provider",
                idempotency_key=f"provider:{attempt.attempt_id}",
                outcome="returned",
                body={"status": "accepted"},
            )
            current = daemon.commit_phase(
                current,
                "provider",
                body={"idempotency_key": f"provider:{attempt.attempt_id}"},
            )
        daemon._begin_callback_dispatch(
            current,
            dispatch_kind=dispatch_kind,
            idempotency_key=f"{dispatch_kind}:{attempt.attempt_id}",
        )
        daemon._record_callback_dispatch_outcome(
            current,
            dispatch_kind=dispatch_kind,
            idempotency_key=f"{dispatch_kind}:{attempt.attempt_id}",
            outcome="raised",
            body={"exception_type": "RuntimeError"},
        )

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup",
            force=True,
        )

        item = result["attempts"][0]
        assert item["database_disposition"] == "blocked_unknown_outcome"
        assert item["retry_receipt"]["forced_block"] is True
        assert item["retry_receipt"]["authority_outcome"] == "unknown"
        assert item["retry_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        daemon.close()
        daemon = _database_portal_successor(repo)
        callback_attempts: list[str] = []

        def forbidden(*_args: object, **_kwargs: object) -> object:
            callback_attempts.append("callback")
            raise AssertionError("raised callback outcome was redispatched")

        daemon._provider_fn = forbidden
        daemon._effect_fn = forbidden
        daemon._validation_fn = forbidden
        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_attempts == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_exact_admitted_binding_only_is_repaired_before_portal_construction(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    try:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        binding = json.loads(paths.binding.read_text(encoding="utf-8"))
        daemon._record_database_portal_attempt_binding(attempt, binding)
        paths.task_projection.unlink()

        result = bridge.reconcile_quiesced_attempt(attempt)

        assert result["reconciled"] is True
        assert result["blocked"] is False
        assert paths.task_projection.is_file()
        assert result["binding_id"] == binding["binding_id"]
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize(
    ("failure_point", "expected_stage"),
    (
        ("mkdir", "prepared"),
        ("root_fsync", "prepared"),
        ("parent_fsync", "prepared"),
        ("binding_write", "prepared"),
        ("projection_write", "prepared"),
        ("published_transition", "prepared"),
        ("portal_entered_transition", "published"),
    ),
)
def test_preentry_publication_fault_defers_then_repairs_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    expected_stage: str,
) -> None:
    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    paths.binding.unlink()
    paths.task_projection.unlink()
    paths.root.rmdir()
    factory_calls: list[str] = []

    def deferred_factory(_paths: object, _alias: str) -> object:
        factory_calls.append("constructed")
        raise DatabasePortalBridgeDeferred("controlled typed deferral")

    bridge.portal_factory = deferred_factory
    original_write = database_portal_bridge_module._atomic_write
    original_fsync_directory = database_portal_bridge_module._fsync_directory
    original_recorder = bridge._binding_recorder
    assert original_recorder is not None

    with monkeypatch.context() as fault:
        if failure_point == "mkdir":
            fault.setattr(
                database_portal_bridge_module,
                "_ensure_durable_directory",
                lambda _path: (_ for _ in ()).throw(
                    OSError("injected attempt directory publication failure")
                ),
            )
        elif failure_point in {"root_fsync", "parent_fsync"}:
            def fail_parent_fsync(path: Path) -> None:
                failed_path = (
                    paths.root
                    if failure_point == "root_fsync"
                    else paths.root.parent
                )
                if path == failed_path:
                    raise OSError("injected attempt directory fsync failure")
                original_fsync_directory(path)

            fault.setattr(
                database_portal_bridge_module,
                "_fsync_directory",
                fail_parent_fsync,
            )
        elif failure_point in {"binding_write", "projection_write"}:
            failed_name = (
                paths.binding.name
                if failure_point == "binding_write"
                else paths.task_projection.name
            )

            def fail_selected_write(path: Path, payload: bytes) -> None:
                if path.name == failed_name:
                    raise OSError("injected Portal artifact write failure")
                original_write(path, payload)

            fault.setattr(
                database_portal_bridge_module,
                "_atomic_write",
                fail_selected_write,
            )
        else:
            failed_stage = (
                "published"
                if failure_point == "published_transition"
                else "portal_entered"
            )

            def fail_selected_transition(
                current_attempt: object,
                binding: object,
                stage: str,
            ) -> None:
                if stage == failed_stage:
                    raise OSError("injected Portal admission transition failure")
                original_recorder(current_attempt, binding, stage)

            fault.setattr(bridge, "_binding_recorder", fail_selected_transition)

        first = daemon._resume_attempt_without_process_crash(attempt)

    assert first["deferred"] is True
    assert first["status"] == "running"
    assert factory_calls == []
    current = daemon.get_attempt(attempt.attempt_id)
    assert current is not None and current.status == "running"
    admission = daemon._database_portal_attempt_binding(current)
    assert admission is not None
    assert admission["stage"] == expected_stage
    dispatch = daemon._dispatch_journal_entry(
        current,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    assert dispatch is not None and dispatch["outcome"] == "deferred"

    second = daemon._resume_attempt_without_process_crash(current)
    assert second["deferred"] is True
    assert second["status"] == "running"
    assert factory_calls == ["constructed"]
    repaired = daemon._database_portal_attempt_binding(current)
    assert repaired is not None and repaired["stage"] == "portal_entered"
    daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_preentry_integrity_failure_never_replays_as_transient_deferral(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    binding = json.loads(paths.binding.read_text(encoding="utf-8"))
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    daemon._record_database_portal_attempt_binding(attempt, binding, "prepared")
    daemon._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
        outcome="deferred",
        body={"preentry_publication_retry_count": 0},
    )
    paths.task_projection.write_text(
        paths.task_projection.read_text(encoding="utf-8")
        + "\nTampered projection content.\n",
        encoding="utf-8",
    )
    factory_calls: list[str] = []
    bridge.portal_factory = lambda *_args: factory_calls.append("constructed")

    first = daemon._resume_attempt_without_process_crash(attempt)
    dispatch = daemon._dispatch_journal_entry(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    assert first.get("deferred") is not True
    assert dispatch is not None and dispatch["outcome"] == "raised"

    second = daemon.run_once()
    assert second.get("implementation_result", {}).get("deferred") is not True
    terminal = daemon.get_attempt(attempt.attempt_id)
    assert terminal is not None and terminal.status == "failed"
    assert factory_calls == []
    daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_repeated_preentry_publication_fault_exhausts_bounded_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    paths.binding.unlink()
    paths.task_projection.unlink()
    paths.root.rmdir()
    factory_calls: list[str] = []
    bridge.portal_factory = lambda *_args: factory_calls.append("constructed")

    def always_fail_binding(path: Path, _payload: bytes) -> None:
        if path.name == paths.binding.name:
            raise OSError("persistent binding publication failure")
        raise AssertionError("projection write preceded failed binding")

    monkeypatch.setattr(
        database_portal_bridge_module,
        "_atomic_write",
        always_fail_binding,
    )
    current = attempt
    for retry_index in range(3):
        result = daemon._resume_attempt_without_process_crash(current)
        assert result["deferred"] is True, retry_index
        current = daemon.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"

    exhausted = daemon._resume_attempt_without_process_crash(current)
    assert exhausted["status"] == "failed"
    assert exhausted["retry_budget"].get("forced_block", False) is False
    dispatch = daemon._dispatch_journal_entry(
        current,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    assert dispatch is not None and dispatch["outcome"] == "preentry_failed"
    terminal = daemon.get_attempt(attempt.attempt_id)
    assert terminal is not None and terminal.status == "failed"
    assert factory_calls == []
    daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_portal_entered_missing_artifact_blocks_without_factory(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    binding = json.loads(paths.binding.read_text(encoding="utf-8"))
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    for stage in ("prepared", "published", "portal_entered"):
        daemon._record_database_portal_attempt_binding(attempt, binding, stage)
    paths.task_projection.unlink()
    factory_calls: list[str] = []
    bridge.portal_factory = lambda *_args: factory_calls.append("constructed")

    result = daemon.reconcile_quiesced_database_portal_attempts(
        trigger="database_daemon_startup",
        force=True,
    )

    assert result["blocked"] is True
    assert result["attempts"][0]["reason"] == (
        "database_portal_nested_reconciliation_failed"
    )
    assert "partial binding artifacts" in result["attempts"][0]["error"]
    assert factory_calls == []
    current = daemon.get_attempt(attempt.attempt_id)
    assert current is not None and current.status == "running"
    daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_same_stage_binding_admission_rejects_rehashed_content_change(
    tmp_path: Path,
) -> None:
    _repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    binding = json.loads(paths.binding.read_text(encoding="utf-8"))
    daemon._record_database_portal_attempt_binding(
        attempt,
        binding,
        "portal_entered",
    )
    changed = dict(binding)
    changed["goal_cid"] = "goal:changed"
    unsigned = dict(changed)
    unsigned.pop("binding_id")
    changed["binding_id"] = "sha256:" + hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        DatabaseImplementationConflictError,
        match="changed exact content identity",
    ):
        daemon._record_database_portal_attempt_binding(
            attempt,
            changed,
            "portal_entered",
        )
    daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_admitted_preportal_absence_revokes_replaced_task_as_superseded(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    try:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        binding = json.loads(paths.binding.read_text(encoding="utf-8"))
        daemon._record_database_portal_attempt_binding(attempt, binding)
        paths.binding.unlink()
        paths.task_projection.unlink()
        daemon.materialize_population(
            {
                "repository_tree_id": "tree:replacement-before-publish",
                "tasks": [
                    {
                        "task_cid": attempt.task_cid,
                        "task_id": "PCTDD-001",
                        "goal_cid": "goal:pctdd",
                        "title": "Replacement before Portal publication",
                        "status": "ready",
                        "validation_commands": ["python -m pytest replacement.py"],
                    }
                ],
            }
        )
        successor_before = daemon.task_source.get_task(attempt.task_cid)
        assert successor_before is not None
        successor_bytes = json.dumps(successor_before.to_dict(), sort_keys=True)

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup",
            force=True,
        )

        item = result["attempts"][0]
        assert item["historical_binding"] is True
        assert item["database_disposition"] == "superseded_attempt_revoked"
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        successor_after = daemon.task_source.get_task(attempt.task_cid)
        assert successor_after is not None
        assert json.dumps(successor_after.to_dict(), sort_keys=True) == successor_bytes
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_admitted_preportal_temp_is_removed_but_started_dispatch_stays_blocked(
    tmp_path: Path,
) -> None:
    repo, daemon, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    try:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        binding = json.loads(paths.binding.read_text(encoding="utf-8"))
        daemon._record_database_portal_attempt_binding(attempt, binding)
        paths.binding.unlink()
        paths.task_projection.unlink()
        temporary = paths.root / ".task-projection.md.exact-crash.tmp"
        temporary.write_text("incomplete publication", encoding="utf-8")

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup",
            force=True,
        )

        item = result["attempts"][0]
        assert item["reason"] == "admitted_preportal_artifacts_absent"
        assert item["database_disposition"] == "blocked_unknown_outcome"
        assert temporary.exists() is False
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        daemon.close()
        daemon = _database_portal_successor(repo)
        callback_attempts: list[str] = []

        def forbidden(*_args: object, **_kwargs: object) -> object:
            callback_attempts.append("callback")
            raise AssertionError("started preportal dispatch was rearmed")

        daemon._provider_fn = forbidden
        daemon._effect_fn = forbidden
        daemon._validation_fn = forbidden
        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_attempts == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_deleted_task_uses_exact_admitted_historical_binding_for_revocation(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    try:
        record = daemon.task_source.get_task(attempt.task_cid)
        assert record is not None
        _paths, binding = bridge._ensure_attempt_projection(attempt, record)
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_database_portal_attempt_binding(
            attempt,
            binding,
            "portal_entered",
        )
        with daemon.task_source.intent._connection(write=True) as connection:
            connection.execute(
                "DELETE FROM tasks WHERE task_cid = ?",
                [attempt.task_cid],
            )

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup",
            force=True,
        )

        item = result["attempts"][0]
        assert item["historical_binding"] is True
        assert item["database_disposition"] == "superseded_attempt_revoked"
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        assert daemon.task_source.get_task(attempt.task_cid) is None
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_live_portal_raised_after_side_effect_is_not_dispatched_twice(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(
            tmp_path,
            seed_nested_state=False,
        )
    )
    calls: list[str] = []

    class EffectfulBlockedPortal:
        def run_once(self) -> dict[str, object]:
            calls.append("external-effect")
            return {
                "blocked": True,
                "reason": "effectful_provider_returned_unaccepted_state",
            }

        def close_event_runtime(self) -> None:
            return None

    bridge.portal_factory = lambda _paths, _alias: EffectfulBlockedPortal()
    try:
        first = daemon.run_once()
        assert first["implementation_result"][
            "provider_reconciliation_pending"
        ] is True
        assert calls == ["external-effect"]

        second = daemon.run_once()
        assert second["implementation_result"]["retry_exhausted"] is True
        assert calls == ["external-effect"]
        terminal = daemon.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_durable_effect_claim_resumes_missing_phase_without_reapplying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = predecessor.commit_phase(attempt, "context", body={})
    provider_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    current, provider_result, _duplicate = predecessor.run_provider(
        current,
        provider_fn=provider,
    )
    effect_calls: list[str] = []

    def effect(
        _attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        effect_calls.append("effect")
        return {"status": "applied", "effect_key": "exact-effect"}

    original_commit = predecessor.commit_phase

    def fail_effect_phase(
        selected: DatabaseTaskAttempt,
        phase: str,
        **kwargs: object,
    ) -> DatabaseTaskAttempt:
        if phase == "effect":
            raise RuntimeError("injected after effect claim insert")
        return original_commit(selected, phase, **kwargs)

    monkeypatch.setattr(predecessor, "commit_phase", fail_effect_phase)
    with pytest.raises(RuntimeError, match="effect claim insert"):
        predecessor.run_effect(current, provider_result, effect_fn=effect)
    assert effect_calls == ["effect"]
    predecessor.close()

    successor = _database_portal_successor(repo)
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_portal_exact_phase_evidence_projected"
        )
        item = first["database_portal_reconciliation"]["attempts"][0]
        assert item["database_disposition"] == (
            "preserved_for_exact_phase_resume"
        )
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.phase_committed("effect")
        assert provider_calls == ["provider"]
        assert effect_calls == ["effect"]
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("callback_kind", ("provider", "effect"))
def test_elapsed_claim_after_durable_callback_blocks_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callback_kind: str,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = predecessor.commit_phase(attempt, "context", body={})
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    def effect(
        _attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        effect_calls.append("effect")
        return {"status": "applied", "effect_key": "durable-effect"}

    original_commit = predecessor.commit_phase

    def fail_selected_phase(
        selected: DatabaseTaskAttempt,
        phase: str,
        **kwargs: object,
    ) -> DatabaseTaskAttempt:
        if phase == callback_kind:
            raise RuntimeError(f"injected after durable {callback_kind} row")
        return original_commit(selected, phase, **kwargs)

    if callback_kind == "provider":
        monkeypatch.setattr(predecessor, "commit_phase", fail_selected_phase)
        with pytest.raises(RuntimeError, match="durable provider row"):
            predecessor.run_provider(current, provider_fn=provider)
    else:
        current, provider_result, _duplicated = predecessor.run_provider(
            current,
            provider_fn=provider,
        )
        monkeypatch.setattr(predecessor, "commit_phase", fail_selected_phase)
        with pytest.raises(RuntimeError, match="durable effect row"):
            predecessor.run_effect(
                current,
                provider_result,
                effect_fn=effect,
            )
    predecessor.close()

    successor = _database_portal_successor(repo)
    monkeypatch.setattr(successor, "_now_ms", lambda: 2_000_000_000_000)
    callback_attempts: list[str] = []

    def forbidden_callback(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("elapsed durable callback was dispatched again")

    successor._provider_fn = forbidden_callback
    successor._effect_fn = forbidden_callback
    successor._validation_fn = forbidden_callback
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        first_item = first["database_portal_reconciliation"]["attempts"][0]
        assert first_item["database_disposition"] == (
            "preserved_for_exact_phase_resume"
        )
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        assert expired["provider_evidence_reused"] is (
            callback_kind == "provider"
        )
        assert expired["effect_evidence_reused"] is (
            callback_kind == "effect"
        )
        terminal = successor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"

        second = successor.run_once()
        assert second["implementation_result"] is None
        count_row = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert count_row is not None and int(count_row[0]) == 1
        assert callback_attempts == []
        assert provider_calls == ["provider"]
        assert effect_calls == (["effect"] if callback_kind == "effect" else [])
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("callback_kind", ("provider", "effect"))
def test_elapsed_claim_after_phase_committed_callback_blocks_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    callback_kind: str,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = predecessor.commit_phase(attempt, "context", body={})
    provider_calls: list[str] = []
    effect_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        provider_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    def effect(
        _attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        effect_calls.append("effect")
        return {"status": "applied", "effect_key": "committed-effect"}

    current, provider_result, duplicated = predecessor.run_provider(
        current,
        provider_fn=provider,
    )
    assert duplicated is False
    assert current.phase_committed("provider")
    if callback_kind == "effect":
        current, _effect_result, duplicated = predecessor.run_effect(
            current,
            provider_result,
            effect_fn=effect,
        )
        assert duplicated is False
        assert current.phase_committed("effect")
    predecessor.close()

    successor = _database_portal_successor(repo)
    monkeypatch.setattr(successor, "_now_ms", lambda: 2_000_000_000_000)
    callback_attempts: list[str] = []

    def forbidden_callback(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("phase-committed callback was dispatched again")

    successor._provider_fn = forbidden_callback
    successor._effect_fn = forbidden_callback
    successor._validation_fn = forbidden_callback
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        first_item = first["database_portal_reconciliation"]["attempts"][0]
        assert first_item["database_disposition"] == (
            "preserved_for_exact_phase_resume"
        )
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        # These flags describe an unprojected result.  The durable callbacks
        # exercised here were already projected into their local phases.
        assert expired["provider_evidence_reused"] is False
        assert expired["effect_evidence_reused"] is False
        terminal = successor.get_attempt(attempt.attempt_id)
        assert terminal is not None and terminal.status == "failed"
        task = successor.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"

        second = successor.run_once()
        assert second["implementation_result"] is None
        count_row = successor._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert count_row is not None and int(count_row[0]) == 1
        assert callback_attempts == []
        assert provider_calls == ["provider"]
        assert effect_calls == (
            ["effect"] if callback_kind == "effect" else []
        )
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_elapsed_postentry_generic_portal_deferred_blocks_without_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    record = daemon.task_source.get_task(attempt.task_cid)
    assert record is not None
    _projection_paths, binding = bridge._ensure_attempt_projection(
        attempt,
        record,
    )
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    daemon._record_database_portal_attempt_binding(
        attempt,
        binding,
        "portal_entered",
    )
    daemon._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
        outcome="deferred",
        body={"exception_type": "DatabasePortalBridgeDeferred"},
    )
    daemon._database_portal_reconciliation_checked = True
    daemon._database_portal_reconciliation_result = {}
    monkeypatch.setattr(daemon, "_now_ms", lambda: 2_000_000_000_000)
    callback_attempts: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("post-entry deferred Portal callback repeated")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        assert first["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        assert expired["callback_authority_incomplete"] is True

        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_attempts == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_exact_preentry_portal_deferred_remains_before_callback_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    record = daemon.task_source.get_task(attempt.task_cid)
    assert record is not None
    _projection_paths, binding_payload = bridge._ensure_attempt_projection(
        attempt,
        record,
    )
    daemon._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )
    daemon._record_database_portal_attempt_binding(
        attempt,
        binding_payload,
        "prepared",
    )
    daemon._record_database_portal_attempt_binding(
        attempt,
        binding_payload,
        "published",
    )
    binding = daemon._database_portal_attempt_binding(attempt)
    assert binding is not None and binding["stage"] == "published"
    daemon._record_callback_dispatch_outcome(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
        outcome="deferred",
        body={
            "exception_type": "DatabasePortalPreEntryPublicationDeferred",
            "preentry_publication_retry_count": 1,
            "preentry_stage": binding["stage"],
        },
    )
    try:
        state = daemon._database_callback_boundary_state(attempt)
        assert state["safe_preentry_provider_deferred"] is True
        assert state["callback_boundary_crossed"] is False
        assert state["callback_authority_incomplete"] is False
        daemon._database_portal_reconciliation_checked = True
        daemon._database_portal_reconciliation_result = {}
        monkeypatch.setattr(daemon, "_now_ms", lambda: 2_000_000_000_000)
        callback_calls: list[str] = []

        def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
            callback_calls.append("provider")
            return {"status": "accepted", "accepted": True}

        daemon._provider_fn = provider
        daemon._effect_fn = lambda *_args: {
            "status": "applied",
            "effect_key": "safe-preentry-retry",
        }
        daemon._validation_fn = lambda *_args: {
            "outcome": "passed",
            "evidence_digest": "sha256:" + "a" * 64,
        }
        first = daemon.run_once()
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "coordination_lease_expired_before_completion"
        )
        assert expired["retry_required"] is True
        assert callback_calls == []

        second = daemon.run_once()
        assert second["implementation_result"] is not None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 2
        assert callback_calls == ["provider"]
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_durable_provider_result_without_dispatch_journal_blocks_redispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repo, daemon, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = daemon.commit_phase(attempt, "context", body={})
    callback_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        callback_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    original_commit = daemon.commit_phase

    def fail_provider_projection(
        selected: DatabaseTaskAttempt,
        phase: str,
        **kwargs: object,
    ) -> DatabaseTaskAttempt:
        if phase == "provider":
            raise RuntimeError("injected after durable provider result")
        return original_commit(selected, phase, **kwargs)

    monkeypatch.setattr(daemon, "commit_phase", fail_provider_projection)
    with pytest.raises(RuntimeError, match="durable provider result"):
        daemon.run_provider(current, provider_fn=provider)
    daemon._require_connection().execute(
        "DELETE FROM attempt_dispatch_journal WHERE attempt_id = ?",
        [attempt.attempt_id],
    )
    monkeypatch.setattr(daemon, "commit_phase", original_commit)
    daemon._database_portal_reconciliation_checked = True
    daemon._database_portal_reconciliation_result = {}
    monkeypatch.setattr(daemon, "_now_ms", lambda: 2_000_000_000_000)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_calls.append("repeated")
        raise AssertionError("journal loss repeated durable provider")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["provider_evidence_reused"] is True
        assert expired["retry_required"] is False
        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_calls == ["provider"]
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("claim_history", ("released", "missing", "completed"))
def test_lost_or_terminal_claim_after_callbacks_never_rearms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    claim_history: str,
) -> None:
    _repo, daemon, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = daemon.commit_phase(attempt, "context", body={})
    callback_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        callback_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    def effect(
        _attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        callback_calls.append("effect")
        return {"status": "applied", "effect_key": "claim-loss-effect"}

    current, provider_result, _duplicated = daemon.run_provider(
        current,
        provider_fn=provider,
    )
    current, _effect_result, _duplicated = daemon.run_effect(
        current,
        provider_result,
        effect_fn=effect,
    )
    assert current.phase_committed("effect")
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    assert claim is not None
    if claim_history == "released":
        daemon.coordinator.release(
            claim.as_fenced_lease(),
            reason="test_released_after_callbacks",
            expected_fencing_token=int(claim.fencing_token),
            expected_fence_epoch=int(claim.fence_epoch),
            now_ms=daemon._now_ms(),
        )
    else:
        connection = daemon.coordinator._require()
        if claim_history == "missing":
            connection.execute(
                "DELETE FROM task_claims WHERE claim_id = ?",
                [attempt.claim_id],
            )
        else:
            connection.execute(
                "UPDATE task_claims SET state = 'completed' WHERE claim_id = ?",
                [attempt.claim_id],
            )
    daemon._database_portal_reconciliation_checked = True
    daemon._database_portal_reconciliation_result = {}
    callback_attempts: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("lost claim authority repeated callbacks")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        assert first["selection_idle_reason"] == (
            "database_expired_attempts_reconciled"
        )
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "claim_authority_lost_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_attempts == []
        assert callback_calls == ["provider", "effect"]
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("binding_state", ("portal_entered", "corrupt"))
def test_portal_binding_authority_without_journal_blocks_expired_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_state: str,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    record = daemon.task_source.get_task(attempt.task_cid)
    assert record is not None
    _projection_paths, binding = bridge._ensure_attempt_projection(
        attempt,
        record,
    )
    if binding_state == "portal_entered":
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_database_portal_attempt_binding(
            attempt,
            binding,
            "portal_entered",
        )
    else:
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_database_portal_attempt_binding(
            attempt,
            binding,
            "prepared",
        )
        daemon._require_connection().execute(
            "UPDATE database_portal_attempt_bindings "
            "SET record_json = '{malformed' WHERE attempt_id = ?",
            [attempt.attempt_id],
        )
    daemon._require_connection().execute(
        "DELETE FROM attempt_dispatch_journal WHERE attempt_id = ?",
        [attempt.attempt_id],
    )
    daemon._database_portal_reconciliation_checked = True
    daemon._database_portal_reconciliation_result = {}
    monkeypatch.setattr(daemon, "_now_ms", lambda: 2_000_000_000_000)
    callback_attempts: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("Portal binding authority was retried")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        expired = first["expired_attempt_reconciliations"][0]
        assert expired["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert expired["retry_required"] is False
        assert expired["callback_authority_incomplete"] is True
        second = daemon.run_once()
        assert second["implementation_result"] is None
        assert callback_attempts == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_completed_claim_without_promoted_completion_is_never_retry_authority(
    tmp_path: Path,
) -> None:
    _repo, daemon, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    daemon.coordinator._require().execute(
        "UPDATE task_claims SET state = 'completed' WHERE claim_id = ?",
        [attempt.claim_id],
    )
    daemon._database_portal_reconciliation_checked = True
    daemon._database_portal_reconciliation_result = {}
    try:
        first = daemon.run_once()
        terminal = first["expired_attempt_reconciliations"][0]
        assert terminal["reason"] == (
            "completed_claim_without_promoted_completion_blocked"
        )
        assert terminal["retry_required"] is False
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("corrupt_authority", ("dispatch", "result"))
def test_malformed_callback_authority_blocks_expired_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_authority: str,
) -> None:
    _repo, daemon, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    if corrupt_authority == "dispatch":
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._require_connection().execute(
            "UPDATE attempt_dispatch_journal SET body_json = '{malformed' "
            "WHERE attempt_id = ?",
            [attempt.attempt_id],
        )
    else:
        daemon._require_connection().execute(
            """
            INSERT INTO provider_invocations(
                invocation_id, attempt_id, task_cid, idempotency_key,
                owner_session_id, recorded_at_ms, result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "provider:malformed-result",
                attempt.attempt_id,
                attempt.task_cid,
                f"provider:{attempt.attempt_id}",
                attempt.owner_session_id,
                daemon._now_ms(),
                "{malformed",
            ],
        )
    daemon._database_portal_reconciliation_checked = True
    daemon._database_portal_reconciliation_result = {}
    monkeypatch.setattr(daemon, "_now_ms", lambda: 2_000_000_000_000)
    callback_attempts: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("malformed callback authority was retried")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        terminal = first["expired_attempt_reconciliations"][0]
        assert terminal["reason"] == (
            "elapsed_claim_after_durable_callback_blocked"
        )
        assert terminal["retry_required"] is False
        assert terminal["callback_authority_incomplete"] is True
        assert terminal["callback_receipt_errors"]
        second = daemon.run_once()
        assert second["implementation_result"] is None
        assert callback_attempts == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_startup_effect_row_without_provider_authority_blocks_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._require_connection().execute(
        """
        INSERT INTO effect_claims(
            effect_id, attempt_id, task_cid, effect_key, idempotency_key,
            owner_session_id, recorded_at_ms, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "effect:orphaned-durable-row",
            attempt.attempt_id,
            attempt.task_cid,
            "orphaned-effect",
            f"effect:{attempt.attempt_id}",
            attempt.owner_session_id,
            predecessor._now_ms(),
            json.dumps({"status": "applied", "effect_key": "orphaned"}),
        ],
    )
    predecessor.close()
    daemon = _database_portal_successor(repo)
    callback_attempts: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callback_attempts.append("callback")
        raise AssertionError("orphaned durable effect triggered a callback")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        item = first["database_portal_reconciliation"]["attempts"][0]
        assert item["database_disposition"] == "blocked_unknown_outcome"
        assert item["database_attempt_status"] == "failed"
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        daemon.close()
        daemon = _database_portal_successor(repo)
        daemon._provider_fn = forbidden
        daemon._effect_fn = forbidden
        daemon._validation_fn = forbidden
        second = daemon.run_once()
        assert second["implementation_result"] is None
        attempts = daemon._require_connection().execute(
            "SELECT COUNT(*) FROM database_task_attempts WHERE task_cid = ?",
            [attempt.task_cid],
        ).fetchone()
        assert attempts is not None and int(attempts[0]) == 1
        assert callback_attempts == []
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("missing_result", ("provider", "effect"))
def test_startup_committed_callback_phase_missing_result_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_result: str,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    current = predecessor.commit_phase(attempt, "context", body={})
    callback_calls: list[str] = []

    def provider(_attempt: DatabaseTaskAttempt) -> dict[str, object]:
        callback_calls.append("provider")
        return {"status": "accepted", "accepted": True}

    def effect(
        _attempt: DatabaseTaskAttempt,
        _provider_result: object,
    ) -> dict[str, object]:
        callback_calls.append("effect")
        return {"status": "applied", "effect_key": "missing-row"}

    current, provider_result, _duplicated = predecessor.run_provider(
        current,
        provider_fn=provider,
    )
    if missing_result == "provider":
        predecessor._require_connection().execute(
            "DELETE FROM provider_invocations WHERE attempt_id = ?",
            [attempt.attempt_id],
        )
    else:
        current, _effect_result, _duplicated = predecessor.run_effect(
            current,
            provider_result,
            effect_fn=effect,
        )
        predecessor._require_connection().execute(
            "DELETE FROM effect_claims WHERE attempt_id = ?",
            [attempt.attempt_id],
        )
    predecessor.close()
    daemon = _database_portal_successor(repo)
    repeated_callbacks: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        repeated_callbacks.append("callback")
        raise AssertionError("committed phase with missing result resumed")

    daemon._provider_fn = forbidden
    daemon._effect_fn = forbidden
    daemon._validation_fn = forbidden
    try:
        first = daemon.run_once()
        item = first["database_portal_reconciliation"]["attempts"][0]
        assert item["database_disposition"] == "blocked_unknown_outcome"
        assert item["database_attempt_status"] == "failed"
        task = daemon.task_source.get_task(attempt.task_cid)
        assert task is not None and task.status == "blocked"
        assert task.body["completion_receipt"]["reason"] == (
            "callback_authority_incomplete_blocked"
        )
        daemon.close()
        daemon = _database_portal_successor(repo)
        daemon._provider_fn = forbidden
        daemon._effect_fn = forbidden
        daemon._validation_fn = forbidden
        second = daemon.run_once()
        assert second["implementation_result"] is None
        assert repeated_callbacks == []
        assert callback_calls == (
            ["provider", "effect"]
            if missing_result == "effect"
            else ["provider"]
        )
    finally:
        daemon.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
@pytest.mark.parametrize("durable_rows", ("provider", "provider_effect"))
def test_startup_projects_durable_rows_without_dispatch_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    durable_rows: str,
) -> None:
    repo, predecessor, _bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    connection = predecessor._require_connection()
    connection.execute(
        """
        INSERT INTO provider_invocations(
            invocation_id, attempt_id, task_cid, idempotency_key,
            owner_session_id, recorded_at_ms, result_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "provider:journal-free",
            attempt.attempt_id,
            attempt.task_cid,
            f"provider:{attempt.attempt_id}",
            attempt.owner_session_id,
            predecessor._now_ms(),
            json.dumps({"status": "accepted", "accepted": True}),
        ],
    )
    if durable_rows == "provider_effect":
        connection.execute(
            """
            INSERT INTO effect_claims(
                effect_id, attempt_id, task_cid, effect_key,
                idempotency_key, owner_session_id, recorded_at_ms,
                result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "effect:journal-free",
                attempt.attempt_id,
                attempt.task_cid,
                "journal-free",
                f"effect:{attempt.attempt_id}",
                attempt.owner_session_id,
                predecessor._now_ms(),
                json.dumps(
                    {"status": "applied", "effect_key": "journal-free"}
                ),
            ],
        )
    predecessor.close()
    successor = _database_portal_successor(repo)
    callbacks: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        callbacks.append("callback")
        raise AssertionError("durable journal-free row invoked callback")

    successor._provider_fn = forbidden
    successor._effect_fn = forbidden
    successor._validation_fn = forbidden
    try:
        first = successor.run_once()
        assert first["selection_idle_reason"] == (
            "database_portal_exact_phase_evidence_projected"
        ), first
        projection = first["database_portal_reconciliation"][
            "exact_phase_projections"
        ][0]
        assert projection["callback_invoked"] is False
        assert projection["projected_phases"] == (
            ["context", "provider", "effect"]
            if durable_rows == "provider_effect"
            else ["context", "provider"]
        )
        assert callbacks == []
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_tampered_post_cas_commit_barrier_remains_hard_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, predecessor, _bridge, attempt, paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    predecessor._begin_callback_dispatch(
        attempt,
        dispatch_kind="provider",
        idempotency_key=f"provider:{attempt.attempt_id}",
    )

    def fail_release(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected after task CAS")

    monkeypatch.setattr(predecessor.coordinator, "release", fail_release)
    with pytest.raises(RuntimeError, match="task CAS"):
        predecessor.reconcile_quiesced_database_portal_attempts(
            trigger="supervisor_signal_shutdown",
            force=True,
        )
    task = predecessor.task_source.get_task(attempt.task_cid)
    assert task is not None
    link = task.body["completion_receipt"]["terminal_reconciliation"]
    barrier_id = link["commit_barrier_receipt_id"]
    barrier_path = Path(paths.reconciliation) / (
        barrier_id.removeprefix("sha256:") + ".json"
    )
    barrier = json.loads(barrier_path.read_text(encoding="utf-8"))
    barrier["trigger"] = "tampered-trigger"
    barrier_path.write_text(json.dumps(barrier), encoding="utf-8")
    predecessor.close()

    successor = _database_portal_successor(repo)
    try:
        first = successor.run_once()
        second = successor.run_once()
        for result in (first, second):
            assert result["selection_idle_reason"] == (
                "database_portal_reconciliation_blocked"
            )
            replay = result["database_portal_reconciliation"]
            assert replay["blocked"] is True
            assert replay["attempts"][0]["reason"] == (
                "database_portal_pre_cas_saga_replay_invalid"
            )
        current = successor.get_attempt(attempt.attempt_id)
        assert current is not None and current.status == "running"
    finally:
        successor.close()


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required")
def test_superseded_historical_binding_revokes_only_old_attempt(
    tmp_path: Path,
) -> None:
    _repo, daemon, bridge, attempt, _paths = (
        _seed_interrupted_database_portal_attempt(tmp_path)
    )
    try:
        record = daemon.task_source.get_task(attempt.task_cid)
        assert record is not None
        _paths, binding = bridge._ensure_attempt_projection(attempt, record)
        daemon._begin_callback_dispatch(
            attempt,
            dispatch_kind="provider",
            idempotency_key=f"provider:{attempt.attempt_id}",
        )
        daemon._record_database_portal_attempt_binding(
            attempt,
            binding,
            "portal_entered",
        )
        replacement = {
            "repository_tree_id": "tree:replacement",
            "tasks": [
                {
                    "task_cid": attempt.task_cid,
                    "task_id": "PCTDD-001",
                    "goal_cid": "goal:pctdd",
                    "title": "Replacement body and validation epoch",
                    "status": "ready",
                    "validation_commands": [
                        "python -m pytest replacement-focused.py"
                    ],
                }
            ],
        }
        daemon.materialize_population(replacement)
        before = daemon.task_source.get_task(attempt.task_cid)
        assert before is not None and before.status == "ready"
        before_bytes = json.dumps(before.to_dict(), sort_keys=True)

        result = daemon.reconcile_quiesced_database_portal_attempts(
            trigger="database_daemon_startup",
            force=True,
        )

        assert result["reconciled"] is True, result
        item = result["attempts"][0]
        assert item["historical_binding"] is True
        assert item["database_disposition"] == "superseded_attempt_revoked"
        after = daemon.task_source.get_task(attempt.task_cid)
        assert after is not None
        assert json.dumps(after.to_dict(), sort_keys=True) == before_bytes
        old = daemon.get_attempt(attempt.attempt_id)
        assert old is not None and old.status == "failed"
    finally:
        daemon.close()


def test_relative_database_state_dir_uses_repository_anchored_attempt_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        database_portal_bridge as bridge_module,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    program = DatabaseProgramConfig(
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        store_id="control.duckdb",
    )
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=Path("state/pctdd_task_state.json"),
            strategy_path=Path("state/pctdd_strategy.json"),
            events_path=Path("state/pctdd_events.jsonl"),
            state_dir=Path("state"),
            repo_root=repo,
            state_prefix="pctdd",
            database_program=program,
        )
    )
    observed: dict[str, object] = {}

    class FakeDaemon:
        def __init__(self, **kwargs: object) -> None:
            self.task_source = object()
            observed["daemon"] = kwargs

        def bind_execution_callbacks(self, **_kwargs: object) -> None:
            return None

        def bind_database_portal_bridge(self, _bridge: object) -> None:
            return None

        def reconcile_quiesced_database_portal_attempts(
            self, **_kwargs: object
        ) -> dict[str, object]:
            return {"reconciled": True, "blocked": False}

        def close(self) -> None:
            return None

    class FakeBridge:
        def __init__(self, **kwargs: object) -> None:
            observed["bridge"] = kwargs

        run_provider = apply_effect = validate_effect = lambda *_a, **_k: {}

    monkeypatch.setattr(daemon_module, "DatabaseImplementationDaemon", FakeDaemon)
    monkeypatch.setattr(bridge_module, "DatabasePortalExecutionBridge", FakeBridge)
    monkeypatch.chdir(elsewhere)

    supervisor._reconcile_interrupted_database_portal_attempts()

    assert observed["bridge"]["attempt_root"] == (
        repo / "state" / "pctdd_database_portal_attempts"
    )
    assert observed["daemon"]["database_path"] == repo / "control.duckdb"


def test_plan_bound_database_child_uses_one_effective_sharding_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        multi_supervisor_runner,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        database_portal_bridge as bridge_module,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_supervisor as supervisor_module,
    )

    repo = tmp_path / "repo"
    state = repo / "state"
    state.mkdir(parents=True)
    program = DatabaseProgramConfig(
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        store_id="control.duckdb",
        store_generation="logical-g1",
    )
    config = PortalSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state / "task-state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        state_dir=state,
        repo_root=repo,
        state_prefix="pctdd",
        task_prefix="## PCTDD-",
        task_shard_count=8,
        task_shard_index=5,
        strict_task_sharding=True,
        database_program=program,
    )
    # Exercise the already-admitted plan-bound runtime shape without creating
    # a second control-plane capsule in this focused authority test.
    config.plan_bound_dispatch = True
    config.scheduler_config_path = repo / "scheduler.json"
    config.plan_revision_store_path = state / "plan-revisions.json"
    config.plan_bound_accepted_tree_root = repo
    config.plan_bound_revision_cid = "cid:revision"
    config.plan_bound_plan_root_cid = "cid:plan"
    config.plan_bound_execution_plan_cid = "cid:execution"
    config.plan_bound_capacity_snapshot_id = "capacity:1"
    config.plan_bound_slice_manifest_cid = "cid:slice"
    config.plan_bound_slice_id = "slice:1"
    config.plan_bound_lane_id = "lane:1"
    config.plan_bound_source_head = "a" * 40
    config.plan_bound_source_tree = "b" * 40
    config.plan_bound_task_source_revision = "revision:tasks"
    config.plan_bound_configuration_root = "cid:configuration"
    config.accepted_control_plane_pin = SimpleNamespace(
        as_dict=lambda: {"schema": "accepted-test-pin@1"}
    )
    config.accepted_control_plane_descriptor = 17
    supervisor = PortalImplementationSupervisor(config)
    monkeypatch.setattr(supervisor, "_validated_plan_bound_slice", lambda: None)
    monkeypatch.setattr(
        multi_supervisor_runner,
        "build_sealed_control_plane_module_command",
        lambda **kwargs: list(kwargs["argv"]),
    )

    command = supervisor._build_daemon_command()
    shard_count_index = command.index("--task-shard-count")
    shard_index_index = command.index("--task-shard-index")
    assert command[shard_count_index + 1] == "1"
    assert command[shard_index_index + 1] == "0"
    assert "--strict-task-sharding" not in command
    command_line = " ".join(command)
    assert supervisor._managed_daemon_matches_command_line(command_line)

    # With missing markers, residual process custody uses the same matcher and
    # therefore still detects the exact live child before any direct DB open.
    monkeypatch.setattr(
        supervisor,
        "_list_process_details",
        lambda: [(4242, command_line)],
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda pid: int(pid) == 4242,
    )
    assert supervisor._find_matching_managed_daemon_pid() == 4242
    drifted = list(command)
    drifted[shard_count_index + 1] = "8"
    drifted[shard_index_index + 1] = "5"
    drifted.append("--strict-task-sharding")
    assert not supervisor._managed_daemon_matches_command_line(
        " ".join(drifted)
    )

    observed: dict[str, object] = {}

    class FakeDaemon:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)
            self.task_source = object()

        def bind_execution_callbacks(self, **_kwargs: object) -> None:
            return None

        def bind_database_portal_bridge(self, _bridge: object) -> None:
            return None

        def reconcile_quiesced_database_portal_attempts(
            self, **_kwargs: object
        ) -> dict[str, object]:
            return {
                "reconciled": True,
                "blocked": False,
                "repair_batch_pending": False,
            }

        def close(self) -> None:
            return None

    class FakeBridge:
        def __init__(self, **_kwargs: object) -> None:
            return None

        run_provider = apply_effect = validate_effect = lambda *_a, **_k: {}

    monkeypatch.setattr(daemon_module, "DatabaseImplementationDaemon", FakeDaemon)
    monkeypatch.setattr(bridge_module, "DatabasePortalExecutionBridge", FakeBridge)

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program
    )

    assert result["blocked"] is False
    assert observed["task_shard_count"] == 1
    assert observed["task_shard_index"] == 0
    assert observed["strict_task_sharding"] is False
    assert observed["control_store_id"] == program.store_id
    assert observed["control_store_generation"] == program.store_generation


def test_shutdown_terminal_repair_stall_is_bounded_and_not_quiescent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        database_portal_bridge as bridge_module,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_daemon as daemon_module,
    )

    repo = tmp_path / "repo"
    state = repo / "state"
    state.mkdir(parents=True)
    program = DatabaseProgramConfig(
        authority_mode="embedded_exclusive",
        task_source_kind="duckdb",
        store_id="control.duckdb",
        store_generation="logical-g1",
    )
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state / "task-state.json",
            strategy_path=state / "strategy.json",
            events_path=state / "events.jsonl",
            state_dir=state,
            repo_root=repo,
            state_prefix="pctdd",
            database_program=program,
        )
    )
    calls: list[int] = []

    class FakeDaemon:
        def __init__(self, **_kwargs: object) -> None:
            self.task_source = object()

        def bind_execution_callbacks(self, **_kwargs: object) -> None:
            return None

        def bind_database_portal_bridge(self, _bridge: object) -> None:
            return None

        def _database_portal_terminal_repair_cursor(
            self,
        ) -> tuple[int, str]:
            return 100, "attempt:stalled-page"

        def reconcile_quiesced_database_portal_attempts(
            self, **_kwargs: object
        ) -> dict[str, object]:
            calls.append(len(calls) + 1)
            return {
                "reconciled": False,
                "blocked": False,
                "reason": "database_portal_terminal_repair_batch_pending",
                "repair_batch_pending": True,
                "reconciliation_complete": False,
                "quiesced": False,
                "safe_to_restart": False,
                "attempts": [
                    {
                        "attempt_id": "attempt:stalled-page",
                        "reason": "terminal_reconciliation_repair_batch_pending",
                        "reconciliation_receipt_id": "",
                    }
                ],
            }

        def close(self) -> None:
            return None

    class FakeBridge:
        def __init__(self, **_kwargs: object) -> None:
            return None

        run_provider = apply_effect = validate_effect = lambda *_a, **_k: {}

    monkeypatch.setattr(daemon_module, "DatabaseImplementationDaemon", FakeDaemon)
    monkeypatch.setattr(bridge_module, "DatabasePortalExecutionBridge", FakeBridge)

    result = supervisor._reconcile_interrupted_database_portal_attempts_bound(
        program
    )

    assert calls == [1, 2]
    assert result["blocked"] is True
    assert result["repair_batch_pending"] is True
    assert result["reconciliation_complete"] is False
    assert result["quiesced"] is False
    assert result["safe_to_restart"] is False
    assert result["reason"] == (
        "database_portal_terminal_repair_shutdown_budget_exhausted"
    )


def test_live_legacy_derived_owner_identity_is_fenced_before_shared_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        implementation_supervisor as supervisor_module,
    )

    repo = tmp_path / "repo"
    state = repo / "state"
    state.mkdir(parents=True)
    config = PortalSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state / "task-state.json",
        strategy_path=state / "strategy.json",
        events_path=state / "events.jsonl",
        state_dir=state,
        repo_root=repo,
        state_prefix="pctdd",
        database_program=DatabaseProgramConfig(
            authority_mode="embedded_exclusive",
            task_source_kind="duckdb",
            store_id="control.duckdb",
        ),
    )
    assert config._database_owner_session_derived is True
    supervisor = PortalImplementationSupervisor(config)
    current_command = supervisor._build_daemon_command()
    owner_index = current_command.index("--owner-session-id")
    legacy_command = tuple(
        current_command[:owner_index] + current_command[owner_index + 2 :]
    )
    legacy_scope = supervisor._managed_daemon_owner_scope()
    legacy_scope.pop("database_owner_session_id")
    pid = 43210
    birth = SimpleNamespace(pid=pid, start_time_ticks=77, boot_id="boot:legacy")
    identity = SimpleNamespace(
        process_birth=birth,
        owner_scope=legacy_scope,
        command=legacy_command,
    )
    supervisor._managed_daemon_pid_path().write_text(str(pid), encoding="utf-8")
    monkeypatch.setattr(
        supervisor_module,
        "load_supervised_child_identity",
        lambda _path: identity,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: birth,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: legacy_command,
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: (
            supervisor_module.OwnerLiveness.ALIVE
            if live["value"]
            else supervisor_module.OwnerLiveness.DEAD
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda _pid: live["value"],
    )

    def terminate_legacy(*_args: object, **_kwargs: object) -> bool:
        live["value"] = False
        return True

    live = {"value": True}
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        terminate_legacy,
    )

    ensured = supervisor.ensure_managed_daemon_pid_file()
    adopted = supervisor._adopt_existing_daemon()

    assert ensured["reason"] == "legacy_managed_database_daemon_fenced"
    assert adopted is None
    assert not supervisor._managed_daemon_pid_path().exists()
    loop_config = supervisor.build_supervisor_loop_config()
    assert loop_config.child_env[
        "IPFS_ACCELERATE_SUPERVISED_CHILD_IDENTITY_PATH"
    ] == str(supervisor._managed_daemon_identity_path())
    assert json.loads(
        loop_config.child_env[
            "IPFS_ACCELERATE_SUPERVISED_CHILD_OWNER_SCOPE"
        ]
    ) == supervisor._managed_daemon_owner_scope()

    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: ("python", "-m", "unrelated.exec-replacement"),
    )
    assert supervisor._exact_managed_daemon_identity_is_live(pid) is False

    explicit = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state / "task-state.json",
            strategy_path=state / "strategy.json",
            events_path=state / "events.jsonl",
            state_dir=state,
            repo_root=repo,
            state_prefix="pctdd",
            database_owner_session_id="session:explicit-owner",
            database_program=config.database_program,
        )
    )
    assert explicit._managed_daemon_identity_matches_scope(
        identity,
        pid=pid,
    ) is False


def test_database_shared_loop_launch_and_restart_adopt_exact_owner_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
        SupervisorLoop,
    )
    from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
        OwnerLiveness,
        ProcessBirthIdentity,
    )

    repo = tmp_path / "repo"
    state = repo / "state"
    state.mkdir(parents=True)
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state / "task-state.json",
            strategy_path=state / "strategy.json",
            events_path=state / "events.jsonl",
            state_dir=state,
            repo_root=repo,
            state_prefix="pctdd",
            database_owner_session_id="session:named-shared-loop-owner",
            database_program=DatabaseProgramConfig(
                authority_mode="embedded_exclusive",
                task_source_kind="duckdb",
                store_id="control.duckdb",
            ),
        )
    )
    spec = SupervisorLoop(
        supervisor.build_supervisor_loop_config()
    )._child_spec("20260830T000000Z")
    expected_scope = supervisor._managed_daemon_owner_scope()
    assert json.loads(spec.env["IPFS_ACCELERATE_SUPERVISED_CHILD_OWNER_SCOPE"]) == (
        expected_scope
    )
    assert spec.env["IPFS_ACCELERATE_SUPERVISED_CHILD_IDENTITY_PATH"] == str(
        supervisor._managed_daemon_identity_path()
    )

    class FakeProcess:
        pid = 45678

    launches: list[tuple[str, ...]] = []

    def launch_process(command: object, **_kwargs: object) -> FakeProcess:
        launches.append(tuple(str(item) for item in command))
        return FakeProcess()

    birth = ProcessBirthIdentity(
        pid=FakeProcess.pid,
        start_time_ticks=345,
        boot_id="boot:shared-db-loop",
        parent_pid=os.getpid(),
    )
    monkeypatch.setattr(supervisor_runtime, "launch_process_child", launch_process)
    monkeypatch.setattr(supervisor_runtime, "read_process_birth", lambda _pid: birth)
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        supervisor_runtime,
        "process_args",
        lambda _pid: " ".join(spec.command),
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_command_argv",
        lambda _pid: tuple(spec.command),
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )

    first = supervisor_runtime.adopt_or_launch_supervised_child(
        spec,
        launch_lock_path=supervisor._managed_daemon_launch_lock_path(),
    )
    second = supervisor_runtime.adopt_or_launch_supervised_child(
        spec,
        launch_lock_path=supervisor._managed_daemon_launch_lock_path(),
    )
    persisted = supervisor_runtime.load_supervised_child_identity(
        supervisor._managed_daemon_identity_path()
    )

    assert first.pid == second.pid == FakeProcess.pid
    assert launches == [tuple(spec.command)]
    assert persisted is not None
    assert persisted.command == tuple(spec.command)
    assert dict(persisted.owner_scope) == expected_scope
    assert persisted.process_birth == birth
