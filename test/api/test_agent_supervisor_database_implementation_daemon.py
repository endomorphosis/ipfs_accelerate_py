"""Tests for DatabaseImplementationDaemon@1 cutover (DQP-018).

Evidence subset: ready selection, strict shards, lost response, provider
capacity, hard quota, timeout, cancellation, crash, restart, stale worker,
status parity.

Acceptance: Four daemon processes claim distinct work; no task status is
updated in Markdown under database authority; JSON queue/status/events/PID
projections can be absent; crash/restart resumes from committed phase and
does not duplicate provider/effect work.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
    DatabaseCoordinationConflictError,
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    AUTHORITY_MODE_EMBEDDED,
    AUTHORITY_MODE_QUACK,
    DatabaseProgramConfig,
    DatabaseProgramConfigError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_ATTEMPT_PHASE_COMPLETED,
    DATABASE_ATTEMPT_PHASE_EFFECT,
    DATABASE_ATTEMPT_PHASE_PROVIDER,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    database_authority_mode_active,
    database_program_from_daemon_namespace,
    open_database_implementation_daemon,
    parse_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    apply_database_program_defaults,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseImplementationDaemon hermetic tests",
)


class FakeClock:
    def __init__(self, start_ms: int = 1_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> None:
        self.now += int(ms)


def _embedded_program(**overrides: object) -> DatabaseProgramConfig:
    payload = {
        "authority_mode": AUTHORITY_MODE_EMBEDDED,
        "task_source_kind": "duckdb",
        "store_id": "control.duckdb",
        "store_generation": "gen-1",
        "schema_revision": "schema-v1",
        "failover_policy": "fail_closed",
        "explicit_legacy": False,
    }
    payload.update(overrides)
    return DatabaseProgramConfig.from_mapping(payload)


def _quack_program(**overrides: object) -> DatabaseProgramConfig:
    payload = {
        "authority_mode": AUTHORITY_MODE_QUACK,
        "task_source_kind": "duckdb",
        "endpoint_secret_handle": "env://QUACK_TOKEN",
        "store_id": "control.duckdb",
        "store_generation": "gen-7",
        "schema_revision": "schema-v1",
        "failover_policy": "fail_closed",
        "explicit_legacy": False,
    }
    payload.update(overrides)
    return DatabaseProgramConfig.from_mapping(payload)


def _open_daemon(
    tmp_path: Path,
    *,
    program: DatabaseProgramConfig | None = None,
    owner_session_id: str = "session:worker",
    clock: FakeClock | None = None,
    markdown_board: Path | None = None,
) -> tuple[DatabaseImplementationDaemon, FakeClock]:
    clock = clock or FakeClock()
    board = markdown_board or (tmp_path / "tasks.md")
    if not board.exists():
        board.write_text(
            "## DQP-001 Ready task\n\n- Status: todo\n",
            encoding="utf-8",
        )
    daemon = open_database_implementation_daemon(
        tmp_path / "coordination.duckdb",
        database_program=program or _embedded_program(),
        owner_session_id=owner_session_id,
        clock_ms=clock,
        markdown_board_path=board,
        projection_paths={
            "queue": tmp_path / "task_queue.json",
            "status": tmp_path / "task_state.json",
            "events": tmp_path / "events.jsonl",
            "pid": tmp_path / "daemon.pid",
        },
    )
    return daemon, clock


def test_interface_identities() -> None:
    assert DATABASE_IMPLEMENTATION_DAEMON_INTERFACE == (
        "DatabaseImplementationDaemon@1"
    )
    assert DATABASE_TASK_ATTEMPT_INTERFACE == "DatabaseTaskAttempt@1"
    assert DatabaseImplementationDaemon.INTERFACE == (
        DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
    )
    assert DatabaseTaskAttempt.INTERFACE == DATABASE_TASK_ATTEMPT_INTERFACE
    assert database_authority_mode_active(AUTHORITY_MODE_EMBEDDED)
    assert database_authority_mode_active(AUTHORITY_MODE_QUACK)
    assert not database_authority_mode_active("legacy_markdown")


def test_four_daemon_sessions_claim_distinct_work(tmp_path: Path) -> None:
    daemon, clock = _open_daemon(tmp_path)
    try:
        for index in range(4):
            daemon.register_ready_task(
                task_cid=f"task:{index}",
                task_id=f"DQP-{index:03d}",
            )
            clock.advance(1)
        claimed: list[DatabaseTaskAttempt] = []
        for index in range(4):
            attempt = daemon.claim_ready(owner_session_id=f"session:{index}")
            assert attempt is not None
            claimed.append(attempt)
        task_cids = {item.task_cid for item in claimed}
        claim_ids = {item.claim_id for item in claimed}
        owners = {item.owner_session_id for item in claimed}
        assert len(task_cids) == 4
        assert len(claim_ids) == 4
        assert owners == {f"session:{index}" for index in range(4)}
        assert daemon.claim_ready(owner_session_id="session:extra") is None
        with pytest.raises(DatabaseCoordinationConflictError):
            daemon.claim_task(
                task_cid=claimed[0].task_cid,
                owner_session_id="session:intruder",
            )
    finally:
        daemon.close()


def test_no_markdown_status_update_under_database_authority(
    tmp_path: Path,
) -> None:
    board = tmp_path / "board.md"
    original = "## DQP-001 Example\n\n- Status: todo\n"
    board.write_text(original, encoding="utf-8")
    daemon, _clock = _open_daemon(tmp_path, markdown_board=board)
    try:
        daemon.register_ready_task(task_cid="task:a", task_id="DQP-001")
        result = daemon.execute_attempt(
            task_cid="task:a",
            input_payload={"prompt": "implement"},
            provider=lambda payload: {"ok": True, **payload},
            effect=lambda payload: {"applied": True, **payload},
        )
        assert result["attempt"]["committed_phase"] == (
            DATABASE_ATTEMPT_PHASE_COMPLETED
        )
        assert board.read_text(encoding="utf-8") == original
        with pytest.raises(RuntimeError, match="forbids Markdown"):
            daemon.forbid_markdown_status_update(board)
        assert daemon.markdown_status_updates_forbidden is True
    finally:
        daemon.close()


def test_json_projections_can_be_absent(tmp_path: Path) -> None:
    daemon, _clock = _open_daemon(tmp_path)
    try:
        presence = daemon.projections_present()
        assert presence["queue"] is False
        assert presence["status"] is False
        assert presence["events"] is False
        assert presence["pid"] is False
        assert daemon.projections_optional is True
        daemon.register_ready_task(task_cid="task:p", task_id="DQP-P")
        # Execution succeeds even though no JSON projections exist.
        outcome = daemon.execute_attempt(
            task_cid="task:p",
            provider=lambda _payload: {"provider": "ok"},
            effect=lambda _payload: {"effect": "ok"},
        )
        assert outcome["attempt"]["status"] == "completed"
        assert not (tmp_path / "task_queue.json").exists()
        assert not (tmp_path / "task_state.json").exists()
        assert not (tmp_path / "events.jsonl").exists()
        assert not (tmp_path / "daemon.pid").exists()
    finally:
        daemon.close()


def test_crash_restart_resumes_without_duplicate_provider_or_effect(
    tmp_path: Path,
) -> None:
    store = tmp_path / "coordination.duckdb"
    program = _embedded_program()
    clock = FakeClock()
    provider_calls = {"count": 0}
    effect_calls = {"count": 0}

    def provider(payload: dict) -> dict:
        provider_calls["count"] += 1
        return {"text": "done", "n": provider_calls["count"], **payload}

    def effect(payload: dict) -> dict:
        effect_calls["count"] += 1
        return {"wrote": True, "n": effect_calls["count"], **payload}

    first = open_database_implementation_daemon(
        store,
        database_program=program,
        owner_session_id="session:worker",
        clock_ms=clock,
    )
    try:
        first.register_ready_task(task_cid="task:resume", task_id="DQP-R")
        attempt = first.claim_task(
            task_cid="task:resume",
            owner_session_id="session:worker",
            idempotency_key="resume-1",
        )
        attempt, provider_result, provider_executed = first.run_provider_phase(
            attempt,
            input_payload={"seed": 1},
            provider=provider,
        )
        assert provider_executed is True
        assert provider_calls["count"] == 1
        assert attempt.committed_phase == DATABASE_ATTEMPT_PHASE_PROVIDER
        attempt, effect_result, effect_executed = first.run_effect_phase(
            attempt,
            effect_payload=provider_result,
            effect=effect,
        )
        assert effect_executed is True
        assert effect_calls["count"] == 1
        assert attempt.committed_phase == DATABASE_ATTEMPT_PHASE_EFFECT
        attempt_id = attempt.attempt_id
    finally:
        first.close()

    # Crash boundary: process restarts with a fresh daemon handle.
    restarted = open_database_implementation_daemon(
        store,
        database_program=program,
        owner_session_id="session:worker",
        clock_ms=clock,
    )
    try:
        resumed = restarted.resume_from_committed_phase(
            attempt_id,
            input_payload={"seed": 1},
            effect_payload={"text": "done", "n": 1, "seed": 1},
            provider=provider,
            effect=effect,
        )
        assert resumed["provider_executed"] is False
        assert resumed["effect_executed"] is False
        assert provider_calls["count"] == 1
        assert effect_calls["count"] == 1
        assert resumed["attempt"]["committed_phase"] == (
            DATABASE_ATTEMPT_PHASE_COMPLETED
        )
        # Response-loss retry with the same idempotency key reuses the claim.
        replay = restarted.claim_task(
            task_cid="task:resume",
            owner_session_id="session:worker",
            idempotency_key="resume-1",
        )
        assert replay.attempt_id == attempt_id
    finally:
        restarted.close()


def test_portal_daemon_honors_database_authority_without_markdown_writes(
    tmp_path: Path,
) -> None:
    board = tmp_path / "tasks.md"
    original = "## DQP-010 Portal\n\n- Status: todo\n"
    board.write_text(original, encoding="utf-8")
    program = _embedded_program()
    db = open_database_implementation_daemon(
        tmp_path / "coordination.duckdb",
        database_program=program,
        markdown_board_path=board,
    )
    try:
        state_dir = tmp_path / "state"
        # Projections intentionally absent.
        daemon = PortalImplementationDaemon(
            todo_path=board,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            repo_root=tmp_path,
            database_program=program,
            database_implementation=db,
        )
        assert daemon.database_authority_active is True
        assert daemon.projections_optional is True
        assert daemon._task_source_writes_markdown_checkout() is False
        result = daemon._mark_tasks_completed_in_todo(
            ["DQP-010"],
            primary_task_id="DQP-010",
            completion_reason="database_authority",
        )
        assert result["updated"] is True
        assert result["writes_markdown_checkout"] is False
        assert board.read_text(encoding="utf-8") == original
    finally:
        db.close()


def test_cli_and_runner_consume_database_program_natively(
    tmp_path: Path,
) -> None:
    program = _quack_program()
    argv = [
        "--todo-path",
        str(tmp_path / "control.duckdb"),
        "--state-dir",
        str(tmp_path / "state"),
        "--state-prefix",
        "dqp",
        "--once",
        *program.cli_args(),
        "--database-store-path",
        str(tmp_path / "coordination.duckdb"),
    ]
    # Defaults injector is idempotent with already-present flags.
    expanded = apply_database_program_defaults(argv, database_program=program)
    assert expanded.count("--authority-mode") == 1
    assert "quack" in expanded
    parsed = parse_args(expanded)
    assert parsed.authority_mode == AUTHORITY_MODE_QUACK
    assert parsed.task_source_kind == "duckdb"
    assert parsed.endpoint_secret_handle == "env://QUACK_TOKEN"
    restored = database_program_from_daemon_namespace(parsed)
    assert restored is not None
    assert restored.authority_mode == AUTHORITY_MODE_QUACK
    assert restored.store_id == "control.duckdb"
    assert restored.endpoint_secret_handle == "env://QUACK_TOKEN"

    embedded_args = parse_args(
        [
            "--authority-mode",
            AUTHORITY_MODE_EMBEDDED,
            "--task-source-kind",
            "duckdb",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "gen-1",
            "--state-schema-revision",
            "schema-v1",
            "--database-store-path",
            str(tmp_path / "runner-coordination.duckdb"),
        ]
    )
    embedded = database_program_from_daemon_namespace(embedded_args)
    assert embedded is not None
    assert database_authority_mode_active(embedded.authority_mode)
    assert embedded.authority_mode == AUTHORITY_MODE_EMBEDDED
    assert parse_args(["--authority-mode", "quack"]).authority_mode == "quack"


def test_legacy_markdown_requires_explicit_flag() -> None:
    with pytest.raises(DatabaseProgramConfigError, match="explicit_legacy"):
        DatabaseProgramConfig(
            authority_mode="legacy_markdown",
            task_source_kind="legacy-markdown",
            explicit_legacy=False,
        )
    legacy = DatabaseProgramConfig.explicit_legacy_markdown()
    assert legacy.explicit_legacy is True
    ns = parse_args(
        [
            "--authority-mode",
            "legacy_markdown",
            "--task-source-kind",
            "legacy-markdown",
            "--explicit-legacy-task-source",
        ]
    )
    program = database_program_from_daemon_namespace(ns)
    assert program is not None
    assert program.authority_mode == "legacy_markdown"
    assert program.explicit_legacy is True
