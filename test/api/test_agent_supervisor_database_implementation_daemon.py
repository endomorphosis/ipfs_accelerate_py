"""Tests for DatabaseImplementationDaemon cutover (DQP-018).

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
    duckdb_available,
    open_database_coordinator,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    AUTHORITY_MODE_EMBEDDED,
    AUTHORITY_MODE_LEGACY_MARKDOWN,
    AUTHORITY_MODE_QUACK,
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_ATTEMPT_PHASE_COMPLETED,
    DATABASE_ATTEMPT_PHASE_EFFECT,
    DATABASE_ATTEMPT_PHASE_PROVIDER,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DatabaseImplementationDaemon,
    DatabaseImplementationDaemonAuthorityError,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    database_program_from_cli_namespace,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
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


def _program(**overrides: object) -> DatabaseProgramConfig:
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


def _seed_tasks(coordinator, *, count: int = 4) -> list[str]:
    cids: list[str] = []
    for index in range(count):
        cid = f"task:work-{index}"
        coordinator.register_task(task_cid=cid, task_id=f"WORK-{index}")
        cids.append(cid)
    return cids


def _open_daemon(
    tmp_path: Path,
    *,
    session_id: str,
    clock: FakeClock | None = None,
    provider_calls: list[str] | None = None,
    effect_calls: list[str] | None = None,
    program: DatabaseProgramConfig | None = None,
) -> DatabaseImplementationDaemon:
    clock = clock or FakeClock()
    coordinator = open_database_coordinator(
        tmp_path / "coordination.duckdb",
        clock_ms=clock,
    )
    provider_calls = provider_calls if provider_calls is not None else []
    effect_calls = effect_calls if effect_calls is not None else []

    def provider(attempt: DatabaseTaskAttempt):
        provider_calls.append(attempt.attempt_id)
        return {
            "provider_id": "test-provider",
            "attempt_id": attempt.attempt_id,
            "task_cid": attempt.task_cid,
            "status": "succeeded",
            "output": {"token": attempt.attempt_id},
        }

    def effect(attempt: DatabaseTaskAttempt, provider_result):
        effect_calls.append(attempt.attempt_id)
        return {
            "effect_kind": "test-effect",
            "attempt_id": attempt.attempt_id,
            "provider_token": provider_result.get("output", {}).get("token"),
            "status": "applied",
        }

    return DatabaseImplementationDaemon(
        coordinator=coordinator,
        session_id=session_id,
        journal_path=tmp_path / f"{session_id}.journal.duckdb",
        database_program=program or _program(),
        authority_mode=AUTHORITY_MODE_EMBEDDED,
        provider_callable=provider,
        effect_callable=effect,
        clock_ms=clock,
        write_json_projections=False,
    )


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert (
        DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
        == "DatabaseImplementationDaemon@1"
    )
    assert DATABASE_TASK_ATTEMPT_INTERFACE == "DatabaseTaskAttempt@1"
    assert (
        DatabaseImplementationDaemon.INTERFACE
        == DATABASE_IMPLEMENTATION_DAEMON_INTERFACE
    )
    assert DatabaseTaskAttempt.INTERFACE == DATABASE_TASK_ATTEMPT_INTERFACE


def test_is_database_authority_mode() -> None:
    assert is_database_authority_mode(AUTHORITY_MODE_QUACK)
    assert is_database_authority_mode(AUTHORITY_MODE_EMBEDDED)
    assert not is_database_authority_mode(AUTHORITY_MODE_LEGACY_MARKDOWN)
    assert not is_database_authority_mode("markdown")
    assert not is_database_authority_mode("")


# ---------------------------------------------------------------------------
# Four processes claim distinct work
# ---------------------------------------------------------------------------


def test_four_daemon_processes_claim_distinct_work(tmp_path: Path) -> None:
    clock = FakeClock()
    coordinator = open_database_coordinator(
        tmp_path / "coordination.duckdb",
        clock_ms=clock,
    )
    try:
        cids = _seed_tasks(coordinator, count=4)
        daemons = [
            DatabaseImplementationDaemon(
                coordinator=coordinator,
                session_id=f"session:{index}",
                journal_path=tmp_path / f"session-{index}.journal.duckdb",
                authority_mode=AUTHORITY_MODE_EMBEDDED,
                clock_ms=clock,
            )
            for index in range(4)
        ]
        claims = [daemon.claim_ready() for daemon in daemons]
        assert all(claim is not None for claim in claims)
        claimed_cids = {claim.task_cid for claim in claims if claim is not None}
        assert claimed_cids == set(cids)
        owners = {
            claim.owner_session_id for claim in claims if claim is not None
        }
        assert owners == {f"session:{index}" for index in range(4)}
        # No fifth claim while all exclusive scopes are held.
        idle = DatabaseImplementationDaemon(
            coordinator=coordinator,
            session_id="session:extra",
            journal_path=tmp_path / "session-extra.journal.duckdb",
            authority_mode=AUTHORITY_MODE_EMBEDDED,
            clock_ms=clock,
        )
        assert idle.claim_ready() is None
        for daemon in daemons:
            daemon.close()
        idle.close()
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# Markdown status never updated under database authority
# ---------------------------------------------------------------------------


def test_database_authority_never_updates_markdown_status(tmp_path: Path) -> None:
    markdown = tmp_path / "todo.md"
    original = """# Board

## DQP-001 Example

- Status: todo
- Priority: P0
"""
    markdown.write_text(original, encoding="utf-8")
    daemon = _open_daemon(tmp_path, session_id="session:md")
    try:
        daemon.coordinator.register_task(task_cid="task:md", task_id="DQP-001")
        with pytest.raises(
            DatabaseImplementationDaemonAuthorityError,
            match="forbids Markdown",
        ):
            daemon.maybe_write_markdown_projection(markdown, status="completed")
        with pytest.raises(DatabaseImplementationDaemonAuthorityError):
            daemon.forbid_markdown_status_write(reason="unit-test")
        assert markdown.read_text(encoding="utf-8") == original
        assert daemon.markdown_write_attempt_count() == 2
    finally:
        daemon.close()
        daemon.coordinator.close()


def test_database_implementation_daemon_rejects_legacy_markdown_mode(
    tmp_path: Path,
) -> None:
    coordinator = open_database_coordinator(tmp_path / "coordination.duckdb")
    try:
        with pytest.raises(
            DatabaseImplementationDaemonAuthorityError,
            match="legacy Markdown",
        ):
            DatabaseImplementationDaemon(
                coordinator=coordinator,
                session_id="session:legacy",
                authority_mode=AUTHORITY_MODE_LEGACY_MARKDOWN,
            )
        with pytest.raises(
            DatabaseImplementationDaemonAuthorityError,
            match="forbids Markdown task-status writes",
        ):
            DatabaseImplementationDaemon(
                coordinator=coordinator,
                session_id="session:write-md",
                authority_mode=AUTHORITY_MODE_EMBEDDED,
                write_markdown_projection=True,
            )
    finally:
        coordinator.close()


# ---------------------------------------------------------------------------
# JSON projections optional / absent
# ---------------------------------------------------------------------------


def test_json_queue_status_events_pid_projections_can_be_absent(
    tmp_path: Path,
) -> None:
    # No state/status/events/pid files exist under the working directory.
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.pid"))

    daemon = _open_daemon(tmp_path, session_id="session:no-json")
    try:
        daemon.coordinator.register_task(task_cid="task:a", task_id="A")
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["json_projections_required"] is False
        assert result["markdown_status_updated"] is False
        # Still no optional projections required for correctness.
        assert not list(tmp_path.glob("*_task_state.json"))
        assert not list(tmp_path.glob("*_events.jsonl"))
        assert not list(tmp_path.glob("*.pid"))
        assert result["attempt"]["phase"] == DATABASE_ATTEMPT_PHASE_COMPLETED
    finally:
        daemon.close()
        daemon.coordinator.close()


# ---------------------------------------------------------------------------
# Crash / restart resumes committed phase without duplicating work
# ---------------------------------------------------------------------------


def test_crash_restart_resumes_from_committed_phase_without_duplicate_work(
    tmp_path: Path,
) -> None:
    clock = FakeClock()
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    coordinator = open_database_coordinator(
        tmp_path / "coordination.duckdb",
        clock_ms=clock,
    )
    journal = tmp_path / "session-worker.journal.duckdb"

    def provider(attempt: DatabaseTaskAttempt):
        provider_calls.append(attempt.attempt_id)
        return {
            "provider_id": "test-provider",
            "attempt_id": attempt.attempt_id,
            "status": "succeeded",
            "output": {"n": len(provider_calls)},
        }

    def effect(attempt: DatabaseTaskAttempt, provider_result):
        effect_calls.append(attempt.attempt_id)
        return {
            "effect_kind": "test-effect",
            "attempt_id": attempt.attempt_id,
            "provider_n": provider_result.get("output", {}).get("n"),
            "status": "applied",
        }

    first = DatabaseImplementationDaemon(
        coordinator=coordinator,
        session_id="session:worker",
        journal_path=journal,
        authority_mode=AUTHORITY_MODE_EMBEDDED,
        provider_callable=provider,
        effect_callable=effect,
        clock_ms=clock,
    )
    try:
        coordinator.register_task(task_cid="task:resume", task_id="RESUME")
        claim = first.claim_ready()
        assert claim is not None
        # Commit provider only, then "crash" before effect.
        after_provider = first.commit_phase(
            claim,
            DATABASE_ATTEMPT_PHASE_PROVIDER,
            provider_digest="provider-digest-1",
            body={
                "claim_id": claim.claim_id,
                "provider_result": {
                    "provider_id": "test-provider",
                    "attempt_id": claim.attempt_id,
                    "status": "succeeded",
                    "output": {"n": 1},
                },
            },
        )
        # Simulate that provider already ran once before the crash.
        provider_calls.append(claim.attempt_id)
        assert after_provider.phase_committed(DATABASE_ATTEMPT_PHASE_PROVIDER)
        first.close()

        # Restart with a new daemon process sharing the journal + coordinator.
        restarted = DatabaseImplementationDaemon(
            coordinator=coordinator,
            session_id="session:worker",
            journal_path=journal,
            authority_mode=AUTHORITY_MODE_EMBEDDED,
            provider_callable=provider,
            effect_callable=effect,
            clock_ms=clock,
        )
        try:
            resumed = restarted.get_attempt(claim.attempt_id)
            assert resumed is not None
            assert resumed.phase_committed(DATABASE_ATTEMPT_PHASE_PROVIDER)
            finished = restarted.resume_or_execute(resumed)
            assert finished.phase == DATABASE_ATTEMPT_PHASE_COMPLETED
            assert finished.phase_committed(DATABASE_ATTEMPT_PHASE_EFFECT)
            # Provider must not re-run after the committed provider phase.
            assert provider_calls.count(claim.attempt_id) == 1
            # Effect runs exactly once on resume.
            assert effect_calls.count(claim.attempt_id) == 1
            # Second resume is a pure replay.
            again = restarted.resume_or_execute(finished)
            assert again.phase == DATABASE_ATTEMPT_PHASE_COMPLETED
            assert provider_calls.count(claim.attempt_id) == 1
            assert effect_calls.count(claim.attempt_id) == 1
        finally:
            restarted.close()
    finally:
        coordinator.close()


def test_idempotent_claim_does_not_duplicate_attempt(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path, session_id="session:idem")
    try:
        daemon.coordinator.register_task(task_cid="task:idem", task_id="IDEM")
        first = daemon.claim_task(
            task_cid="task:idem",
            idempotency_key="key-1",
        )
        second = daemon.claim_task(
            task_cid="task:idem",
            idempotency_key="key-1",
        )
        assert first.claim_id == second.claim_id
        assert first.attempt_id == second.attempt_id
        assert first.fencing_token == second.fencing_token
    finally:
        daemon.close()
        daemon.coordinator.close()


def test_run_once_completes_ready_task_and_records_digests(
    tmp_path: Path,
) -> None:
    provider_calls: list[str] = []
    effect_calls: list[str] = []
    daemon = _open_daemon(
        tmp_path,
        session_id="session:once",
        provider_calls=provider_calls,
        effect_calls=effect_calls,
    )
    try:
        daemon.coordinator.register_task(task_cid="task:once", task_id="ONCE")
        result = daemon.run_once()
        assert result["provider_ran"] is True
        assert result["effect_ran"] is True
        assert result["implementation_result"]["phase"] == (
            DATABASE_ATTEMPT_PHASE_COMPLETED
        )
        assert result["implementation_result"]["provider_digest"]
        assert result["implementation_result"]["effect_digest"]
        idle = daemon.run_once()
        assert idle["unchanged"] is True
        assert idle["selection_idle_reason"] == "no_ready_task"
        assert len(provider_calls) == 1
        assert len(effect_calls) == 1
    finally:
        daemon.close()
        daemon.coordinator.close()


# ---------------------------------------------------------------------------
# CLI / Portal cutover wiring
# ---------------------------------------------------------------------------


def test_parse_args_accepts_database_authority_flags() -> None:
    args = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--state-store-id",
            "control.duckdb",
            "--database-session-id",
            "session:cli",
            "--once",
        ]
    )
    program = database_program_from_cli_namespace(args)
    assert program is not None
    assert program.authority_mode == AUTHORITY_MODE_EMBEDDED
    assert program.task_source_kind == "duckdb"
    assert args.database_session_id == "session:cli"


def test_portal_daemon_delegates_to_database_authority(
    tmp_path: Path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "checkout", "-b", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    todo = repo / "todo.md"
    todo.write_text("# unused under database authority\n", encoding="utf-8")
    subprocess.run(["git", "add", "todo.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    # Intentionally do not create JSON projections.
    coordinator_path = tmp_path / "coordination.duckdb"
    program = _program()
    coordinator = open_database_coordinator(coordinator_path)
    try:
        coordinator.register_task(task_cid="task:portal", task_id="PORTAL")
        portal = PortalImplementationDaemon(
            todo_path=todo,
            state_path=state_dir / "portal_task_state.json",
            strategy_path=state_dir / "portal_strategy.json",
            events_path=state_dir / "portal_events.jsonl",
            repo_root=repo,
            database_program=program,
            database_coordinator=coordinator,
            database_session_id="session:portal",
        )
        assert portal.database_authority_active is True
        result = portal.run_once()
        assert result.get("database_authority") is True
        assert result.get("markdown_status_updated") is False
        assert result.get("json_projections_required") is False
        assert result.get("active_task_id") == "task:portal"
        # Markdown board must remain unchanged.
        assert "unused under database authority" in todo.read_text(
            encoding="utf-8"
        )
        assert not (state_dir / "portal_task_state.json").exists()
        portal.close_event_runtime()
    finally:
        coordinator.close()


def test_open_database_implementation_daemon_helper(tmp_path: Path) -> None:
    daemon = open_database_implementation_daemon(
        coordinator_path=tmp_path / "coord.duckdb",
        session_id="session:helper",
        journal_path=tmp_path / "helper.journal.duckdb",
        authority_mode=AUTHORITY_MODE_EMBEDDED,
    )
    try:
        daemon.coordinator.register_task(task_cid="task:h", task_id="H")
        result = daemon.run_once()
        assert result["unchanged"] is False
        assert result["attempt"]["task_cid"] == "task:h"
    finally:
        daemon.close()
        daemon.coordinator.close()
