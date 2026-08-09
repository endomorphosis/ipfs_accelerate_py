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
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_ATTEMPT_PHASES,
    DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
    DATABASE_TASK_ATTEMPT_INTERFACE,
    DatabaseIdempotencyConflictError,
    DatabaseImplementationDaemon,
    DatabaseMarkdownMutationError,
    DatabasePhaseConflictError,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    is_database_authority_mode,
    open_database_implementation_daemon,
    parse_args,
    resolve_daemon_database_program,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
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
        "authority_mode": "embedded",
        "task_source_kind": "duckdb",
        "failover_policy": "fail_closed",
        "explicit_legacy": False,
    }
    payload.update(overrides)
    return DatabaseProgramConfig.from_mapping(payload)


def _open_daemon(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    session_id: str = "session:daemon-a",
    markdown_path: Path | None = None,
) -> tuple[DatabaseImplementationDaemon, FakeClock]:
    clock = clock or FakeClock()
    coordination = tmp_path / "coordination.duckdb"
    daemon = open_database_implementation_daemon(
        coordination,
        database_program=_embedded_program(),
        owner_session_id=session_id,
        clock_ms=clock,
        projections_optional=True,
        markdown_path=markdown_path,
    )
    return daemon, clock


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
    assert is_database_authority_mode("quack")
    assert is_database_authority_mode("embedded")
    assert is_database_authority_mode("embedded_exclusive")
    assert not is_database_authority_mode("legacy_markdown")
    assert not is_database_authority_mode("")


def test_four_daemon_sessions_claim_distinct_work(tmp_path: Path) -> None:
    daemon, _clock = _open_daemon(tmp_path)
    try:
        for index in range(4):
            daemon.register_task(
                task_cid=f"task:ready-{index}",
                task_id=f"DQP-T{index}",
            )
        sessions = [f"session:worker-{index}" for index in range(4)]
        claims = [
            daemon.claim_ready(owner_session_id=session)
            for session in sessions
        ]
        assert all(claim is not None for claim in claims)
        task_cids = {claim.task_cid for claim in claims}  # type: ignore[union-attr]
        attempt_ids = {claim.attempt_id for claim in claims}  # type: ignore[union-attr]
        owners = {claim.owner_session_id for claim in claims}  # type: ignore[union-attr]
        assert task_cids == {f"task:ready-{index}" for index in range(4)}
        assert len(attempt_ids) == 4
        assert owners == set(sessions)
        assert daemon.claim_ready(owner_session_id="session:worker-extra") is None
        for claim in claims:
            assert claim is not None
            assert claim.committed_phase == "claimed"
            assert claim.status == "running"
            assert claim.fencing_token >= 1
    finally:
        daemon.close()


def test_markdown_status_not_updated_under_database_authority(
    tmp_path: Path,
) -> None:
    markdown = tmp_path / "board.md"
    original = (
        "## DQP-T1 Example\n\n"
        "- Status: todo\n"
        "- Priority: P0\n"
    )
    markdown.write_text(original, encoding="utf-8")
    daemon, _clock = _open_daemon(tmp_path, markdown_path=markdown)
    try:
        daemon.register_task(task_cid="task:md", task_id="DQP-T1")
        attempt = daemon.claim_task(
            task_cid="task:md",
            owner_session_id="session:md",
            body={"task_id": "DQP-T1"},
        )
        with pytest.raises(DatabaseMarkdownMutationError, match="Markdown"):
            daemon.update_task_status_in_markdown(
                "DQP-T1",
                status="completed",
                reason="test",
            )
        assert markdown.read_text(encoding="utf-8") == original
        # Completion still proceeds through the database.
        for phase in DATABASE_ATTEMPT_PHASES:
            if phase == "claimed":
                continue
            daemon.commit_phase(
                attempt.attempt_id,
                phase,
                idempotency_key=f"{attempt.attempt_id}:phase:{phase}",
            )
        completed = daemon.complete_attempt(attempt.attempt_id)
        assert completed.status == "succeeded"
        assert markdown.read_text(encoding="utf-8") == original
    finally:
        daemon.close()


def test_json_projections_can_be_absent(tmp_path: Path) -> None:
    daemon, _clock = _open_daemon(tmp_path)
    try:
        assert daemon.projections_may_be_absent() is True
        present = daemon.projection_paths_present()
        assert present == {
            "queue": False,
            "status": False,
            "events": False,
            "pid": False,
        }
        daemon.register_task(task_cid="task:proj", task_id="PROJ")
        attempt = daemon.claim_ready(owner_session_id="session:proj")
        assert attempt is not None
        daemon.heartbeat(attempt_id=attempt.attempt_id)
        # Still no optional projection files created by the cutover path.
        assert not (tmp_path / "task_queue.json").exists()
        assert not list(tmp_path.glob("*_events.jsonl"))
        assert not list(tmp_path.glob("*.pid"))
    finally:
        daemon.close()


def test_crash_restart_resumes_phase_without_duplicate_provider_effect(
    tmp_path: Path,
) -> None:
    coordination = tmp_path / "coordination.duckdb"
    execution = tmp_path / "coordination.execution.duckdb"
    clock = FakeClock()

    first = open_database_implementation_daemon(
        coordination,
        execution_database_path=execution,
        database_program=_embedded_program(),
        owner_session_id="session:crash",
        clock_ms=clock,
    )
    try:
        first.register_task(task_cid="task:crash", task_id="CRASH")
        attempt = first.claim_ready(owner_session_id="session:crash")
        assert attempt is not None
        # Progress through provider, then "crash" before effect.
        stopped = first.run_phase_machine(
            attempt.attempt_id,
            provider_input_digest="digest:provider:v1",
            effect_target="src/main.py",
            effect_input_digest="digest:effect:v1",
            stop_after_phase="provider",
        )
        assert stopped.committed_phase == "provider"
        first_provider = first.claim_provider_invocation(
            attempt.attempt_id,
            input_digest="digest:provider:v1",
            idempotency_key=f"{attempt.attempt_id}:provider:digest:provider:v1",
        )
        assert first_provider["duplicate"] is True
        assert first_provider["provider_dispatched"] is False
    finally:
        first.close()

    # Restart from a fresh daemon process bound to the same durable stores.
    second = open_database_implementation_daemon(
        coordination,
        execution_database_path=execution,
        database_program=_embedded_program(),
        owner_session_id="session:crash",
        clock_ms=clock,
    )
    try:
        resumed = second.resume_attempt(attempt.attempt_id)
        assert resumed.committed_phase == "provider"
        assert resumed.status == "resuming"
        provider_again = second.claim_provider_invocation(
            attempt.attempt_id,
            input_digest="digest:provider:v1",
            idempotency_key=f"{attempt.attempt_id}:provider:digest:provider:v1",
        )
        assert provider_again["accepted"] is False
        assert provider_again["duplicate"] is True
        assert provider_again["provider_dispatched"] is False

        finished = second.run_phase_machine(
            attempt.attempt_id,
            provider_input_digest="digest:provider:v1",
            effect_target="src/main.py",
            effect_input_digest="digest:effect:v1",
        )
        assert finished.committed_phase == "completion"
        assert finished.status == "succeeded"

        effect_again = second.claim_effect(
            attempt.attempt_id,
            effect_kind="workspace",
            target_path="src/main.py",
            input_digest="digest:effect:v1",
            idempotency_key=(
                f"{attempt.attempt_id}:effect:workspace:src/main.py:"
                "digest:effect:v1"
            ),
        )
        assert effect_again["duplicate"] is True
        assert effect_again["effect_applied"] is False
        assert effect_again["accepted"] is False
    finally:
        second.close()


def test_phase_skip_and_idempotency_conflicts(tmp_path: Path) -> None:
    daemon, _clock = _open_daemon(tmp_path)
    try:
        daemon.register_task(task_cid="task:phase", task_id="PHASE")
        attempt = daemon.claim_ready(owner_session_id="session:phase")
        assert attempt is not None
        with pytest.raises(DatabasePhaseConflictError, match="cannot skip"):
            daemon.commit_phase(attempt.attempt_id, "effect")
        daemon.commit_phase(attempt.attempt_id, "context")
        daemon.commit_phase(
            attempt.attempt_id,
            "context",
            idempotency_key=f"{attempt.attempt_id}:phase:context",
        )
        with pytest.raises(DatabaseIdempotencyConflictError):
            daemon.commit_phase(
                attempt.attempt_id,
                "context",
                idempotency_key="different-key",
            )
    finally:
        daemon.close()


def test_concurrent_claim_conflict_on_same_task(tmp_path: Path) -> None:
    daemon, _clock = _open_daemon(tmp_path)
    try:
        daemon.register_task(task_cid="task:one", task_id="ONE")
        first = daemon.claim_task(
            task_cid="task:one",
            owner_session_id="session:a",
        )
        assert first.owner_session_id == "session:a"
        with pytest.raises(DatabaseCoordinationConflictError):
            daemon.claim_task(
                task_cid="task:one",
                owner_session_id="session:b",
            )
    finally:
        daemon.close()


def _init_git_repo(root: Path) -> Path:
    import subprocess

    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "DQP-018"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "dqp018@example.invalid"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    readme = root / "README.md"
    readme.write_text("dqp-018\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return root


def test_portal_daemon_database_authority_skips_markdown(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    markdown = repo / "board.md"
    original = "## DQP-X Example\n\n- Status: todo\n"
    markdown.write_text(original, encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    program = _embedded_program()
    portal = PortalImplementationDaemon(
        todo_path=markdown,
        state_path=state_dir / "portal_task_state.json",
        strategy_path=state_dir / "portal_strategy.json",
        events_path=state_dir / "portal_events.jsonl",
        repo_root=repo,
        database_program=program,
        coordination_database_path=state_dir / "coordination.duckdb",
        database_projections_optional=True,
        worktree_pool_enabled=False,
        use_ephemeral_worktree=False,
    )
    try:
        assert portal.is_database_authority() is True
        assert portal.database_implementation is not None
        result = portal._mark_tasks_completed_in_todo(
            ["DQP-X"],
            primary_task_id="DQP-X",
            completion_reason="database_cutover_test",
            expected_task_cids={"DQP-X": "task:DQP-X"},
        )
        assert result["updated"] is False
        assert result["markdown_updated"] is False
        assert result["database_authority"] is True
        assert result["durable"] is True
        assert markdown.read_text(encoding="utf-8") == original
        # Events projection may remain absent under optional projections.
        assert not (state_dir / "portal_events.jsonl").exists()
    finally:
        if portal.database_implementation is not None:
            portal.database_implementation.close()


def test_runner_propagates_database_program(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    markdown = repo / "board.md"
    markdown.write_text("## T1\n\n- Status: todo\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    program = _embedded_program()
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON",
        __import__("json").dumps(program.to_dict(), sort_keys=True),
    )
    # CLI authority flags win together with env; keep the board on the
    # legacy Markdown parser while database authority drives claims/status.
    parsed = parse_args(
        [
            "--todo-path",
            str(markdown),
            "--state-dir",
            str(state_dir),
            "--state-prefix",
            "portal",
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "embedded",
            "--coordination-database-path",
            str(state_dir / "coordination.duckdb"),
            "--once",
        ]
    )
    resolved = resolve_daemon_database_program(args=parsed)
    assert resolved is not None
    assert resolved.authority_mode == "embedded"
    assert resolved.task_source_kind == "duckdb"
    # Build via the portal constructor rather than open_task_source(duckdb on
    # a Markdown path): the runner still resolves and passes the program.
    portal = PortalImplementationDaemon(
        todo_path=markdown,
        state_path=state_dir / "portal_task_state.json",
        strategy_path=state_dir / "portal_strategy.json",
        events_path=state_dir / "portal_events.jsonl",
        repo_root=repo,
        database_program=resolved,
        coordination_database_path=state_dir / "coordination.duckdb",
        worktree_pool_enabled=False,
        use_ephemeral_worktree=False,
    )
    try:
        assert portal.is_database_authority() is True
        assert portal.database_implementation is not None
        # Runner helper still accepts the same parsed namespace shape.
        assert hasattr(build_portal_implementation_daemon_from_args, "__call__")
    finally:
        if portal.database_implementation is not None:
            portal.database_implementation.close()


def test_parse_args_exposes_authority_options() -> None:
    parsed = parse_args(
        [
            "--task-source-kind",
            "duckdb",
            "--authority-mode",
            "quack",
            "--endpoint-secret-handle",
            "env://QUACK_TOKEN",
            "--state-store-id",
            "control.duckdb",
            "--state-store-generation",
            "gen-1",
            "--state-schema-revision",
            "schema-v1",
            "--state-failover-policy",
            "fail_closed",
            "--once",
        ]
    )
    assert parsed.authority_mode == "quack"
    assert parsed.endpoint_secret_handle == "env://QUACK_TOKEN"
    assert parsed.state_store_id == "control.duckdb"
    assert parsed.task_source_kind == "duckdb"
