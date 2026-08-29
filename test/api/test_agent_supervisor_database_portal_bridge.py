"""Focused safety tests for database-authoritative Portal execution."""

from __future__ import annotations

import json
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
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATASETS_AUTHORITATIVE_STATE_SCHEMA_REVISION,
    SEMANTIC_TRUTH_AUTHORITY_ENV,
    SEMANTIC_WRITER_POLICY_ENV,
    DatabaseImplementationAuthorityError,
    DatabaseImplementationDaemon,
    DatabaseTaskAttempt,
    PortalImplementationDaemon,
    parse_args,
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
        self.paths.events.write_text(
            json.dumps(
                {
                    "type": "task_completed",
                    "task_id": self.task_alias,
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
def test_real_database_portal_bridge_obeys_outer_global_retry_cap(
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
        assert first["implementation_result"]["retry_exhausted"] is False
        assert second["implementation_result"]["retry_exhausted"] is True
        assert idle["implementation_result"] is None
        assert len(provider_attempts) == 2
        assert inner_caps == [1, 1]
        assert len(set(portal_roots)) == 2
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
            self.events.write_text(
                json.dumps(
                    {
                        "type": "task_completed",
                        "task_id": "PCTDD-001",
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
            "provider_dispatch_outcome_unknown"
        )
        assert daemon.run_once()["implementation_result"] is None
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
        assert daemon.run_once()["implementation_result"] is None
        assert portal_calls == ["run"]
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
