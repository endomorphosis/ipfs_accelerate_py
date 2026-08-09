"""Tests for DatabaseRunRegistry@1 and ImprovementEpochRepository@1 (DQP-031).

Evidence subset: concurrent run creation, head CAS, lost response, replay,
challenger isolation, epoch transition, rollback, redaction, list pagination.

Acceptance:

* Directory scan cannot create a run
* Duplicate idempotency key with different request conflicts
* Exact replay returns prior result
* Challenger uses ordinary worktree/session/lease identities
* Self-improvement can be planned as goals/tasks in the same database
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.database_run_registry import (
    AUTHORITY_CLASS as RUN_AUTHORITY,
    DATABASE_RUN_REGISTRY_INTERFACE,
    DIRECTORY_SCAN_AUTHORITY,
    EXPORT_AUTHORITY,
    REDACTION_MARKER as RUN_REDACTION_MARKER,
    DatabaseRunRegistry,
    DatabaseRunRegistryConflictError,
    DatabaseRunRegistryNotFoundError,
    RunLifecycleState,
    duckdb_available as run_duckdb_available,
    open_database_run_registry,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.database_epochs import (
    AUTHORITY_CLASS as EPOCH_AUTHORITY,
    IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE,
    ChallengerStatus,
    EpochMode,
    EpochStage,
    EpochStatus,
    ImprovementEpochConflictError,
    ImprovementEpochRepository,
    duckdb_available as epoch_duckdb_available,
    open_improvement_epoch_repository,
)


pytestmark = pytest.mark.skipif(
    not (run_duckdb_available() and epoch_duckdb_available()),
    reason="DuckDB is required for DQP-031 hermetic tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _registry(tmp_path: Path) -> DatabaseRunRegistry:
    return open_database_run_registry(tmp_path / "runs.duckdb")


def _epochs(tmp_path: Path, *, name: str = "epochs.duckdb") -> ImprovementEpochRepository:
    return open_improvement_epoch_repository(tmp_path / name)


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_RUN_REGISTRY_INTERFACE == "DatabaseRunRegistry@1"
    assert IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE == "ImprovementEpochRepository@1"
    assert DatabaseRunRegistry.INTERFACE == DATABASE_RUN_REGISTRY_INTERFACE
    assert ImprovementEpochRepository.INTERFACE == IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE
    assert RUN_AUTHORITY == "database_authority"
    assert EPOCH_AUTHORITY == "database_authority"
    assert EXPORT_AUTHORITY == "export_adapter_only"
    assert DIRECTORY_SCAN_AUTHORITY == "scan_non_authoritative"


def test_metadata_records_policy(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        meta = registry.metadata()
        assert meta["interface"] == DATABASE_RUN_REGISTRY_INTERFACE
        assert meta["directory_scan_creates_runs"] == "false"
        assert meta["authority"] == RUN_AUTHORITY
    with _epochs(tmp_path) as repo:
        meta = repo.metadata()
        assert meta["interface"] == IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE
        assert meta["challenger_identity_class"] == "ordinary_worktree_session_lease"
        assert meta["goals_tasks_same_database"] == "true"


# ---------------------------------------------------------------------------
# Run registry: create, CAS, current pointer, pagination
# ---------------------------------------------------------------------------


def test_create_run_and_cas_head(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        created = registry.create_run(
            run_id="run:dqp-031-a",
            run_namespace="ns:dqp-031",
            repository_id="repo:dqp-031",
            worktree_id="wt:main",
            session_id="session:main",
            lease_id="lease:main",
            state=RunLifecycleState.STARTING,
            body={"lane": "dqp-runs"},
        )
        assert created["root"]["run_id"] == "run:dqp-031-a"
        assert created["head"]["run_revision"] == 1
        assert created["head"]["state"] == "starting"
        assert created["root"]["authority"] == RUN_AUTHORITY

        updated = registry.cas_update(
            "run:dqp-031-a",
            expected_revision=1,
            state=RunLifecycleState.RUNNING,
            health="healthy",
            handle={"phase": "implement"},
        )
        assert updated["head"]["run_revision"] == 2
        assert updated["head"]["state"] == "running"
        assert updated["head"]["previous_revision"] == 1

        with pytest.raises(DatabaseRunRegistryConflictError):
            registry.cas_update(
                "run:dqp-031-a",
                expected_revision=1,
                state=RunLifecycleState.COMPLETED,
            )

        current = registry.set_current(
            run_namespace="ns:dqp-031",
            run_id="run:dqp-031-a",
        )
        assert current["selected_run_id"] == "run:dqp-031-a"
        assert current["pointer_revision"] == 1
        assert registry.get_current("ns:dqp-031")["selected_run_id"] == "run:dqp-031-a"


def test_list_runs_pagination(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        for index in range(5):
            registry.create_run(
                run_id=f"run:page-{index}",
                run_namespace="ns:page",
                repository_id="repo:page",
            )
        page = registry.list_runs(run_namespace="ns:page", limit=2, offset=0)
        assert len(page) == 2
        assert page[0]["root"]["run_id"] == "run:page-0"
        next_page = registry.list_runs(run_namespace="ns:page", limit=2, offset=2)
        assert len(next_page) == 2
        assert next_page[0]["root"]["run_id"] == "run:page-2"
        tail = registry.list_runs(run_namespace="ns:page", limit=2, offset=4)
        assert len(tail) == 1


def test_concurrent_run_creation(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:

        def _create(index: int) -> str:
            result = registry.create_run(
                run_id=f"run:concurrent-{index}",
                run_namespace="ns:concurrent",
                repository_id="repo:concurrent",
            )
            return result["root"]["run_id"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_create, index) for index in range(8)]
            ids = [future.result() for future in as_completed(futures)]
        assert len(set(ids)) == 8
        listed = registry.list_runs(run_namespace="ns:concurrent", limit=20)
        assert len(listed) == 8


def test_missing_run_raises(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        with pytest.raises(DatabaseRunRegistryNotFoundError):
            registry.get_run("run:missing")


# ---------------------------------------------------------------------------
# Acceptance: directory scan cannot create a run
# ---------------------------------------------------------------------------


def test_directory_scan_cannot_create_a_run(tmp_path: Path) -> None:
    fake_tree = tmp_path / "legacy-runs"
    (fake_tree / "runs" / "run-legacy-1").mkdir(parents=True)
    (fake_tree / "runs" / "run-legacy-1" / "root.json").write_text(
        '{"run_id":"run-legacy-1"}',
        encoding="utf-8",
    )
    (fake_tree / "runs" / "run-legacy-2").mkdir(parents=True)

    with _registry(tmp_path) as registry:
        receipt = registry.scan_directory(fake_tree)
        assert receipt.creates_runs is False
        assert receipt.authority == DIRECTORY_SCAN_AUTHORITY
        assert receipt.observed_count >= 1
        assert registry.list_runs(limit=100) == ()

        refusal = registry.import_from_directory_scan(fake_tree)
        assert refusal["accepted"] is False
        assert refusal["creates_runs"] is False
        assert "directory_scan_cannot_create_run" in refusal["reason_codes"]
        assert registry.list_runs(limit=100) == ()

        # Explicit create remains the only admission path.
        created = registry.create_run(
            run_id="run:explicit",
            run_namespace="ns:explicit",
            repository_id="repo:explicit",
        )
        assert created["root"]["run_id"] == "run:explicit"
        assert len(registry.list_runs(limit=100)) == 1

        audits = registry.list_audits(subject_id=str(fake_tree))
        assert any(item["action"] == "directory_scan" for item in audits)


# ---------------------------------------------------------------------------
# Acceptance: idempotency conflict + exact replay
# ---------------------------------------------------------------------------


def test_idempotency_exact_replay_returns_prior_result(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        request = {
            "operation": "lifecycle.start",
            "program_id": "program:1",
            "expected_effects": ["effect:start"],
        }
        first = registry.commit_idempotent_result(
            idempotency_key="idem:start-1",
            operation="lifecycle.start",
            request=request,
            result={"status": "started", "run_id": "run:1"},
            caller="operator:test",
            repository_id="repo:dqp-031",
        )
        assert first["replayed"] is False
        assert first["result"]["status"] == "started"
        assert first["status"] == "committed"

        # Lost-response retry with identical request returns prior result.
        replay = registry.commit_idempotent_result(
            idempotency_key="idem:start-1",
            operation="lifecycle.start",
            request=request,
            result={"status": "started", "run_id": "run:1"},
            caller="operator:test",
            repository_id="repo:dqp-031",
        )
        assert replay["replayed"] is True
        assert replay["result"] == first["result"]
        assert replay["request_digest"] == first["request_digest"]
        assert replay["result_digest"] == first["result_digest"]
        assert replay["record_id"] == first["record_id"]
        assert replay["status"] == "replayed"

        loaded = registry.get_idempotency(
            idempotency_key="idem:start-1",
            operation="lifecycle.start",
            caller="operator:test",
            repository_id="repo:dqp-031",
        )
        assert loaded is not None
        assert loaded["result"]["run_id"] == "run:1"


def test_duplicate_idempotency_key_with_different_request_conflicts(
    tmp_path: Path,
) -> None:
    with _registry(tmp_path) as registry:
        registry.commit_idempotent_result(
            idempotency_key="idem:conflict-1",
            operation="lifecycle.stop",
            request={"program_id": "program:1", "drain": False},
            result={"status": "stopping"},
            caller="operator:test",
            repository_id="repo:dqp-031",
        )
        with pytest.raises(DatabaseRunRegistryConflictError) as excinfo:
            registry.commit_idempotent_result(
                idempotency_key="idem:conflict-1",
                operation="lifecycle.stop",
                request={"program_id": "program:1", "drain": True},
                result={"status": "draining"},
                caller="operator:test",
                repository_id="repo:dqp-031",
            )
        assert "different request" in str(excinfo.value).casefold()

        audits = [
            item
            for item in registry.list_audits(subject_id="idem:conflict-1")
            if item["action"] == "idempotency_conflict"
        ]
        assert len(audits) == 1


def test_redaction_on_idempotency_and_run_body(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        created = registry.create_run(
            run_id="run:secret",
            run_namespace="ns:secret",
            repository_id="repo:secret",
            body={"token": "super-secret-token", "lane": "safe"},
        )
        assert created["root"]["body"]["token"] == RUN_REDACTION_MARKER
        assert created["root"]["body"]["lane"] == "safe"

        record = registry.commit_idempotent_result(
            idempotency_key="idem:secret",
            operation="control.mutate",
            request={"password": "also-secret", "target": "program:1"},
            result={"api_key": "k-123", "ok": True},
            caller="operator:test",
        )
        assert record["request"]["password"] == RUN_REDACTION_MARKER
        assert record["result"]["api_key"] == RUN_REDACTION_MARKER
        assert record["result"]["ok"] is True


def test_export_is_non_authoritative(tmp_path: Path) -> None:
    with _registry(tmp_path) as registry:
        registry.create_run(
            run_id="run:export-1",
            run_namespace="ns:export",
            repository_id="repo:export",
        )
        export_path = tmp_path / "exports" / "runs.json"
        receipt = registry.export_runs(export_path, run_namespace="ns:export")
        assert receipt["authority"] == EXPORT_AUTHORITY
        assert receipt["authoritative"] is False
        assert receipt["run_count"] == 1
        assert export_path.is_file()
        # Tampering with the export does not remove the authoritative run.
        export_path.write_text("{}", encoding="utf-8")
        assert registry.get_run("run:export-1")["root"]["run_id"] == "run:export-1"


# ---------------------------------------------------------------------------
# Self-improvement epochs
# ---------------------------------------------------------------------------


def test_epoch_transition_and_rollback(tmp_path: Path) -> None:
    with _epochs(tmp_path) as repo:
        epoch = repo.create_epoch(
            mode=EpochMode.SHADOW,
            program_id="program:dqp-031",
            policy_id="policy:self-improve",
            repository_id="repo:dqp-031",
            objective_id="objective:dqp-031",
        )
        assert epoch["stage"] == "baseline"
        assert epoch["status"] == "open"

        proposed = repo.transition(epoch["epoch_id"], EpochStage.PROPOSE)
        assert proposed["stage"] == "propose"
        assert proposed["status"] == "active"

        evaluated = repo.transition(epoch["epoch_id"], EpochStage.EVALUATE)
        assert evaluated["stage"] == "evaluate"

        rolled = repo.rollback_epoch(epoch["epoch_id"], reason="quality_regression")
        assert rolled["stage"] == "rollback"
        assert rolled["status"] == "rolled_back"
        assert rolled["challenger_status"] == ChallengerStatus.ROLLED_BACK.value

        transitions = repo.list_transitions(epoch["epoch_id"])
        stages = [(item["from_stage"], item["to_stage"]) for item in transitions]
        assert ("baseline", "baseline") in stages
        assert ("baseline", "propose") in stages
        assert ("evaluate", "rollback") in stages

        with pytest.raises(ImprovementEpochConflictError):
            repo.transition(epoch["epoch_id"], EpochStage.PROMOTE)


def test_challenger_uses_ordinary_worktree_session_lease_identities(
    tmp_path: Path,
) -> None:
    with _epochs(tmp_path) as repo:
        epoch = repo.create_epoch(mode=EpochMode.ASSIST, program_id="program:1")
        challenger = repo.register_challenger(
            epoch["epoch_id"],
            worktree_id="wt:lane-0",
            session_id="session:daemon-1",
            lease_id="lease:fenced-7",
            fencing_epoch=7,
            body={"role": "challenger"},
        )
        assert challenger["worktree_id"] == "wt:lane-0"
        assert challenger["session_id"] == "session:daemon-1"
        assert challenger["lease_id"] == "lease:fenced-7"
        assert challenger["fencing_epoch"] == 7
        assert challenger["identity_class"] == "ordinary_worktree_session_lease"
        assert challenger["status"] == "registered"

        loaded = repo.get_challenger(epoch["epoch_id"])
        assert loaded is not None
        assert loaded["worktree_id"] == "wt:lane-0"
        assert loaded["session_id"] == "session:daemon-1"
        assert loaded["lease_id"] == "lease:fenced-7"

        # Only one challenger per epoch.
        with pytest.raises(ImprovementEpochConflictError):
            repo.register_challenger(
                epoch["epoch_id"],
                worktree_id="wt:lane-1",
                session_id="session:daemon-2",
                lease_id="lease:fenced-8",
            )

        # Privileged / special identity markers are rejected.
        epoch_b = repo.create_epoch(mode=EpochMode.CANARY, program_id="program:2")
        with pytest.raises(ImprovementEpochConflictError):
            repo.register_challenger(
                epoch_b["epoch_id"],
                worktree_id="wt:ok",
                session_id="session:ok",
                lease_id="lease:ok",
                body={"identity_class": "privileged_challenger"},
            )


def test_self_improvement_planned_as_goals_and_tasks_same_database(
    tmp_path: Path,
) -> None:
    """Goals/tasks for self-improvement live in the same DuckDB file."""

    db_path = tmp_path / "control_plane.duckdb"
    with open_improvement_epoch_repository(db_path) as repo:
        epoch = repo.create_epoch(
            mode=EpochMode.SHADOW,
            program_id="program:shared",
            objective_id="objective:shared",
        )
        goal = repo.plan_goal(
            epoch["epoch_id"],
            title="Reduce provider churn without quality loss",
            goal_cid="goal:si-1",
            body={"track": "self-improvement"},
        )
        assert goal["same_database"] is True
        assert goal["goal_cid"] == "goal:si-1"

        task = repo.plan_task(
            epoch["epoch_id"],
            goal_cid="goal:si-1",
            title="Migrate run registry idempotency to DuckDB",
            task_cid="task:si-1",
            body={"outputs": ["database_run_registry.py"]},
        )
        assert task["same_database"] is True
        assert task["goal_cid"] == "goal:si-1"
        assert task["task_cid"] == "task:si-1"

        goals = repo.list_planned_goals(epoch["epoch_id"])
        tasks = repo.list_planned_tasks(epoch["epoch_id"])
        assert len(goals) == 1
        assert len(tasks) == 1
        assert goals[0]["title"].startswith("Reduce provider")
        assert tasks[0]["title"].startswith("Migrate run registry")

        metrics = repo.record_token_metrics(
            epoch["epoch_id"],
            tokens_in=1200,
            tokens_out=400,
            provider_calls=2,
            cost_micros=2500,
        )
        assert metrics["tokens_in"] == 1200
        receipt = repo.record_receipt(
            epoch["epoch_id"],
            kind="epoch.evaluation",
            status="retained",
            body={"quality_delta_milli": 0},
        )
        assert receipt["kind"] == "epoch.evaluation"
        rollout = repo.record_rollout(
            epoch["epoch_id"],
            mode=EpochMode.SHADOW,
            status="scheduled",
        )
        assert rollout["mode"] == "shadow"

    # Re-open the same database path: planned work remains durable authority.
    with open_improvement_epoch_repository(db_path) as reopened:
        assert len(reopened.list_planned_goals(epoch["epoch_id"])) == 1
        assert len(reopened.list_planned_tasks(epoch["epoch_id"])) == 1
        assert reopened.get_epoch(epoch["epoch_id"])["epoch_id"] == epoch["epoch_id"]


def test_shared_database_hosts_runs_and_epochs(tmp_path: Path) -> None:
    """Runs and self-improvement epochs can share one control-plane database."""

    db_path = tmp_path / "shared.duckdb"
    with open_database_run_registry(db_path) as registry:
        registry.create_run(
            run_id="run:shared-1",
            run_namespace="ns:shared",
            repository_id="repo:shared",
            worktree_id="wt:shared",
            session_id="session:shared",
            lease_id="lease:shared",
        )
        assert registry.get_run("run:shared-1")["root"]["run_id"] == "run:shared-1"

    with open_improvement_epoch_repository(db_path) as repo:
        epoch = repo.create_epoch(mode=EpochMode.OBSERVE, program_id="program:shared")
        repo.register_challenger(
            epoch["epoch_id"],
            worktree_id="wt:shared",
            session_id="session:shared",
            lease_id="lease:shared",
            fencing_epoch=1,
        )
        goal = repo.plan_goal(
            epoch["epoch_id"],
            title="Plan improvement on shared state",
        )
        task = repo.plan_task(
            epoch["epoch_id"],
            goal_cid=goal["goal_cid"],
            title="Execute improvement task",
        )
        assert goal["same_database"] is True
        assert task["same_database"] is True
        assert repo.get_challenger(epoch["epoch_id"])["worktree_id"] == "wt:shared"

    # Run registry tables remain present after epoch repository open.
    with open_database_run_registry(db_path) as registry:
        assert registry.get_run("run:shared-1")["head"]["run_revision"] == 1


def test_mode_off_refuses_epoch_creation(tmp_path: Path) -> None:
    with _epochs(tmp_path) as repo:
        with pytest.raises(ImprovementEpochConflictError):
            repo.create_epoch(mode=EpochMode.OFF)


def test_list_epochs_filter_and_status(tmp_path: Path) -> None:
    with _epochs(tmp_path) as repo:
        first = repo.create_epoch(mode=EpochMode.SHADOW, program_id="p1")
        second = repo.create_epoch(mode=EpochMode.ASSIST, program_id="p2")
        repo.transition(second["epoch_id"], EpochStage.PROPOSE)
        repo.transition(second["epoch_id"], EpochStage.STOP)
        open_epochs = repo.list_epochs(status=EpochStatus.OPEN)
        terminal = repo.list_epochs(status=EpochStatus.TERMINAL)
        assert any(item["epoch_id"] == first["epoch_id"] for item in open_epochs)
        assert any(item["epoch_id"] == second["epoch_id"] for item in terminal)
