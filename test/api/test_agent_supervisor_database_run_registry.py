"""Tests for DatabaseRunRegistry@1 and ImprovementEpochRepository@1 (DQP-031).

Acceptance:

* Directory scan cannot create a run
* Duplicate idempotency key with different request conflicts
* Exact replay returns prior result
* Challenger uses ordinary worktree/session/lease identities
* Self-improvement can be planned as goals/tasks in the same database

Evidence subset: concurrent run creation, head CAS, lost response, replay,
challenger isolation, epoch transition, rollback, redaction, list pagination.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.database_run_registry import (
    DATABASE_RUN_REGISTRY_INTERFACE,
    EXPORT_AUTHORITY,
    FORBIDDEN_CREATE_SOURCE_KINDS,
    REDACTION_MARKER,
    DatabaseIdempotencyConflictError,
    DatabaseRunCasConflictError,
    DatabaseRunRegistry,
    DatabaseRunSourceError,
    RegistryTxOutcome,
    duckdb_available as run_registry_duckdb_available,
    open_database_run_registry,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.database_epochs import (
    IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE,
    ORDINARY_IDENTITY_KINDS,
    EpochStage,
    EpochStatus,
    ImprovementChallengerIdentityError,
    ImprovementEpochConflictError,
    ImprovementEpochRepository,
    RolloutDecision,
    duckdb_available as epochs_duckdb_available,
    open_improvement_epoch_repository,
)

pytestmark = pytest.mark.skipif(
    not (run_registry_duckdb_available() and epochs_duckdb_available()),
    reason="DuckDB is required for DQP-031 hermetic tests",
)


class FakeClock:
    def __init__(self, start_ms: int = 1_700_000_000_000) -> None:
        self.now = int(start_ms)

    def __call__(self) -> int:
        return int(self.now)

    def advance(self, ms: int) -> int:
        self.now += int(ms)
        return self.now


def _open_registry(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
) -> tuple[DatabaseRunRegistry, FakeClock]:
    clock = clock or FakeClock()
    registry = open_database_run_registry(
        tmp_path / "run_registry.duckdb",
        clock_ms=clock,
    )
    return registry, clock


def _open_epochs(
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    shared_db: Path | None = None,
) -> tuple[ImprovementEpochRepository, FakeClock]:
    clock = clock or FakeClock()
    path = shared_db if shared_db is not None else tmp_path / "epochs.duckdb"
    repo = open_improvement_epoch_repository(path, clock_ms=clock)
    return repo, clock


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_RUN_REGISTRY_INTERFACE == "DatabaseRunRegistry@1"
    assert IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE == "ImprovementEpochRepository@1"
    assert DatabaseRunRegistry.INTERFACE == DATABASE_RUN_REGISTRY_INTERFACE
    assert ImprovementEpochRepository.INTERFACE == IMPROVEMENT_EPOCH_REPOSITORY_INTERFACE
    assert "directory_scan" in FORBIDDEN_CREATE_SOURCE_KINDS
    assert ORDINARY_IDENTITY_KINDS == frozenset({"worktree", "session", "lease"})


def test_authority_policy_surfaces(tmp_path: Path) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        policy = registry.authority_policy()
        assert policy["semantic_authority"] == "database"
        assert policy["directory_scan_create"] == "prohibited"
        assert policy["filesystem_run_trees"] == EXPORT_AUTHORITY
    finally:
        registry.close()

    epochs, _ = _open_epochs(tmp_path)
    try:
        policy = epochs.authority_policy()
        assert policy["challenger_identity"] == "ordinary_worktree_session_lease"
        assert policy["special_challenger_identity_classes"] == "none"
        assert (
            policy["self_improvement_planning"]
            == "goals_and_tasks_in_same_database"
        )
    finally:
        epochs.close()


# ---------------------------------------------------------------------------
# Directory scan cannot create a run
# ---------------------------------------------------------------------------


def test_directory_scan_cannot_create_a_run(tmp_path: Path) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        for source in (
            "directory_scan",
            "prompt_directory_scan",
            "filesystem_scan",
            "scan",
        ):
            with pytest.raises(DatabaseRunSourceError):
                registry.create_run(
                    run_namespace="ns:demo",
                    repository_id="repository:demo",
                    source_kind=source,
                    handle_body={"from": source},
                )
        assert registry.list_runs().total_estimate == 0
    finally:
        registry.close()


def test_admitted_source_creates_run(tmp_path: Path) -> None:
    registry, clock = _open_registry(tmp_path)
    try:
        receipt = registry.create_run(
            run_namespace="ns:demo",
            repository_id="repository:demo",
            source_kind="control_api",
            worktree_id="worktree:main-1",
            handle_body={"phase": "bootstrap"},
            run_id="run:demo-1",
        )
        assert receipt.outcome is RegistryTxOutcome.COMMITTED
        assert receipt.run_id == "run:demo-1"
        assert receipt.run_revision == 1
        assert "created" in receipt.reason_codes

        record = registry.get_run("run:demo-1")
        assert record.run_namespace == "ns:demo"
        assert record.worktree_id == "worktree:main-1"
        assert record.source_kind == "control_api"
        assert record.created_at_ms == clock()
        assert registry.exists("run:demo-1")
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Idempotency: conflict vs exact replay / lost response
# ---------------------------------------------------------------------------


def test_duplicate_idempotency_key_with_different_request_conflicts(
    tmp_path: Path,
) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        first = registry.execute_control_mutation(
            command_kind="lifecycle.start",
            request={"target": "run:a", "mode": "start"},
            idempotency_key="idem:start-1",
            result_body={"accepted": True, "state": "starting"},
            run_id="run:a",
        )
        assert first.outcome is RegistryTxOutcome.COMMITTED
        assert first.replayed is False

        with pytest.raises(DatabaseIdempotencyConflictError):
            registry.execute_control_mutation(
                command_kind="lifecycle.start",
                request={"target": "run:a", "mode": "pause"},  # different request
                idempotency_key="idem:start-1",
                result_body={"accepted": True, "state": "paused"},
                run_id="run:a",
            )
    finally:
        registry.close()


def test_exact_replay_returns_prior_result_without_redispatch(tmp_path: Path) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        dispatches: list[int] = []

        def effect() -> dict[str, object]:
            dispatches.append(1)
            return {"accepted": True, "token": "tok-1"}

        request = {"target": "run:b", "mode": "retry"}
        first = registry.execute_control_mutation(
            command_kind="lifecycle.retry",
            request=request,
            idempotency_key="idem:retry-1",
            effect_fn=effect,
            run_id="run:b",
        )
        assert first.outcome is RegistryTxOutcome.COMMITTED
        assert dispatches == [1]

        # Lost-response recovery: same key + same request returns prior body.
        replay = registry.execute_control_mutation(
            command_kind="lifecycle.retry",
            request=request,
            idempotency_key="idem:retry-1",
            effect_fn=effect,
            run_id="run:b",
        )
        assert replay.outcome is RegistryTxOutcome.REPLAYED
        assert replay.replayed is True
        assert "idempotent_replay" in replay.reason_codes
        assert replay.integrity_cid == first.integrity_cid
        assert replay.body["result"] == first.body["result"]
        # effect_fn must not run again on exact replay.
        assert dispatches == [1]

        looked_up = registry.lookup_idempotency("idem:retry-1")
        assert looked_up is not None
        assert looked_up.replayed is True
        assert looked_up.integrity_cid == first.integrity_cid
    finally:
        registry.close()


def test_create_run_idempotent_replay(tmp_path: Path) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        kwargs = dict(
            run_namespace="ns:idem",
            repository_id="repository:idem",
            source_kind="daemon",
            run_id="run:idem-1",
            handle_body={"n": 1},
            idempotency_key="idem:create-1",
            request={
                "operation": "create",
                "run_id": "run:idem-1",
                "run_namespace": "ns:idem",
            },
        )
        first = registry.create_run(**kwargs)
        second = registry.create_run(**kwargs)
        assert first.outcome is RegistryTxOutcome.COMMITTED
        assert second.outcome is RegistryTxOutcome.REPLAYED
        assert second.run_id == first.run_id
        assert second.handle_cid == first.handle_cid
        assert registry.list_runs().total_estimate == 1
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Head CAS, concurrent create, current pointer
# ---------------------------------------------------------------------------


def test_head_cas_advances_and_conflicts(tmp_path: Path) -> None:
    registry, clock = _open_registry(tmp_path)
    try:
        created = registry.create_run(
            run_namespace="ns:cas",
            repository_id="repository:cas",
            source_kind="operator",
            run_id="run:cas-1",
            handle_body={"rev": 1},
        )
        clock.advance(10)
        updated = registry.cas_update(
            "run:cas-1",
            expected_revision=1,
            handle_body={"rev": 2},
            state="running",
            health="healthy",
        )
        assert updated.outcome is RegistryTxOutcome.COMMITTED
        assert updated.run_revision == 2
        assert updated.previous_revision == 1
        head = registry.get_head("run:cas-1")
        assert head["run_revision"] == 2
        assert head["state"] == "running"

        with pytest.raises(DatabaseRunCasConflictError) as exc_info:
            registry.cas_update(
                "run:cas-1",
                expected_revision=1,  # stale
                handle_body={"rev": "stale"},
            )
        assert exc_info.value.receipt is not None
        assert "revision_mismatch" in exc_info.value.receipt.reason_codes
        assert created.run_id == "run:cas-1"
    finally:
        registry.close()


def test_set_current_and_list_pagination(tmp_path: Path) -> None:
    registry, clock = _open_registry(tmp_path)
    try:
        for index in range(5):
            clock.advance(1)
            registry.create_run(
                run_namespace="ns:page",
                repository_id="repository:page",
                source_kind="test",
                run_id=f"run:page-{index}",
                handle_body={"i": index},
            )
        page1 = registry.list_runs(run_namespace="ns:page", limit=2)
        assert len(page1.items) == 2
        assert page1.has_more is True
        assert page1.next_cursor == "2"
        assert page1.total_estimate == 5

        page2 = registry.list_runs(
            run_namespace="ns:page",
            limit=2,
            cursor=page1.next_cursor,
        )
        assert len(page2.items) == 2
        assert page2.has_more is True

        page3 = registry.list_runs(
            run_namespace="ns:page",
            limit=2,
            cursor=page2.next_cursor,
        )
        assert len(page3.items) == 1
        assert page3.has_more is False
        assert page3.next_cursor == ""

        current = registry.set_current(
            run_namespace="ns:page",
            repository_id="repository:page",
            run_id="run:page-2",
        )
        assert current.outcome is RegistryTxOutcome.COMMITTED
        pointer = registry.get_current("ns:page")
        assert pointer is not None
        assert pointer["selected_run_id"] == "run:page-2"
        assert pointer["pointer_revision"] == 1
    finally:
        registry.close()


def test_concurrent_run_creation(tmp_path: Path) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        def create_one(index: int) -> str:
            receipt = registry.create_run(
                run_namespace="ns:concurrent",
                repository_id="repository:concurrent",
                source_kind="daemon",
                run_id=f"run:concurrent-{index}",
                handle_body={"i": index},
            )
            return receipt.run_id

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(create_one, i) for i in range(16)]
            run_ids = [future.result() for future in as_completed(futures)]
        assert len(set(run_ids)) == 16
        assert registry.list_runs(run_namespace="ns:concurrent").total_estimate == 16
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Audit redaction + filesystem export non-authority
# ---------------------------------------------------------------------------


def test_audit_redaction_and_export_non_authority(tmp_path: Path) -> None:
    registry, _ = _open_registry(tmp_path)
    try:
        registry.create_run(
            run_namespace="ns:audit",
            repository_id="repository:audit",
            source_kind="control_api",
            run_id="run:audit-1",
            handle_body={"ok": True},
        )
        audit = registry.append_audit(
            actor_id="operator:1",
            action="inspect",
            outcome="ok",
            run_id="run:audit-1",
            body={
                "note": "visible",
                "access_token": "super-secret-token-value",
                "nested": {"api_key": "also-secret"},
            },
        )
        assert audit.redacted is True
        # Secret-bearing keys must not persist raw values.
        body_text = str(audit.body)
        assert "super-secret-token-value" not in body_text
        assert "also-secret" not in body_text
        assert REDACTION_MARKER in body_text or "secret" in body_text.casefold()

        listed = registry.list_audits(run_id="run:audit-1")
        assert any(item.action == "inspect" for item in listed)
        assert any(item.action == "create_run" for item in listed)

        export_dir = tmp_path / "export-tree"
        receipt = registry.export_filesystem_tree(export_dir)
        assert receipt["authority"] == EXPORT_AUTHORITY
        assert (export_dir / "EXPORT_NON_AUTHORITATIVE.json").is_file()

        # Tampering/deleting the export must not affect registry authority.
        for path in export_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        assert registry.exists("run:audit-1")
        assert registry.get_run("run:audit-1").run_id == "run:audit-1"
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# Self-improvement epochs: challenger, transition, rollback, planning
# ---------------------------------------------------------------------------


def test_challenger_uses_ordinary_worktree_session_lease_identities(
    tmp_path: Path,
) -> None:
    repo, _ = _open_epochs(tmp_path)
    try:
        epoch = repo.create_epoch(
            repository_id="repository:improve",
            repository_tree="tree:abc",
            policy_id="policy:si@1",
        )
        challenger = repo.register_challenger(
            epoch.epoch_id,
            worktree_id="worktree:challenger-1",
            session_id="session:challenger-1",
            lease_id="lease:challenger-1",
        )
        assert challenger.isolation_mode == "ordinary_identities"
        assert set(challenger.to_dict()["identity_kinds"]) == set(
            ORDINARY_IDENTITY_KINDS
        )
        assert challenger.worktree_id.startswith("worktree:")
        assert challenger.session_id.startswith("session:")
        assert challenger.lease_id.startswith("lease:")

        with pytest.raises(ImprovementChallengerIdentityError):
            # Second challenger on same epoch violates isolation (max 1).
            repo.register_challenger(
                epoch.epoch_id,
                worktree_id="worktree:challenger-2",
                session_id="session:challenger-2",
                lease_id="lease:challenger-2",
            )

        # Special identity classes are rejected.
        other = repo.create_epoch(
            repository_id="repository:improve",
            repository_tree="tree:def",
            epoch_id="epoch:special-test",
        )
        with pytest.raises(ImprovementChallengerIdentityError):
            repo.register_challenger(
                other.epoch_id,
                worktree_id="special:privileged-wt",
                session_id="session:ok",
                lease_id="lease:ok",
            )
    finally:
        repo.close()


def test_epoch_transition_rollback_and_rollout(tmp_path: Path) -> None:
    repo, _ = _open_epochs(tmp_path)
    try:
        epoch = repo.create_epoch(
            repository_id="repository:tx",
            repository_tree="tree:1",
            run_id="run:linked",
            worktree_id="worktree:baseline",
        )
        assert epoch.stage is EpochStage.BASELINE
        assert epoch.status is EpochStatus.OPEN
        assert epoch.revision == 1

        epoch, transition = repo.transition(
            epoch.epoch_id,
            to_stage=EpochStage.PROPOSE,
            expected_revision=1,
            reason="start_propose",
        )
        assert transition.from_stage is EpochStage.BASELINE or str(
            transition.from_stage
        ) in {"baseline", EpochStage.BASELINE}
        assert epoch.stage is EpochStage.PROPOSE
        assert epoch.revision == 2

        epoch, _ = repo.transition(
            epoch.epoch_id,
            to_stage=EpochStage.SHADOW,
            expected_revision=2,
        )
        epoch, _ = repo.transition(
            epoch.epoch_id,
            to_stage=EpochStage.EVALUATE,
            expected_revision=3,
        )
        epoch, _ = repo.transition(
            epoch.epoch_id,
            to_stage=EpochStage.CANARY,
            expected_revision=4,
        )

        with pytest.raises(ImprovementEpochConflictError):
            repo.transition(
                epoch.epoch_id,
                to_stage=EpochStage.PROPOSE,  # illegal from canary
                expected_revision=5,
            )

        epoch, rb = repo.rollback_epoch(
            epoch.epoch_id,
            expected_revision=5,
            reason="quality_regression",
        )
        assert epoch.stage is EpochStage.ROLLBACK
        assert epoch.status is EpochStatus.ROLLED_BACK
        assert rb.to_stage is EpochStage.ROLLBACK or str(rb.to_stage) == "rollback"

        transitions = repo.list_transitions(epoch.epoch_id)
        assert len(transitions) >= 2
        assert any(
            str(item.to_stage) in {"rollback", EpochStage.ROLLBACK}
            or (
                isinstance(item.to_stage, EpochStage)
                and item.to_stage is EpochStage.ROLLBACK
            )
            for item in transitions
        )

        rollout = repo.record_rollout(
            epoch.epoch_id,
            decision=RolloutDecision.ROLLBACK,
            baseline_receipt_id="receipt:baseline",
            challenger_receipt_id="receipt:challenger",
        )
        assert rollout.decision is RolloutDecision.ROLLBACK

        metrics = repo.record_token_metrics(
            epoch.epoch_id,
            input_tokens=100,
            output_tokens=40,
            provider_calls=2,
            context_bytes=4096,
        )
        assert metrics.input_tokens == 100

        receipt = repo.record_receipt(
            epoch.epoch_id,
            receipt_kind="rollback_receipt",
            body={"reason": "quality_regression"},
        )
        assert receipt.digest.startswith("sha256:")
    finally:
        repo.close()


def test_self_improvement_planned_as_goals_and_tasks_same_database(
    tmp_path: Path,
) -> None:
    """Self-improvement planning lands as goals/tasks in the same DB file."""

    shared_db = tmp_path / "control_plane.duckdb"

    # Create a linked run first (single writer), then plan epochs/goals/tasks
    # in the same DuckDB file after releasing the registry connection.
    registry = open_database_run_registry(shared_db)
    try:
        run_receipt = registry.create_run(
            run_namespace="ns:si",
            repository_id="repository:si",
            source_kind="control_api",
            run_id="run:si-1",
            worktree_id="worktree:si-baseline",
            handle_body={"program": "self_improvement"},
        )
        assert run_receipt.run_id == "run:si-1"
    finally:
        registry.close()

    epochs = open_improvement_epoch_repository(shared_db)
    try:
        epoch = epochs.create_epoch(
            repository_id="repository:si",
            repository_tree="tree:si",
            run_id="run:si-1",
            worktree_id="worktree:si-baseline",
            policy_id="policy:self-improvement@1",
        )
        planned = epochs.plan_as_goals_and_tasks(
            epoch.epoch_id,
            goals=[
                {
                    "title": "Reduce unchanged reprompt rate",
                    "tasks": [
                        {"title": "Add failure signature store"},
                        {"title": "Wire deterministic packet cache"},
                    ],
                },
                {
                    "title": "Improve impact closure completeness",
                    "tasks": [
                        {"title": "Index nested submodule edges"},
                    ],
                },
            ],
        )
        assert planned["same_database"] is True
        assert planned["database_path"] == str(shared_db)
        assert planned["goal_count"] == 2
        assert planned["task_count"] == 3

        goals = epochs.list_planned_goals(epoch.epoch_id)
        tasks = epochs.list_planned_tasks(epoch.epoch_id)
        assert len(goals) == 2
        assert len(tasks) == 3
        assert all(goal.epoch_id == epoch.epoch_id for goal in goals)
        assert all(task.epoch_id == epoch.epoch_id for task in tasks)
        updated = epochs.get_epoch(epoch.epoch_id)
        assert updated.status is EpochStatus.SUCCESSORS_CREATED
        assert updated.run_id == "run:si-1"
    finally:
        epochs.close()

    # Re-open run registry on the same file: run identity survived co-located
    # epoch/goal/task writes.
    registry = open_database_run_registry(shared_db)
    try:
        assert registry.get_run("run:si-1").worktree_id == "worktree:si-baseline"
    finally:
        registry.close()


def test_epoch_list_pagination(tmp_path: Path) -> None:
    repo, clock = _open_epochs(tmp_path)
    try:
        for index in range(4):
            clock.advance(1)
            repo.create_epoch(
                repository_id="repository:list",
                repository_tree=f"tree:{index}",
                epoch_id=f"epoch:list-{index}",
            )
        page = repo.list_epochs(repository_id="repository:list", limit=2)
        assert len(page["items"]) == 2
        assert page["has_more"] is True
        page2 = repo.list_epochs(
            repository_id="repository:list",
            limit=2,
            cursor=page["next_cursor"],
        )
        assert len(page2["items"]) == 2
        assert page2["has_more"] is False
    finally:
        repo.close()
