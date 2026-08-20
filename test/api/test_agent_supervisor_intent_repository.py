"""Tests for IntentRepository@1 / DatabaseTaskSource@1 / PlanRevisionRepository@1.

DQP-012 acceptance:

* No cross-file saga is needed (single DB transactions + domain events)
* Completion cannot be selected without current required evidence
* Existing public task/plan/objective APIs retain canonical identities
* Database rebuild from admitted events matches current projections

Evidence subset: CAS heads, supersession, continuation, recovery, dependency
readiness, queue retry, goal reopen, current evidence.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
    DATABASE_TASK_SOURCE_INTERFACE,
    DATABASE_TASK_SOURCE_SCHEMA,
    MAX_QUERY_LIMIT,
    DatabaseTaskSource,
    TaskSourceBoundsError,
    TaskSourceCompletionError,
    TaskSourceConflictError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
    open_duckdb_connection,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    INTENT_REPOSITORY_INTERFACE,
    PLAN_REVISION_REPOSITORY_INTERFACE,
    IntentCompletionError,
    IntentEventType,
    IntentRepository,
    IntentRepositoryBoundsError,
    IntentRepositoryConflictError,
    PlanRevisionRepository,
    open_intent_repository,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for intent-repository hermetic tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _repo(tmp_path: Path) -> IntentRepository:
    return open_intent_repository(tmp_path / "control.duckdb", owner_id="owner:test")


def _seed_graph(repo: IntentRepository) -> dict[str, str]:
    repo.upsert_objective(
        objective_id="objective:dqp-012",
        objective_alias="DQP-O012",
        title="Migrate intent state",
        priority="P0",
        body={"track": "intent-repository"},
    )
    repo.upsert_goal(
        goal_cid="goal:cid:root",
        goal_alias="DQP-G020",
        title="Intent authority",
        objective_id="objective:dqp-012",
        ordinal=1,
    )
    repo.upsert_goal(
        goal_cid="goal:cid:child",
        goal_alias="DQP-G020-A",
        title="Child goal",
        objective_id="objective:dqp-012",
        parent_goal_cid="goal:cid:root",
        ordinal=2,
    )
    repo.link_goal_edge(
        parent_goal_cid="goal:cid:root",
        child_goal_cid="goal:cid:child",
        edge_kind="depends_on",
    )
    repo.upsert_plan(
        plan_cid="plan:cid:v1",
        goal_cid="goal:cid:root",
        plan_alias="plan-v1",
        status="active",
        body={"steps": ["seed", "migrate"]},
    )
    repo.upsert_task(
        task_cid="task:cid:001",
        task_alias="DQP-012-A",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=1,
        status="ready",
        priority="P0",
        acceptance=[
            {
                "criterion": "tests pass",
                "required_digest": "sha256:" + ("ab" * 32),
                "evidence_kind": "validation",
            }
        ],
        validations=[["python", "-m", "pytest", "-q"]],
        outputs=[{"path": "intent_repository.py", "effect": "create"}],
    )
    repo.upsert_task(
        task_cid="task:cid:002",
        task_alias="DQP-012-B",
        goal_cid="goal:cid:root",
        plan_cid="plan:cid:v1",
        objective_id="objective:dqp-012",
        ordinal=2,
        status="ready",
        dependencies=["task:cid:001"],
        acceptance=[{"criterion": "rebuild matches", "evidence_kind": "validation"}],
        validations=["pytest test_rebuild.py"],
    )
    return {
        "objective_id": "objective:dqp-012",
        "goal_cid": "goal:cid:root",
        "plan_cid": "plan:cid:v1",
        "task_a": "task:cid:001",
        "task_b": "task:cid:002",
        "evidence_digest": "sha256:" + ("ab" * 32),
    }


# ---------------------------------------------------------------------------
# Interface identities
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert INTENT_REPOSITORY_INTERFACE == "IntentRepository@1"
    assert PLAN_REVISION_REPOSITORY_INTERFACE == "PlanRevisionRepository@1"
    assert DATABASE_TASK_SOURCE_INTERFACE == "DatabaseTaskSource@1"
    assert IntentRepository.INTERFACE == INTENT_REPOSITORY_INTERFACE
    assert PlanRevisionRepository.INTERFACE == PLAN_REVISION_REPOSITORY_INTERFACE
    assert DatabaseTaskSource.INTERFACE == DATABASE_TASK_SOURCE_INTERFACE


# ---------------------------------------------------------------------------
# Core intent mutations + canonical identities
# ---------------------------------------------------------------------------


def test_objectives_goals_plans_tasks_retain_canonical_ids(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)

        objective = repo.get_objective(ids["objective_id"])
        assert objective is not None
        assert objective["objective_id"] == "objective:dqp-012"
        # Alias lookup still returns the canonical objective_id.
        by_alias = repo.get_objective("DQP-O012")
        assert by_alias is not None
        assert by_alias["objective_id"] == objective["objective_id"]

        goal = repo.get_goal(ids["goal_cid"])
        assert goal is not None
        assert goal["goal_cid"] == "goal:cid:root"
        assert repo.get_goal("DQP-G020")["goal_cid"] == "goal:cid:root"  # type: ignore[index]

        plan = repo.get_plan(ids["plan_cid"])
        assert plan is not None
        assert plan["plan_cid"] == "plan:cid:v1"

        task = repo.get_task(ids["task_a"])
        assert task is not None
        assert task["task_cid"] == "task:cid:001"
        assert task["task_alias"] == "DQP-012-A"
        # Alias lookup preserves the durable CID.
        by_task_alias = repo.get_task("DQP-012-A")
        assert by_task_alias is not None

        assert tuple(dict(item) for item in repo.list_goal_edges()) == (
            {
                "parent_goal_cid": "goal:cid:root",
                "child_goal_cid": "goal:cid:child",
                "edge_kind": "depends_on",
            },
        )
        with pytest.raises(IntentRepositoryBoundsError):
            repo.list_goal_edges(limit=MAX_QUERY_LIMIT + 1)
        assert by_task_alias["task_cid"] == "task:cid:001"
        assert by_task_alias["dependencies"] == ()
        assert len(by_task_alias["acceptance"]) == 1
        assert len(by_task_alias["validations"]) == 1
        assert len(by_task_alias["outputs"]) == 1

        dependent = repo.get_task(ids["task_b"])
        assert dependent is not None
        assert dependent["dependencies"] == ("task:cid:001",)


def test_cas_heads_reject_stale_revisions(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        revision = int(task["revision"])

        # Provide evidence so a valid completion would succeed.
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["pytest"],
        )
        ok = repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=revision,
            new_status="in_progress",
        )
        assert ok.changed is True

        with pytest.raises(IntentRepositoryConflictError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=revision,
                new_status="blocked",
            )

        # Objective CAS
        objective = repo.get_objective(ids["objective_id"])
        assert objective is not None
        with pytest.raises(IntentRepositoryConflictError):
            repo.upsert_objective(
                objective_id=ids["objective_id"],
                objective_alias="DQP-O012",
                title="stale",
                expected_revision=0,
            )


def test_quack_concurrent_task_status_cas_has_one_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Quack CAS loser cannot append a duplicate revision or domain event."""

    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import (
        QuackCapabilityStatus,
        probe_quack_capabilities,
    )

    capability = probe_quack_capabilities(allow_network_install=False)
    if capability.status is not QuackCapabilityStatus.COMPATIBLE:
        pytest.skip(f"reviewed preinstalled Quack unavailable: {capability.status.value}")

    class _BarrierCursor:
        def __init__(self, cursor: Any, barrier: threading.Barrier) -> None:
            self._cursor = cursor
            self._barrier = barrier

        def fetchall(self) -> Any:
            rows = self._cursor.fetchall()
            self._barrier.wait(timeout=10.0)
            return rows

        def __getattr__(self, name: str) -> Any:
            return getattr(self._cursor, name)

    class _BarrierConnection:
        def __init__(self, connection: Any, barrier: threading.Barrier) -> None:
            self._connection = connection
            self._barrier = barrier

        def execute(self, sql: str, parameters: Any = None) -> Any:
            if parameters is None:
                cursor = self._connection.execute(sql)
            else:
                cursor = self._connection.execute(sql, parameters)
            normalized = " ".join(str(sql).strip().upper().split())
            if "FROM TASKS WHERE TASK_CID = ? OR TASK_ALIAS = ?" in normalized:
                return _BarrierCursor(cursor, self._barrier)
            return cursor

        def __getattr__(self, name: str) -> Any:
            return getattr(self._connection, name)

    class _RacingIntentRepository(IntentRepository):
        def __init__(self, *args: Any, barrier: threading.Barrier, **kwargs: Any) -> None:
            self._select_barrier = barrier
            super().__init__(*args, **kwargs)

        @contextmanager
        def _connection(self, *, write: bool = False) -> Iterator[Any]:
            with super()._connection(write=write) as connection:
                yield _BarrierConnection(connection, self._select_barrier)

    database_path = tmp_path / "control.duckdb"
    with open_intent_repository(database_path, owner_id="owner:seed") as seed:
        ids = _seed_graph(seed)
        baseline_watermark = seed.event_watermark()

    server = build_server(
        database_path=database_path,
        state_dir=tmp_path / "quack-owner",
        port=0,
        store_id="intent-cas-race",
        secret_handle="handle:intent-cas-race",
    )
    identity = server.start()
    assert server._vault is not None  # noqa: SLF001 - exact owner test boundary
    token = server._vault.resolve(identity.secret_handle)  # noqa: SLF001
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", token)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", identity.store_id)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", str(identity.generation))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR",
        str(server.mutation_inbox_path()),
    )

    stop = threading.Event()
    pump_errors: list[BaseException] = []

    def pump() -> None:
        try:
            while not stop.wait(0.002):
                server.process_mutation_inbox()
        except BaseException as exc:  # pragma: no cover - asserted below
            pump_errors.append(exc)

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    barrier = threading.Barrier(2)
    repositories = tuple(
        _RacingIntentRepository(
            identity.listen_uri,
            owner_id=f"owner:race:{ordinal}",
            install_schema=False,
            barrier=barrier,
        )
        for ordinal in range(2)
    )

    def change_status(ordinal: int, status: str) -> tuple[str, str]:
        try:
            receipt = repositories[ordinal].cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=1,
                new_status=status,
            )
        except IntentRepositoryConflictError:
            return "conflict", status
        return "success", str(receipt.details["status"])

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(change_status, 0, "in_progress"),
                executor.submit(change_status, 1, "blocked"),
            )
            outcomes = tuple(future.result(timeout=20.0) for future in futures)
        assert sorted(outcome for outcome, _status in outcomes) == [
            "conflict",
            "success",
        ]
        assert pump_thread.is_alive()
        assert not pump_errors
        winning_status = next(status for outcome, status in outcomes if outcome == "success")
    finally:
        for repository in repositories:
            repository.close()
        stop.set()
        pump_thread.join(timeout=2.0)
        server.stop()

    assert not pump_thread.is_alive()
    assert not pump_errors
    with open_intent_repository(database_path, owner_id="owner:verify") as verify:
        task = verify.get_task(ids["task_a"])
        assert task is not None
        assert task["revision"] == 2
        assert task["status"] == winning_status
        events = verify.list_events(after_global_sequence=baseline_watermark)
        assert len(events) == 1
        assert events[0]["event_type"] == IntentEventType.TASK_STATUS_CHANGED.value
        assert events[0]["task_cid"] == ids["task_a"]

    connection = open_duckdb_connection(database_path)
    try:
        revisions = connection.execute(
            """
            SELECT revision, status FROM task_revisions
            WHERE task_cid = ? ORDER BY revision
            """,
            [ids["task_a"]],
        ).fetchall()
    finally:
        connection.close()
    assert [(int(row[0]), str(row[1])) for row in revisions] == [
        (1, "ready"),
        (2, winning_status),
    ]


# ---------------------------------------------------------------------------
# Completion evidence gate
# ---------------------------------------------------------------------------


def test_completion_requires_current_required_evidence(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None

        with pytest.raises(IntentCompletionError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(task["revision"]),
                new_status="completed",
            )

        # Wrong digest still fails.
        repo.record_evidence(
            task_cid=ids["task_a"],
            evidence_kind="validation",
            digest="sha256:" + ("cd" * 32),
        )
        with pytest.raises(IntentCompletionError):
            repo.cas_task_status(
                task_cid=ids["task_a"],
                expected_revision=int(task["revision"]),
                new_status="completed",
            )

        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
            argv=["python", "-m", "pytest", "-q"],
        )
        satisfied, missing = repo.required_evidence_satisfied(ids["task_a"])
        assert satisfied is True
        assert missing == ()

        receipt = repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            receipt={"validation": "passed"},
            evidence_digests=[ids["evidence_digest"]],
        )
        assert receipt.changed is True
        assert receipt.event_type == IntentEventType.COMPLETION_RECORDED.value
        completed = repo.get_task(ids["task_a"])
        assert completed is not None
        assert completed["status"] == "completed"
        # Canonical identity unchanged across completion.
        assert completed["task_cid"] == "task:cid:001"


def test_ready_selection_respects_dependencies_and_excludes_completed(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        ready = repo.select_ready_tasks()
        assert [item["task_cid"] for item in ready] == ["task:cid:001"]

        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            evidence_digests=[ids["evidence_digest"]],
        )
        ready_after = repo.select_ready_tasks()
        assert [item["task_cid"] for item in ready_after] == ["task:cid:002"]


# ---------------------------------------------------------------------------
# Queue backoff / retry
# ---------------------------------------------------------------------------


def test_queue_backoff_and_retry(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        now = 1_700_000_000_000
        repo._clock_ms = lambda: now  # type: ignore[method-assign]

        repo.record_queue_backoff(
            task_cid=ids["task_a"],
            delay_ms=60_000,
            reason="provider capacity",
            selection_penalty=100,
        )
        entry = repo.get_queue_entry(ids["task_a"])
        assert entry is not None
        assert entry.is_cooled_down(now_ms=now) is True
        assert entry.selection_penalty == 100

        ready = repo.select_ready_tasks(now_ms=now)
        assert all(item["task_cid"] != ids["task_a"] for item in ready)

        repo.record_queue_retry(task_cid=ids["task_a"])
        entry_after = repo.get_queue_entry(ids["task_a"])
        assert entry_after is not None
        assert entry_after.retry_not_before_ms == 0
        ready_after = repo.select_ready_tasks(now_ms=now)
        assert any(item["task_cid"] == ids["task_a"] for item in ready_after)


# ---------------------------------------------------------------------------
# Plan supersession / continuation / CAS heads
# ---------------------------------------------------------------------------


def test_plan_revision_repository_supersession_and_continuation(
    tmp_path: Path,
) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        plans = repo.plan_revisions()
        assert isinstance(plans, PlanRevisionRepository)

        head = plans.head(ids["goal_cid"])
        assert head is not None
        assert head.plan_cid == ids["plan_cid"]

        plans.append_revision(
            plan_cid=ids["plan_cid"],
            expected_revision=1,
            delta={"add_step": "verify"},
            body={"steps": ["seed", "migrate", "verify"]},
        )
        revisions = plans.list_revisions(ids["plan_cid"])
        assert len(revisions) >= 2

        plans.upsert(
            plan_cid="plan:cid:v2",
            goal_cid=ids["goal_cid"],
            plan_alias="plan-v2",
            status="active",
            body={"steps": ["seed", "migrate", "verify", "export"]},
            set_head=False,
        )
        plans.supersede(
            plan_cid=ids["plan_cid"],
            successor_plan_cid="plan:cid:v2",
            expected_revision=2,
            reason="steering",
        )
        head_after = plans.head(ids["goal_cid"])
        assert head_after is not None
        assert head_after.plan_cid == "plan:cid:v2"
        superseded = plans.get(ids["plan_cid"])
        assert superseded is not None
        assert superseded["status"] == "superseded"

        plans.continue_from(
            plan_cid="plan:cid:v2",
            continuation_plan_cid="plan:cid:v2-cont",
            expected_revision=1,
            body={"phase": "export"},
        )
        cont = plans.get("plan:cid:v2-cont")
        assert cont is not None
        assert cont["status"] == "active"
        assert cont["body"].get("continuation_of") == "plan:cid:v2"


# ---------------------------------------------------------------------------
# Goal reopen, blocks, attempts
# ---------------------------------------------------------------------------


def test_goal_reopen_blocks_and_attempts(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        goal = repo.get_goal(ids["goal_cid"])
        assert goal is not None
        repo.upsert_goal(
            goal_cid=ids["goal_cid"],
            goal_alias="DQP-G020",
            title="Intent authority",
            objective_id=ids["objective_id"],
            status="verified_complete",
            expected_revision=int(goal["revision"]),
        )
        closed = repo.get_goal(ids["goal_cid"])
        assert closed is not None
        reopen = repo.reopen_goal(
            goal_cid=ids["goal_cid"],
            expected_revision=int(closed["revision"]),
            reason="new evidence required",
        )
        assert reopen.event_type == IntentEventType.GOAL_REOPENED.value
        reopened = repo.get_goal(ids["goal_cid"])
        assert reopened is not None
        assert reopened["status"] == "reopened"

        block = repo.block_task(
            task_cid=ids["task_b"],
            blocker_kind="dependency",
            blocker_id=ids["task_a"],
            reason="waiting on A",
        )
        assert block.changed is True
        blocked = repo.get_task(ids["task_b"])
        assert blocked is not None
        assert blocked["status"] == "blocked"
        ready = repo.select_ready_tasks()
        assert all(item["task_cid"] != ids["task_b"] for item in ready)

        repo.unblock_task(task_cid=ids["task_b"])
        unblocked = repo.get_task(ids["task_b"])
        assert unblocked is not None
        assert unblocked["status"] == "ready"

        attempt = repo.record_attempt(task_cid=ids["task_a"], status="started")
        assert attempt.event_type == IntentEventType.ATTEMPT_RECORDED.value


# ---------------------------------------------------------------------------
# Event rebuild parity
# ---------------------------------------------------------------------------


def test_rebuild_from_admitted_events_matches_projections(tmp_path: Path) -> None:
    with _repo(tmp_path) as repo:
        ids = _seed_graph(repo)
        task = repo.get_task(ids["task_a"])
        assert task is not None
        repo.record_validation_result(
            task_cid=ids["task_a"],
            outcome="passed",
            evidence_digest=ids["evidence_digest"],
        )
        repo.cas_task_status(
            task_cid=ids["task_a"],
            expected_revision=int(task["revision"]),
            new_status="completed",
            evidence_digests=[ids["evidence_digest"]],
        )
        repo.record_queue_backoff(task_cid=ids["task_b"], delay_ms=5_000, reason="retry later")

        before = repo.snapshot()
        assert before.task_count == 2
        assert before.event_watermark > 0

        events = repo.list_events(limit=1000)
        assert any(item["event_type"] == IntentEventType.TASK_UPSERTED.value for item in events)
        assert any(
            item["event_type"] == IntentEventType.COMPLETION_RECORDED.value for item in events
        )

        after = repo.rebuild_projections_from_events()
        assert after.projection_cid == before.projection_cid
        assert after.task_count == before.task_count
        assert after.goal_count == before.goal_count
        assert after.plan_count == before.plan_count
        assert after.dependency_count == before.dependency_count

        rebuilt_task = repo.get_task(ids["task_a"])
        assert rebuilt_task is not None
        assert rebuilt_task["task_cid"] == "task:cid:001"
        assert rebuilt_task["status"] == "completed"
        rebuilt_dep = repo.get_task(ids["task_b"])
        assert rebuilt_dep is not None
        assert rebuilt_dep["dependencies"] == ("task:cid:001",)

        # Recovery is a pure database operation (no external files).
        recovery = repo.recover()
        assert recovery.event_type == IntentEventType.RECOVERY_APPLIED.value


# ---------------------------------------------------------------------------
# DatabaseTaskSource public API
# ---------------------------------------------------------------------------


def test_database_task_source_public_api_and_completion_gate(tmp_path: Path) -> None:
    source = DatabaseTaskSource(tmp_path / "control.duckdb")
    try:
        receipt = source.materialize(
            {
                "repository_tree_id": "tree:dqp-012",
                "objectives": [
                    {
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "objective_id": "objective:dqp-012",
                        "title": "Move intent into the database",
                        "acceptance_criteria": ["projections rebuild"],
                    }
                ],
                "taskboard": [
                    {
                        "task_id": "DQP-012",
                        "task_cid": "task:cid:012",
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "acceptance_criteria": [
                            {
                                "criterion": "tests pass",
                                "required_digest": "sha256:" + ("11" * 32),
                                "evidence_kind": "validation",
                            }
                        ],
                        "validation_commands": [
                            "python -m pytest -q "
                            "test/api/test_agent_supervisor_intent_repository.py"
                        ],
                        "effects": [
                            {
                                "path": (
                                    "ipfs_accelerate_py/agent_supervisor/"
                                    "task_sources/intent_repository.py"
                                ),
                                "effect": "create",
                            }
                        ],
                    },
                    {
                        "task_id": "DQP-012-B",
                        "task_cid": "task:cid:012b",
                        "goal_id": "G20",
                        "goal_cid": "goal:cid:g20",
                        "depends_on": ["DQP-012"],
                        "acceptance_criteria": ["ready after A"],
                        "validation_commands": ["true"],
                    },
                ],
            }
        )
        assert receipt["task_count"] == 2
        assert receipt["plan_root_cid"]
        assert source.SCHEMA == DATABASE_TASK_SOURCE_SCHEMA

        snap = source.snapshot()
        assert snap.task_count == 2
        assert snap.goal_count >= 1
        assert snap.source_schema == DATABASE_TASK_SOURCE_SCHEMA

        task = source.get_task("DQP-012")
        assert task is not None
        assert task.task_cid == "task:cid:012"
        assert source.get_task("task:cid:012") is not None
        assert source.get_task("task:cid:012").task_cid == task.task_cid  # type: ignore[union-attr]

        page = source.list_tasks(limit=1)
        assert len(page.tasks) == 1
        second = source.list_tasks(cursor=page.next_cursor, limit=1)
        assert len(second.tasks) == 1
        assert {page.tasks[0].task_cid, second.tasks[0].task_cid} == {
            "task:cid:012",
            "task:cid:012b",
        }

        ready = source.ready_tasks()
        assert [item.task_cid for item in ready.tasks] == ["task:cid:012"]

        with pytest.raises(TaskSourceCompletionError):
            source.compare_and_set_status(
                task.task_cid,
                task.revision,
                "completed",
            )

        source.record_validation_result(
            task_cid=task.task_cid,
            outcome="passed",
            evidence_digest="sha256:" + ("11" * 32),
            argv=["pytest"],
        )
        result = source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "completed",
            {"validation": "passed"},
            evidence_digests=["sha256:" + ("11" * 32)],
        )
        assert result.changed is True
        assert result.task.task_cid == "task:cid:012"
        assert result.task.status == "completed"

        ready_after = source.ready_tasks()
        assert [item.task_cid for item in ready_after.tasks] == ["task:cid:012b"]

        # Objective / plan identity APIs.
        assert source.get_objective("objective:dqp-012") is not None
        assert source.get_goal("goal:cid:g20") is not None
        assert source.get_plan(str(receipt["plan_root_cid"])) is not None

        assert source.projection_matches_events() is True

        with pytest.raises(TaskSourceBoundsError):
            source.list_tasks(limit=MAX_QUERY_LIMIT + 1)
    finally:
        source.close()


def test_database_task_source_forwards_sorted_bounded_goal_edges(
    tmp_path: Path,
) -> None:
    with DatabaseTaskSource(tmp_path / "goal-edges.duckdb") as source:
        source.materialize(
            {
                "repository_tree_id": "tree:goal-edges",
                "objectives": [
                    {"goal_cid": "goal:root", "goal_id": "G0", "title": "Root"},
                    {"goal_cid": "goal:a", "goal_id": "GA", "title": "A"},
                    {"goal_cid": "goal:b", "goal_id": "GB", "title": "B"},
                ],
                "goal_edges": [
                    {
                        "parent_goal_cid": "goal:a",
                        "child_goal_cid": "goal:b",
                        "edge_kind": "goal_dependency",
                    },
                    {
                        "parent_goal_cid": "goal:root",
                        "child_goal_cid": "goal:a",
                        "edge_kind": "goal_parent",
                    },
                ],
                "taskboard": [
                    {
                        "task_cid": "task:goal-edges",
                        "task_id": "T-GOAL-EDGES",
                        "goal_cid": "goal:a",
                        "acceptance_criteria": ["goal graph is exact"],
                    }
                ],
            }
        )

        assert tuple(dict(item) for item in source.list_goal_edges()) == (
            {
                "parent_goal_cid": "goal:a",
                "child_goal_cid": "goal:b",
                "edge_kind": "goal_dependency",
            },
            {
                "parent_goal_cid": "goal:root",
                "child_goal_cid": "goal:a",
                "edge_kind": "goal_parent",
            },
        )
        with pytest.raises(TaskSourceBoundsError):
            source.list_goal_edges(limit=MAX_QUERY_LIMIT + 1)


def test_database_task_source_stale_cursor_and_cas_conflict(tmp_path: Path) -> None:
    with DatabaseTaskSource(tmp_path / "control.duckdb") as source:
        source.materialize(
            {
                "repository_tree_id": "tree:x",
                "objectives": [
                    {
                        "goal_cid": "goal:1",
                        "goal_id": "G1",
                        "title": "One",
                    }
                ],
                "taskboard": [
                    {
                        "task_cid": "task:1",
                        "task_id": "T1",
                        "goal_cid": "goal:1",
                        "acceptance_criteria": ["ok"],
                    },
                    {
                        "task_cid": "task:2",
                        "task_id": "T2",
                        "goal_cid": "goal:1",
                        "acceptance_criteria": ["ok"],
                    },
                ],
            }
        )
        page = source.list_tasks(limit=1)
        assert page.next_cursor
        # Corrupt cursor revision.
        with pytest.raises(TaskSourceConflictError):
            source.list_tasks(cursor=page.next_cursor[:-1] + "x", limit=1)

        task = source.get_task("task:1")
        assert task is not None
        source.record_evidence(
            task_cid=task.task_cid,
            evidence_kind="validation",
            digest="sha256:" + ("22" * 32),
        )
        source.compare_and_set_status(task.task_cid, task.revision, "in_progress")
        with pytest.raises(TaskSourceConflictError):
            source.compare_and_set_status(task.task_cid, task.revision, "blocked")


def test_single_transaction_emits_events_without_external_files(
    tmp_path: Path,
) -> None:
    """Mutations only touch the database path — no sidecar saga files."""

    db = tmp_path / "control.duckdb"
    with open_intent_repository(db) as repo:
        _seed_graph(repo)
        watermark = repo.event_watermark()
        assert watermark > 0
    # Only the duckdb file (and maybe wal) under tmp_path — no markdown/json saga.
    names = {path.name for path in tmp_path.iterdir()}
    assert "control.duckdb" in names
    unexpected = {
        name
        for name in names
        if name.endswith((".md", ".json", ".jsonl")) and "control" not in name
    }
    assert not unexpected
