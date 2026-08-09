from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor import duckdb_task_source as task_source_module
from ipfs_accelerate_py.agent_supervisor.duckdb_task_source import (
    DUCKDB_TASK_SOURCE_SCHEMA,
    MAX_QUERY_LIMIT,
    ConsistentTaskSourceProjection,
    DuckDBTaskSource,
    TaskSourceBoundsError,
    TaskSourceConflictError,
    TaskSourceInjectionError,
    TaskSourceIntegrityError,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow import (
    PromptAcceptanceRecord,
    PromptEvidenceRecord,
    PromptGoalGraph,
    PromptGoalRecord,
    PromptOutputRecord,
    PromptTaskRecord,
    PromptValidationRecord,
)


def _hold_duckdb_writer(
    database_path: str,
    entered: object,
    release: object,
) -> None:
    source = DuckDBTaskSource(database_path)
    with source._write_connection():
        entered.set()
        if not release.wait(timeout=10.0):
            raise RuntimeError("reader did not release the multiprocess writer")


def _complete_task_writer(
    database_path: str,
    attempted: object,
    finished: object,
) -> None:
    source = DuckDBTaskSource(database_path)
    attempted.set()
    source.compare_and_set_status("REF-275", 1, "completed")
    finished.set()


def _source() -> dict[str, object]:
    return {
        "schema": "fixture/formal-plan-input@1",
        "repository_tree_id": "tree:candidate",
        "objectives": [
            {
                "goal_id": "G12.S1",
                "goal_cid": "goal:cid:g12-s1",
                "owner_actor_id": "owner:supervisor",
                "title": "Compile formal plans",
                "acceptance_criteria": ["Every task has retained evidence."],
            }
        ],
        "taskboard": [
            {
                "task_id": "REF-275",
                "task_cid": "task:cid:275",
                "goal_id": "G12.S1",
                "actor_id": "agent:alpha",
                "resource_needs": ["cpu", "duckdb"],
                "changed_ast_scopes": ["symbol:cid:contracts"],
                "acceptance_criteria": ["contract tests pass"],
                "validation_commands": ["pytest test_contracts.py"],
                "effects": [
                    {
                        "effect_id": "effect:275",
                        "operation": "assign",
                        "fluent_id": "output:contracts.py",
                        "path": "contracts.py",
                        "value": "modify",
                    }
                ],
            },
            {
                "task_id": "REF-276",
                "task_cid": "task:cid:276",
                "goal_id": "G12.S1",
                "depends_on": ["REF-275"],
                "actor_id": "agent:beta",
                "changed_ast_scopes": ["symbol:cid:compiler"],
                "acceptance_criteria": [
                    {
                        "kind": "test",
                        "validation_commands": ["pytest test_compiler.py"],
                        "statement": "JSON and DuckDB agree",
                    }
                ],
                "validation_commands": ["pytest test_compiler.py"],
            },
        ],
        "ast_records": [
            {
                "symbol_cid": "symbol:cid:contracts",
                "tree_cid": "tree:candidate",
                "task_cid": "task:cid:275",
                "symbol": "FormalWorkPlan",
            },
            {
                "symbol_cid": "symbol:cid:compiler",
                "tree_cid": "tree:candidate",
                "task_id": "REF-276",
                "symbol": "FormalPlanCompiler",
            },
        ],
        "proof_policy": {
            "policy_cid": "policy:cid:g12",
            "minimum_code_assurance": "candidate",
            "freshness_seconds": 3600,
            "fallback_check_ids": ["fallback:pytest"],
        },
        "evidence_records": [
            {
                "evidence_cid": "evidence:cid:prior-contract-test",
                "task_cid": "task:cid:275",
                "kind": "test",
            }
        ],
    }


def _materialized(tmp_path: Path) -> DuckDBTaskSource:
    source = DuckDBTaskSource(tmp_path / "workflow.duckdb")
    source.materialize(_source())
    return source


def _prompt_graph() -> PromptGoalGraph:
    cid = lambda label: content_identity({"fixture": label})
    evidence = PromptEvidenceRecord(
        evidence_key="evidence:source",
        source_kind="directory_scan",
        artifact_cid=cid("artifact"),
        summary="The source was inspected.",
        repository_paths=("package/source.py",),
    )
    acceptance = PromptAcceptanceRecord(
        criterion_key="criterion:tests",
        criterion="The focused tests pass.",
        evidence_cids=(evidence.evidence_cid,),
        validation_keys=("validation:pytest",),
    )
    goal = PromptGoalRecord(
        goal_key="goal:root",
        parent_goal_cid="",
        dependency_goal_cids=(),
        title="Improve the source",
        objective="Make one bounded improvement.",
        rationale="The scan located the implementation.",
        scope_paths=("package",),
        acceptance=(acceptance,),
        evidence_cids=(evidence.evidence_cid,),
    )
    task = PromptTaskRecord(
        task_key="task:source",
        goal_cid=goal.goal_cid,
        dependency_task_cids=(),
        objective="Implement the bounded improvement.",
        rationale="The root goal requires this change.",
        scope_paths=("package/source.py",),
        outputs=(
            PromptOutputRecord(
                path="package/source.py",
                effect="modify",
                media_type="text/x-python",
            ),
        ),
        validations=(
            PromptValidationRecord(
                validation_key="validation:pytest",
                argv=("python", "-m", "pytest", "test_source.py"),
                policy_cid=cid("validation-policy"),
            ),
        ),
        acceptance=(acceptance,),
        evidence_cids=(evidence.evidence_cid,),
        policy_roots=(cid("policy"),),
        predicted_files=("package/source.py",),
    )
    return PromptGoalGraph(
        request_cid=cid("request"),
        scan_cid=cid("scan"),
        program_root=cid("program"),
        policy_roots=(cid("policy"),),
        goals=(goal,),
        tasks=(task,),
        evidence=(evidence,),
    )


def test_atomic_projection_has_all_tables_and_recompiles_losslessly(
    tmp_path: Path,
) -> None:
    original = FormalPlanCompiler().compile(_source())
    source = _materialized(tmp_path)

    snapshot = source.snapshot()
    integrity = source.validate_integrity()
    recompiled = source.recompile_formal_plan()

    assert snapshot.source_schema == DUCKDB_TASK_SOURCE_SCHEMA
    assert snapshot.schema_version == 1
    assert snapshot.repository_tree_id == "tree:candidate"
    assert snapshot.goal_count == 1
    assert snapshot.task_count == 2
    assert snapshot.dependency_count == 1
    assert integrity.valid
    assert recompiled.status is CompilationStatus.COMPILED
    assert recompiled.plan_id == original.plan_id == snapshot.formal_plan_id
    assert recompiled.source_identity == original.source_identity
    expected_tables = {
        "workflow_metadata",
        "artifacts",
        "goals",
        "tasks",
        "task_dependencies",
        "task_outputs",
        "task_validations",
        "task_acceptance",
        "task_events",
        "materialization_receipts",
        "formal_plan_input_records",
        "formal_plan_input_metadata",
        "schema_migration_receipts",
    }
    connection = duckdb.connect(str(source.database_path), read_only=True)
    try:
        tables = {str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()}
        formal_metadata = dict(
            connection.execute(
                "SELECT field_name, field_value FROM formal_plan_input_metadata"
            ).fetchall()
        )
    finally:
        connection.close()
    assert expected_tables.issubset(tables)
    assert formal_metadata["repository_tree_id"] == "tree:candidate"
    assert formal_metadata["source_identity"] == original.source_identity


def test_prompt_graph_uses_canonical_plan_root_and_rich_projection(
    tmp_path: Path,
) -> None:
    graph = _prompt_graph()
    source = DuckDBTaskSource(tmp_path / "prompt.duckdb")
    receipt = source.materialize(
        graph,
        repository_tree_id="tree:prompt",
    )

    assert receipt["plan_root_cid"] == graph.plan_root_cid
    assert source.snapshot().plan_root_cid == graph.plan_root_cid
    assert source.get_task(graph.tasks[0].task_cid) is not None
    assert len(source.query("task_outputs")) == 1
    assert len(source.query("task_validations")) == 1
    assert len(source.query("task_acceptance")) == 1
    assert source.recompile_formal_plan().status is CompilationStatus.COMPILED


def test_bounded_snapshot_query_paging_and_ready_order(tmp_path: Path) -> None:
    source = _materialized(tmp_path)

    first = source.list_tasks(limit=1)
    second = source.list_tasks(cursor=first.next_cursor, limit=1)
    ready = source.ready_tasks(limit=10)

    assert [item.task_alias for item in first.tasks] == ["REF-275"]
    assert [item.task_alias for item in second.tasks] == ["REF-276"]
    assert [item.task_alias for item in ready.tasks] == ["REF-275"]
    assert source.get_task("task:cid:275") == source.get_task("REF-275")
    with pytest.raises(TaskSourceBoundsError):
        source.list_tasks(limit=MAX_QUERY_LIMIT + 1)
    with pytest.raises(TaskSourceConflictError):
        source.list_tasks(cursor=first.next_cursor[:-1] + "x", limit=1)


def test_cas_revision_events_watch_and_status_independent_identity(
    tmp_path: Path,
) -> None:
    source = _materialized(tmp_path)
    task = source.get_task("REF-275")
    assert task is not None
    original_cid = task.task_cid

    result = source.compare_and_set_status(
        task.task_cid,
        task.revision,
        "completed",
        {"validation": "passed"},
    )

    assert result.changed
    assert result.task.task_cid == original_cid
    assert result.task.revision == task.revision + 1
    assert source.get_task("REF-275").task_cid == original_cid  # type: ignore[union-attr]
    assert [item.task_alias for item in source.ready_tasks().tasks] == ["REF-276"]
    page = source.watch(cursor=0, timeout=0)
    assert page.cursor == 1
    assert page.events[0]["event_type"] == "status_changed"
    assert source.watch(cursor=page.cursor, timeout=0).timed_out
    with pytest.raises(TaskSourceConflictError):
        source.compare_and_set_status("REF-275", task.revision, "failed")


def test_write_preconditions_rollback_cas_and_event_inserts(tmp_path: Path) -> None:
    source = _materialized(tmp_path)
    task = source.get_task("REF-275")
    assert task is not None
    before = source.snapshot()
    calls: list[str] = []

    def reject() -> None:
        calls.append("checked")
        raise RuntimeError("write authority expired")

    with pytest.raises(RuntimeError, match="authority expired"):
        source.compare_and_set_status(
            task.task_cid,
            task.revision,
            "completed",
            write_precondition=reject,
        )
    with pytest.raises(RuntimeError, match="authority expired"):
        source.append_event(
            {
                "event_type": "test_precondition",
                "task_cid": task.task_cid,
                "value": "must not commit",
            },
            write_precondition=reject,
        )
    with pytest.raises(
        TaskSourceConflictError, match="status/revision precondition"
    ):
        source.append_event(
            {
                "event_type": "test_stale_task_precondition",
                "task_cid": task.task_cid,
            },
            expected_task_status="completed",
            expected_task_revision=task.revision,
        )

    after = source.snapshot()
    unchanged = source.get_task(task.task_cid)
    assert calls == ["checked", "checked"]
    assert unchanged is not None
    assert (unchanged.status, unchanged.revision) == (task.status, task.revision)
    assert (after.revision, after.event_cursor) == (
        before.revision,
        before.event_cursor,
    )


def test_writer_fencing_rejects_stale_and_concurrent_writers(tmp_path: Path) -> None:
    first = _materialized(tmp_path)
    second = DuckDBTaskSource(
        first.database_path, writer_id="writer:second", fencing_token=1
    )
    lease = second.acquire_writer("writer:second", expected_fencing_token=1)
    task = second.get_task("REF-275")
    assert task is not None and lease.fencing_token == 2

    with pytest.raises(TaskSourceConflictError):
        first.compare_and_set_status("REF-275", task.revision, "completed")
    changed = second.compare_and_set_status(
        "REF-275",
        task.revision,
        "completed",
        fencing_token=lease.fencing_token,
    )
    assert changed.changed


def test_multiprocess_reader_waits_for_writer_file_lock(tmp_path: Path) -> None:
    source = _materialized(tmp_path)
    context = mp.get_context("spawn")
    entered = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_duckdb_writer,
        args=(str(source.database_path), entered, release),
    )
    process.start()
    try:
        assert entered.wait(timeout=10.0)
        timer = threading.Timer(0.35, release.set)
        timer.start()
        started = time.monotonic()
        snapshot = source.snapshot()
        elapsed = time.monotonic() - started
        timer.join(timeout=2.0)
    finally:
        release.set()
        process.join(timeout=10.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    assert process.exitcode == 0
    assert snapshot.task_count == 2
    assert elapsed >= 0.25


def test_consistent_projection_validates_deduplicates_and_bounds_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _materialized(tmp_path)

    projection = source.read_consistent_projection(
        ["tasks", "goals", "tasks"]
    )

    assert tuple(projection.tables) == ("goals", "tasks")
    assert projection.row_counts == {"goals": 1, "tasks": 2}
    assert projection.snapshot.goal_count == projection.row_counts["goals"]
    assert projection.snapshot.task_count == projection.row_counts["tasks"]
    assert [row["task_cid"] for row in projection.tables["tasks"]] == sorted(
        row["task_cid"] for row in projection.tables["tasks"]
    )
    with pytest.raises(TypeError):
        projection.tables["tasks"][0]["status"] = "failed"  # type: ignore[index]
    with pytest.raises(TypeError):
        source.read_consistent_projection("tasks")  # type: ignore[arg-type]
    with pytest.raises(TaskSourceInjectionError):
        source.read_consistent_projection(["tasks; DROP TABLE tasks"])
    with pytest.raises(ValueError):
        source.read_consistent_projection([])

    monkeypatch.setitem(
        task_source_module._CONSISTENT_PROJECTION_TABLE_BOUNDS,
        "tasks",
        1,
    )
    with pytest.raises(TaskSourceBoundsError, match="projection bound is 1"):
        source.read_consistent_projection(["tasks"])


def test_consistent_projection_blocks_writer_and_captures_one_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _materialized(tmp_path)
    projection_paused = threading.Event()
    release_projection = threading.Event()
    original_reader = source._consistent_projection_table

    def controlled_reader(connection: object, table: str):
        result = original_reader(connection, table)
        if table == "task_events":
            projection_paused.set()
            if not release_projection.wait(timeout=10.0):
                raise RuntimeError("writer did not attempt the concurrent update")
        return result

    monkeypatch.setattr(source, "_consistent_projection_table", controlled_reader)
    projections: list[ConsistentTaskSourceProjection] = []
    projection_errors: list[BaseException] = []

    def read_projection() -> None:
        try:
            projections.append(
                source.read_consistent_projection(["tasks", "task_events"])
            )
        except BaseException as exc:
            projection_errors.append(exc)

    reader = threading.Thread(target=read_projection, daemon=True)
    reader.start()
    context = mp.get_context("spawn")
    writer_attempted = context.Event()
    writer_finished = context.Event()
    writer: mp.Process | None = None
    try:
        assert projection_paused.wait(timeout=10.0)
        writer = context.Process(
            target=_complete_task_writer,
            args=(
                str(source.database_path),
                writer_attempted,
                writer_finished,
            ),
        )
        writer.start()
        assert writer_attempted.wait(timeout=10.0)
        assert not writer_finished.wait(timeout=0.3)
    finally:
        release_projection.set()
        reader.join(timeout=10.0)
        if writer is not None:
            writer.join(timeout=10.0)
            if writer.is_alive():
                writer.terminate()
                writer.join(timeout=5.0)

    assert not reader.is_alive()
    assert projection_errors == []
    assert writer is not None and writer.exitcode == 0
    assert writer_finished.is_set()
    assert len(projections) == 1
    projection = projections[0]
    assert projection.revision == 1
    assert projection.event_cursor == 0
    assert projection.row_counts == {"task_events": 0, "tasks": 2}
    projected_task = next(
        row
        for row in projection.tables["tasks"]
        if row["task_alias"] == "REF-275"
    )
    assert projected_task["status"] == "pending"
    assert projected_task["revision"] == 1

    current = source.snapshot()
    assert current.revision == 2
    assert current.event_cursor == 1
    assert source.get_task("REF-275").status == "completed"  # type: ignore[union-attr]


def test_identical_replay_is_noop_and_population_drift_is_rejected(
    tmp_path: Path,
) -> None:
    source = _materialized(tmp_path)
    before = source.snapshot()
    replay = source.materialize(_source())
    changed_source = _source()
    tasks = changed_source["taskboard"]
    assert isinstance(tasks, list)
    tasks.pop()

    assert replay["changed"] is False
    assert source.snapshot() == before
    with pytest.raises(TaskSourceConflictError):
        source.materialize(changed_source)


def test_atomic_install_recovers_after_crash_before_rename(tmp_path: Path) -> None:
    path = tmp_path / "recover.duckdb"
    source = DuckDBTaskSource(path)

    def crash(point: str) -> None:
        if point == "before_install":
            raise RuntimeError("simulated process crash")

    with pytest.raises(RuntimeError, match="simulated"):
        source.materialize(_source(), fault_injector=crash)
    assert not path.exists()

    recovered = DuckDBTaskSource(path)
    assert recovered.snapshot().task_count == 2
    assert recovered.validate_integrity().valid


def test_partial_corrupt_and_foreign_databases_fail_closed(tmp_path: Path) -> None:
    partial = tmp_path / "partial.duckdb"
    connection = duckdb.connect(str(partial))
    connection.execute("CREATE TABLE tasks(task_cid VARCHAR)")
    connection.close()
    with pytest.raises(TaskSourceIntegrityError, match="partial"):
        DuckDBTaskSource(partial).snapshot()

    source = _materialized(tmp_path / "valid")
    snapshot = source.snapshot()
    with pytest.raises(TaskSourceIntegrityError, match="foreign plan root"):
        DuckDBTaskSource(
            source.database_path, expected_plan_root_cid="plan:foreign"
        ).snapshot()
    with pytest.raises(TaskSourceIntegrityError, match="foreign repository"):
        DuckDBTaskSource(
            source.database_path, expected_repository_tree_id="tree:foreign"
        ).snapshot()
    assert snapshot.plan_root_cid


def test_closed_query_api_rejects_identifier_injection_but_values_are_bound(
    tmp_path: Path,
) -> None:
    source_record = _source()
    tasks = source_record["taskboard"]
    assert isinstance(tasks, list)
    tasks[0]["title"] = "'); DROP TABLE tasks; --"
    source = DuckDBTaskSource(tmp_path / "safe.duckdb")
    source.materialize(source_record)

    with pytest.raises(TaskSourceInjectionError):
        source.query("tasks; DROP TABLE tasks")
    with pytest.raises(TaskSourceInjectionError):
        source.get_task("REF-275'; DROP TABLE tasks; --")
    assert source.snapshot().task_count == 2
    assert source.get_task("REF-275") is not None


def test_application_integrity_detects_manual_key_edge_and_json_corruption(
    tmp_path: Path,
) -> None:
    source = _materialized(tmp_path)
    connection = duckdb.connect(str(source.database_path))
    connection.execute(
        "UPDATE tasks SET goal_cid = 'goal:missing' WHERE task_alias = 'REF-275'"
    )
    connection.close()

    with pytest.raises(TaskSourceIntegrityError):
        source.validate_integrity()


def test_migration_preview_noop_receipt_and_rollback_are_identity_bound(
    tmp_path: Path,
) -> None:
    source = _materialized(tmp_path)
    preview = source.preview_migration()
    receipt = source.migrate(preview)
    rollback = source.rollback_migration(receipt)

    assert preview.supported
    assert not preview.changed
    assert not receipt.changed
    assert rollback.rolled_back
    assert source.validate_integrity().valid


def test_status_dependent_duplicate_task_identity_is_rejected(tmp_path: Path) -> None:
    record = _source()
    tasks = record["taskboard"]
    assert isinstance(tasks, list)
    duplicate = dict(tasks[0])
    duplicate["task_id"] = "REF-999"
    duplicate["task_cid"] = "task:cid:999"
    duplicate["status"] = "completed"
    tasks.append(duplicate)

    with pytest.raises(
        TaskSourceIntegrityError,
        match="differ only by mutable status or aliases",
    ):
        DuckDBTaskSource(tmp_path / "status-identity.duckdb").materialize(record)
