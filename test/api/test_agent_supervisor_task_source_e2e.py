from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

duckdb = pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.duckdb_task_source import (
    DuckDBTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.markdown_task_source import (
    MarkdownTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.prompt_workflow import (
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.task_source import (
    MAX_QUERY_LIMIT,
    TaskSourceBoundsError,
    TaskSourceConflictError,
    TaskSourceIntegrityError,
    open_task_source,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)
from test.api.test_agent_supervisor_prompt_plan_admission import _graph_fixture


def _canonical_fixture():
    _workflow, scan, graph = _graph_fixture()
    template = graph.tasks[0]
    first = replace(
        template,
        task_key="FIX-001",
        objective="Implement the first fixture task.",
        parallel_lane="fixture-first",
    )
    second_output = replace(
        template.outputs[0],
        path="pkg/fixture_second.py",
    )
    second = replace(
        template,
        task_key="FIX-002",
        dependency_task_cids=(first.task_cid,),
        objective="Implement the dependent fixture task.",
        outputs=(second_output,),
        predicted_files=("pkg/fixture_second.py",),
        parallel_lane="fixture-second",
    )
    fixture = replace(graph, tasks=(first, second))
    receipt = SimpleNamespace(
        candidate_plan_cid=fixture.plan_root_cid,
        topological_task_cids=(first.task_cid, second.task_cid),
        formal_plan_id="formal:fixture",
        ir_receipt_id="ir:fixture",
        policy_id="policy:fixture",
        repository_tree_id=scan.dirty_worktree_root,
        topology_id="topology:fixture",
    )
    admitted_root = prompt_workflow_cid(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "admitted-prompt-plan@1"
            ),
            "candidate_plan_cid": fixture.plan_root_cid,
            "formal_plan_id": receipt.formal_plan_id,
            "ir_receipt_id": receipt.ir_receipt_id,
            "policy_id": receipt.policy_id,
            "repository_tree_id": receipt.repository_tree_id,
            "task_cids": sorted((first.task_cid, second.task_cid)),
            "topology_id": receipt.topology_id,
        }
    )
    admission = SimpleNamespace(
        admitted=True,
        admitted_graph=fixture,
        plan_root_cid=admitted_root,
        task_cids=tuple(sorted((first.task_cid, second.task_cid))),
        receipt=receipt,
    )
    aliases = {
        first.task_cid: first.task_key,
        second.task_cid: second.task_key,
    }
    return fixture, admission, aliases, scan.dirty_worktree_root


def _sources(tmp_path: Path):
    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown_backend = MarkdownTaskSource(
        tmp_path / "fixture.md",
        root=tmp_path,
        task_prefix="FIX",
        board_namespace="fixture",
    )
    markdown_backend.materialize(admission, aliases=aliases)
    duckdb_backend = DuckDBTaskSource(tmp_path / "fixture.duckdb")
    duckdb_backend.materialize(graph, repository_tree_id=tree_id)
    return open_task_source(markdown_backend), open_task_source(duckdb_backend)


def _canonical_graph(source) -> list[tuple[str, str, tuple[str, ...]]]:
    snapshot = source.snapshot(include_tasks=True)
    return [
        (task.task_id, task.status, task.dependency_task_ids)
        for task in snapshot.tasks
    ]


def _exercise_lifecycle(source):
    ready_order: list[tuple[str, ...]] = []
    claims: list[str] = []
    retries: list[str] = []
    completions: list[str] = []

    ready_order.append(tuple(task.task_id for task in source.ready_set().tasks))
    first = source.get("FIX-001")
    assert first is not None
    claim = source.compare_and_swap_status(
        first.task_id,
        expected_status=first.status,
        new_status="in_progress",
        expected_revision=first.revision,
        receipt={"attempt": 1},
    )
    claims.append(claim.task.task_id)
    retry = source.compare_and_swap_status(
        first.task_id,
        expected_status="in_progress",
        new_status="ready",
        expected_revision=claim.task.revision,
        receipt={"attempt": 1, "outcome": "retry"},
    )
    retries.append(retry.task.task_id)
    second_claim = source.compare_and_swap_status(
        first.task_id,
        expected_status="ready",
        new_status="in_progress",
        expected_revision=retry.task.revision,
        receipt={"attempt": 2},
    )
    claims.append(second_claim.task.task_id)
    first_done = source.compare_and_swap_status(
        first.task_id,
        expected_status="in_progress",
        new_status="completed",
        expected_revision=second_claim.task.revision,
        receipt={"attempt": 2, "outcome": "completed"},
    )
    completions.append(first_done.task.task_id)
    ready_order.append(tuple(task.task_id for task in source.ready_set().tasks))

    second = source.get("FIX-002")
    assert second is not None
    second_claim = source.compare_and_swap_status(
        second.task_id,
        expected_status=second.status,
        new_status="in_progress",
        expected_revision=second.revision,
        receipt={"attempt": 1},
    )
    claims.append(second_claim.task.task_id)
    second_done = source.compare_and_swap_status(
        second.task_id,
        expected_status="in_progress",
        new_status="completed",
        expected_revision=second_claim.task.revision,
        receipt={"attempt": 1, "outcome": "completed"},
    )
    completions.append(second_done.task.task_id)
    ready_order.append(tuple(task.task_id for task in source.ready_set().tasks))
    terminal = source.snapshot()
    return {
        "ready_order": ready_order,
        "claims": claims,
        "retries": retries,
        "completions": completions,
        "terminal": terminal.terminal,
        "graph": [
            (task_id, status, dependencies)
            for task_id, status, dependencies in _canonical_graph(source)
        ],
    }


def _daemon(tmp_path: Path, source) -> PortalImplementationDaemon:
    tmp_path.mkdir(parents=True, exist_ok=True)
    daemon = PortalImplementationDaemon(
        task_source=source,
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        worktree_pool_enabled=False,
        validation_cache_dir=tmp_path / "validation-cache",
        merge_queue_dir=tmp_path / "merge-queue",
    )
    daemon._consume_one_merge_candidate = lambda: None  # type: ignore[method-assign]
    daemon._reconcile_failed_merges = lambda **_kwargs: []  # type: ignore[method-assign]
    daemon._cleanup_already_merged_worktrees = lambda: []  # type: ignore[method-assign]
    daemon._periodic_maintenance = lambda: None  # type: ignore[method-assign]
    return daemon


def test_same_fixture_has_identical_queries_claims_retries_and_terminal_graph(
    tmp_path: Path,
) -> None:
    markdown, database = _sources(tmp_path)

    assert [task.task_id for task in markdown.query(limit=1).tasks] == ["FIX-001"]
    assert [task.task_id for task in database.query(limit=1).tasks] == ["FIX-001"]
    with pytest.raises(TaskSourceBoundsError):
        markdown.query(limit=MAX_QUERY_LIMIT + 1)
    with pytest.raises(TaskSourceBoundsError):
        database.query(limit=MAX_QUERY_LIMIT + 1)

    markdown_result = _exercise_lifecycle(markdown)
    database_result = _exercise_lifecycle(database)

    assert markdown_result == database_result == {
        "ready_order": [("FIX-001",), ("FIX-002",), ()],
        "claims": ["FIX-001", "FIX-001", "FIX-002"],
        "retries": ["FIX-001"],
        "completions": ["FIX-001", "FIX-002"],
        "terminal": True,
        "graph": [
            ("FIX-001", "completed", ()),
            ("FIX-002", "completed", ("FIX-001",)),
        ],
    }


def test_daemon_consumes_either_source_and_receipts_bind_source_identity(
    tmp_path: Path,
) -> None:
    markdown, database = _sources(tmp_path)
    observed: list[tuple[str, str, str]] = []
    for name, source in (("markdown", markdown), ("duckdb", database)):
        daemon = _daemon(tmp_path / name, source)
        first = daemon.run_once()
        completed = daemon._mark_task_completed_in_todo("FIX-001")
        second = daemon.run_once()

        assert first["active_task_id"] == "FIX-001"
        assert second["active_task_id"] == "FIX-002"
        assert completed["completion_receipts"][0][
            "task_source_identity"
        ] == source.identity.to_dict()
        checkpoint = daemon._runtime_checkpoint
        assert checkpoint["task_source_identity"] == source.identity.to_dict()
        observed.append(
            (
                first["active_task_id"],
                second["active_task_id"],
                completed["updated_task_ids"][0],
            )
        )
    assert observed == [
        ("FIX-001", "FIX-002", "FIX-001"),
        ("FIX-001", "FIX-002", "FIX-001"),
    ]
    assert not list(tmp_path.rglob("*duckdb*.md"))


def test_stale_revision_cursor_corruption_foreign_root_and_swap_fail_closed(
    tmp_path: Path,
) -> None:
    markdown, database = _sources(tmp_path)
    database_daemon = _daemon(tmp_path / "corrupt-runtime", database)
    first_page = markdown.query(limit=1)
    first = markdown.get("FIX-001")
    assert first is not None
    markdown.compare_and_swap_status(
        first.task_id,
        expected_status=first.status,
        new_status="in_progress",
        expected_revision=first.revision,
    )
    with pytest.raises(TaskSourceConflictError, match="stale"):
        markdown.query(cursor=first_page.next_cursor, limit=1)
    with pytest.raises(TaskSourceConflictError, match="stale"):
        markdown.compare_and_swap_status(
            first.task_id,
            expected_status=first.status,
            new_status="completed",
            expected_revision=first.revision,
        )

    with pytest.raises(TaskSourceIntegrityError, match="foreign plan root"):
        open_task_source(database.backend, expected_root_id="plan:foreign")

    connection = duckdb.connect(str(database.path))
    connection.execute(
        "UPDATE tasks SET goal_cid = 'goal:foreign' WHERE task_alias = 'FIX-001'"
    )
    connection.close()
    assert not database.check_integrity().valid
    invalid = database_daemon.run_once()
    assert invalid["reason"] == "task_source_invalid"
    assert invalid["task_source_identity"] == database.pinned_identity.to_dict()

    replacement = tmp_path / "replacement.duckdb"
    replacement.write_bytes(b"foreign backend bytes")
    shutil.move(replacement, database.path)
    with pytest.raises(TaskSourceIntegrityError):
        database.snapshot()


def test_checkpoint_rejects_mid_run_backend_swap(tmp_path: Path) -> None:
    markdown, database = _sources(tmp_path)
    runtime = tmp_path / "runtime"
    first_daemon = _daemon(runtime, markdown)
    assert first_daemon.run_once()["active_task_id"] == "FIX-001"

    with pytest.raises(
        TaskSourceIntegrityError,
        match="differs from the durable checkpoint",
    ):
        _daemon(runtime, database)

    with pytest.raises(
        TaskSourceIntegrityError,
        match="mode differs from the durable checkpoint",
    ):
        PortalImplementationDaemon(
            todo_path=markdown.path,
            state_path=runtime / "state.json",
            strategy_path=runtime / "strategy.json",
            events_path=runtime / "events.jsonl",
            repo_root=runtime,
            worktree_pool_enabled=False,
            validation_cache_dir=runtime / "validation-cache",
            merge_queue_dir=runtime / "merge-queue",
        )
