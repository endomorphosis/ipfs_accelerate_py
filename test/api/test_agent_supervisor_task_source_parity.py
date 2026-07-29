from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import DuckDBTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import MarkdownTaskSource
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    CanonicalProjectionSnapshot,
    DualTaskSource,
    DualTaskSourcePartialError,
    TaskSourceIntegrityError,
    canonical_projection_snapshot,
    compare_task_source_projections,
    open_task_source,
    rebuild_task_source_projection,
)
from test.api.test_agent_supervisor_task_source_e2e import _canonical_fixture


def _sources(tmp_path: Path):
    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown_backend = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="FIX",
        board_namespace="fixture",
    )
    markdown_backend.materialize(admission, aliases=aliases)
    database_backend = DuckDBTaskSource(tmp_path / "tasks.duckdb")
    database_backend.materialize(graph, repository_tree_id=tree_id)
    return (
        graph,
        tree_id,
        open_task_source(markdown_backend),
        open_task_source(database_backend),
    )


def _event_count(source) -> int:
    return len(canonical_projection_snapshot(source).events)


def test_exact_canonical_parity_and_bidirectional_identity_round_trip(
    tmp_path: Path,
) -> None:
    _graph, _tree_id, markdown, database = _sources(tmp_path)

    markdown_snapshot = canonical_projection_snapshot(markdown)
    database_snapshot = canonical_projection_snapshot(database)
    parity = compare_task_source_projections(markdown_snapshot, database_snapshot)

    assert parity.valid
    assert parity.promotion_allowed
    assert parity.mismatches == ()
    assert markdown_snapshot.parity_dict() == database_snapshot.parity_dict()
    assert markdown_snapshot.plan_root == database_snapshot.plan_root
    assert dict(markdown_snapshot.task_aliases) == dict(database_snapshot.task_aliases)
    assert dict(markdown_snapshot.goal_aliases) == dict(database_snapshot.goal_aliases)
    assert dict(markdown_snapshot.task_records) == dict(database_snapshot.task_records)
    assert dict(markdown_snapshot.goal_records) == dict(database_snapshot.goal_records)
    assert dict(markdown_snapshot.dependencies) == dict(database_snapshot.dependencies)
    assert markdown_snapshot.ready_task_cids == database_snapshot.ready_task_cids

    to_database = rebuild_task_source_projection(
        markdown_snapshot,
        tmp_path / "round-trip.duckdb",
        kind="duckdb",
    )
    database_round_trip = canonical_projection_snapshot(
        open_task_source(tmp_path / "round-trip.duckdb")
    )
    assert to_database.parity.valid
    assert to_database.source_snapshot_id == to_database.target_snapshot_id
    assert database_round_trip.snapshot_id == markdown_snapshot.snapshot_id

    to_markdown = rebuild_task_source_projection(
        database_round_trip,
        tmp_path / "round-trip.md",
        kind="markdown",
    )
    markdown_round_trip = canonical_projection_snapshot(
        open_task_source(tmp_path / "round-trip.md")
    )
    assert to_markdown.parity.valid
    assert to_markdown.source_snapshot_id == to_markdown.target_snapshot_id
    assert markdown_round_trip.snapshot_id == database_round_trip.snapshot_id
    assert markdown_round_trip.to_dict() == markdown_snapshot.to_dict()


def test_status_transaction_preserves_cids_terminal_outcome_and_replay_is_noop(
    tmp_path: Path,
) -> None:
    _graph, _tree_id, markdown, database = _sources(tmp_path)
    dual = DualTaskSource(markdown, database)
    initial = dual.canonical_snapshot()
    initial_cids = initial.task_cids

    first = dual.get("FIX-001")
    assert first is not None
    claim = dual.compare_and_swap_status(
        first.task_id,
        expected_status=first.status,
        new_status="in_progress",
        expected_revision=first.revision,
        receipt={"attempt": 1, "effect_id": "effect:first"},
    )
    assert claim.changed
    assert claim.task.task_cid == first.task_cid
    assert dual.canonical_snapshot().task_cids == initial_cids
    before_replay = (_event_count(markdown), _event_count(database))

    replay = dual.compare_and_swap_status(
        first.task_id,
        expected_status=first.status,
        new_status="in_progress",
        expected_revision=first.revision,
        receipt={"attempt": 1, "effect_id": "effect:first"},
    )
    assert not replay.changed
    assert (_event_count(markdown), _event_count(database)) == before_replay

    first = dual.get("FIX-001")
    assert first is not None
    dual.compare_and_swap_status(
        first.task_id,
        expected_status=first.status,
        new_status="completed",
        expected_revision=first.revision,
        receipt={"attempt": 1, "outcome": "completed"},
    )
    second = dual.get("FIX-002")
    assert second is not None
    dual.compare_and_swap_status(
        second.task_id,
        expected_status=second.status,
        new_status="in_progress",
        expected_revision=second.revision,
        receipt={"attempt": 1},
    )
    second = dual.get("FIX-002")
    assert second is not None
    dual.compare_and_swap_status(
        second.task_id,
        expected_status=second.status,
        new_status="completed",
        expected_revision=second.revision,
        receipt={"attempt": 1, "outcome": "completed"},
    )

    terminal = dual.canonical_snapshot()
    assert terminal.task_cids == initial_cids
    assert terminal.terminal
    assert terminal.ready_task_cids == ()
    assert dual.parity().valid
    assert dict(terminal.statuses) == {
        task_cid: "completed" for task_cid in terminal.task_cids
    }


def test_crash_after_first_leg_resumes_exactly_once(tmp_path: Path) -> None:
    _graph, _tree_id, markdown, database = _sources(tmp_path)

    def crash(point: str) -> None:
        if point == "after_primary":
            raise RuntimeError("simulated process crash")

    crashing = DualTaskSource(markdown, database, fault_injector=crash)
    first = crashing.get("FIX-001")
    assert first is not None
    with pytest.raises(DualTaskSourcePartialError) as raised:
        crashing.compare_and_swap_status(
            first.task_id,
            expected_status=first.status,
            new_status="in_progress",
            expected_revision=first.revision,
            receipt={"attempt": 1},
        )
    assert raised.value.transaction_id
    assert markdown.get("FIX-001").status == "in_progress"
    assert database.get("FIX-001").status == "proposed"

    recovered = DualTaskSource(markdown, database)
    assert recovered.get("FIX-001").status == "in_progress"
    assert recovered.parity().valid
    assert _event_count(markdown) == _event_count(database) == 1

    journal = json.loads(recovered.journal_path.read_bytes())["payload"]
    transaction = journal["operations"][raised.value.transaction_id]
    assert transaction["state"] == "committed"


def test_concurrent_native_writer_drift_fails_closed_and_blocks_promotion(
    tmp_path: Path,
) -> None:
    _graph, _tree_id, markdown, database = _sources(tmp_path)
    dual = DualTaskSource(markdown, database, mode="migration")
    database_task = database.get("FIX-001")
    assert database_task is not None
    database.compare_and_swap_status(
        database_task.task_id,
        expected_status=database_task.status,
        new_status="in_progress",
        expected_revision=database_task.revision,
        receipt={"writer": "foreign"},
    )

    report = dual.parity()
    assert not report.valid
    assert {"statuses", "task_revisions", "events"} <= set(report.mismatches)
    assert not report.promotion_allowed
    with pytest.raises(TaskSourceIntegrityError, match="parity disagreement"):
        dual.promote()
    with pytest.raises(TaskSourceIntegrityError, match="parity disagreement"):
        dual.snapshot()


def test_corrupt_projection_is_quarantined_and_rebuilt_only_from_snapshot(
    tmp_path: Path,
) -> None:
    _graph, _tree_id, markdown, database = _sources(tmp_path)
    verified = canonical_projection_snapshot(markdown)
    connection = duckdb.connect(str(database.path))
    connection.execute(
        "UPDATE tasks SET goal_cid = 'goal:corrupt' WHERE task_alias = 'FIX-001'"
    )
    connection.close()
    assert not database.check_integrity().valid

    result = rebuild_task_source_projection(
        verified,
        database.path,
        kind="duckdb",
    )
    assert result.changed
    assert result.quarantine_path is not None
    assert result.quarantine_path.exists()
    assert result.parity.valid
    rebuilt = canonical_projection_snapshot(open_task_source(database.path))
    assert rebuilt.snapshot_id == verified.snapshot_id

    tampered = verified.to_dict()
    tampered["task_aliases"][verified.task_cids[0]] = "FORGED-001"
    with pytest.raises(TaskSourceIntegrityError, match="digest"):
        rebuild_task_source_projection(
            tampered,
            tmp_path / "forged.duckdb",
            kind="duckdb",
        )
    assert not (tmp_path / "forged.duckdb").exists()


def test_interrupted_migration_resumes_and_identical_replay_writes_nothing(
    tmp_path: Path,
) -> None:
    _graph, _tree_id, markdown, _database = _sources(tmp_path)
    verified = canonical_projection_snapshot(markdown)
    target = tmp_path / "interrupted.duckdb"

    def interrupt(point: str) -> None:
        if point == "after_install":
            raise RuntimeError("migration interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        rebuild_task_source_projection(
            verified,
            target,
            kind="duckdb",
            fault_injector=interrupt,
        )
    assert target.exists()

    resumed = rebuild_task_source_projection(verified, target, kind="duckdb")
    assert resumed.changed
    assert resumed.resumed
    assert resumed.parity.valid
    before = target.stat().st_mtime_ns
    replay = rebuild_task_source_projection(verified, target, kind="duckdb")
    after = target.stat().st_mtime_ns
    assert replay.replayed
    assert not replay.changed
    assert before == after


def test_parity_disagreement_names_exact_component_and_never_auto_promotes(
    tmp_path: Path,
) -> None:
    graph, tree_id, markdown, _database = _sources(tmp_path)
    changed_task = replace(
        graph.tasks[0],
        objective="A different immutable task objective.",
    )
    drifted_graph = replace(
        graph,
        tasks=(changed_task, *graph.tasks[1:]),
    )
    drifted_backend = DuckDBTaskSource(tmp_path / "drifted.duckdb")
    drifted_backend.materialize(drifted_graph, repository_tree_id=tree_id)
    drifted = open_task_source(drifted_backend)

    with pytest.raises(TaskSourceIntegrityError, match="task_records"):
        compare_task_source_projections(markdown, drifted).require_valid()

    dual = DualTaskSource(markdown, drifted, mode="migration")
    report = dual.parity()
    assert not report.valid
    assert "task_records" in report.mismatches
    assert not report.promotion_allowed
    with pytest.raises(TaskSourceIntegrityError, match="parity disagreement"):
        dual.promote(automatic=True)


def test_snapshot_restoration_rejects_unknown_fields_and_stale_digest(
    tmp_path: Path,
) -> None:
    _graph, _tree_id, markdown, _database = _sources(tmp_path)
    snapshot = canonical_projection_snapshot(markdown)
    restored = CanonicalProjectionSnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot

    unknown = snapshot.to_dict()
    unknown["unreviewed"] = True
    with pytest.raises(TaskSourceIntegrityError, match="unknown fields"):
        CanonicalProjectionSnapshot.from_dict(unknown)

    stale = snapshot.to_dict()
    stale["terminal"] = True
    with pytest.raises(TaskSourceIntegrityError, match="digest"):
        CanonicalProjectionSnapshot.from_dict(stale)
