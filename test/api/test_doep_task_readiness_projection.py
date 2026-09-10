"""Read-only readiness diagnostics use the canonical source, not todo status."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon
from test.api.test_agent_supervisor_database_implementation_daemon import (
    _open_daemon,
    _population,
)


def _assert_diagnostic_scope(projection: Mapping[str, object]) -> None:
    assert projection["projection_authority"] is False
    assert projection["readiness_scope"] == (
        "canonical_task_source_dependencies_and_cooldowns"
    )
    assert projection["eligible_ready_count_scope"] == "source_readiness_only"
    assert projection["claim_eligibility_known"] is False
    assert projection["claim_eligible_count"] is None


def test_projection_separates_dependency_ready_work_from_todo_backlog(
    tmp_path: Path,
) -> None:
    providers: list[str] = []
    effects: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=providers, effect_calls=effects)
    try:
        population = _population(2)
        population["tasks"][1]["dependencies"] = ["task:cid:001"]
        daemon.materialize_population(population)
        before = daemon.task_source.snapshot()
        before_claims = daemon.coordinator.coordination_registry_projection()
        source_ready = daemon.task_source.ready_tasks()
        assert [task.task_alias for task in source_ready.tasks] == ["DQP-T001"]

        projected = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result={},
        )
        assert projected["projection_complete"] is True
        assert projected["ready_task_ids"] == ["DQP-T001"]
        assert projected["todo_task_ids"] == ["DQP-T001", "DQP-T002"]
        assert projected["todo_count"] == 2
        assert projected["ready_count"] == projected["eligible_ready_count"] == 1
        _assert_diagnostic_scope(projected)
        assert daemon.task_source.snapshot() == before
        after_claims = daemon.coordinator.coordination_registry_projection()
        assert after_claims["counts"] == before_claims["counts"]
        assert after_claims["task_claims"] == before_claims["task_claims"] == []
        assert providers == effects == []

        # Only the ordinary test daemon's validated execution changes readiness.
        daemon.run_once()
        assert providers == effects == ["task:cid:001"]
        projected = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result={},
        )
        assert projected["ready_task_ids"] == ["DQP-T002"]
        assert projected["todo_task_ids"] == ["DQP-T002"]
        assert projected["todo_count"] == projected["ready_count"] == 1
        _assert_diagnostic_scope(projected)
    finally:
        daemon.close()


def test_projection_observes_durable_source_cooldown_without_reselecting(
    tmp_path: Path,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        daemon.task_source.record_queue_backoff(
            task_cid="task:cid:001", delay_ms=60_000, reason="test_cooldown",
        )
        before = daemon.task_source.snapshot()
        assert daemon.task_source.ready_tasks().tasks == ()
        projected = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result={},
        )
        assert projected["projection_complete"] is True
        assert projected["todo_count"] == 1
        assert projected["todo_task_ids"] == ["DQP-T001"]
        assert projected["ready_task_ids"] == []
        assert projected["ready_count"] == projected["eligible_ready_count"] == 0
        assert daemon.task_source.snapshot() == before
        _assert_diagnostic_scope(projected)
    finally:
        daemon.close()


def test_dependency_blocked_backlog_matches_idle_pass_without_claim_writes(
    tmp_path: Path,
) -> None:
    providers: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=providers)
    try:
        population = _population(3)
        population["tasks"][0]["status"] = "blocked"
        population["tasks"][1]["dependencies"] = ["task:cid:001"]
        population["tasks"][2]["dependencies"] = ["task:cid:002"]
        daemon.materialize_population(population)
        assert daemon.task_source.ready_tasks().tasks == ()
        idle_pass = daemon.run_once()
        assert idle_pass["selection_idle_reason"] == "no_ready_tasks"
        before = daemon.task_source.snapshot()
        projected = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result=idle_pass,
        )
        assert projected["projection_complete"] is True
        assert projected["ready_count"] == projected["eligible_ready_count"] == 0
        assert projected["ready_task_ids"] == []
        assert projected["todo_count"] == 2
        assert projected["todo_task_ids"] == ["DQP-T002", "DQP-T003"]
        assert projected["task_count"] == 3
        assert projected["completed_count"] == 0
        assert projected["blocked_task_ids"] == ["DQP-T001"]
        _assert_diagnostic_scope(projected)
        assert daemon.task_source.snapshot() == before
        assert daemon.coordinator.coordination_registry_projection()["counts"][
            "task_claims"
        ] == 0
        assert daemon.list_running_attempts() == []
        assert providers == []
    finally:
        daemon.close()


def test_source_ready_manual_task_is_not_reported_as_claim_eligible(
    tmp_path: Path,
) -> None:
    providers: list[str] = []
    daemon = _open_daemon(tmp_path, provider_calls=providers)
    try:
        population = _population(1)
        population["tasks"][0]["completion"] = "manual"
        daemon.materialize_population(population)
        assert len(daemon.task_source.ready_tasks().tasks) == 1
        ordinary_pass = daemon.run_once()
        assert ordinary_pass["selection_idle_reason"] == "no_ready_tasks"
        before = daemon.task_source.snapshot()
        projected = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result=ordinary_pass,
        )
        assert projected["ready_count"] == projected["eligible_ready_count"] == 1
        assert projected["todo_count"] == 1
        _assert_diagnostic_scope(projected)
        assert daemon.task_source.snapshot() == before
        assert daemon.coordinator.coordination_registry_projection()["counts"][
            "task_claims"
        ] == 0
        assert daemon.list_running_attempts() == []
        assert providers == []
    finally:
        daemon.close()


@pytest.mark.parametrize("failure", ["unavailable", "revision", "cursor", "foreign"])
def test_unbound_ready_read_cannot_leave_a_previous_terminal_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    daemon = _open_daemon(tmp_path)
    path = tmp_path / "projection.json"
    try:
        daemon.materialize_population(_population(1))
        daemon.run_once()
        idle_pass = daemon.run_once()
        previous = daemon.materialize_task_state_compatibility_projection(
            state_path=path, pass_result=idle_pass,
        )
        assert previous["projection_complete"] is True
        assert previous["implementation_in_progress"] is False
        original = daemon.task_source.ready_tasks

        def broken_ready(**kwargs):
            page = original(**kwargs)
            if failure == "unavailable":
                raise RuntimeError("ready query unavailable")
            if failure == "revision":
                return replace(page, revision=page.revision + 1)
            if failure == "cursor":
                return replace(page, next_cursor="more")
            return replace(page, tasks=(SimpleNamespace(task_cid="foreign"),))

        monkeypatch.setattr(daemon.task_source, "ready_tasks", broken_ready)
        failed = daemon.materialize_task_state_compatibility_projection(
            state_path=path, pass_result=idle_pass,
        )
        assert failed["projection_complete"] is False
        assert failed["implementation_in_progress"] is True
        assert failed["readiness_scope"] == "unknown"
        assert failed["todo_count"] is None
        assert failed["claim_eligibility_known"] is False
        assert failed["claim_eligible_count"] is None
        assert json.loads(path.read_text())["projection_complete"] is False
    finally:
        daemon.close()


def test_projection_rejects_ready_scan_that_could_have_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(2))
        monkeypatch.setattr(implementation_daemon, "TASK_SOURCE_QUERY_LIMIT", 1)
        failed = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result={},
        )
        assert failed["projection_complete"] is False
        assert failed["implementation_in_progress"] is True
    finally:
        daemon.close()


def test_projection_rechecks_source_after_ready_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        original = daemon.task_source.snapshot
        calls = []

        def changed_snapshot():
            snapshot = original()
            calls.append(snapshot)
            return snapshot if len(calls) == 1 else replace(
                snapshot, projection_cid="changed-after-ready",
            )

        monkeypatch.setattr(daemon.task_source, "snapshot", changed_snapshot)
        failed = daemon.materialize_task_state_compatibility_projection(
            state_path=tmp_path / "projection.json", pass_result={},
        )
        assert failed["projection_complete"] is False
        assert failed["implementation_in_progress"] is True
    finally:
        daemon.close()
