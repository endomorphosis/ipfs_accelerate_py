"""The board stall watchdog releases blocked work and does not set it ready."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.board_stall_watchdog import (
    BoardStallWatchdog,
    decide_board_stall,
    file_lock_clear,
    lock_conflict,
    observe_tasks,
    resolution_repair_subagent,
    run_once,
    serve,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for board stall watchdog tests",
)


class _Repo:
    def __init__(self, records, release=None):
        self._records = list(records)
        self.release_calls = 0
        self._release = release or {
            "completed": [],
            "pending": [],
            "held": [],
            "spawned": [],
        }

    def list_tasks(self, *, limit, offset):
        return tuple(self._records[offset : offset + limit])

    def release_dependency_satisfied_blocks(self):
        self.release_calls += 1
        return self._release


def _blocked(alias: str) -> dict[str, str]:
    return {"task_alias": alias, "status": "blocked", "task_cid": alias.lower()}


def test_file_lock_clear_matches_only_an_empty_fuser_miss() -> None:
    assert file_lock_clear(1, "", "") is True
    assert file_lock_clear(0, "", "") is False
    assert file_lock_clear(1, "123", "") is False
    assert file_lock_clear(1, "", "busy") is False


def test_active_work_is_left_alone_and_a_finished_board_is_idle() -> None:
    active = observe_tasks(
        (_blocked("LIR-BLOCK"), {"task_alias": "LIR-RUN", "status": "in_progress"}),
        operator_hold_present=True,
    )
    decision = decide_board_stall(active)
    assert decision["action"] == "continue"
    assert decision["executor_start_allowed"] is False
    assert decision["set_ready"] is False
    assert decision["preserve_operator_hold"] is True

    finished = observe_tasks(
        ({"task_alias": "LIR-DONE", "status": "completed"},),
        operator_hold_present=False,
    )
    assert decide_board_stall(finished)["action"] == "idle_complete"


def test_idle_blocked_board_releases_once_and_then_records_the_stall() -> None:
    repo = _Repo((_blocked("LIR-1905"), _blocked("LIR-2102")))
    started: list[dict] = []

    def spawn(target, spec):
        started.append(dict(spec))
        return resolution_repair_subagent(target, spec)

    watchdog = BoardStallWatchdog()
    first = watchdog.tick(repo, operator_hold_present=True, spawn_repair=spawn)
    second = watchdog.tick(repo, operator_hold_present=True, spawn_repair=spawn)

    assert first["action"] == "start_repair_subagent"
    assert first["repair_spec"]["allowed"] == ["release_dependency_satisfied_blocks"]
    assert "set_ready" in first["repair_spec"]["forbidden"]
    assert first["repair_spec"]["executor_start_allowed"] is False
    assert first["repair_report"]["set_ready"] is False
    assert first["repair_report"]["hold_removed"] is False
    assert second["action"] == "stall_recorded"
    assert repo.release_calls == 2
    assert len(started) == 1


def test_progress_does_not_start_a_repair_subagent() -> None:
    repo = _Repo(
        (_blocked("LIR-GOAL"),),
        release={
            "completed": [],
            "spawned": [{"task_alias": "LIR-GOAL--00-fact"}],
        },
    )
    started: list[object] = []
    report = BoardStallWatchdog().tick(
        repo,
        operator_hold_present=False,
        spawn_repair=lambda _repo, spec: started.append(spec) or {"set_ready": False},
    )
    assert report["action"] == "progressed"
    assert started == []
    assert repo.release_calls == 1


def test_run_once_keeps_the_operator_hold_bytes(tmp_path: Path) -> None:
    hold = tmp_path / "watchdog.hold"
    original = b'{"schema":"legal-ir-owned-temporary-scientific-repair-launch-fence/v1"}\n'
    hold.write_bytes(original)
    report = run_once(
        _Repo((_blocked("LIR-2102"),)),
        watchdog=BoardStallWatchdog(),
        hold_path=hold,
    )
    assert report["preserve_operator_hold"] is True
    assert report["executor_start_allowed"] is False
    assert hold.read_bytes() == original


def test_a_catalogue_lock_is_busy_and_a_known_stall_does_not_write(
    tmp_path: Path,
) -> None:
    class _Locked:
        def list_tasks(self, *, limit, offset):
            raise RuntimeError(
                'Could not set lock on file "control.duckdb": '
                "Conflicting lock is held"
            )

        def release_dependency_satisfied_blocks(self):
            raise AssertionError("write release is not used while the catalogue is locked")

    assert lock_conflict(RuntimeError("Conflicting lock is held")) is True
    hold = tmp_path / "watchdog.hold"
    hold.write_text('{"reason":"preserve"}\n', encoding="utf-8")
    busy = serve(_Locked(), hold_path=hold, once=True)
    assert busy["action"] == "database_busy"
    assert busy["executor_start_allowed"] is False
    assert hold.read_text(encoding="utf-8").startswith("{")

    repo = _Repo((_blocked("LIR-1905"),))
    watchdog = BoardStallWatchdog()
    serve(repo, watchdog=watchdog, once=True)
    writes = repo.release_calls

    def observe():
        return observe_tasks(
            (_blocked("LIR-1905"),),
            operator_hold_present=False,
        )

    again = serve(
        repo,
        watchdog=watchdog,
        once=True,
        readonly_observe=observe,
    )
    assert again["action"] == "stall_recorded"
    assert repo.release_calls == writes


def test_a_tick_failure_stays_in_the_loop_without_clearing_the_hold(
    tmp_path: Path,
) -> None:
    class _Broken:
        def list_tasks(self, *, limit, offset):
            raise RuntimeError("catalogue busy")

    hold = tmp_path / "watchdog.hold"
    hold.write_text('{"reason":"preserve"}\n', encoding="utf-8")
    status = tmp_path / "status.json"
    report = serve(
        _Broken(),
        hold_path=hold,
        status_path=status,
        once=True,
    )
    assert report["action"] == "tick_failed"
    assert report["error"] == "RuntimeError"
    assert report["executor_start_allowed"] is False
    assert hold.read_text(encoding="utf-8").startswith("{")


def test_serve_stops_after_one_repair_and_keeps_the_hold(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    hold = tmp_path / "watchdog.hold"
    hold.write_text('{"reason":"preserve"}\n', encoding="utf-8")
    repo = _Repo((_blocked("LIR-2401"),))
    watchdog = BoardStallWatchdog()
    first = serve(
        repo,
        watchdog=watchdog,
        hold_path=hold,
        status_path=status,
        once=True,
    )
    second = serve(
        repo,
        watchdog=watchdog,
        hold_path=hold,
        status_path=status,
        once=True,
    )
    assert first["action"] == "start_repair_subagent"
    assert second["action"] == "stall_recorded"
    assert repo.release_calls == 2
    assert hold.read_text(encoding="utf-8").startswith("{")
    assert '"action": "stall_recorded"' in status.read_text(encoding="utf-8")
    assert '"executor_start_allowed": false' in status.read_text(encoding="utf-8")


def _manual_body(reason: str) -> dict[str, object]:
    return {
        "completion": "manual",
        "execution_packet_required": True,
        "blocked_reason": reason,
        "resource_stage": "validation",
        "required_stage_packets": ["validation"],
        "gold_or_teacher_admission_granted": False,
        "admitted": False,
        "formalized": False,
        "constitution_roundtrip_ok": False,
    }


def _catalogue(tmp_path: Path):
    repo = open_intent_repository(
        tmp_path / "control.duckdb",
        owner_id="owner:dependency-release",
        evidence_freshness_seconds=0,
    )
    repo.upsert_objective(
        objective_id="objective:scientific",
        objective_alias="scientific",
        title="Scientific",
    )
    repo.upsert_goal(
        goal_cid="goal:scientific",
        goal_alias="scientific-goal",
        title="Scientific goal",
        objective_id="objective:scientific",
    )
    repo.upsert_plan(
        plan_cid="plan:scientific",
        goal_cid="goal:scientific",
        plan_alias="scientific-plan",
        status="active",
    )
    return repo


def _task(repo, cid: str, alias: str, *, status: str, body: dict, dependencies=(), acceptance=None) -> None:
    repo.upsert_task(
        task_cid=cid,
        task_alias=alias,
        goal_cid="goal:scientific",
        plan_cid="plan:scientific",
        objective_id="objective:scientific",
        status=status,
        body=body,
        dependencies=list(dependencies),
        acceptance=acceptance
        or [
            {
                "criterion": "current validation evidence is already recorded",
                "evidence_kind": "validation",
            }
        ],
    )


def test_watchdog_creates_subgoals_for_an_idle_blocked_goal_without_setting_ready(
    tmp_path: Path,
) -> None:
    repo = _catalogue(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:goal",
        "LIR-GOAL",
        status="blocked",
        body=_manual_body("The high-level goal stays blocked until each clause is recorded."),
        dependencies=["task:predecessor"],
        acceptance=(
            "Every accepted component retains complete material meaning.",
            "Input context and output capacities stay separate.",
        ),
    )
    started: list[object] = []
    report = BoardStallWatchdog().tick(
        repo,
        operator_hold_present=True,
        spawn_repair=lambda _repo, spec: started.append(spec),
    )

    assert report["action"] == "progressed"
    assert started == []
    assert report["set_ready"] is False
    assert report["grant_admission"] is False
    assert report["executor_start_allowed"] is False
    spawned = {
        item["task_alias"]
        for item in repo.list_tasks(limit=20, offset=0)
        if item["status"] != "completed"
    }
    assert "LIR-GOAL--00-every-accepted-component-retains" in spawned
    assert "LIR-GOAL--01-input-context-output-capacities" in spawned
    for item in repo.list_tasks(limit=20, offset=0):
        assert item["status"] != "ready"
        assert item["body"].get("admitted") is not True
        assert item["body"].get("formalized") is not True
        assert item["body"].get("gold_or_teacher_admission_granted") is not True
        assert item["body"].get("constitution_roundtrip_ok") is not True


def test_watchdog_does_not_complete_a_runtime_budget_hold_with_stored_evidence(tmp_path: Path) -> None:
    """A fresh metadata receipt cannot let a repair subagent reset resources."""
    repo = open_intent_repository(tmp_path / "budget-hold.duckdb", evidence_freshness_seconds=0)
    repo.upsert_objective(objective_id="objective:hold", objective_alias="hold", title="hold")
    repo.upsert_goal(goal_cid="goal:hold", goal_alias="hold", title="hold", objective_id="objective:hold")
    repo.upsert_task(task_cid="task:done", task_alias="DONE", goal_cid="goal:hold", status="completed")
    body = {"completion": "manual", "execution_packet_required": True, "blocked_reason": "",
            "budget_exhausted": True, "retry_budget": 0, "admitted": False, "formalized": False}
    repo.upsert_task(task_cid="task:held", task_alias="HELD", goal_cid="goal:hold", status="blocked",
                     body=body, dependencies=["task:done"],
                     acceptance=[{"criterion": "current metadata result", "evidence_kind": "validation"}])
    repo.record_evidence(task_cid="task:held", evidence_kind="validation", digest="sha256:" + "a" * 64,
                         body={"admitted": False, "formalized": False})
    before = dict(repo.get_task("task:held"))
    report = BoardStallWatchdog().tick(repo, operator_hold_present=True,
                                     spawn_repair=lambda target, spec: resolution_repair_subagent(target, spec))
    assert report["action"] == "start_repair_subagent"
    assert report["repair_report"]["completed"] == []
    assert report["repair_report"]["executor_started"] is False
    assert report["repair_report"]["hold_removed"] is False
    assert dict(repo.get_task("task:held")) == before
