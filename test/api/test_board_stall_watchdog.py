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


# Pure modeled repository/clock controls. They grant no actual board, legal,
# native, storage, provider or operator-hold authority.
@pytest.mark.parametrize("entry", ["tick", "readonly_serve"])
def test_recorded_stall_rechecks_new_dependency_evidence_without_another_repair(
    monkeypatch, entry
):
    from ipfs_accelerate_py.agent_supervisor.task_sources import board_stall_watchdog as owner
    now = [100.0]
    monkeypatch.setattr(owner.time, "monotonic", lambda: now[0])
    repo = _Repo((_blocked("LIR-NEW-EVIDENCE"),))
    watchdog = BoardStallWatchdog()
    repairs = []
    def repair(target, spec):
        repairs.append(dict(spec))
        return resolution_repair_subagent(target, spec)
    first = watchdog.tick(repo, operator_hold_present=False, spawn_repair=repair)
    assert first["action"] == "start_repair_subagent"
    assert repo.release_calls == 2 and len(repairs) == 1
    # Evidence can become eligible without changing the blocked alias or hold.
    repo._release = {"completed": [{"task_alias": "LIR-NEW-EVIDENCE"}], "spawned": []}
    now[0] += 121.0
    if entry == "tick":
        second = watchdog.tick(repo, operator_hold_present=False, spawn_repair=repair)
    else:
        second = serve(repo, watchdog=watchdog, once=True, spawn_repair=repair,
            readonly_observe=lambda: observe_tasks(repo._records, operator_hold_present=False))
    assert second["action"] == "progressed"
    assert repo.release_calls == 3 and len(repairs) == 1
    assert second["set_ready"] is False and second["grant_admission"] is False
    assert second["executor_start_allowed"] is False


@pytest.mark.parametrize("entry", ["tick", "readonly_serve"])
def test_expired_unchanged_stall_rechecks_release_but_keeps_repair_deduplication(
    monkeypatch, entry, tmp_path
):
    from ipfs_accelerate_py.agent_supervisor.task_sources import board_stall_watchdog as owner
    now = [100.0]
    monkeypatch.setattr(owner.time, "monotonic", lambda: now[0])
    repo = _Repo((_blocked("LIR-STILL-PENDING"),))
    hold = tmp_path / "operator.hold"
    hold.write_bytes(b"preserve existing operator hold\n")
    watchdog = BoardStallWatchdog()
    repairs = []
    def repair(target, spec):
        repairs.append(dict(spec))
        return resolution_repair_subagent(target, spec)
    watchdog.tick(repo, operator_hold_present=True, spawn_repair=repair)
    assert repo.release_calls == 2 and len(repairs) == 1
    now[0] += 121.0
    if entry == "tick":
        second = watchdog.tick(repo, operator_hold_present=True, spawn_repair=repair)
    else:
        second = serve(repo, watchdog=watchdog, once=True, spawn_repair=repair,
            hold_path=hold,
            readonly_observe=lambda: observe_tasks(repo._records, operator_hold_present=True))
    assert second["action"] == "stall_recorded"
    assert repo.release_calls == 3 and len(repairs) == 1
    # The renewed check suppresses an immediate additional writable retry.
    watchdog.tick(repo, operator_hold_present=True, spawn_repair=repair)
    assert repo.release_calls == 3 and len(repairs) == 1
    assert second["preserve_operator_hold"] is True
    assert hold.read_bytes() == b"preserve existing operator hold\n"
    assert second["set_ready"] is False and second["executor_start_allowed"] is False


import time

def _active_and_blocked_repo(release=None):
    return _Repo(
        (
            {"task_alias": "LAW-ACTIVE", "task_cid": "active", "status": "in_progress"},
            _blocked("LAW-METADATA"),
        ),
        release=release,
    )


def _forbid_active_repair(_repo, _spec):
    raise AssertionError("active work must not spawn a repair callback")


def test_active_task_does_not_starve_independent_metadata_resolution():
    repo = _active_and_blocked_repo(
        {"completed": [], "spawned": [{"task_alias": "LAW-METADATA--definition"}], "retired": []}
    )
    rows_before = [dict(row) for row in repo._records]
    report = BoardStallWatchdog().tick(
        repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
    )
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert report["release"] == {"completed": [], "spawned": ["LAW-METADATA--definition"], "retired": []}
    assert report["executor_start_allowed"] is False
    assert report["set_ready"] is False
    assert report["grant_admission"] is False
    assert report["preserve_operator_hold"] is True
    assert repo._records == rows_before
    assert "repair_spec" not in report and "repair_report" not in report


def test_active_resolution_reports_trusted_metadata_completion_without_launch():
    repo = _active_and_blocked_repo(
        {"completed": [{"task_alias": "LAW-METADATA"}], "spawned": [], "retired": []}
    )
    report = BoardStallWatchdog().tick(
        repo, operator_hold_present=False, spawn_repair=_forbid_active_repair
    )
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert report["release"]["completed"] == ["LAW-METADATA"]
    assert report["executor_start_allowed"] is False
    assert report["set_ready"] is False
    assert report["grant_admission"] is False


def test_active_blocked_hold_remains_held_without_a_repair_attempt():
    repo = _active_and_blocked_repo(
        {"completed": [], "spawned": [], "retired": [], "held": [{"decision": "runtime_authority_hold"}]}
    )
    report = BoardStallWatchdog().tick(
        repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
    )
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert report["release"] == {"completed": [], "spawned": [], "retired": []}
    assert repo._records[1]["status"] == "blocked"
    assert report["preserve_operator_hold"] is True
    assert "repair_report" not in report


def test_active_recheck_does_not_reset_or_spend_the_idle_repair_latch():
    repo = _active_and_blocked_repo()
    owner = BoardStallWatchdog()
    signature = observe_tasks(repo._records, operator_hold_present=True)["signature"]
    owner._idle_signatures.add(signature)
    owner._idle_checked_at[signature] = time.monotonic()
    before = (set(owner._idle_signatures), dict(owner._idle_checked_at))
    report = owner.tick(repo, operator_hold_present=True, spawn_repair=_forbid_active_repair)
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert (owner._idle_signatures, owner._idle_checked_at) == before


def test_existing_serve_fast_path_does_not_hide_an_active_blocked_frontier():
    repo = _active_and_blocked_repo()
    owner = BoardStallWatchdog()
    signature = observe_tasks(repo._records, operator_hold_present=True)["signature"]
    owner._idle_signatures.add(signature)
    owner._idle_checked_at[signature] = time.monotonic()
    report = serve(
        repo,
        watchdog=owner,
        once=True,
        readonly_observe=lambda: observe_tasks(repo._records, operator_hold_present=True),
        spawn_repair=_forbid_active_repair,
    )
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert owner._idle_signatures == {signature}


def test_active_only_frontier_does_not_call_the_metadata_resolver():
    repo = _Repo(({"task_alias": "LAW-ACTIVE", "status": "in_progress"},))
    report = BoardStallWatchdog().tick(
        repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
    )
    assert report["action"] == "continue"
    assert repo.release_calls == 0
    assert report["preserve_operator_hold"] is True


def test_active_recheck_preserves_the_actual_lock_exception():
    error = RuntimeError("database is locked")

    class LockedRepo(_Repo):
        def release_dependency_satisfied_blocks(self):
            self.release_calls += 1
            raise error

    repo = LockedRepo(_active_and_blocked_repo()._records)
    try:
        BoardStallWatchdog().tick(
            repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
        )
    except RuntimeError as caught:
        assert caught is error
    else:
        raise AssertionError("the existing database-lock refusal must propagate")
    assert repo.release_calls == 1


def test_active_release_failure_does_not_claim_progress_or_spawn_repair():
    from copy import deepcopy

    class FailingRepo(_Repo):
        def release_dependency_satisfied_blocks(self):
            self.release_calls += 1
            raise ValueError("fictional metadata refusal")

    repo = FailingRepo(_active_and_blocked_repo()._records)
    repo._records[0].update({"claim_id": "fictional-claim", "lease_id": "fictional-lease"})
    repo._records[1].update({"operator_hold": True, "budget_exhausted": True,
                             "retry_budget": 0, "source_lease_id": "fictional-source"})
    records_before = deepcopy(repo._records)
    owner = BoardStallWatchdog()
    signature = observe_tasks(repo._records, operator_hold_present=True)["signature"]
    owner._idle_signatures.add(signature)
    owner._idle_checked_at[signature] = 123.0
    latches_before = (set(owner._idle_signatures), dict(owner._idle_checked_at))
    report = owner.tick(
        repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
    )
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert report["release_failure"] == {"action": "release_failed", "error": "ValueError"}
    assert "error" not in report and "release" not in report
    assert report["signature"] == signature
    assert report["executor_start_allowed"] is False
    assert report["set_ready"] is False
    assert report["grant_admission"] is False
    assert report["preserve_operator_hold"] is True
    assert "repair_spec" not in report and "repair_report" not in report
    assert repo._records == records_before
    assert (owner._idle_signatures, owner._idle_checked_at) == latches_before

    # The pre-existing inactive error disposition remains unchanged.
    idle_repo = FailingRepo((_blocked("LAW-IDLE"),))
    idle_owner = BoardStallWatchdog()
    idle_report = idle_owner.tick(
        idle_repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
    )
    assert idle_repo.release_calls == 1
    assert idle_report["action"] == "release_failed"
    assert idle_report["error"] == "ValueError"
    assert "release_failure" not in idle_report and "release" not in idle_report
    assert idle_report["executor_start_allowed"] is False
    assert idle_report["set_ready"] is False
    assert idle_report["grant_admission"] is False
    assert idle_report["preserve_operator_hold"] is True
    assert idle_owner._idle_signatures == set() and idle_owner._idle_checked_at == {}


def test_active_run_once_preserves_the_operator_hold_and_truthful_status(tmp_path):
    hold = tmp_path / "hold"
    status = tmp_path / "status.json"
    hold.write_bytes(b"operator-owned legal review hold\n")
    repo = _active_and_blocked_repo()
    report = run_once(
        repo,
        watchdog=BoardStallWatchdog(),
        hold_path=hold,
        status_path=status,
        spawn_repair=_forbid_active_repair,
    )
    assert repo.release_calls == 1
    assert hold.read_bytes() == b"operator-owned legal review hold\n"
    import json
    saved = json.loads(status.read_bytes())
    assert saved["action"] == report["action"] == "continue"
    assert saved["set_ready"] is False and saved["grant_admission"] is False


def test_active_recheck_never_grants_execution_even_for_malformed_resolver_output():
    repo = _active_and_blocked_repo()
    repo._release = object()
    report = BoardStallWatchdog().tick(
        repo, operator_hold_present=True, spawn_repair=_forbid_active_repair
    )
    assert repo.release_calls == 1
    assert report["action"] == "continue"
    assert report["release"] == {"completed": [], "spawned": [], "retired": []}
    assert report["executor_start_allowed"] is False
    assert report["set_ready"] is False
    assert report["grant_admission"] is False


def test_active_owner_does_not_starve_real_repository_metadata_obligations(tmp_path):
    """Normal-import/DuckDB integration; private call-count fixtures do not qualify it."""
    repo = _catalogue(tmp_path)
    try:
        _task(
            repo, "task:independent-done", "LAW-DONE", status="completed",
            body={"completion": "manual", "execution_packet_required": True},
        )
        from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
            IntentRepositoryConflictError,
        )

        # Ordinary activation remains forbidden without its current Native packet.
        with pytest.raises(
            IntentRepositoryConflictError,
            match="^packet_current_native_contract_required$",
        ):
            _task(
                repo, "task:independent-active", "LAW-ACTIVE", status="in_progress",
                body={"completion": "manual", "execution_packet_required": True},
            )
        assert repo.get_task("task:independent-active") is None

        # This owned DB fixture models an interrupted/legacy active projection.
        # Seed only its status; create no execution packet, claim, approval or lease.
        _task(
            repo, "task:independent-active", "LAW-ACTIVE", status="blocked",
            body={"completion": "manual", "execution_packet_required": True},
        )
        with repo._connection(write=True) as connection:
            connection.execute(
                "UPDATE tasks SET status = ? WHERE task_cid = ? AND status = ?",
                ["in_progress", "task:independent-active", "blocked"],
            )
        assert repo.get_task("task:independent-active")["status"] == "in_progress"
        assert repo.get_task("task:independent-active")["body"]["execution_packet_required"] is True
        _task(
            repo, "task:independent-metadata", "LAW-METADATA", status="blocked",
            body=_manual_body("This authored fixture remains unresolved until each clause is recorded."),
            dependencies=["task:independent-done"],
            acceptance=(
                "Every accepted component retains complete material meaning.",
                "Input context and output capacities stay separate.",
            ),
        )
        _task(
            repo, "task:independent-held", "LAW-HELD", status="blocked",
            body={**_manual_body("Await the existing resource owner."),
                  "budget_exhausted": True, "retry_budget": 0},
            dependencies=["task:independent-done"],
        )
        active_before = dict(repo.get_task("task:independent-active"))
        held_before = dict(repo.get_task("task:independent-held"))
        hold = tmp_path / "operator.hold"
        hold.write_bytes(b"operator-owned review and runtime hold\n")
        report = run_once(
            repo,
            watchdog=BoardStallWatchdog(),
            hold_path=hold,
            spawn_repair=_forbid_active_repair,
        )
        assert report["action"] == "continue"
        assert set(report["release"]["spawned"]) == {
            "LAW-METADATA--00-every-accepted-component-retains",
            "LAW-METADATA--01-input-context-output-capacities",
        }
        assert report["release"]["completed"] == []
        assert dict(repo.get_task("task:independent-active")) == active_before
        assert dict(repo.get_task("task:independent-held")) == held_before
        assert repo.get_task("task:independent-metadata")["status"] == "blocked"
        assert hold.read_bytes() == b"operator-owned review and runtime hold\n"
        assert report["executor_start_allowed"] is False
        assert report["set_ready"] is False
        assert report["grant_admission"] is False
        for task in repo.list_tasks(limit=20, offset=0):
            assert task["status"] != "ready"
            for flag in ("admitted", "formalized", "gold_or_teacher_admission_granted",
                         "constitution_roundtrip_ok"):
                assert task["body"].get(flag) is not True
    finally:
        repo.close()
