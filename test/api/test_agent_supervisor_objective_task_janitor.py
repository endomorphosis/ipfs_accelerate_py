from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    completion_gate_receipts_from_decisions,
    load_goal_completion_gate_records,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import ContradictionEvidence
from ipfs_accelerate_py.agent_supervisor.objective_graph import ObjectiveGoal
from ipfs_accelerate_py.agent_supervisor.objective_task_janitor import (
    JANITOR_RECEIPT_SCHEMA,
    reconcile_objective_task_strategy,
    registered_goal_ids_from_bundle_index,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _dynamic_task(*, status: str = "todo") -> PortalTask:
    metadata = {
        "goal id": "codebase/runtime/src-runtime",
        "graph parents": "codebase/runtime",
        "candidate kind": "codebase_scan",
        "goal registration": "dynamic",
    }
    if status == "blocked":
        metadata["blocked reason"] = (
            "Retired by objective-task janitor during launch steering because "
            "orphaned_goal_reference."
        )
    return PortalTask(
        task_id="AUTO-002",
        title="Resolve code annotation in src/runtime.py:2",
        status=status,
        completion="manual",
        priority="P1",
        track="runtime",
        metadata=metadata,
    )


def test_bundle_index_registers_open_dynamic_goals_only():
    payload = {
        "bundles": {
            "codebase/runtime/src-runtime": {
                "tasks": [
                    {
                        "task_id": "AUTO-002",
                        "status": "blocked",
                        "candidate_kind": "codebase_scan",
                        "goal_registration": "dynamic",
                        "goal_id": "codebase/runtime/src-runtime",
                        "parent_goal_id": "codebase/runtime",
                        "parent_goal_ids": ["codebase"],
                    }
                ]
            },
            "codebase/runtime/src-complete": {
                "tasks": [
                    {
                        "task_id": "AUTO-003",
                        "status": "completed",
                        "goal_registration": "dynamic",
                        "goal_id": "codebase/runtime/src-complete",
                    }
                ]
            },
            "objective/static": {
                "tasks": [
                    {
                        "task_id": "AUTO-001",
                        "status": "todo",
                        "candidate_kind": "seed",
                        "goal_id": "G1.S1",
                    }
                ]
            },
        }
    }

    assert registered_goal_ids_from_bundle_index(payload) == [
        "codebase/runtime/src-runtime",
        "codebase/runtime",
        "codebase",
    ]


def test_registered_dynamic_goal_is_not_retired_as_orphan():
    result = reconcile_objective_task_strategy(
        goals=[
            ObjectiveGoal(
                goal_id="G1",
                title="Static objective",
                fields={"status": "active", "priority": "P0"},
            )
        ],
        tasks=[_dynamic_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        registered_goal_ids=["codebase/runtime", "codebase/runtime/src-runtime"],
    )

    assert result["blocked_task_ids"] == []
    assert result["deprioritized_task_ids"] == []
    assert result["registered_goal_ids"] == [
        "codebase/runtime",
        "codebase/runtime/src-runtime",
    ]


def test_registered_dynamic_goal_unblocks_prior_janitor_retirement():
    prior_receipt = {
        "schema": JANITOR_RECEIPT_SCHEMA,
        "action": "block",
        "task_id": "AUTO-002",
        "retired_task_reason": "orphaned_goal_reference",
    }
    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[_dynamic_task(status="blocked")],
        strategy={
            "blocked_tasks": ["AUTO-002"],
            "objective_task_janitor_receipts": [prior_receipt],
        },
        now="2026-07-22T00:00:00+00:00",
        registered_goal_ids=["codebase/runtime", "codebase/runtime/src-runtime"],
    )

    assert result["blocked_task_ids"] == []
    assert result["unblocked_task_ids"] == ["AUTO-002"]
    assert result["strategy"]["blocked_tasks"] == []
    assert result["receipts"][0]["action"] == "unblock"


def test_registered_dynamic_goal_recovers_materialized_block_without_receipt():
    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[_dynamic_task(status="blocked")],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        registered_goal_ids=["codebase/runtime", "codebase/runtime/src-runtime"],
    )

    assert result["unblocked_task_ids"] == ["AUTO-002"]
    assert result["strategy"]["blocked_tasks"] == []


def test_unresolved_materialized_block_retains_janitor_ownership():
    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[_dynamic_task(status="blocked")],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
    )

    assert result["blocked_task_ids"] == ["AUTO-002"]
    assert result["strategy"]["blocked_tasks"] == ["AUTO-002"]
    assert result["receipts"][0]["retired_task_reason"] == "orphaned_goal_reference"


def _legacy_off_mission_scan_task() -> PortalTask:
    return PortalTask(
        task_id="AUTO-051",
        title="Review swallowed exception path in adversarial_harness/harness.py:49",
        status="blocked",
        completion="manual",
        priority="P1",
        track="runtime",
        outputs=["adversarial_harness/harness.py"],
        acceptance="Codebase scan filed this finding.",
        metadata={
            "blocked reason": (
                "Deferred by objective-task janitor during launch steering because "
                "off_mission_codebase_scan_task."
            )
        },
    )


def test_changed_mission_scope_unblocks_legacy_deferred_scan_task():
    task = _legacy_off_mission_scan_task()
    prior_receipt = {
        "schema": JANITOR_RECEIPT_SCHEMA,
        "action": "block",
        "task_id": task.task_id,
        "retired_task_reason": "goal_not_active",
    }

    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[task],
        strategy={
            "blocked_tasks": [task.task_id],
            "objective_task_janitor_receipts": [prior_receipt],
        },
        now="2026-07-22T00:00:00+00:00",
        mission_terms=["adversarial_harness"],
    )

    assert result["blocked_task_ids"] == []
    assert result["unblocked_task_ids"] == [task.task_id]
    assert result["strategy"]["blocked_tasks"] == []
    assert result["receipts"][0]["retired_task_reason"] == "task_now_matches_mission_scope"


def test_legacy_deferred_scan_task_stays_blocked_outside_mission_scope():
    task = _legacy_off_mission_scan_task()

    result = reconcile_objective_task_strategy(
        goals=[],
        tasks=[task],
        strategy={"blocked_tasks": [task.task_id]},
        now="2026-07-22T00:00:00+00:00",
        mission_terms=["unrelated-area"],
    )

    assert result["unblocked_task_ids"] == []
    assert result["blocked_task_ids"] == [task.task_id]
    assert result["receipts"][0]["retired_task_reason"] == "off_mission_codebase_scan_task"


def test_supervisor_loads_bundle_registry_and_materializes_unblock(tmp_path):
    todo_path = tmp_path / "todo.md"
    objective_path = tmp_path / "objective.md"
    bundle_dir = tmp_path / "bundles"
    state_dir = tmp_path / "state"
    strategy_path = state_dir / "strategy.json"
    events_path = state_dir / "events.jsonl"
    state_path = state_dir / "task_state.json"
    bundle_dir.mkdir()
    state_dir.mkdir()
    objective_path.write_text(
        """# Objective heap

## G1 Static objective

- Status: active
- Priority: P0
""",
        encoding="utf-8",
    )
    todo_path.write_text(
        """# Todo

## AUTO-002 Resolve code annotation in src/runtime.py:2

- Status: blocked
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Goal id: codebase/runtime/src-runtime
- Graph parents: codebase/runtime
- Candidate kind: codebase_scan
- Goal registration: dynamic
- Blocked reason: Retired by objective-task janitor because orphaned_goal_reference.
""",
        encoding="utf-8",
    )
    (bundle_dir / "index.json").write_text(
        json.dumps(
            {
                "bundles": {
                    "codebase/runtime/src-runtime": {
                        "tasks": [
                            {
                                "task_id": "AUTO-002",
                                "status": "blocked",
                                "candidate_kind": "codebase_scan",
                                "goal_id": "codebase/runtime/src-runtime",
                                "parent_goal_id": "codebase/runtime",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    strategy_path.write_text("{}\n", encoding="utf-8")
    supervisor = PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=todo_path,
            state_path=state_path,
            strategy_path=strategy_path,
            events_path=events_path,
            state_dir=state_dir,
            task_prefix="## AUTO-",
            objective_path=objective_path,
            objective_bundle_dir=bundle_dir,
        )
    )

    result = supervisor.reconcile_objective_task_janitor()

    assert result["unblocked_task_ids"] == ["AUTO-002"]
    assert result["materialized"]["unblocked_task_ids"] == ["AUTO-002"]
    assert result["materialized"]["removed_reason_task_ids"] == ["AUTO-002"]
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "- Status: todo" in todo_text
    assert "- Blocked reason:" not in todo_text


def _goal_task(goal_id: str = "G1") -> PortalTask:
    return PortalTask(
        task_id="AUTO-010",
        title="Implement reopened objective",
        status="todo",
        completion="manual",
        priority="P0",
        track="runtime",
        metadata={"goal id": goal_id},
    )


def _passing_completion_decision(*, evaluated_children=None):
    evaluated_evidence = {
        "acceptance_criteria": ["criterion one"],
        "coverage": {
            "criteria": [{"criterion": "criterion one", "status": "verified"}]
        },
        "validation_evidence": [{"valid": True, "evidence": {"provenance_cid": "bafy-proof"}}],
        "analyzer_health": {"status": "healthy"},
        "exhaustion_quorum": {"satisfied": True, "member_count": 2},
        "analysis_result": {"terminal_reason": "exhausted", "safe_for_completion_reasoning": True},
        "child_goals": list(evaluated_children or []),
    }
    return {
        "state": "verified_complete",
        "verified": True,
        "reason_codes": [],
        "actionable_reasons": [],
        "completion_gate": {
            "passed": True,
            "reason_codes": [],
            "actionable_reasons": [],
            "checks": [
                {"name": name, "passed": True, "reason_code": "", "evidence": {}}
                for name in (
                    "mandatory_coverage",
                    "required_validations",
                    "analyzer_health",
                    "exhaustion_quorum",
                    "analysis_terminal_state",
                    "child_goals",
                )
            ],
            "evaluated_evidence": evaluated_evidence,
        },
    }


def test_reopened_goal_remains_schedulable_and_keeps_linked_work_open():
    result = reconcile_objective_task_strategy(
        goals=[ObjectiveGoal("G1", "Regressed objective", {"status": "reopened", "priority": "P0"})],
        tasks=[_goal_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
    )

    assert result["blocked_task_ids"] == []
    assert result["active_goal_ids"] == ["G1"]
    assert result["scheduled_goal_ids"] == ["G1"]
    assert result["open_goal_ids"] == ["G1"]


@pytest.mark.parametrize("status", ["verified_complete", "completed"])
def test_verified_goal_retires_linked_work_as_completed(status: str):
    result = reconcile_objective_task_strategy(
        goals=[ObjectiveGoal("G1", "Done objective", {"status": status})],
        tasks=[_goal_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
    )

    assert result["blocked_task_ids"] == ["AUTO-010"]
    assert result["receipts"][0]["retired_task_reason"] == "goal_completed"


def test_supplied_failed_completion_gate_cannot_be_hidden_by_verified_status():
    decision = _passing_completion_decision()
    decision["completion_gate"]["passed"] = False
    decision["completion_gate"]["reason_codes"] = ["analyzer_unhealthy"]
    decision["completion_gate"]["checks"][2] = {
        "name": "analyzer_health",
        "passed": False,
        "reason_code": "analyzer_unhealthy",
        "evidence": {"status": "partial"},
    }

    result = reconcile_objective_task_strategy(
        goals=[ObjectiveGoal("G1", "Unproven objective", {"status": "verified_complete"})],
        tasks=[_goal_task()],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        completion_decisions={"G1": decision},
    )

    assert result["blocked_task_ids"] == ["AUTO-010"]
    assert result["receipts"][0]["retired_task_reason"] == "analyzer_unhealthy"
    assert result["completion_gate_failed_goal_ids"] == ["G1"]
    gate_receipt = result["completion_gate_receipts"][0]
    assert gate_receipt["passed"] is False
    assert "analyzer_unhealthy" in gate_receipt["reason_codes"]
    assert "completion_gate_failed" in gate_receipt["reason_codes"]
    assert gate_receipt["evaluated_evidence"] == decision["completion_gate"]["evaluated_evidence"]


def test_passing_completion_gate_allows_prior_janitor_block_to_be_removed():
    task = replace(
        _goal_task(),
        status="blocked",
        metadata={
            "goal id": "G1",
            "blocked reason": "Retired by objective-task janitor because goal_not_active.",
        },
    )
    prior_receipt = {
        "schema": JANITOR_RECEIPT_SCHEMA,
        "action": "block",
        "task_id": task.task_id,
    }

    result = reconcile_objective_task_strategy(
        goals=[ObjectiveGoal("G1", "Proven objective", {"status": "verified_complete"})],
        tasks=[task],
        strategy={
            "blocked_tasks": [task.task_id],
            "objective_task_janitor_receipts": [prior_receipt],
        },
        now="2026-07-22T00:00:00+00:00",
        completion_decisions={"G1": _passing_completion_decision()},
    )

    assert result["removed_task_ids"] == ["AUTO-010"]
    assert result["completion_gate_passed_goal_ids"] == ["G1"]
    assert result["completion_gate_receipts"][0]["passed"] is True


def test_parent_gate_does_not_hide_reopened_descendant():
    child = {"goal_id": "G1.S1", "state": "verified_complete", "verified": True}
    result = reconcile_objective_task_strategy(
        goals=[
            ObjectiveGoal("G1", "Parent", {"status": "verified_complete"}),
            ObjectiveGoal("G1.S1", "Regressed child", {"status": "reopened", "parents": "G1"}),
        ],
        tasks=[_goal_task("G1")],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        completion_decisions={
            "G1": _passing_completion_decision(evaluated_children=[child]),
        },
    )

    receipt = result["completion_gate_receipts"][0]
    assert receipt["passed"] is False
    assert "descendant_reopened" in receipt["reason_codes"]
    assert receipt["descendants"] == [
        {
            "goal_id": "G1.S1",
            "state": "reopened",
            "passed": False,
            "reason_codes": [],
            "evaluated_evidence": {},
        }
    ]
    assert result["receipts"][0]["retired_task_reason"] == "descendant_reopened"


def test_gate_artifact_loader_rejects_malformed_per_goal_record(tmp_path):
    artifact = tmp_path / "completion-gate.json"
    artifact.write_text(json.dumps({"goals": {"G1": "passed"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="record for 'G1' must be an object"):
        load_goal_completion_gate_records(artifact)


def test_daemon_gate_receipt_rechecks_internal_checks_and_preserves_exact_evidence():
    decision = _passing_completion_decision()
    decision["completion_gate"]["checks"][3]["passed"] = False
    decision["completion_gate"]["checks"][3]["reason_code"] = "exhaustion_quorum_unsatisfied"

    receipt = completion_gate_receipts_from_decisions({"G1": decision})["G1"]

    assert receipt["passed"] is False
    assert "completion_gate_check_failed" in receipt["reason_codes"]
    assert receipt["evaluated_evidence"] == decision["completion_gate"]["evaluated_evidence"]


def test_contradiction_reopens_goal_parent_and_dependent_and_schedules_work():
    goals = [
        ObjectiveGoal(
            "G1",
            "Parent goal",
            {"status": "verified_complete", "acceptance": "parent criterion"},
        ),
        ObjectiveGoal(
            "G1.S1",
            "Affected child",
            {
                "status": "verified_complete",
                "parents": "G1",
                "acceptance": "child criterion",
            },
        ),
        ObjectiveGoal(
            "G2",
            "Dependent goal",
            {
                "status": "provisionally_complete",
                "depends_on": "G1.S1",
                "acceptance": "dependent criterion",
            },
        ),
    ]
    repair = replace(
        _goal_task("G1.S1"),
        task_id="AUTO-209",
        title="Repair the failed required validation",
    )
    contradiction = ContradictionEvidence(
        goal_id="G1.S1",
        kind="failed_validation",
        summary="The required API validation regressed.",
        impacted_criteria=("child criterion",),
        invalidated_evidence=("bafy-old-proof",),
        source_receipt={"receipt_cid": "bafy-failed-validation", "passed": False},
        scheduled_work=({"task_id": repair.task_id, "reason": "rerun validation"},),
    )
    historical_completion = {
        "receipt_id": "bafy-historical-completion",
        "state": "verified_complete",
        "verified": True,
    }

    result = reconcile_objective_task_strategy(
        goals=goals,
        tasks=[repair],
        strategy={"objective_completion_decisions": {"G1.S1": historical_completion}},
        now="2026-07-22T00:00:00+00:00",
        contradictions=[contradiction],
    )

    assert result["blocked_task_ids"] == []
    assert result["contradiction_reopened_goal_ids"] == ["G1", "G1.S1", "G2"]
    assert result["effective_reopened_goal_ids"] == ["G1", "G1.S1", "G2"]
    assert result["recalculated_goal_ids"] == ["G1", "G1.S1", "G2"]
    assert result["newly_scheduled_task_ids"] == ["AUTO-209"]
    assert set(result["scheduled_goal_ids"]) == {"G1", "G1.S1", "G2"}

    child_decision = result["goal_reopening_decisions"]["G1.S1"]
    assert child_decision["state"] == "reopened"
    assert child_decision["impacted_criteria"] == ["child criterion"]
    assert child_decision["invalidated_evidence"] == ["bafy-old-proof"]
    assert child_decision["source_receipts"] == [
        {"passed": False, "receipt_cid": "bafy-failed-validation"}
    ]
    assert child_decision["newly_scheduled_work"] == [
        {"reason": "rerun validation", "task_id": "AUTO-209"}
    ]
    # Reopening adds a new ledger entry; it does not overwrite the historical
    # proof which justified the earlier verified state.
    assert child_decision["historical_completion_receipts"] == [
        {"goal_id": "G1.S1", **historical_completion}
    ]
    assert len(result["goal_reopening_receipts"]) == 3
    assert result["strategy"]["objective_completion_decisions"] == {
        "G1.S1": historical_completion
    }


def test_identical_contradiction_replay_is_idempotent_and_preserves_force_state():
    goal = ObjectiveGoal(
        "G1",
        "Completed goal",
        {"status": "verified_complete", "acceptance": "criterion one"},
    )
    contradiction = ContradictionEvidence(
        goal_id="G1",
        kind="changed_surface",
        impacted_criteria=("criterion one",),
        invalidated_evidence=("src/api.py@old",),
        source_receipt={"receipt_cid": "bafy-surface-change"},
        scheduled_work=({"task_id": "AUTO-210"},),
    )
    first = reconcile_objective_task_strategy(
        goals=[goal],
        tasks=[],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        contradictions=[contradiction],
    )
    second = reconcile_objective_task_strategy(
        goals=[goal],
        tasks=[],
        strategy=first["strategy"],
        now="2026-07-22T01:00:00+00:00",
        contradictions=[contradiction],
    )

    assert first["contradiction_reopened_goal_ids"] == ["G1"]
    assert second["contradiction_reopened_goal_ids"] == []
    assert second["effective_reopened_goal_ids"] == ["G1"]
    assert second["newly_scheduled_task_ids"] == []
    assert len(second["goal_reopening_receipts"]) == 1
    assert second["goal_reopening_receipts"] == first["goal_reopening_receipts"]
    assert second["changed"] is False

    materialized = reconcile_objective_task_strategy(
        goals=[replace(goal, fields={**goal.fields, "status": "reopened"})],
        tasks=[],
        strategy=second["strategy"],
        now="2026-07-22T02:00:00+00:00",
    )
    reverified = reconcile_objective_task_strategy(
        goals=[goal],
        tasks=[],
        strategy=materialized["strategy"],
        now="2026-07-22T03:00:00+00:00",
    )
    assert materialized["strategy"]["objective_task_janitor_pending_reopen_goal_ids"] == []
    assert reverified["effective_reopened_goal_ids"] == []
    assert reverified["reopened_goal_ids"] == []


def test_unrelated_finding_does_not_reopen_completed_goal():
    result = reconcile_objective_task_strategy(
        goals=[
            ObjectiveGoal("G1", "Completed goal", {"status": "verified_complete"}),
            ObjectiveGoal("G2", "Active goal", {"status": "active"}),
        ],
        tasks=[],
        strategy={},
        now="2026-07-22T00:00:00+00:00",
        contradictions=[
            ContradictionEvidence(
                goal_id="G2",
                kind="mapped_finding",
                source_receipt={"finding_id": "finding-unrelated-to-G1"},
                scheduled_work=({"task_id": "AUTO-211"},),
            )
        ],
    )

    assert "G1" not in result["reopened_goal_ids"]
    assert result["contradiction_reopened_goal_ids"] == []
    assert result["goal_reopening_receipts"] == []
