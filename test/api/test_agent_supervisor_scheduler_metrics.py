from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import DynamicBundleScheduler
from ipfs_accelerate_py.agent_supervisor.event_log import (
    append_jsonl_event,
    read_jsonl_event_sources,
)
from ipfs_accelerate_py.agent_supervisor.scheduler_metrics import (
    GOAL_COMPLETION_DIAGNOSTICS_SCHEMA,
    LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS,
    SCHEDULER_PHASES,
    SCHEDULER_SNAPSHOT_SCHEMA,
    SCHEDULER_SNAPSHOT_SCHEMA_VERSION,
    build_scheduler_snapshot,
    build_scheduler_snapshot_from_paths,
    normalize_metric_identity,
    project_goal_completion_diagnostics,
    read_scheduler_snapshot,
    write_scheduler_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.supervisor_watchdog import SupervisorWatchdog


IDENTITY = {
    "goal_cid": "goal:g9",
    "subgoal_cid": "subgoal:g9.s4",
    "task_cid": "task:ref-044",
    "lane_id": "lane:g9.s4",
    "provider_id": "provider:codex",
}


def _event(kind: str, timestamp: str, **payload: Any) -> dict[str, Any]:
    return {"type": kind, "timestamp": timestamp, **IDENTITY, **payload}


def test_lifecycle_metrics_are_derived_from_current_daemon_event_shapes() -> None:
    snapshot = build_scheduler_snapshot(
        [
            _event("task_ready", "2026-01-01T00:00:00Z"),
            _event("implementation_started", "2026-01-01T00:00:10Z", attempt=1),
            _event(
                "implementation_finished",
                "2026-01-01T00:00:40Z",
                returncode=0,
                validation_result={
                    "attempted": True,
                    "passed": True,
                    "results": [
                        {
                            "started_at": "2026-01-01T00:00:25Z",
                            "finished_at": "2026-01-01T00:00:35Z",
                        }
                    ],
                },
            ),
            _event("merge_candidate_enqueued", "2026-01-01T00:00:42Z"),
            _event("merge_started", "2026-01-01T00:00:50Z"),
            _event("merge_finished", "2026-01-01T00:01:00Z", merged=True, returncode=0),
            # Completion is deduplicated after the successful merge.
            _event("task_completed", "2026-01-01T00:01:01Z"),
        ],
        now="2026-01-01T00:02:00Z",
    )

    assert snapshot["authoritative"] is True
    assert tuple(snapshot["phases"]) == SCHEDULER_PHASES
    assert snapshot["phase_counts"]["idle"] == 1
    row = snapshot["metrics"][0]
    assert row["queue_wait_seconds"] == 10.0
    assert row["implementation_duration_seconds"] == 15.0
    assert row["validation_duration_seconds"] == 10.0
    assert row["merge_wait_seconds"] == 8.0
    assert row["completions"] == 1
    assert {key: row[key] for key in IDENTITY} == IDENTITY


def test_snapshot_zero_fills_and_reports_all_scheduler_phases() -> None:
    phase_events = {
        "ready": ("task_ready", {}),
        "active": ("implementation_started", {"attempt": 1}),
        "idle": ("daemon_no_tasks", {}),
        "blocked": ("task_blocked", {}),
        "validation": ("daemon_phase_heartbeat", {"phase": "validating"}),
        "merge": ("merge_started", {}),
        "resolver": ("llm_merge_resolver_invoked", {}),
    }
    events = []
    for index, (expected, (kind, extra)) in enumerate(phase_events.items()):
        identity = {
            **IDENTITY,
            "task_cid": f"task:{expected}",
            "lane_id": f"lane:{index}",
        }
        events.append(
            {
                "type": kind,
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                **identity,
                **extra,
            }
        )

    snapshot = build_scheduler_snapshot(events)

    assert snapshot["phase_counts"] == {phase: 1 for phase in SCHEDULER_PHASES}
    assert all(snapshot["phases"][phase]["count"] == 1 for phase in SCHEDULER_PHASES)


def test_timestamped_projection_overrides_undated_legacy_lane_state() -> None:
    snapshot = build_scheduler_snapshot(
        [
            {
                "type": "scheduler_lane_state",
                "phase": "ready",
                **IDENTITY,
            },
            {
                "type": "scheduler_state",
                "timestamp": "2026-01-01T00:00:00Z",
                "phase": "blocked",
                "state": "blocked",
                **IDENTITY,
            },
        ]
    )

    assert snapshot["phase_counts"]["ready"] == 0
    assert snapshot["phase_counts"]["blocked"] == 1
    assert snapshot["phases"]["blocked"]["items"][0]["task_cid"] == IDENTITY["task_cid"]


def test_rates_usage_and_identity_grouping_are_zero_safe_and_canonical() -> None:
    other = {**IDENTITY, "task_cid": "task:other", "provider_id": "provider:other"}
    events = [
        _event("implementation_started", "2026-01-01T00:00:00Z", attempt=1),
        _event("implementation_finished", "2026-01-01T00:00:05Z", returncode=1),
        _event(
            "implementation_started",
            "2026-01-01T00:00:10Z",
            attempt=2,
            usage={"prompt_tokens": 7, "completion_tokens": 5},
            estimated_cost_usd=0.25,
        ),
        _event("merge_finished", "2026-01-01T00:00:20Z", returncode=1, reason="git conflict"),
        {"type": "task_ready", "timestamp": "2026-01-01T00:00:00Z", **other},
    ]

    snapshot = build_scheduler_snapshot(events)
    rows = {row["task_cid"]: row for row in snapshot["metrics"]}
    row = rows[IDENTITY["task_cid"]]
    assert row["implementation_attempts"] == 2
    assert row["retries"] == 1
    assert row["retry_rate"] == 0.5
    assert row["merge_attempts"] == 1
    assert row["conflicts"] == 1
    assert row["conflict_rate"] == 1.0
    assert row["tokens"] == 12
    assert row["cost_usd"] == 0.25
    assert rows[other["task_cid"]]["retry_rate"] == 0.0
    assert len(rows) == 2


def test_identity_normalization_always_returns_five_stable_dimensions() -> None:
    identity = normalize_metric_identity(
        {"canonical_task_cid": "task:canonical", "effective_provider_name": "openai"},
        {"goal_id": "G9", "subgoal_id": "G9.S4", "parallel_lane": "g9-s4"},
    )

    assert identity["goal_cid"] == "G9"
    assert identity["subgoal_cid"] == "G9.S4"
    assert identity["task_cid"] == "task:canonical"
    assert identity["lane_id"] == "g9-s4"
    assert identity["provider_id"] == "openai"


def test_multiple_and_rotated_event_logs_feed_one_atomic_snapshot(tmp_path: Path) -> None:
    active = tmp_path / "events.jsonl"
    archive = tmp_path / "events.jsonl.rotated-20260101000000"
    archive.write_text(
        json.dumps(_event("task_ready", "2026-01-01T00:00:00Z")) + "\n",
        encoding="utf-8",
    )
    append_jsonl_event(
        active,
        "implementation_started",
        {**IDENTITY, "timestamp": "2026-01-01T00:00:05Z", "attempt": 1},
    )

    events = read_jsonl_event_sources([active])
    snapshot = build_scheduler_snapshot_from_paths([active])
    output = tmp_path / "operator" / "scheduler.json"
    write_scheduler_snapshot(output, snapshot)

    assert len(events) == 2
    assert snapshot["totals"]["queue_wait_seconds"] == 5.0
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot_id"] == snapshot.snapshot_id


def test_refill_receipt_metrics_distinguish_terminal_outcomes_and_dedupe_cids() -> None:
    reasons = (
        "generated",
        "exhausted",
        "duplicate_only",
        "threshold_satisfied",
        "cooldown",
        "disabled",
        "partial",
        "failed",
        "timed_out",
    )
    events: list[dict[str, Any]] = []
    for index, reason in enumerate(reasons):
        events.append(
            {
                "type": "refill_scan_receipt",
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "receipt_cid": f"bafy-receipt-{reason}",
                "artifact_path": f"state/scan_receipts/{reason}.json",
                "scan_kind": "objective" if index % 2 else "codebase",
                "terminal_reason": reason,
                "scan_mode": "exhaustive",
                "analyzer_version": "test/v2",
                "repository_id": "repo:test",
                "tree_id": "tree:test",
                "started_at": f"2026-01-01T00:00:{index:02d}Z",
                "finished_at": f"2026-01-01T00:00:{index:02d}Z",
                "generated_count": 2 if reason == "generated" else 0,
                "health": "unhealthy" if reason in {"failed", "timed_out"} else "healthy",
                "freshness": {"fresh": True, "age_seconds": 0},
                "candidate_funnel": {
                    "raw_candidates": index + 1,
                    "appended_tasks": 2 if reason == "generated" else 0,
                },
            }
        )
    # Delivery retries can replay a receipt.  Its CID, rather than event ID,
    # is the canonical attempt identity.
    events.append({**events[0], "timestamp": "2026-01-01T00:01:00Z"})

    snapshot = build_scheduler_snapshot(events)
    scans = snapshot["scan_metrics"]

    assert (
        scans["attempts"]
        == scans["attempted"]
        == scans["receipts"]
        == scans["receipt_count"]
        == 9
    )
    assert scans["skipped"] == 3
    assert scans["failed_total"] == 2
    assert scans["successful"] == 3
    assert scans["generated_count"] == 2
    assert scans["by_terminal_reason"] == {reason: 1 for reason in reasons}
    assert scans["outcome_counts"] == scans["by_terminal_reason"]
    assert scans["duplicate_only"] == scans["exhausted"] == scans["partial"] == 1
    assert scans["failed"] == scans["timed_out"] == 1
    assert scans["candidate_funnel"] == {"raw_candidates": 45, "appended_tasks": 2}
    assert scans["latest_attempted_scan"]["terminal_reason"] == "timed_out"
    assert scans["latest_attempted_scan"]["freshness"]["fresh"] is True
    assert scans["latest_successful_scan"]["terminal_reason"] == "duplicate_only"
    assert "items" not in scans["latest_attempted_scan"]
    assert snapshot["totals"]["refill_scan_skipped"] == 3
    assert snapshot["totals"]["refill_scan_duplicate_only"] == 1
    assert snapshot["totals"]["scan_skipped"] == 3
    assert snapshot["metrics"] == []
    assert all(count == 0 for count in snapshot["phase_counts"].values())


def test_refill_metrics_accept_nested_and_legacy_event_shapes_without_false_exhaustion() -> None:
    snapshot = build_scheduler_snapshot(
        [
            {
                "type": "scan_receipt",
                "timestamp": "2026-01-01T00:00:00Z",
                "scan_receipt": {
                    "receipt_cid": "bafy-nested",
                    "scan_kind": "codebase",
                    "terminal_reason": "duplicate-only",
                    "generated_count": 0,
                    "metadata": {"candidate_funnel": {"deduplicated_candidates": 4}},
                },
            },
            {
                "type": "codebase_refill_failed",
                "timestamp": "2026-01-01T00:00:01Z",
                "error": "parser unavailable",
            },
            {
                "type": "objective_refill_scan",
                "timestamp": "2026-01-01T00:00:02Z",
                "generated_count": 3,
            },
        ]
    )

    scans = snapshot["scan_metrics"]
    assert scans["attempts"] == 3
    assert scans["duplicate_only"] == 1
    assert scans["failed"] == 1
    assert scans["generated"] == 1
    assert scans["exhausted"] == 0
    assert scans["candidate_funnel"]["deduplicated_candidates"] == 4


def test_canonical_receipt_supersedes_only_its_matching_legacy_event() -> None:
    snapshot = build_scheduler_snapshot(
        [
            # A historical legacy-only event remains a real attempted scan.
            {
                "type": "codebase_refill_failed",
                "timestamp": "2026-01-01T00:00:00Z",
                "error": "historical parser failure",
            },
            # Current supervisors retain this compatibility event immediately
            # before publishing the canonical receipt for the same attempt.
            {
                "type": "codebase_refill_failed",
                "timestamp": "2026-01-01T01:00:00Z",
                "error": "current parser failure",
            },
            {
                "type": "refill_scan_receipt",
                "timestamp": "2026-01-01T01:00:01Z",
                "receipt_cid": "bafy-current-failure",
                "scan_kind": "codebase",
                "terminal_reason": "failed",
                "scan_mode": "exhaustive",
                "analyzer_version": "test/v2",
                "repository_id": "repo:test",
                "tree_id": "tree:test",
                "started_at": "2026-01-01T00:59:00Z",
                "finished_at": "2026-01-01T01:00:00Z",
                "generated_count": 0,
                "health": "unhealthy",
                "candidate_funnel": {"parser_failure_count": 1},
            },
        ]
    )

    scans = snapshot["scan_metrics"]
    assert scans["attempted"] == 2
    assert scans["failed"] == 2
    assert scans["failed_total"] == 2
    assert scans["receipt_count"] == scans["receipts"] == 1
    assert scans["legacy_event_count"] == 1
    assert scans["candidate_funnel"] == {"parser_failure_count": 1}


@dataclass
class _Process:
    pid: int = 9001
    alive: bool = True

    def poll(self) -> None | int:
        return None if self.alive else 0


def test_scheduler_decisions_reference_the_exposed_event_snapshot(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    index = repo / "index.json"
    index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/g9/g9-s4": {
                        "shard_path": "bundle.todo.md",
                        "parallel_lane": "g9-s4",
                        "tasks": [{"task_id": "REF-044", "title": "metrics"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    started: list[Any] = []

    def launch(lane: Any, grant: Any) -> _Process:
        started.append((lane, grant))
        return _Process()

    scheduler = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        coordination_path=repo / "coordination.sqlite3",
        max_lanes=1,
        launcher=launch,
        process_alive=lambda process: process.alive,
    )

    manifest = scheduler.reconcile_once()

    assert started
    decision = next(item for item in manifest["scheduler_decisions"] if item["decision"] == "launched")
    assert decision["snapshot_id"] == manifest["scheduler_decision_snapshot_id"]
    assert manifest["scheduler_decision_snapshot"]["snapshot_id"] == decision["snapshot_id"]
    assert manifest["scheduler_snapshot"]["authoritative"] is True
    assert set(manifest["scheduler_snapshot"]["phase_counts"]) == set(SCHEDULER_PHASES)
    published = json.loads((repo / "state" / "scheduler_metrics.json").read_text(encoding="utf-8"))
    assert published == manifest["scheduler_snapshot"]
    decision_input = json.loads(
        (repo / "state" / "scheduler_decision_metrics.json").read_text(encoding="utf-8")
    )
    assert decision_input["snapshot_id"] == decision["snapshot_id"]


def test_watchdog_returns_same_snapshot_and_defers_dynamic_lane_recovery(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    snapshot = build_scheduler_snapshot([_event("implementation_started", "2026-01-01T00:00:00Z")])
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1",
                "authoritative": True,
                "scheduler_snapshot": snapshot.to_dict(),
                "lanes": [
                    {
                        "bundle_key": "g9-s4",
                        "state_dir": str(tmp_path / "missing"),
                        "state_prefix": "agent_g9_s4",
                        "command": ["false"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    watchdog = SupervisorWatchdog(manifest_path=manifest_path, repo_root=tmp_path)

    report = watchdog._check_cycle()

    assert report["scheduler_snapshot"] == snapshot.to_dict()
    assert report["scheduler_snapshot_id"] == snapshot.snapshot_id
    assert report["reports"][0]["action"] == "scheduler_recovery_required"
    assert report["restarts"] == 0


def test_completion_diagnostics_expose_all_fail_closed_operator_dimensions() -> None:
    decision = {
        "type": "objective_goal_completion_evaluated",
        "timestamp": "2026-01-01T00:01:00Z",
        "goal_id": "G10.S4",
        "state": "reopened",
        "confidence": "0.42",
        "missing_criteria": ["Manifest exposes missing proof"],
        "invalid_criteria": ["Evidence is bound to the current tree"],
        "reason_codes": ["stale_evidence", "verification_invalidated"],
        "actionable_reasons": ["Regenerate validation proof."],
        "completion_gate": {
            "passed": False,
            "checks": [
                {
                    "name": "mandatory_coverage",
                    "passed": False,
                    "evidence": {
                        "missing_criteria": ["Manifest exposes missing proof"],
                        "unverified_criteria": ["Analyzer result is trustworthy"],
                    },
                }
            ],
            "evaluated_evidence": {
                "evaluated_at": "2026-01-01T00:00:59Z",
                "analyzer_health": {
                    "status": "unhealthy",
                    "reasons": ["parser_failure"],
                },
                "exhaustion_quorum": {
                    "satisfied": False,
                    "member_count": 1,
                    "required_members": 2,
                },
            },
        },
    }

    projection = project_goal_completion_diagnostics([decision], now="2026-01-01T00:02:00Z")
    snapshot = build_scheduler_snapshot([decision], now="2026-01-01T00:02:00Z")

    assert projection["schema"] == GOAL_COMPLETION_DIAGNOSTICS_SCHEMA
    assert projection["goal_count"] == 1
    row = projection["by_goal_id"]["G10.S4"]
    assert row["lifecycle_state"] == "reopened"
    assert row["confidence"] == 0.42
    assert row["uncovered_criteria"] == [
        "Manifest exposes missing proof",
        "Evidence is bound to the current tree",
        "Analyzer result is trustworthy",
    ]
    assert row["stale_evidence"] == ["stale_evidence"]
    assert row["analyzer_health"]["status"] == "unhealthy"
    assert row["exhaustion_quorum"]["satisfied"] is False
    assert row["reopen_reasons"] == ["Regenerate validation proof."]
    assert projection["unhealthy_analyzer_goal_count"] == 1
    assert projection["stale_evidence_goal_count"] == 1
    assert projection["reopened_goal_count"] == 1
    assert snapshot["goal_completion"] == snapshot["goal_completion_diagnostics"]
    assert snapshot["goal_completion_diagnostics"]["by_goal_id"]["G10.S4"] == row
    # A goal-only event is not an anonymous scheduler task.
    assert snapshot["task_states"] == []


def test_legacy_completed_diagnostic_is_provisional_and_never_gains_implicit_proof() -> None:
    projection = project_goal_completion_diagnostics(
        {
            "objective_completion_decisions": {
                "G1": {
                    "previous_state": "active",
                    "state": "completed",
                    "tasks_complete": True,
                    "acceptance_criteria": ["Ship the feature"],
                }
            }
        },
        now="2026-01-01T00:00:00Z",
    )

    row = projection["by_goal_id"]["G1"]
    assert row["lifecycle_state"] == "provisionally_complete"
    assert row["confidence"] is None
    assert row["confidence_reported"] is False
    assert row["analyzer_health"] == {}
    assert row["exhaustion_quorum"] == {}
    assert row["completion_gate_passed"] is None
    assert projection["unknown_confidence_goal_count"] == 1


def test_migration_and_runner_diagnostic_shapes_retain_nested_structured_proof() -> None:
    projection = project_goal_completion_diagnostics(
        [
            {
                "schema": "objective_goal_migration@1",
                "type": "objective_goal_migration",
                "goal_id": "G1",
                "legacy_state": "completed",
                "state": "provisionally_complete",
                "reason_codes": ["legacy_evidence_stale"],
                "completion_decision": {
                    "state": "active",
                    "missing_criteria": ["Current validation"],
                },
                "diagnostics": {
                    "confidence": 0.25,
                    "stale_evidence": [
                        {"receipt_cid": "bafy-old", "reason": "tree_changed"}
                    ],
                    "analyzer_health": {"status": "unknown"},
                },
            },
            {
                "goal_completion_diagnostics": {
                    "G2": {
                        "lifecycle_state": "verified_complete",
                        "confidence": 0.98,
                        "uncovered_criteria": [],
                        "analyzer_health": {"status": "healthy"},
                        "exhaustion_quorum": {"satisfied": True},
                    }
                }
            },
        ]
    )

    migrated = projection["by_goal_id"]["G1"]
    assert migrated["lifecycle_state"] == "provisionally_complete"
    assert migrated["confidence"] == 0.25
    assert migrated["uncovered_criteria"] == ["Current validation"]
    assert migrated["stale_evidence"] == [
        {"receipt_cid": "bafy-old", "reason": "tree_changed"}
    ]
    assert projection["by_goal_id"]["G2"]["lifecycle_state"] == "verified_complete"
    assert projection["by_goal_id"]["G2"]["exhaustion_quorum"]["satisfied"] is True


def test_snapshot_reader_accepts_v1_and_adds_diagnostics_without_rewriting_schema(
    tmp_path: Path,
) -> None:
    legacy_schema = next(iter(LEGACY_SCHEDULER_SNAPSHOT_SCHEMAS))
    path = tmp_path / "scheduler-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema": legacy_schema,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "snapshot_id": "legacy-id",
                "authoritative": True,
                "phases": {},
                "metrics": [],
                "task_states": [{"goal_cid": "goal:legacy", "task_cid": "task:1"}],
            }
        ),
        encoding="utf-8",
    )

    restored = read_scheduler_snapshot(path)

    assert restored is not None
    assert restored["schema"] == legacy_schema
    assert restored["schema_version"] == 1
    assert restored["goal_completion_diagnostics"]["goal_count"] == 1
    legacy = restored["goal_completion_diagnostics"]["by_goal_id"]["goal:legacy"]
    assert legacy["lifecycle_state"] == "unknown"
    assert legacy["confidence"] is None
    assert legacy["reason_codes"] == ["legacy_diagnostics_unavailable"]
    assert restored["goal_completion_diagnostics"]["diagnostics_available"] is False
    assert restored["goal_completion"] == restored["goal_completion_diagnostics"]
    current = build_scheduler_snapshot([])
    assert current["schema"] == SCHEDULER_SNAPSHOT_SCHEMA
    assert current["schema_version"] == SCHEDULER_SNAPSHOT_SCHEMA_VERSION
