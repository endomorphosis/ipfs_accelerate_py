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
    SCHEDULER_PHASES,
    build_scheduler_snapshot,
    build_scheduler_snapshot_from_paths,
    normalize_metric_identity,
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
