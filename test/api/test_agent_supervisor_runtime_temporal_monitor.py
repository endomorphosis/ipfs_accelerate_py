from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime_temporal_monitor import (
    MonitorVerdict,
    NoticeCode,
    RUNTIME_TEMPORAL_COUNTEREXAMPLE_SCHEMA,
    RuntimeTemporalMonitor,
    TemporalMonitorConfig,
    TemporalMonitorPolicy,
    TemporalPropertyKind,
    load_temporal_counterexamples,
    monitor_event_logs,
    monitor_event_trace,
)


def _event(
    event_type: str,
    second: int | None,
    *,
    task: str = "REF-287",
    event_id: str | None = None,
    lane: str = "g12-s4",
    tree: str = "tree:abc",
    epoch: str = "epoch:a",
    **payload: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "type": event_type,
        "task_id": task,
        "lane_id": lane,
        "repository_tree_id": tree,
        "restart_epoch": epoch,
        **payload,
    }
    if second is not None:
        result["timestamp"] = f"2026-01-01T00:00:{second:02d}+00:00"
    if event_id is not None:
        result["event_id"] = event_id
    return result


def _safe_trace() -> list[dict[str, object]]:
    return [
        _event("lease_acquired", 0, event_id="lease", expires_at="2026-01-01T00:01:00Z"),
        _event("implementation_started", 1, event_id="start", attempt=1, sequence=1),
        _event("resource_acquired", 2, event_id="resource", resource_id="gpu:0", sequence=2),
        _event("implementation_finished", 3, event_id="finish", returncode=0, sequence=3),
        _event("proof_verified", 4, event_id="proof", verdict="proved", sequence=4),
        _event("merge_started", 5, event_id="merge", sequence=5),
        _event("task_completed", 6, event_id="terminal", status="completed", sequence=6),
        _event("resource_released", 7, event_id="release", resource_id="gpu:0", sequence=7),
        _event("lease_released", 8, event_id="lease-release", sequence=8),
    ]


def test_versioned_properties_cover_the_runtime_contract_without_claiming_proof() -> None:
    policy = TemporalMonitorPolicy()

    assert {item.kind for item in policy.properties} == set(TemporalPropertyKind)
    assert all(item.version >= 1 and item.property_id and item.formula for item in policy.properties)
    assert policy.policy_id == TemporalMonitorPolicy().policy_id

    report = monitor_event_trace(_safe_trace(), policy=policy, now="2026-01-01T00:00:20Z")

    assert report.verdict is MonitorVerdict.NO_VIOLATION_OBSERVED
    assert report.counterexamples == ()
    assert report.notices == ()
    assert report.events_received == len(_safe_trace())
    assert report.events_evaluated == len(_safe_trace())
    assert report.partitions_observed == 1
    assert report.finalized
    assert not report.proved
    assert report.to_dict()["proved"] is False
    assert report.to_dict()["verdict"] == "no_violation_observed"


def test_event_ordering_lease_expiration_stop_and_proof_before_merge_are_independent(
    tmp_path,
) -> None:
    policy = TemporalMonitorPolicy(max_retries=1)
    counterexamples = tmp_path / "counterexamples.jsonl"
    reopens = tmp_path / "reopens.jsonl"
    reopened = []
    monitor = RuntimeTemporalMonitor(
        policy,
        TemporalMonitorConfig(
            out_of_order_window_seconds=0,
            counterexample_path=counterexamples,
            reopen_path=reopens,
        ),
        reopen_callback=reopened.append,
    )

    monitor.ingest_many(
        [
            _event("lease_acquired", 0, event_id="lease", expires_at="2026-01-01T00:00:02Z"),
            # A finish before a start is an ordering violation.
            _event("implementation_finished", 1, event_id="early-finish"),
            # At expiry is invalid; leases use a half-open validity interval.
            _event("implementation_started", 2, event_id="late-start", attempt=1),
            _event("task_revoked", 3, event_id="revoke"),
            _event("validation_started", 4, event_id="post-revoke-action"),
            # Merge has neither an accepted proof nor a valid implementation order.
            _event("merge_started", 5, event_id="unproved-merge"),
        ]
    )
    report = monitor.finalize()
    kinds = {item.property_kind for item in report.counterexamples}

    assert report.verdict is MonitorVerdict.VIOLATED
    assert {
        TemporalPropertyKind.EVENT_ORDERING,
        TemporalPropertyKind.LEASE_EXPIRATION,
        TemporalPropertyKind.NO_ACTION_AFTER_STOP,
        TemporalPropertyKind.PROOF_BEFORE_MERGE,
    } <= kinds
    assert len(load_temporal_counterexamples(counterexamples)) == len(
        report.counterexamples
    )
    assert len(reopened) == len(report.counterexamples)
    reopen_records = [
        json.loads(line) for line in reopens.read_text(encoding="utf-8").splitlines()
    ]
    assert {item["status"] for item in reopen_records} == {"reopened"}
    assert {item["task_id"] for item in reopen_records} == {"REF-287"}
    assert all(item["schema"] == RUNTIME_TEMPORAL_COUNTEREXAMPLE_SCHEMA for item in load_temporal_counterexamples(counterexamples))
    assert all(item["is_proof"] is False for item in load_temporal_counterexamples(counterexamples))

    # Replaying the same trace finds the same violation identities but does not
    # duplicate durable counterexamples or reopen requests.
    replay = monitor_event_trace(
        [
            _event("lease_acquired", 0, event_id="lease", expires_at="2026-01-01T00:00:02Z"),
            _event("implementation_finished", 1, event_id="early-finish"),
            _event("implementation_started", 2, event_id="late-start", attempt=1),
            _event("task_revoked", 3, event_id="revoke"),
            _event("validation_started", 4, event_id="post-revoke-action"),
            _event("merge_started", 5, event_id="unproved-merge"),
        ],
        policy=policy,
        config=TemporalMonitorConfig(out_of_order_window_seconds=0),
        counterexample_path=counterexamples,
        reopen_path=reopens,
    )
    assert replay.verdict is MonitorVerdict.VIOLATED
    assert len(load_temporal_counterexamples(counterexamples)) == len(
        report.counterexamples
    )
    assert len(reopens.read_text(encoding="utf-8").splitlines()) == len(
        report.counterexamples
    )


def test_bounded_retry_eventual_terminal_and_resource_release_deadlines() -> None:
    policy = TemporalMonitorPolicy(
        max_retries=1,
        terminal_deadline_seconds=5,
        resource_release_deadline_seconds=2,
    )
    monitor = RuntimeTemporalMonitor(
        policy,
        TemporalMonitorConfig(out_of_order_window_seconds=0),
    )
    monitor.ingest_many(
        [
            _event("lease_acquired", 0, expires_at="2026-01-01T00:01:00Z"),
            _event("implementation_started", 1, event_id="attempt-1", attempt=1),
            _event("resource_acquired", 2, event_id="gpu", resource_id="gpu"),
            _event("retry_started", 3, event_id="retry-1"),
            _event("retry_started", 4, event_id="retry-2"),
        ]
    )
    monitor.advance_time("2026-01-01T00:00:07Z")
    # Cancellation is terminal and starts a release deadline for held resources.
    monitor.ingest(_event("task_cancelled", 8, event_id="cancel"))
    report = monitor.finalize(now="2026-01-01T00:00:11Z")
    kinds = {item.property_kind for item in report.counterexamples}

    assert TemporalPropertyKind.BOUNDED_RETRY in kinds
    assert TemporalPropertyKind.EVENTUAL_TERMINAL in kinds
    assert TemporalPropertyKind.RESOURCE_RELEASE in kinds
    assert any("gpu" in item.reason for item in report.counterexamples)


def test_reordering_duplicate_missing_timestamp_and_restart_epochs_are_explicit() -> None:
    config = TemporalMonitorConfig(out_of_order_window_seconds=5)
    monitor = RuntimeTemporalMonitor(config=config)

    # Deliberately delivered out of order, but within the configured window.
    monitor.ingest(_event("implementation_finished", 3, event_id="finish", sequence=3))
    monitor.ingest(
        _event(
            "lease_acquired",
            0,
            event_id="lease",
            expires_at="2026-01-01T00:01:00Z",
            sequence=0,
        )
    )
    monitor.ingest(_event("implementation_started", 1, event_id="start", sequence=1))
    monitor.ingest(_event("proof_verified", 4, event_id="proof", sequence=4))
    monitor.ingest(_event("merge_started", 5, event_id="merge", sequence=5))
    monitor.ingest(_event("task_completed", 6, event_id="done", sequence=6))
    monitor.ingest(_event("lease_released", 7, event_id="release", sequence=7))
    monitor.ingest(_event("lease_released", 7, event_id="release", sequence=7))

    # Sequence numbers restart in a new daemon epoch and are not compared with
    # the preceding epoch.
    monitor.ingest(
        _event(
            "heartbeat",
            8,
            event_id="new-epoch",
            epoch="epoch:b",
            sequence=1,
        )
    )
    # Missing timestamps remain observable and make the finite-prefix verdict
    # inconclusive; they are not silently assigned wall-clock time.
    monitor.ingest(_event("heartbeat", None, event_id="untimed", epoch="epoch:b"))
    report = monitor.finalize()

    assert report.counterexamples == ()
    assert report.verdict is MonitorVerdict.INCONCLUSIVE
    assert report.duplicate_events == 1
    assert {notice.code for notice in report.notices} >= {
        NoticeCode.DUPLICATE_EVENT,
        NoticeCode.MISSING_TIMESTAMP,
    }


def test_event_older_than_committed_watermark_is_not_retroactively_evaluated() -> None:
    monitor = RuntimeTemporalMonitor(
        config=TemporalMonitorConfig(out_of_order_window_seconds=1)
    )
    monitor.ingest(
        _event(
            "lease_acquired",
            10,
            event_id="lease",
            expires_at="2026-01-01T00:01:00Z",
        )
    )
    monitor.ingest(_event("implementation_started", 11, event_id="start"))
    monitor.ingest(_event("heartbeat", 20, event_id="watermark"))
    monitor.ingest(_event("implementation_finished", 2, event_id="too-late"))
    report = monitor.finalize()

    assert report.verdict is MonitorVerdict.INCONCLUSIVE
    assert report.counterexamples == ()
    assert NoticeCode.OUT_OF_ORDER_WINDOW_EXCEEDED in {
        item.code for item in report.notices
    }
    assert report.events_evaluated == 3


def test_open_eventual_obligation_is_inconclusive_and_finalize_is_idempotent() -> None:
    monitor = RuntimeTemporalMonitor(
        config=TemporalMonitorConfig(out_of_order_window_seconds=0)
    )
    monitor.ingest(
        _event(
            "lease_acquired",
            0,
            event_id="lease",
            expires_at="2026-01-01T00:01:00Z",
        )
    )
    monitor.ingest(_event("implementation_started", 1, event_id="start"))

    first = monitor.finalize(now="2026-01-01T00:00:02Z")
    second = monitor.finalize(now="2026-01-01T00:00:59Z")

    assert first.verdict is MonitorVerdict.INCONCLUSIVE
    assert first.to_dict() == second.to_dict()
    assert [item.code for item in first.notices].count(NoticeCode.OPEN_OBLIGATION) == 1
    with pytest.raises(RuntimeError):
        monitor.ingest(_event("implementation_finished", 3))


def test_rotated_logs_are_consumed_and_duplicates_do_not_duplicate_violations(
    tmp_path,
) -> None:
    active = tmp_path / "events.jsonl"
    rotated = tmp_path / "events.jsonl.rotated-20260101000000"
    trace = _safe_trace()
    rotated.write_text(
        "\n".join(json.dumps(item) for item in trace[:4]) + "\n",
        encoding="utf-8",
    )
    # Repeat one boundary event as a defensive simulation of copy/truncate
    # rotation, then continue in the active file.
    active.write_text(
        "\n".join(json.dumps(item) for item in [trace[3], *trace[4:]]) + "\n",
        encoding="utf-8",
    )

    report = monitor_event_logs(active, now="2026-01-01T00:00:20Z")

    assert report.verdict is MonitorVerdict.NO_VIOLATION_OBSERVED
    assert report.duplicate_events == 1
    assert report.events_received == len(trace) + 1
    assert report.events_evaluated == len(trace)


def test_streaming_state_is_bounded_and_partitioned_by_all_four_identities() -> None:
    policy = TemporalMonitorPolicy()
    monitor = RuntimeTemporalMonitor(
        policy,
        TemporalMonitorConfig(
            out_of_order_window_seconds=0,
            max_partitions=2,
            max_events_per_partition=2,
            max_pending_events_per_partition=2,
        ),
    )
    for index, (task, lane, tree) in enumerate(
        [
            ("A", "lane:1", "tree:1"),
            ("A", "lane:2", "tree:1"),
            ("A", "lane:2", "tree:2"),
        ]
    ):
        monitor.ingest(
            _event(
                "lease_acquired",
                index * 3,
                task=task,
                lane=lane,
                tree=tree,
                expires_at="2026-01-01T00:01:00Z",
            )
        )
        monitor.ingest(
            _event(
                "implementation_started",
                index * 3 + 1,
                task=task,
                lane=lane,
                tree=tree,
            )
        )

    report = monitor.flush()

    assert report.partitions_observed == 3
    assert report.active_partitions == 2
    assert NoticeCode.PARTITION_EVICTED in {item.code for item in report.notices}
    assert report.verdict is MonitorVerdict.INCONCLUSIVE
    assert all(
        item.partition.policy_id == policy.policy_id for item in report.notices
    )


def test_foreign_policy_and_unknown_task_are_isolated_and_not_evaluated() -> None:
    monitor = RuntimeTemporalMonitor()
    monitor.ingest(
        {
            "type": "implementation_started",
            "timestamp": "2026-01-01T00:00:01Z",
            "policy_id": "policy:foreign",
        }
    )
    report = monitor.finalize()

    assert report.verdict is MonitorVerdict.INCONCLUSIVE
    assert report.counterexamples == ()
    assert {item.code for item in report.notices} == {
        NoticeCode.UNKNOWN_TASK,
        NoticeCode.POLICY_IDENTITY_MISMATCH,
    }
    assert report.events_evaluated == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"terminal_deadline_seconds": -1},
        {"resource_release_deadline_seconds": float("inf")},
    ],
)
def test_policy_rejects_unbounded_or_invalid_limits(updates) -> None:
    with pytest.raises(ValueError):
        TemporalMonitorPolicy(**updates)


def test_policy_identity_changes_with_semantics() -> None:
    baseline = TemporalMonitorPolicy()
    changed = replace(baseline, max_retries=baseline.max_retries + 1)

    assert baseline.policy_id != changed.policy_id
    assert baseline.to_dict()["schema"].endswith("@1")
    assert datetime.fromtimestamp(0, timezone.utc).tzinfo is timezone.utc
