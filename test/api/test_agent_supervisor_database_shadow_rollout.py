"""Tests for DatabaseShadowRollout@1 / ShadowParityReport@1 (DQP-037).

Acceptance covered:

* Exact import reconciles
* No unexplained authority-relevant drift
* Shadow never controls production effect
* Dual observation has bounded duration/retention
* Rollback and re-run preserve history and generate the same parity decision
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.database_shadow_rollout import (
    AUTHORITY_SURFACES,
    DATABASE_SHADOW_ROLLOUT_INTERFACE,
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MAX_RETENTION_RECORDS,
    EXPORT_NON_AUTHORITY_MARKER,
    PRODUCTION_EFFECT_CHANNEL,
    SHADOW_PARITY_REPORT_INTERFACE,
    TASK_ID,
    DatabaseShadowRollout,
    DriftDisposition,
    DriftDispositionKind,
    DualObservationBoundError,
    DualObservationWindow,
    ParityDecision,
    ShadowAuthorityError,
    ShadowDispositionError,
    ShadowParityError,
    ShadowParityReport,
    ShadowRolloutStage,
    ShadowStageError,
    project_authority_snapshot,
    run_shadow_parity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import import (
    ConflictPolicy,
    ImportDomain,
    ImportMediaType,
    ImportMode,
    ImportSourceSpec,
    LegacyStateImport,
    OUTCOME_APPLIED,
    OUTCOME_REPLAYED,
    duckdb_available,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _json_tasks(path: Path) -> Path:
    return _write(
        path,
        json.dumps(
            {
                "records": [
                    {
                        "id": "DQP-037",
                        "title": "Backfill legacy state",
                        "status": "todo",
                        "priority": "P0",
                        "task_cid": "task:dqp-037",
                        "revision": 0,
                    },
                    {
                        "id": "DQP-038",
                        "title": "Staged canary cutover",
                        "status": "todo",
                        "priority": "P0",
                        "task_cid": "task:dqp-038",
                        "revision": 0,
                    },
                ]
            },
            indent=2,
        )
        + "\n",
    )


def _jsonl_events(path: Path) -> Path:
    lines = [
        json.dumps(
            {"event_id": "evt-1", "task_id": "DQP-037", "kind": "created"}
        ),
        json.dumps(
            {"event_id": "evt-2", "task_id": "DQP-037", "kind": "queued"}
        ),
    ]
    return _write(path, "\n".join(lines) + "\n")


def _sources(tmp_path: Path) -> list[ImportSourceSpec]:
    # Single authoritative taskboard source plus events avoids multi-source
    # content conflicts while still exercising multi-media import reconcile.
    tasks = _json_tasks(tmp_path / "tasks.json")
    events = _jsonl_events(tmp_path / "events.jsonl")
    return [
        ImportSourceSpec(
            source_id="tasks-json",
            path=str(tasks),
            media_type=ImportMediaType.JSON,
            domain=ImportDomain.TASKBOARDS,
        ),
        ImportSourceSpec(
            source_id="events-jsonl",
            path=str(events),
            media_type=ImportMediaType.JSONL,
            domain=ImportDomain.EVENTS,
        ),
    ]


def _rollout(
    tmp_path: Path,
    *,
    window: DualObservationWindow | None = None,
    dispositions: list[DriftDisposition] | None = None,
    clock=None,
    durable: bool = False,
) -> DatabaseShadowRollout:
    # Default path is hermetic in-memory import so parity tests do not depend
    # on the optional duckdb wheel. Durable DuckDB apply is covered separately.
    target = tmp_path / "shadow.duckdb" if durable else None
    importer = LegacyStateImport(
        target_database=target,
        source_root=tmp_path,
    )
    return DatabaseShadowRollout(
        work_dir=tmp_path,
        source_root=tmp_path,
        target_database=target,
        window=window,
        dispositions=dispositions or (),
        clock=clock,
        importer=importer,
    )


def _backfill(rollout: DatabaseShadowRollout, tmp_path: Path, *, durable: bool = False):
    from ipfs_accelerate_py.agent_supervisor.task_sources.legacy_state_import import (
        ImportManifest,
    )

    target = ""
    if durable:
        target = str(tmp_path / "shadow.duckdb")
    manifest = ImportManifest(
        import_id="dqp-037-import",
        sources=tuple(_sources(tmp_path)),
        mode=ImportMode.APPLY,
        strict=True,
        default_conflict_policy=ConflictPolicy.REJECT,
        target_database=target,
    )
    return rollout.backfill(manifest)


# ---------------------------------------------------------------------------
# Interface identity
# ---------------------------------------------------------------------------


def test_interface_identity() -> None:
    assert DATABASE_SHADOW_ROLLOUT_INTERFACE == "DatabaseShadowRollout@1"
    assert SHADOW_PARITY_REPORT_INTERFACE == "ShadowParityReport@1"
    assert DatabaseShadowRollout.INTERFACE == DATABASE_SHADOW_ROLLOUT_INTERFACE
    assert ShadowParityReport.INTERFACE == SHADOW_PARITY_REPORT_INTERFACE
    assert set(AUTHORITY_SURFACES) == {
        "counts_digests",
        "duplicate_conflict",
        "task_cid",
        "readiness",
        "lease_fence",
        "status",
        "event_cursor",
        "completion",
        "restart",
        "export",
    }


def test_cold_import_is_side_effect_free() -> None:
    """Importing the module constructs no files and starts no processes."""

    import importlib

    mod = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.self_improvement.database_shadow_rollout"
    )
    assert mod.DATABASE_SHADOW_ROLLOUT_INTERFACE == "DatabaseShadowRollout@1"
    # Pure construction without work_dir must not touch disk.
    rollout = mod.DatabaseShadowRollout()
    assert rollout.stage is mod.ShadowRolloutStage.OFF
    assert rollout.shadow_controls_production is False


# ---------------------------------------------------------------------------
# Exact import reconciles
# ---------------------------------------------------------------------------


def test_exact_import_reconciles(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    receipt = _backfill(rollout, tmp_path)

    assert receipt.outcome in (OUTCOME_APPLIED, OUTCOME_REPLAYED)
    assert receipt.applied is True
    assert len(receipt.accepted_rows) >= 2
    assert all(str(row.get("source_digest") or "").startswith("sha256:") for row in receipt.accepted_rows)
    assert all(observation.source_digest.startswith("sha256:") for observation in receipt.source_observations)
    # Exact replay is a no-op with the same receipt cid.
    replay = _backfill(rollout, tmp_path)
    assert replay.outcome == OUTCOME_REPLAYED
    assert replay.receipt_cid == receipt.receipt_cid
    assert replay.manifest_cid == receipt.manifest_cid


def test_backfill_then_parity_pass_when_snapshots_match(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    receipt = _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start_shadow")
    report = rollout.evaluate_parity()

    assert report.import_reconciled is True
    assert report.import_receipt_cid == receipt.receipt_cid
    assert report.decision is ParityDecision.PASS
    assert report.passed is True
    assert report.unexplained_drift_count == 0
    assert report.matched_surface_count == len(AUTHORITY_SURFACES)
    assert report.shadow_controls_production is False
    assert report.task_id == TASK_ID
    payload = report.to_dict()
    assert payload["interface"] == SHADOW_PARITY_REPORT_INTERFACE
    assert payload["passed"] is True
    assert "dual_observation_non_authoritative" in payload["reason_codes"]


# ---------------------------------------------------------------------------
# No unexplained authority-relevant drift
# ---------------------------------------------------------------------------


def test_unexplained_authority_relevant_drift_fails(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start_shadow")

    production = project_authority_snapshot(
        tasks=[{"id": "DQP-037", "task_cid": "task:dqp-037", "status": "todo"}],
        readiness=["task:dqp-037"],
        statuses={"task:dqp-037": "todo"},
    )
    shadow = project_authority_snapshot(
        tasks=[{"id": "DQP-037", "task_cid": "task:dqp-037", "status": "in_progress"}],
        readiness=[],
        statuses={"task:dqp-037": "in_progress"},
    )
    report = rollout.evaluate_parity(
        production_snapshot=production,
        shadow_snapshot=shadow,
    )
    assert report.decision is ParityDecision.FAIL
    assert report.passed is False
    assert report.unexplained_drift_count > 0
    assert "unexplained_authority_relevant_drift" in report.reason_codes
    drifted = [item for item in report.drifts if not item.matched]
    assert drifted
    assert all(
        item.disposition is DriftDispositionKind.UNEXPLAINED for item in drifted
    )
    with pytest.raises(ShadowParityError, match="parity did not pass"):
        report.require_passed()


def test_reviewed_disposition_clears_authority_relevant_drift(
    tmp_path: Path,
) -> None:
    dispositions = [
        DriftDisposition(
            surface="status",
            kind=DriftDispositionKind.REVIEWED_ACCEPT,
            reason="legacy status projection lags one CAS; accepted for shadow",
            reviewer="operator-a",
        ),
        DriftDisposition(
            surface="readiness",
            kind=DriftDispositionKind.REVIEWED_ACCEPT,
            reason="readiness follows status; same reviewed lag",
            reviewer="operator-a",
        ),
        DriftDisposition(
            surface="task_cid",
            kind=DriftDispositionKind.REVIEWED_IGNORE,
            reason="task body status field differs; cid identity stable",
            reviewer="operator-a",
        ),
        DriftDisposition(
            surface="counts_digests",
            kind=DriftDispositionKind.REVIEWED_IGNORE,
            reason="count digest includes status; reviewed",
            reviewer="operator-a",
        ),
        DriftDisposition(
            surface="completion",
            kind=DriftDispositionKind.REVIEWED_IGNORE,
            reason="no completion delta beyond status lag",
            reviewer="operator-a",
        ),
    ]
    rollout = _rollout(tmp_path, dispositions=dispositions)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start_shadow")

    production = project_authority_snapshot(
        tasks=[{"id": "DQP-037", "task_cid": "task:dqp-037", "status": "todo"}],
        readiness=["task:dqp-037"],
        statuses={"task:dqp-037": "todo"},
        counts={"tasks": 1, "ready": 1},
        digests={"tasks": "sha256:" + ("aa" * 32)},
    )
    shadow = project_authority_snapshot(
        tasks=[
            {
                "id": "DQP-037",
                "task_cid": "task:dqp-037",
                "status": "in_progress",
            }
        ],
        readiness=[],
        statuses={"task:dqp-037": "in_progress"},
        counts={"tasks": 1, "ready": 0},
        digests={"tasks": "sha256:" + ("bb" * 32)},
    )
    report = rollout.evaluate_parity(
        production_snapshot=production,
        shadow_snapshot=shadow,
    )
    assert report.unexplained_drift_count == 0
    assert report.decision is ParityDecision.PASS
    assert report.passed is True
    assert report.reviewed_disposition_count >= 1
    reviewed = [
        item
        for item in report.drifts
        if item.disposition
        in (
            DriftDispositionKind.REVIEWED_ACCEPT,
            DriftDispositionKind.REVIEWED_IGNORE,
            DriftDispositionKind.REVIEWED_QUARANTINE,
        )
    ]
    assert reviewed


def test_invalid_disposition_rejected() -> None:
    with pytest.raises(ShadowDispositionError):
        DriftDisposition(
            surface="not_a_surface",
            kind=DriftDispositionKind.REVIEWED_ACCEPT,
            reason="bad",
        )
    with pytest.raises(ShadowDispositionError):
        DriftDisposition(
            surface="status",
            kind=DriftDispositionKind.MATCH,
            reason="match is automatic",
        )
    with pytest.raises(ShadowDispositionError):
        DriftDisposition(
            surface="status",
            kind=DriftDispositionKind.UNEXPLAINED,
            reason="cannot pre-register unexplained",
        )


# ---------------------------------------------------------------------------
# Shadow never controls production effect
# ---------------------------------------------------------------------------


def test_shadow_never_controls_production_effect(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start_shadow")

    effect = rollout.apply_production_effect(
        action="claim_task",
        payload={"task_id": "DQP-037", "owner": "worker-a"},
    )
    assert effect.channel == PRODUCTION_EFFECT_CHANNEL
    assert effect.action == "claim_task"
    assert len(rollout.production_effects) == 1

    with pytest.raises(ShadowAuthorityError, match="shadow channel cannot control"):
        rollout.apply_production_effect(
            action="claim_task",
            payload={"task_id": "DQP-037"},
            channel="shadow",
        )

    shadow_record = rollout.shadow_write(
        action="mirror_claim",
        payload={"task_id": "DQP-037", "owner": "worker-a"},
    )
    assert shadow_record["authoritative"] is False
    assert shadow_record["controls_production"] is False
    assert shadow_record["channel"] == "shadow"
    # Production effects remain only those applied via production channel.
    assert len(rollout.production_effects) == 1
    assert rollout.shadow_controls_production is False

    report = rollout.evaluate_parity()
    assert report.shadow_controls_production is False
    assert report.passed is True


def test_observe_decision_does_not_apply_production_effect(
    tmp_path: Path,
) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.OBSERVE, reason="observe")

    observation = rollout.observe_decision(
        kind="lifecycle_claim",
        production={"task_id": "DQP-037", "status": "in_progress", "fence": 1},
        shadow={"task_id": "DQP-037", "status": "in_progress", "fence": 1},
        surface="lease_fence",
    )
    assert observation.matched is True
    assert len(rollout.production_effects) == 0
    assert len(rollout.observations) == 1


# ---------------------------------------------------------------------------
# Dual observation bounded duration/retention
# ---------------------------------------------------------------------------


def test_dual_observation_duration_bound(tmp_path: Path) -> None:
    clock = {"now": 1_000_000.0}

    def _clock() -> float:
        return clock["now"]

    window = DualObservationWindow(max_duration_seconds=60, max_retention_records=100)
    rollout = _rollout(tmp_path, window=window, clock=_clock)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    assert rollout.window is not None
    assert rollout.window.opened_at
    assert rollout.window.expires_at

    rollout.observe_decision(
        kind="read_status",
        production={"status": "todo"},
        shadow={"status": "todo"},
        surface="status",
    )

    # Advance past expiry.
    clock["now"] = 1_000_000.0 + 61
    with pytest.raises(DualObservationBoundError, match="expired"):
        rollout.observe_decision(
            kind="read_status",
            production={"status": "todo"},
            shadow={"status": "todo"},
            surface="status",
        )
    with pytest.raises(DualObservationBoundError, match="expired"):
        rollout.evaluate_parity()


def test_dual_observation_retention_record_bound(tmp_path: Path) -> None:
    window = DualObservationWindow(
        max_duration_seconds=3600,
        max_retention_records=3,
    )
    rollout = _rollout(tmp_path, window=window)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.OBSERVE, reason="observe")

    for index in range(3):
        rollout.observe_decision(
            kind=f"read_{index}",
            production={"i": index},
            shadow={"i": index},
            surface="status",
        )
    with pytest.raises(DualObservationBoundError, match="retention record"):
        rollout.observe_decision(
            kind="read_overflow",
            production={"i": 99},
            shadow={"i": 99},
            surface="status",
        )


def test_dual_observation_window_rejects_over_ceiling() -> None:
    with pytest.raises(DualObservationBoundError, match="hard ceiling"):
        DualObservationWindow(
            max_duration_seconds=DEFAULT_MAX_DURATION_SECONDS + 1
        )
    with pytest.raises(DualObservationBoundError, match="hard ceiling"):
        DualObservationWindow(
            max_retention_records=DEFAULT_MAX_RETENTION_RECORDS + 1
        )


# ---------------------------------------------------------------------------
# Rollback and re-run preserve history / same parity decision
# ---------------------------------------------------------------------------


def test_rollback_preserves_history(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    report = rollout.evaluate_parity()
    assert report.passed is True

    history_before = rollout.history_digests()
    assert len(history_before) >= 2

    receipt = rollout.rollback(reason="kill_switch")
    assert receipt.history_preserved is True
    assert receipt.from_stage == ShadowRolloutStage.SHADOW.value
    assert receipt.to_stage == ShadowRolloutStage.OFF.value
    assert receipt.prior_parity_digest == report.parity_digest

    history_after = rollout.history_digests()
    assert len(history_after) == len(history_before) + 1
    # Prior digests are a prefix of post-rollback history (append-only).
    assert history_after[: len(history_before)] == history_before
    assert rollout.stage is ShadowRolloutStage.OFF
    # Import receipt and last parity remain available for audit.
    assert rollout.import_receipt is not None
    assert rollout.last_parity_report is not None
    assert rollout.last_parity_report.parity_digest == report.parity_digest


def test_rerun_generates_same_parity_decision(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    first = rollout.evaluate_parity()
    second = rollout.rerun_parity()

    assert second.decision is first.decision
    assert second.parity_digest == first.parity_digest
    assert second.passed is first.passed
    assert second.production_snapshot_digest == first.production_snapshot_digest
    assert second.shadow_snapshot_digest == first.shadow_snapshot_digest
    # History grew with the second parity evaluation.
    parity_entries = [
        entry for entry in rollout.history if entry.kind.value == "parity"
    ]
    assert len(parity_entries) >= 2
    assert parity_entries[0].payload["parity_digest"] == first.parity_digest
    assert parity_entries[1].payload["parity_digest"] == second.parity_digest


def test_rollback_then_rerun_same_decision(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    original = rollout.evaluate_parity()
    history_at_pass = list(rollout.history_digests())

    rollout.rollback(reason="operator_abort")
    # Re-enter shadow without re-import; history preserved.
    rollout.transition(ShadowRolloutStage.BACKFILL, reason="resume")
    rollout.transition(ShadowRolloutStage.SHADOW, reason="resume_shadow")
    resumed = rollout.rerun_parity()

    assert resumed.decision is original.decision
    assert resumed.parity_digest == original.parity_digest
    # All pre-rollback history digests remain.
    current = list(rollout.history_digests())
    for digest in history_at_pass:
        assert digest in current


# ---------------------------------------------------------------------------
# Stage machine / status / helpers
# ---------------------------------------------------------------------------


def test_illegal_stage_transition_refused(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    with pytest.raises(ShadowStageError):
        rollout.transition(ShadowRolloutStage.SHADOW, reason="skip_backfill")


def test_status_projection(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    rollout.evaluate_parity()
    status = rollout.status()
    assert status["interface"] == DATABASE_SHADOW_ROLLOUT_INTERFACE
    assert status["stage"] == "shadow"
    assert status["shadow_controls_production"] is False
    assert status["import_receipt_cid"]
    assert status["last_parity_decision"] == "pass"
    assert status["dual_observation_enabled"] is True


def test_run_shadow_parity_helper(tmp_path: Path) -> None:
    sources = [
        {
            "source_id": "tasks-json",
            "path": str(_json_tasks(tmp_path / "only_tasks.json")),
            "media_type": "json",
            "domain": "taskboards",
        }
    ]
    report = run_shadow_parity(
        work_dir=tmp_path / "helper",
        sources=sources,
        import_id="helper-import",
        durable=False,
    )
    assert report.passed is True
    assert report.import_reconciled is True
    assert report.decision is ParityDecision.PASS


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for durable apply")
def test_durable_duckdb_backfill_reconciles(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path, durable=True)
    receipt = _backfill(rollout, tmp_path, durable=True)
    assert receipt.outcome in (OUTCOME_APPLIED, OUTCOME_REPLAYED)
    assert receipt.applied is True
    replay = _backfill(rollout, tmp_path, durable=True)
    assert replay.outcome == OUTCOME_REPLAYED
    assert replay.receipt_cid == receipt.receipt_cid
    rollout.transition(ShadowRolloutStage.SHADOW, reason="durable")
    assert rollout.evaluate_parity().passed is True


def test_export_surface_carries_non_authority_marker(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    report = rollout.evaluate_parity()
    export_drift = next(
        item for item in report.drifts if item.surface == "export"
    )
    assert export_drift.matched is True
    assert (
        export_drift.production_value.get("non_authority_marker")
        == EXPORT_NON_AUTHORITY_MARKER
    )


def test_mismatched_observation_feeds_parity_failure(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    _backfill(rollout, tmp_path)
    rollout.transition(ShadowRolloutStage.SHADOW, reason="start")
    observation = rollout.observe_decision(
        kind="lifecycle_complete",
        production={"task_id": "DQP-037", "status": "completed"},
        shadow={"task_id": "DQP-037", "status": "failed"},
        surface="completion",
    )
    assert observation.matched is False
    report = rollout.evaluate_parity()
    # completion surface now differs unless disposition registered.
    completion = next(item for item in report.drifts if item.surface == "completion")
    assert completion.matched is False
    assert completion.disposition is DriftDispositionKind.UNEXPLAINED
    assert report.passed is False


def test_project_authority_snapshot_closed_surfaces() -> None:
    snapshot = project_authority_snapshot(
        tasks=[{"id": "T1", "task_cid": "cid:1"}],
        readiness=["cid:1"],
        events=[{"event_id": "e1"}],
        leases=[{"lease_id": "l1", "fence": 2}],
        statuses={"cid:1": "todo"},
        revisions={"cid:1": 0},
        exports=[{"path": "out.md"}],
        completions=[],
        restarts=[{"reason": "server"}],
        event_cursor="1",
    )
    assert set(snapshot) == set(AUTHORITY_SURFACES)
    assert snapshot["export"]["non_authority_marker"] == EXPORT_NON_AUTHORITY_MARKER
    assert snapshot["lease_fence"]["leases"][0]["fence"] == 2


def test_parity_requires_backfill_first(tmp_path: Path) -> None:
    rollout = _rollout(tmp_path)
    rollout.transition(ShadowRolloutStage.BACKFILL, reason="manual")
    with pytest.raises(ShadowParityError, match="reconciled backfill"):
        rollout.evaluate_parity()
