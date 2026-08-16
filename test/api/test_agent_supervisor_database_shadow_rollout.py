"""Tests for DatabaseShadowRollout@1 / ShadowParityReport@1 (DQP-037)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.database_shadow_rollout import (
    DATABASE_SHADOW_ROLLOUT_INTERFACE,
    DEFAULT_MAX_DUAL_OBSERVATION_SECONDS,
    DEFAULT_RETENTION_SECONDS,
    EVIDENCE,
    GOAL_ID,
    PARITY_DOMAINS,
    SHADOW_PARITY_REPORT_INTERFACE,
    TASK_ID,
    DatabaseShadowRollout,
    DriftDisposition,
    DriftRecord,
    DriftSeverity,
    LegacyDecision,
    LegacyRecord,
    ParityVerdict,
    ShadowParityReport,
    ShadowRolloutError,
    default_hermetic_program,
    run_database_shadow_rollout,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import (
    StateAuthorityMode,
)


def test_interface_identities() -> None:
    assert DATABASE_SHADOW_ROLLOUT_INTERFACE == "DatabaseShadowRollout@1"
    assert SHADOW_PARITY_REPORT_INTERFACE == "ShadowParityReport@1"
    assert DatabaseShadowRollout.INTERFACE == DATABASE_SHADOW_ROLLOUT_INTERFACE
    assert ShadowParityReport.INTERFACE == SHADOW_PARITY_REPORT_INTERFACE
    assert TASK_ID == "DQP-037"
    assert GOAL_ID == "DQP-G080"
    assert EVIDENCE == "dqp/database-shadow-rollout@1"


def test_exact_import_reconciles_and_parity_passes() -> None:
    report = run_database_shadow_rollout()
    assert report.verdict is ParityVerdict.PARITY
    assert report.passed is True
    assert report.backfill.exact_reconcile is True
    assert report.backfill.record_count >= 3
    assert report.decisions_mirrored == 2
    assert report.unexplained_authority_drift == 0
    assert report.production_effect is False
    assert report.authority_mode == StateAuthorityMode.QUACK_SHADOW.value
    assert set(report.domains_compared) == set(PARITY_DOMAINS)
    assert report.history_preserved is True
    assert report.parity_decision_stable is True
    assert report.dual_observation_seconds <= DEFAULT_MAX_DUAL_OBSERVATION_SECONDS
    assert report.retention_seconds == DEFAULT_RETENTION_SECONDS
    payload = report.to_dict()
    assert payload["interface"] == SHADOW_PARITY_REPORT_INTERFACE
    assert payload["production_effect"] is False
    assert payload["task_id"] == "DQP-037"


def test_shadow_never_controls_production_effect() -> None:
    rollout = DatabaseShadowRollout()
    records, decisions = default_hermetic_program()
    report = rollout.run(records, decisions)
    assert report.production_effect is False
    assert rollout._production_effects == []
    for tx_event in rollout._history:
        if tx_event.get("event") == "mirror_decision":
            assert tx_event.get("production_effect") is False


def test_exact_replay_backfill_is_noop() -> None:
    rollout = DatabaseShadowRollout()
    records, _ = default_hermetic_program()
    first = rollout.backfill(records, import_id="import:replay")
    second = rollout.backfill(records, import_id="import:replay")
    assert first.exact_reconcile is True
    assert second.replayed is True
    assert second.digest == first.digest
    assert second.record_count == first.record_count


def test_unexplained_authority_drift_fails_closed() -> None:
    injected = DriftRecord(
        domain="tasks",
        record_id="TASK-1",
        field="status",
        legacy_value="ready",
        shadow_value="corrupted",
        severity=DriftSeverity.AUTHORITY,
        reason_code="injected",
    )
    report = run_database_shadow_rollout(inject_drift=(injected,))
    assert report.verdict is ParityVerdict.DRIFT_UNEXPLAINED
    assert report.passed is False
    assert report.unexplained_authority_drift == 1
    assert "unexplained_authority_drift" in report.reason_codes


def test_reviewed_disposition_allows_drift_reviewed_verdict() -> None:
    injected = DriftRecord(
        domain="tasks",
        record_id="TASK-1",
        field="status",
        legacy_value="ready",
        shadow_value="observed-diff",
        severity=DriftSeverity.AUTHORITY,
        reason_code="known-clock-skew",
    )
    report = run_database_shadow_rollout(
        inject_drift=(injected,),
        dispositions={
            ("tasks", "TASK-1", "status"): DriftDisposition.ACCEPT_LEGACY,
        },
    )
    assert report.verdict is ParityVerdict.DRIFT_REVIEWED
    assert report.passed is True
    assert report.unexplained_authority_drift == 0
    assert all(item.reviewed for item in report.drifts)


def test_rollback_and_rerun_preserve_history_and_stable_decision() -> None:
    rollout = DatabaseShadowRollout()
    records, decisions = default_hermetic_program()
    first = rollout.run(records, decisions)
    history_len_after_first = len(rollout._history)
    rollout.rollback()
    assert rollout.authority_mode is StateAuthorityMode.EMBEDDED_MAINTENANCE
    second = rollout.re_run(records, decisions)
    assert second.verdict is first.verdict
    assert second.parity_decision_stable is True
    assert second.history_preserved is True
    assert len(rollout._history) > history_len_after_first
    assert any(item.get("event") == "history_preserved" for item in rollout._history)
    # Same parity decision for same inputs.
    assert second.verdict is ParityVerdict.PARITY
    assert second.backfill.exact_reconcile is True


def test_dual_observation_bound_enforced() -> None:
    rollout = DatabaseShadowRollout(max_dual_observation_seconds=60)
    records, decisions = default_hermetic_program()
    with pytest.raises(ShadowRolloutError, match="dual observation"):
        rollout.run(records, decisions, dual_observation_seconds=120)


def test_mirrored_decision_updates_shadow_revision() -> None:
    rollout = DatabaseShadowRollout()
    records, decisions = default_hermetic_program()
    rollout.backfill(records)
    tx = rollout.mirror_decision(decisions[0])
    assert tx["authoritative"] is False
    assert tx["production_effect"] is False
    assert tx["revision"] == decisions[0].revision
    key = ("tasks", "TASK-1")
    assert rollout._shadow[key]["revision"] == decisions[0].revision
    assert rollout._shadow[key]["status"] == "claimed"


def test_digests_present_for_evidence_subset() -> None:
    report = run_database_shadow_rollout()
    for key in ("legacy", "shadow", "decisions", "history"):
        assert key in report.digests
        assert report.digests[key].startswith("sha256:")
