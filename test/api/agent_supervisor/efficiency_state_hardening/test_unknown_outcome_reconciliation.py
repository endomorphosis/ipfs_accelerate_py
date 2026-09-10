"""Unknown provider outcomes must preserve truth before recovery advances."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store import (
    DurableProviderAttemptCAS,
)
from ipfs_accelerate_py.agent_supervisor.rescue.database_recovery import (
    ActionKind,
    ActionStatus,
    DatabaseRecoveryError,
    SubjectKind,
    SubjectStatus,
    duckdb_available,
    open_database_recovery,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(), reason="DuckDB is required for recovery tests"
)


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "provider_available": True,
        "effect_observed": False,
        "effect_absent": False,
        "receipt_observed": False,
        "idempotency_observed": False,
        "idempotency_key": "",
        "completion_observed": False,
        "completion_succeeded": False,
        "compensation_observed": False,
        "effect_idempotent": True,
    }
    evidence.update(overrides)
    return evidence


def _recovery(tmp_path: Path):
    return open_database_recovery(tmp_path / "recovery.duckdb", max_retries=3)


def _unknown_subject(recovery, ref: str = "attempt-1"):
    subject = recovery.register_subject(
        subject_kind=SubjectKind.ATTEMPT,
        subject_ref=ref,
        attempt_id=ref,
        task_cid="task:unknown-outcome",
    )
    return recovery.mark_provider_outcome_unknown(
        subject_id=subject.subject_id,
        reason="provider_timeout_after_dispatch",
        observation={"timeout_ms": 30_000},
    )


def test_timeout_freezes_the_exact_effect_reservation(tmp_path: Path) -> None:
    store = DurableProviderAttemptCAS(tmp_path / "attempts")
    launch_context = {
        "provider_id": "provider:test",
        "command_id": _sha("command"),
        "runtime_id": _sha("runtime"),
        "image_id": _sha("image"),
        "mount_id": _sha("mount"),
        "environment_id": _sha("environment"),
        "container_name": "ipfs-accelerate-codex-1-" + "a" * 32,
        "container_id": _sha("container"),
    }
    claimed = store.reserve_or_adopt(
        logical_attempt_id="attempt:unknown",
        route_id="route:test",
        decision_id="decision:test",
        task_id="task:test",
        worktree_id="worktree:test",
        authorized=True,
        launch_context=launch_context,
        now_ms=100,
    )
    unknown = store.mark_provider_outcome_unknown(
        claimed.reservation, reason="provider_connection_loss", now_ms=101
    )
    assert unknown.state == "provider_outcome_unknown"
    assert unknown.effect_launch_receipt == claimed.effect_launch_receipt
    adopted = store.claim_effect(unknown, launch_context=launch_context, now_ms=102)
    assert adopted.launch_authorized is False
    assert adopted.reservation.state == "provider_outcome_unknown"


def test_unknown_outcome_rejects_blind_retry_and_waits_for_unavailable_provider(
    tmp_path: Path,
) -> None:
    recovery = _recovery(tmp_path)
    try:
        subject = _unknown_subject(recovery)
        assert subject.status is SubjectStatus.PROVIDER_OUTCOME_UNKNOWN
        with pytest.raises(DatabaseRecoveryError, match="inadmissible|context"):
            recovery.decide_action(
                subject_id=subject.subject_id,
                action_kind=ActionKind.RETRY,
                idempotency_key="blind-retry",
            )
        action = recovery.reconcile_provider_outcome_unknown(
            subject_id=subject.subject_id,
            idempotency_key="provider-unavailable",
            evidence=_evidence(provider_available=False),
        )
        assert action.action_kind is ActionKind.RECONCILE
        assert recovery.get_subject(subject.subject_id).status is SubjectStatus.RECONCILIATION_PENDING
    finally:
        recovery.close()


def test_late_completion_and_duplicate_delivery_terminalize_once(tmp_path: Path) -> None:
    recovery = _recovery(tmp_path)
    try:
        subject = _unknown_subject(recovery)
        evidence = _evidence(
            effect_observed=True,
            receipt_observed=True,
            idempotency_observed=True,
            idempotency_key="provider:attempt:1",
            completion_observed=True,
            completion_succeeded=True,
        )
        first = recovery.reconcile_provider_outcome_unknown(
            subject_id=subject.subject_id,
            idempotency_key="late-completion",
            evidence=evidence,
        )
        second = recovery.reconcile_provider_outcome_unknown(
            subject_id=subject.subject_id,
            idempotency_key="late-completion",
            evidence=evidence,
        )
        assert first.action_kind is ActionKind.TERMINALIZE
        assert second.action_id == first.action_id
        assert second.status is ActionStatus.REPLAYED
        assert recovery.get_subject(subject.subject_id).status is SubjectStatus.TERMINAL_SUCCEEDED
    finally:
        recovery.close()


def test_compensation_and_non_idempotent_effect_choose_safe_transitions(
    tmp_path: Path,
) -> None:
    recovery = _recovery(tmp_path)
    try:
        compensated = _unknown_subject(recovery, "compensate")
        action = recovery.reconcile_provider_outcome_unknown(
            subject_id=compensated.subject_id,
            idempotency_key="compensate:1",
            evidence=_evidence(
                effect_observed=True,
                receipt_observed=True,
                idempotency_observed=True,
                idempotency_key="effect:compensate",
                compensation_observed=True,
            ),
        )
        assert action.action_kind is ActionKind.COMPENSATE
        assert recovery.get_subject(compensated.subject_id).status is SubjectStatus.COMPENSATED

        non_idempotent = _unknown_subject(recovery, "non-idempotent")
        action = recovery.reconcile_provider_outcome_unknown(
            subject_id=non_idempotent.subject_id,
            idempotency_key="non-idempotent:1",
            evidence=_evidence(
                effect_absent=True,
                receipt_observed=True,
                idempotency_observed=True,
                idempotency_key="effect:non-idempotent",
                effect_idempotent=False,
            ),
        )
        assert action.action_kind is ActionKind.QUARANTINE
        assert recovery.get_subject(non_idempotent.subject_id).status is SubjectStatus.QUARANTINED
    finally:
        recovery.close()


def test_observed_absence_allows_only_idempotent_retry(tmp_path: Path) -> None:
    recovery = _recovery(tmp_path)
    try:
        subject = _unknown_subject(recovery, "retry")
        action = recovery.reconcile_provider_outcome_unknown(
            subject_id=subject.subject_id,
            idempotency_key="retry:1",
            evidence=_evidence(
                effect_absent=True,
                receipt_observed=True,
                idempotency_observed=True,
                idempotency_key="effect:retry",
                effect_idempotent=True,
            ),
        )
        assert action.action_kind is ActionKind.RETRY
        assert recovery.get_subject(subject.subject_id).status is SubjectStatus.RETRY_PENDING
    finally:
        recovery.close()
