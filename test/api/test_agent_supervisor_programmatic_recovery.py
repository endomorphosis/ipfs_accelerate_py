from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    IncidentKind,
    ProgrammaticRecoveryExhaustionReceipt,
    RecordStatus,
    RecoveryAttemptOutcome,
    RescueOperation,
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.rescue.recovery_diagnostics import (
    RecoveryDiagnosis,
    diagnose_supervisor_incident,
)
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_recovery import (
    FaultInjector,
    ProgrammaticRecoveryController,
    ProgrammaticRecoveryPolicy,
    RecoveryDisposition,
    RecoveryIntegrityError,
)


def _diagnosis(
    tmp_path: Path,
    *,
    prior_actions: tuple[dict[str, object], ...] = (),
    **evidence: object,
) -> RecoveryDiagnosis:
    return diagnose_supervisor_incident(
        repository_root=str(tmp_path.resolve()),
        state_root=str((tmp_path / "state").resolve()),
        repository_root_cid=prompt_workflow_cid({"repository": "current"}),
        policy_root=prompt_workflow_cid({"policy": "current"}),
        run_cid=prompt_workflow_cid({"run": "current"}),
        prior_actions=prior_actions,
        observed_at_ms=1_000,
        **evidence,
    )


def _success(context: object) -> dict[str, object]:
    action = context.action  # type: ignore[attr-defined]
    return {
        "succeeded": True,
        "observed_effects": action.expected_effects,
        "post_action_health": {"healthy": True},
    }


def test_least_invasive_action_is_fully_bound_and_deduplicated(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, status={"projection_stale": True}, health={"healthy": True}
    )
    calls: list[RescueOperation] = []

    def reconcile(context: object) -> dict[str, object]:
        calls.append(context.action.operation)  # type: ignore[attr-defined]
        return _success(context)

    controller = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={
            RescueOperation.RECONCILE_PROJECTION: reconcile,
            RescueOperation.RESTART_LANE: reconcile,
        },
        clock_ms=lambda: 1_000,
    )
    result = controller.recover(diagnosis)
    duplicate = controller.recover(diagnosis)

    assert result.recovered
    assert result.receipt is not None
    assert result.receipt.operation is RescueOperation.RECONCILE_PROJECTION
    assert result.receipt.disposition is RecoveryDisposition.RECOVERED
    assert result.receipt.action.preconditions
    assert result.receipt.action.max_attempts == 2
    assert result.receipt.action.cooldown_ms == 30_000
    assert result.receipt.action.deadline_ms == 120_000
    assert result.receipt.action.expected_effects == ("projection_reconciled",)
    assert result.receipt.post_action_health["healthy"] is True
    assert calls == [RescueOperation.RECONCILE_PROJECTION]
    assert duplicate.deduplicated
    assert duplicate.terminal_cid == result.terminal_cid
    assert calls == [RescueOperation.RECONCILE_PROJECTION]


def test_injected_fault_is_retried_only_within_the_action_bound(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, lease={"lease_id": "lease-1", "expired": True}
    )
    injector = FaultInjector()
    injector.arm(
        "before_programmatic_recovery_action",
        OSError("injected"),
        times=1,
    )
    controller = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.REPAIR_EXPIRED_LEASE: _success},
        fault_injector=injector,
        clock_ms=lambda: 1_000,
    )

    result = controller.recover(diagnosis)

    assert result.recovered
    assert [item.outcome for item in result.attempts] == [
        RecoveryAttemptOutcome.FAILED,
        RecoveryAttemptOutcome.SUCCEEDED,
    ]
    assert [item.attempt for item in result.attempts] == [1, 2]


def test_failed_effect_or_health_attestation_emits_current_exhaustion(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, validation={"validation_id": "v1", "failed": True}
    )
    calls = 0

    def incomplete(_context: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "succeeded": True,
            "observed_effects": (),
            "post_action_health": {"healthy": False},
        }

    controller = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.VALIDATION_REPLAY: incomplete},
        policy=ProgrammaticRecoveryPolicy(max_attempts_per_action=2),
        clock_ms=lambda: 1_000,
    )
    result = controller.recover(diagnosis)
    duplicate = controller.recover(diagnosis)

    assert not result.recovered
    assert isinstance(
        result.exhaustion_receipt,
        ProgrammaticRecoveryExhaustionReceipt,
    )
    exhaustion = result.exhaustion_receipt
    assert exhaustion is not None
    assert exhaustion.incident_cid == diagnosis.incident_cid
    assert exhaustion.repository_root_cid == diagnosis.incident.repository_root_cid
    assert exhaustion.policy_root == diagnosis.incident.policy_root
    assert exhaustion.run_cid == diagnosis.incident.run_cid
    assert exhaustion.status is RecordStatus.QUARANTINED
    assert exhaustion.created_at_ms == 1_000
    assert exhaustion.updated_at_ms == 1_000
    assert len(exhaustion.attempts) == 2
    assert duplicate.deduplicated
    assert duplicate.terminal_cid == result.terminal_cid
    assert calls == 2


def test_retry_failure_falls_through_to_one_lane_restart(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, process={"lane_id": "lane-1", "failed": True}
    )
    operations: list[RescueOperation] = []

    def fail(context: object) -> bool:
        operations.append(context.action.operation)  # type: ignore[attr-defined]
        return False

    def restart(context: object) -> dict[str, object]:
        operations.append(context.action.operation)  # type: ignore[attr-defined]
        return _success(context)

    result = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={
            RescueOperation.RETRY: fail,
            RescueOperation.RESTART_LANE: restart,
        },
        policy=ProgrammaticRecoveryPolicy(max_attempts_per_action=1),
        clock_ms=lambda: 1_000,
    ).recover(diagnosis)

    assert result.recovered
    assert result.receipt is not None
    assert result.receipt.operation is RescueOperation.RESTART_LANE
    assert operations == [RescueOperation.RETRY, RescueOperation.RESTART_LANE]


def test_cooldown_prevents_reexecution_and_produces_exhaustion(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path,
        prior_actions=(
            {
                "operation": "retry",
                "outcome": "failed",
                "finished_at_ms": 900,
            },
        ),
        attempt={"attempt_id": "attempt-1", "consumed": True},
    )
    calls = 0

    def retry(context: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _success(context)

    result = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.RETRY: retry},
        policy=ProgrammaticRecoveryPolicy(cooldown_ms=1_000),
        clock_ms=lambda: 1_000,
    ).recover(diagnosis)

    assert calls == 0
    assert result.exhaustion_receipt is not None
    assert result.exhaustion_receipt.inapplicable_operations == (
        RescueOperation.RETRY,
    )
    assert result.exhaustion_receipt.exhaustion_reason == "action_cooldown_active"


def test_cooldown_exhaustion_expires_before_recovery_is_retried(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path,
        prior_actions=(
            {
                "operation": "retry",
                "outcome": "failed",
                "finished_at_ms": 900,
            },
        ),
        attempt={"attempt_id": "attempt-1", "consumed": True},
    )
    now = [1_000]
    calls = 0

    def retry(context: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _success(context)

    controller = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.RETRY: retry},
        policy=ProgrammaticRecoveryPolicy(cooldown_ms=1_000),
        clock_ms=lambda: now[0],
    )
    exhausted = controller.recover(diagnosis)
    now[0] = 2_000
    recovered = controller.recover(diagnosis)

    assert exhausted.exhaustion_receipt is not None
    assert recovered.recovered
    assert not recovered.deduplicated
    assert calls == 1


def test_corrupt_scope_quarantines_instead_of_looping(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path,
        task_source={"task_id": "task-1", "digest_mismatch": True},
    )
    assert diagnosis.kind is IncidentKind.CORRUPT_TASK_SOURCE

    result = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.QUARANTINE: _success},
        clock_ms=lambda: 1_000,
    ).recover(diagnosis)

    assert result.quarantined
    assert result.exhaustion_receipt is None
    assert result.receipt is not None
    assert result.receipt.operation is RescueOperation.QUARANTINE
    assert result.receipt.disposition is RecoveryDisposition.QUARANTINED


def test_no_registered_action_exhausts_without_invoking_a_model(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, provider={"provider_id": "p1", "unavailable": True}
    )
    result = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={},
        clock_ms=lambda: 1_000,
    ).recover(diagnosis)

    assert result.exhaustion_receipt is not None
    assert result.exhaustion_receipt.inapplicable_operations == (
        RescueOperation.REASSIGN_INDEPENDENT_WORK,
    )
    assert result.attempts == ()


def test_concurrent_controllers_collapse_an_unchanged_incident(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, lock={"lock_id": "lock-1", "orphaned": True}
    )
    calls = 0
    guard = threading.Lock()

    def repair(context: object) -> dict[str, object]:
        nonlocal calls
        with guard:
            calls += 1
        return _success(context)

    def run() -> str:
        return ProgrammaticRecoveryController(
            tmp_path / "recovery",
            handlers={RescueOperation.REPAIR_ORPHANED_LOCK: repair},
            clock_ms=lambda: 1_000,
        ).recover(diagnosis).terminal_cid

    with ThreadPoolExecutor(max_workers=4) as pool:
        terminal_cids = tuple(pool.map(lambda _item: run(), range(4)))

    assert calls == 1
    assert len(set(terminal_cids)) == 1


def test_success_requires_explicit_exact_effects_and_health(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, status={"projection_stale": True}, health={"healthy": True}
    )
    controller = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={
            RescueOperation.RECONCILE_PROJECTION: lambda _context: True
        },
        policy=ProgrammaticRecoveryPolicy(max_attempts_per_action=1),
        clock_ms=lambda: 1_000,
    )

    result = controller.recover(diagnosis)

    assert not result.recovered
    assert result.exhaustion_receipt is not None
    assert result.attempts[0].outcome is RecoveryAttemptOutcome.FAILED


def test_loaded_terminal_must_match_all_current_incident_bindings(
    tmp_path: Path,
) -> None:
    first = _diagnosis(
        tmp_path, lock={"lock_id": "lock-1", "orphaned": True}
    )
    second = _diagnosis(
        tmp_path, lock={"lock_id": "lock-2", "orphaned": True}
    )
    controller = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.REPAIR_ORPHANED_LOCK: _success},
        clock_ms=lambda: 1_000,
    )
    controller.recover(first)
    wrapper = json.loads(
        controller._result_path(first.incident_cid).read_text(
            encoding="utf-8"
        )
    )
    wrapper["incident_cid"] = second.incident_cid
    target = controller._result_path(second.incident_cid)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(wrapper), encoding="utf-8")

    with pytest.raises(RecoveryIntegrityError, match="binding mismatch"):
        controller.recover(second)


def test_explicit_precondition_denial_prevents_effect(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path,
        lease={
            "lease_id": "lease-1",
            "expired": True,
            "fence_current": False,
        },
    )
    calls = 0

    def repair(context: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _success(context)

    result = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={RescueOperation.REPAIR_EXPIRED_LEASE: repair},
        clock_ms=lambda: 1_000,
    ).recover(diagnosis)

    assert calls == 0
    assert result.exhaustion_receipt is not None
    assert result.exhaustion_receipt.exhaustion_reason == (
        "action_precondition_denied"
    )


def test_partial_lane_effect_requires_verified_compensation(
    tmp_path: Path,
) -> None:
    diagnosis = _diagnosis(
        tmp_path, process={"lane_id": "lane-1", "failed": True}
    )
    restart_attempts = 0

    def restart(context: object) -> dict[str, object]:
        nonlocal restart_attempts
        restart_attempts += 1
        if restart_attempts == 1:
            return {
                "succeeded": False,
                "observed_effects": ("old_lane_fenced",),
                "post_action_health": {"healthy": False},
                "partial": True,
            }
        return _success(context)

    result = ProgrammaticRecoveryController(
        tmp_path / "recovery",
        handlers={
            RescueOperation.RESTART_LANE: restart,
            RescueOperation.STOP: _success,
        },
        clock_ms=lambda: 1_000,
    ).recover(diagnosis)

    assert result.recovered
    assert result.receipt is not None
    assert result.receipt.compensations == (RescueOperation.STOP,)
    assert [item.operation for item in result.attempts] == [
        RescueOperation.RESTART_LANE,
        RescueOperation.STOP,
        RescueOperation.RESTART_LANE,
    ]
