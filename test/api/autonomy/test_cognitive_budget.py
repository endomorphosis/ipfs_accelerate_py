from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomy.cognitive_budget import (
    BudgetDimension,
    BudgetExhaustion,
    BudgetExhaustionReason,
    CognitiveBudgetProtocolError,
    CognitiveCost,
    ObjectiveCognitiveBudgetLedger,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AutonomyContractError,
    BudgetLedger,
    BudgetPurpose,
    BudgetReservation,
    BudgetReservationStatus,
    CognitiveBudget,
    TerminalStatus,
)


def _budget(**overrides: int) -> CognitiveBudget:
    values = {
        "max_total_model_calls": 10,
        "max_strong_model_calls": 3,
        "max_input_tokens": 1_000,
        "max_output_tokens": 500,
        "max_provider_spend_micros": 100_000,
        "max_proof_time_ms": 80,
        "max_validation_time_ms": 100,
        "max_human_questions": 2,
        "max_repair_rounds": 3,
        "max_plan_branches": 4,
        "max_context_expansions": 5,
        "max_wall_time_ms": 1_000,
        "validation_reserve_ms": 40,
        "proof_reserve_ms": 20,
    }
    values.update(overrides)
    return CognitiveBudget(**values)


def _reserve(
    ledger: ObjectiveCognitiveBudgetLedger,
    key: str,
    cost: CognitiveCost,
    *,
    purpose: BudgetPurpose = BudgetPurpose.ANALYSIS,
) -> BudgetReservation | BudgetExhaustion:
    return ledger.reserve(
        idempotency_key=key,
        question_id=f"question-{key}",
        action_id=f"action-{key}",
        purpose=purpose,
        requested=cost,
        expires_at_ms=10_000,
    )


def test_reserve_before_start_binds_every_cost_to_question_and_action() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(), epoch=7)
    with pytest.raises(CognitiveBudgetProtocolError, match="without a current reservation"):
        ledger.reconcile("unknown", CognitiveCost(wall_time_ms=1))

    result = _reserve(
        ledger,
        "static-1",
        CognitiveCost(validation_time_ms=10, wall_time_ms=20),
    )
    assert isinstance(result, BudgetReservation)
    assert result.status is BudgetReservationStatus.RESERVED
    assert result.idempotency_key == "static-1"
    assert result.question_id == "question-static-1"
    assert result.action_id == "action-static-1"
    assert result.purpose is BudgetPurpose.ANALYSIS
    assert result.max_validation_time_ms == 10
    assert result.max_wall_time_ms == 20
    snapshot = ledger.snapshot()
    assert snapshot.epoch == 7
    assert snapshot.reservations == (result,)
    assert snapshot.committed_wall_time_ms == 0


def test_reservation_replay_is_idempotent_and_conflicts_fail_closed() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget())
    first = _reserve(ledger, "same", CognitiveCost(input_tokens=100))
    second = _reserve(ledger, "same", CognitiveCost(input_tokens=100))
    assert isinstance(first, BudgetReservation)
    assert second is first
    assert len(ledger.snapshot().reservations) == 1

    with pytest.raises(CognitiveBudgetProtocolError, match="different reservation"):
        _reserve(ledger, "same", CognitiveCost(input_tokens=101))


def test_capacity_exhaustion_is_typed_and_release_restores_capacity() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_input_tokens=100))
    first = _reserve(ledger, "first", CognitiveCost(input_tokens=80))
    assert isinstance(first, BudgetReservation)
    rejected = _reserve(ledger, "second", CognitiveCost(input_tokens=21))
    assert isinstance(rejected, BudgetExhaustion)
    assert rejected.dimension is BudgetDimension.INPUT_TOKENS
    assert rejected.reason is BudgetExhaustionReason.CAPACITY_EXHAUSTED
    assert rejected.requested == 21
    assert rejected.available == 20
    assert rejected.terminal_status is TerminalStatus.EXHAUSTED
    assert len(ledger.snapshot().reservations) == 1
    assert ledger.snapshot().exhaustions == (rejected,)
    assert _reserve(ledger, "second", CognitiveCost(input_tokens=21)) is rejected

    released = ledger.release(first.reservation_id)
    assert released.status is BudgetReservationStatus.RELEASED
    assert ledger.release(first.reservation_id).content_id == released.content_id
    admitted = _reserve(ledger, "second-retry", CognitiveCost(input_tokens=21))
    assert isinstance(admitted, BudgetReservation)


@pytest.mark.parametrize(
    ("purpose", "cost", "dimension", "reason", "available", "reserve"),
    (
        (
            BudgetPurpose.PLANNING,
            CognitiveCost(validation_time_ms=61),
            BudgetDimension.VALIDATION_TIME_MS,
            BudgetExhaustionReason.VALIDATION_RESERVE_PROTECTED,
            60,
            40,
        ),
        (
            BudgetPurpose.ANALYSIS,
            CognitiveCost(proof_time_ms=61),
            BudgetDimension.PROOF_TIME_MS,
            BudgetExhaustionReason.PROOF_RESERVE_PROTECTED,
            60,
            20,
        ),
    ),
)
def test_validation_and_proof_reserves_are_protected(
    purpose: BudgetPurpose,
    cost: CognitiveCost,
    dimension: BudgetDimension,
    reason: BudgetExhaustionReason,
    available: int,
    reserve: int,
) -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget())
    result = _reserve(ledger, "protected", cost, purpose=purpose)
    assert isinstance(result, BudgetExhaustion)
    assert result.dimension is dimension
    assert result.reason is reason
    assert result.available == available
    assert result.protected_reserve == reserve


def test_validation_and_proof_actions_may_use_their_own_reserves() -> None:
    validation = ObjectiveCognitiveBudgetLedger(_budget())
    validation_result = _reserve(
        validation,
        "validation",
        CognitiveCost(validation_time_ms=100),
        purpose=BudgetPurpose.VALIDATION,
    )
    assert isinstance(validation_result, BudgetReservation)

    proof = ObjectiveCognitiveBudgetLedger(_budget())
    proof_result = _reserve(
        proof,
        "proof",
        CognitiveCost(proof_time_ms=80),
        purpose=BudgetPurpose.PROOF,
    )
    assert isinstance(proof_result, BudgetReservation)


def test_reconcile_actual_usage_releases_unused_capacity_and_is_idempotent() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_input_tokens=150))
    reservation = _reserve(
        ledger,
        "model",
        CognitiveCost(
            total_model_calls=1,
            input_tokens=120,
            output_tokens=30,
            provider_spend_micros=1_000,
            wall_time_ms=100,
        ),
        purpose=BudgetPurpose.MODEL,
    )
    assert isinstance(reservation, BudgetReservation)
    actual = CognitiveCost(
        total_model_calls=1,
        input_tokens=80,
        output_tokens=20,
        provider_spend_micros=900,
        wall_time_ms=90,
    )
    reconciled = ledger.reconcile(
        reservation.reservation_id,
        actual,
        provider_usage_receipt_ids=("provider-usage-1",),
        token_measurement_ids=("token-ledger-1",),
    )
    assert reconciled.status is BudgetReservationStatus.RECONCILED
    assert reconciled.actual_input_tokens == 80
    assert reconciled.question_id == reservation.question_id
    assert reconciled.action_id == reservation.action_id
    assert (
        ledger.reconcile(
            reservation.reservation_id,
            actual,
            provider_usage_receipt_ids=("provider-usage-1",),
            token_measurement_ids=("token-ledger-1",),
        ).content_id
        == reconciled.content_id
    )
    snapshot = ledger.snapshot()
    assert snapshot.committed_input_tokens == 80
    assert snapshot.committed_output_tokens == 20
    assert snapshot.provider_usage_receipt_ids == ("provider-usage-1",)
    assert snapshot.token_measurement_ids == ("token-ledger-1",)
    assert ledger.available(BudgetPurpose.MODEL).input_tokens == 70

    with pytest.raises(CognitiveBudgetProtocolError, match="conflicts"):
        ledger.reconcile(
            reservation.reservation_id,
            CognitiveCost(input_tokens=81),
            token_measurement_ids=("token-ledger-1",),
        )


def test_provider_and_token_actuals_require_existing_authority_receipts() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget())
    reservation = _reserve(
        ledger,
        "provider",
        CognitiveCost(total_model_calls=1, input_tokens=10),
        purpose=BudgetPurpose.MODEL,
    )
    assert isinstance(reservation, BudgetReservation)
    with pytest.raises(AutonomyContractError, match="token-measurement"):
        ledger.reconcile(
            reservation.reservation_id,
            CognitiveCost(total_model_calls=1, input_tokens=10),
            provider_usage_receipt_ids=("provider-usage-1",),
        )
    with pytest.raises(AutonomyContractError, match="provider-usage"):
        ledger.reconcile(
            reservation.reservation_id,
            CognitiveCost(total_model_calls=1, input_tokens=10),
            token_measurement_ids=("token-ledger-1",),
        )


def test_observed_overrun_is_attributed_then_closes_ledger_fail_closed() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_input_tokens=100))
    reservation = _reserve(ledger, "overrun", CognitiveCost(input_tokens=90))
    assert isinstance(reservation, BudgetReservation)
    reconciled = ledger.reconcile(
        reservation.reservation_id,
        CognitiveCost(input_tokens=110),
        token_measurement_ids=("token-ledger-overrun",),
    )
    assert reconciled.actual_input_tokens == 110
    snapshot = ledger.snapshot()
    assert snapshot.status is TerminalStatus.EXHAUSTED
    assert snapshot.committed_input_tokens == 110

    rejected = _reserve(ledger, "after-overrun", CognitiveCost())
    assert isinstance(rejected, BudgetExhaustion)
    assert rejected.reason is BudgetExhaustionReason.LEDGER_TERMINAL


def test_reservation_overrun_closes_ledger_with_global_capacity_remaining() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_wall_time_ms=1_000))
    reservation = _reserve(ledger, "bounded-action", CognitiveCost(wall_time_ms=10))
    assert isinstance(reservation, BudgetReservation)

    reconciled = ledger.reconcile(
        reservation.reservation_id,
        CognitiveCost(wall_time_ms=11),
    )
    assert reconciled.actual_wall_time_ms == 11
    snapshot = ledger.snapshot()
    assert snapshot.committed_wall_time_ms == 11
    assert snapshot.status is TerminalStatus.EXHAUSTED

    restarted = ObjectiveCognitiveBudgetLedger.from_snapshot(snapshot)
    rejected = _reserve(restarted, "next-action", CognitiveCost())
    assert isinstance(rejected, BudgetExhaustion)
    assert rejected.reason is BudgetExhaustionReason.LEDGER_TERMINAL


def test_canonical_snapshot_restarts_without_duplicate_spend() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(), epoch=3)
    completed = _reserve(ledger, "completed", CognitiveCost(wall_time_ms=30))
    pending = _reserve(ledger, "pending", CognitiveCost(wall_time_ms=20))
    assert isinstance(completed, BudgetReservation)
    assert isinstance(pending, BudgetReservation)
    ledger.reconcile(completed.reservation_id, CognitiveCost(wall_time_ms=25))

    snapshot = ledger.snapshot()
    decoded = BudgetLedger.from_json(snapshot.to_json())
    restarted = ObjectiveCognitiveBudgetLedger.from_snapshot(decoded)
    assert restarted.snapshot().content_id == snapshot.content_id
    replay = _reserve(restarted, "pending", CognitiveCost(wall_time_ms=20))
    assert isinstance(replay, BudgetReservation)
    assert replay.content_id == pending.content_id
    assert len(restarted.snapshot().reservations) == 2
    restarted.reconcile(pending.reservation_id, CognitiveCost(wall_time_ms=18))
    assert restarted.snapshot().committed_wall_time_ms == 43


def test_exhaustion_replay_survives_canonical_restart() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_input_tokens=1))
    rejected = _reserve(ledger, "rejected", CognitiveCost(input_tokens=2))
    assert isinstance(rejected, BudgetExhaustion)
    snapshot = BudgetLedger.from_json(ledger.snapshot().to_json())
    restarted = ObjectiveCognitiveBudgetLedger.from_snapshot(snapshot)
    replay = _reserve(restarted, "rejected", CognitiveCost(input_tokens=2))
    assert isinstance(replay, BudgetExhaustion)
    assert replay.content_id == rejected.content_id
    with pytest.raises(CognitiveBudgetProtocolError, match="different reservation"):
        _reserve(restarted, "rejected", CognitiveCost(input_tokens=3))


def test_every_cost_dimension_is_exactly_attributed_to_one_reservation() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget())
    cost = CognitiveCost(
        total_model_calls=2,
        strong_model_calls=1,
        input_tokens=100,
        output_tokens=50,
        provider_spend_micros=10_000,
        proof_time_ms=10,
        validation_time_ms=10,
        human_questions=1,
        repair_rounds=1,
        plan_branches=1,
        context_expansions=1,
        wall_time_ms=100,
    )
    reservation = _reserve(
        ledger,
        "all-costs",
        cost,
        purpose=BudgetPurpose.REPAIR,
    )
    assert isinstance(reservation, BudgetReservation)
    ledger.reconcile(
        reservation.reservation_id,
        cost,
        provider_usage_receipt_ids=("provider-usage-all",),
        token_measurement_ids=("token-ledger-all",),
    )
    snapshot = ledger.snapshot()
    for dimension in BudgetDimension:
        assert getattr(snapshot, "committed_" + dimension.value) == cost[dimension]
    assert snapshot.reservations[0].question_id == "question-all-costs"
    assert snapshot.reservations[0].action_id == "action-all-costs"


def test_typed_exhaustion_is_canonical_closed_and_identity_checked() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_input_tokens=1))
    result = _reserve(ledger, "too-large", CognitiveCost(input_tokens=2))
    assert isinstance(result, BudgetExhaustion)
    rebuilt = BudgetExhaustion.from_json(result.to_json())
    assert rebuilt.content_id == result.content_id
    assert BudgetExhaustion.from_dict(result.to_record()).content_id == result.content_id

    unknown = result.to_dict()
    unknown["extra"] = True
    with pytest.raises(AutonomyContractError, match="unsupported"):
        BudgetExhaustion.from_dict(unknown)
    forged = result.to_record()
    forged["content_id"] = "forged"
    with pytest.raises(AutonomyContractError, match="identity"):
        BudgetExhaustion.from_dict(forged)


def test_parallel_reservation_is_atomic_and_never_over_admits() -> None:
    ledger = ObjectiveCognitiveBudgetLedger(_budget(max_input_tokens=10))

    def attempt(index: int) -> BudgetReservation | BudgetExhaustion:
        return _reserve(ledger, f"parallel-{index}", CognitiveCost(input_tokens=1))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(attempt, range(40)))
    admitted = [item for item in results if isinstance(item, BudgetReservation)]
    exhausted = [item for item in results if isinstance(item, BudgetExhaustion)]
    assert len(admitted) == 10
    assert len(exhausted) == 30
    assert sum(item.max_input_tokens for item in ledger.snapshot().reservations) == 10


def test_cost_vectors_are_closed_integer_only_and_strong_is_subset() -> None:
    with pytest.raises(CognitiveBudgetProtocolError, match="unsupported"):
        CognitiveCost.from_mapping({"mystery_cost": 1})
    with pytest.raises(CognitiveBudgetProtocolError, match="integer"):
        CognitiveCost(input_tokens=0.5)  # type: ignore[arg-type]
    with pytest.raises(CognitiveBudgetProtocolError, match="strong-model"):
        CognitiveCost(total_model_calls=0, strong_model_calls=1)
