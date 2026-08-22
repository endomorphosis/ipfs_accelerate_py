from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.contracts import (
    AutonomyContractError,
    AutonomyPromotionReceipt,
    PromotionStatus,
)
from ipfs_accelerate_py.agent_supervisor.autonomy.promotion import (
    AUTONOMY_PROMOTION_CONTROLLER_INTERFACE,
    REQUIRED_SAFETY_GATES,
    REQUIRED_THRESHOLD_BPS,
    AutonomyPromotionController,
    PolicyPointerStore,
    PromotionControllerError,
    PromotionRequest,
)
import pytest

SAFETY_PASS = {gate: True for gate in REQUIRED_SAFETY_GATES}
THRESHOLDS_PASS = dict(REQUIRED_THRESHOLD_BPS)


def _request(**overrides: object) -> PromotionRequest:
    values: dict[str, object] = {
        "candidate_policy_id": "policy:candidate-v2",
        "expected_old_policy_id": "policy:current-v1",
        "authorization_id": "operator:starworks5:apmc-g110",
        "safety_gate_results": dict(SAFETY_PASS),
        "safety_gate_receipt_ids": ("receipt:safety-1",),
        "held_out_evaluation_ids": ("eval:held-out-1",),
        "threshold_bps": dict(THRESHOLDS_PASS),
        "tree_id": "tree:current",
        "candidate_version": "policy:candidate-v2",
        "authorization_subject": "operator:starworks5:apmc-g110",
    }
    values.update(overrides)
    return PromotionRequest(**values)  # type: ignore[arg-type]


def test_interfaces_are_versioned() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    assert controller.interface == AUTONOMY_PROMOTION_CONTROLLER_INTERFACE
    assert AUTONOMY_PROMOTION_CONTROLLER_INTERFACE == "AutonomyPromotionController@1"


def test_candidate_cannot_authorize_itself() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    receipt = controller.apply(
        _request(
            authorization_id="policy:candidate-v2",
            authorization_subject="policy:candidate-v2",
        )
    )
    assert receipt.status is PromotionStatus.NON_PROMOTED
    assert "self_authorized_policy_promotions" in receipt.blocker_codes
    assert store.current_policy_id == "policy:current-v1"


def test_missed_safety_gate_blocks_with_exact_reason() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    safety = dict(SAFETY_PASS)
    safety["false_completions"] = False
    receipt = controller.apply(_request(safety_gate_results=safety))
    assert receipt.status is PromotionStatus.NON_PROMOTED
    assert receipt.blocker_codes == ("false_completions",)
    assert store.current_policy_id == "policy:current-v1"


def test_missed_threshold_cannot_be_lowered() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    thresholds = dict(THRESHOLDS_PASS)
    thresholds["token_input_reduction_bps"] = 2_999
    receipt = controller.apply(_request(threshold_bps=thresholds))
    assert receipt.status is PromotionStatus.NON_PROMOTED
    assert "token_input_reduction_bps" in receipt.blocker_codes


def test_external_authorization_promotes_with_expected_old_cas() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    receipt = controller.apply(_request())
    assert receipt.status is PromotionStatus.PROMOTED
    assert receipt.resulting_policy_id == "policy:candidate-v2"
    assert receipt.compare_and_swap_receipt_id
    assert store.current_policy_id == "policy:candidate-v2"
    assert receipt.rollback_policy_id == "policy:current-v1"


def test_cas_race_and_aba_fail_closed() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    stale_generation = store.generation
    controller.apply(_request())
    raced = store.compare_and_swap(
        expected_old="policy:current-v1",
        candidate="policy:attacker",
        observed_generation=stale_generation,
    )
    assert raced["applied"] is False
    assert raced["reason"] == "cas_generation_mismatch"
    assert store.current_policy_id == "policy:candidate-v2"


def test_rollback_restores_expected_old_pointer() -> None:
    store = PolicyPointerStore(current_policy_id="policy:current-v1")
    controller = AutonomyPromotionController(store)
    promoted = controller.apply(_request())
    rolled = controller.rollback(
        promoted,
        authorization_id="operator:starworks5:apmc-g110-rollback",
    )
    assert rolled.status is PromotionStatus.ROLLED_BACK
    assert store.current_policy_id == "policy:current-v1"
    with pytest.raises(PromotionControllerError, match="candidate cannot authorize"):
        controller.apply(_request())
        controller.rollback(
            AutonomyPromotionReceipt(
                candidate_policy_id="policy:candidate-v2",
                expected_old_policy_id="policy:current-v1",
                resulting_policy_id="policy:candidate-v2",
                status=PromotionStatus.PROMOTED,
                safety_gate_results=SAFETY_PASS,
                held_out_evaluation_ids=("eval:held-out-1",),
                safety_gate_receipt_ids=("receipt:safety-1",),
                authorization_id="operator:ok",
                compare_and_swap_receipt_id="cas:1",
                rollback_policy_id="policy:current-v1",
            ),
            authorization_id="policy:candidate-v2",
        )


def test_promoted_receipt_rejects_failed_gates() -> None:
    with pytest.raises(AutonomyContractError, match="failed safety gates"):
        safety = dict(SAFETY_PASS)
        safety["path_or_scope_escapes"] = False
        AutonomyPromotionReceipt(
            candidate_policy_id="policy:candidate-v2",
            expected_old_policy_id="policy:current-v1",
            resulting_policy_id="policy:candidate-v2",
            status=PromotionStatus.PROMOTED,
            safety_gate_results=safety,
            held_out_evaluation_ids=("eval:held-out-1",),
            safety_gate_receipt_ids=("receipt:safety-1",),
            authorization_id="operator:ok",
            compare_and_swap_receipt_id="cas:1",
            rollback_policy_id="policy:current-v1",
        )
