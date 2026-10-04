"""Typed owner continuation for unknown-callback quarantines.

Retry-suppressed unknown-callback and dead-admitted unknown quarantines must
not silently freeze the rest of the board. This contract preserves every
historical unknown receipt, never forges completion, and either admits a
bounded successor attempt or reports dual pending-merge identity without
rewriting it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "unknown-callback-quarantine-continuation@1"
)
OPERATION = "database_unknown_callback_quarantine_continuation"
DUAL_PENDING_MERGE_IDENTITY_REASON = (
    "Portal pending-merge candidate identity changed while the database "
    "claim was waiting"
)
DEAD_ADMITTED_OPERATION = (
    "database_dead_admitted_provider_outcome_unknown_quarantine"
)
NEUTRAL_UNKNOWN_OPERATION = "database_portal_neutral_failure_quarantine"
UNKNOWN_FAILURE_KIND = "provider_callback_outcome_unknown"
SUCCESSOR_REASON = "owner_continuation_preserves_unknown"
DUAL_IDENTITY_REPORT_REASON = "dual_pending_merge_candidate_identity_reported"
IDENTITY_INCOMPLETE_REASON = "unknown_callback_continuation_identity_incomplete"


def _receipt(task: Any) -> Mapping[str, Any] | None:
    body = getattr(task, "body", None)
    if not isinstance(body, Mapping):
        return None
    receipt = body.get("completion_receipt")
    return receipt if isinstance(receipt, Mapping) else None


def _status(task: Any) -> str:
    return str(getattr(task, "status", "") or "").strip().lower()


def is_dual_pending_merge_identity_block(task: Any) -> bool:
    """True for the DOEP-063 class: dual candidate identity, do not settle."""

    receipt = _receipt(task)
    if receipt is None or _status(task) != "blocked":
        return False
    return bool(
        receipt.get("operation") == "database_portal_terminal_failure"
        and receipt.get("reason") == DUAL_PENDING_MERGE_IDENTITY_REASON
    )


def is_unknown_callback_retry_suppressed_quarantine(task: Any) -> bool:
    receipt = _receipt(task)
    if receipt is None or _status(task) != "quarantined":
        return False
    return bool(
        receipt.get("operation") == NEUTRAL_UNKNOWN_OPERATION
        and str(receipt.get("failure_kind") or "") == UNKNOWN_FAILURE_KIND
        and receipt.get("retry_suppressed") is True
    )


def is_dead_admitted_unknown_quarantine(task: Any) -> bool:
    receipt = _receipt(task)
    if receipt is None or _status(task) != "quarantined":
        return False
    return bool(receipt.get("operation") == DEAD_ADMITTED_OPERATION)


def admits_owner_continuation(task: Any) -> bool:
    return bool(
        is_unknown_callback_retry_suppressed_quarantine(task)
        or is_dead_admitted_unknown_quarantine(task)
    )


def is_successor_admitted_unknown_retrying(task: Any) -> bool:
    """True when a continuation already preserved UNKNOWN and is retrying."""

    receipt = _receipt(task)
    if receipt is None or _status(task) != "retrying":
        return False
    return bool(
        receipt.get("operation") == OPERATION
        and receipt.get("successor_attempt_admitted") is True
        and receipt.get("unknown_preserved") is True
        and receipt.get("completion_authoritative") is False
        and receipt.get("history_rewritten") is False
    )


def dual_identity_observation(task: Any) -> dict[str, Any]:
    return {
        "task_cid": str(getattr(task, "task_cid", "") or ""),
        "reopened": False,
        "changed": False,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "status": "blocked",
        "reason": DUAL_IDENTITY_REPORT_REASON,
        "unknown_preserved": True,
        "completion_authoritative": False,
        "history_rewritten": False,
        "operator_review_required": True,
        "successor_attempt_admitted": False,
    }


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or type(value) is not int or value < 1:
        return None
    return value


def _nonempty_str(value: Any) -> str | None:
    if type(value) is not str:
        return None
    text = value.strip()
    return text or None


def successor_attempt_number(
    receipt: Mapping[str, Any],
    *,
    attempt_number_floor: int = 0,
) -> int | None:
    """Return a monotonic successor attempt above receipt and cooldown floors."""

    prior_attempt = receipt.get("attempt_number")
    if prior_attempt is None:
        attempt_number = 1
    else:
        checked = _positive_int(prior_attempt)
        if checked is None:
            return None
        attempt_number = checked + 1
    if (
        isinstance(attempt_number_floor, bool)
        or type(attempt_number_floor) is not int
        or attempt_number_floor < 0
    ):
        return None
    return max(attempt_number, attempt_number_floor + 1)


def continuation_receipt(
    task: Any,
    *,
    expected_revision: int,
    attempt_number_floor: int = 0,
) -> dict[str, Any] | None:
    """Build a successor retry receipt that embeds the historical unknown."""

    receipt = _receipt(task)
    if receipt is None or not admits_owner_continuation(task):
        return None
    if (
        isinstance(expected_revision, bool)
        or type(expected_revision) is not int
        or expected_revision < 0
    ):
        return None
    attempt_id = _nonempty_str(receipt.get("attempt_id"))
    claim_id = _nonempty_str(receipt.get("claim_id"))
    lease_id = _nonempty_str(receipt.get("lease_id"))
    owner_session_id = _nonempty_str(receipt.get("owner_session_id"))
    fencing_token = _positive_int(receipt.get("fencing_token"))
    fence_epoch = _positive_int(receipt.get("fence_epoch"))
    if None in (
        attempt_id,
        claim_id,
        lease_id,
        owner_session_id,
        fencing_token,
        fence_epoch,
    ):
        return None
    attempt_number = successor_attempt_number(
        receipt,
        attempt_number_floor=attempt_number_floor,
    )
    if attempt_number is None:
        return None
    queue_reason = f"{OPERATION}:{attempt_id}"
    preserved = dict(receipt)
    payload = {
        "schema": SCHEMA,
        "operation": OPERATION,
        "attempt_id": attempt_id,
        "claim_id": claim_id,
        "lease_id": lease_id,
        "owner_session_id": owner_session_id,
        "attempt_number": attempt_number,
        "fencing_token": fencing_token,
        "fence_epoch": fence_epoch,
        "queue_reason": queue_reason,
        "backoff_ms": 0,
        "retry_not_before_ms": 0,
        "control_expected_status": "quarantined",
        "control_expected_revision": expected_revision,
        "preserved_unknown_receipt": preserved,
        "unknown_preserved": True,
        "provider_effect_state": "unknown_may_have_started",
        "completion_authoritative": False,
        "successor_attempt_admitted": True,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "history_rewritten": False,
        "reason": SUCCESSOR_REASON,
    }
    for field in (
        "execution_route_binding",
        "execution_route_policy_id",
        "execution_route_origin_revision",
    ):
        if field in receipt:
            payload[field] = receipt[field]
    return payload


def identity_incomplete_observation(task: Any) -> dict[str, Any]:
    return {
        "task_cid": str(getattr(task, "task_cid", "") or ""),
        "reopened": False,
        "changed": False,
        "provider_dispatched": False,
        "attempt_consumed": False,
        "status": _status(task),
        "reason": IDENTITY_INCOMPLETE_REASON,
        "unknown_preserved": True,
        "completion_authoritative": False,
        "history_rewritten": False,
        "operator_review_required": True,
        "successor_attempt_admitted": False,
    }
