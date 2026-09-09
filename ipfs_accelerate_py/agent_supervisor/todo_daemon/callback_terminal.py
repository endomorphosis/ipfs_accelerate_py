"""Recognize the exact terminal suffix of an expired unknown callback.

This is evidence classification only. Callers must independently verify live
quarantine receipts, provider intent, source currency and released fences.
"""
from collections.abc import Mapping, Sequence
from typing import Any


def is_expired_callback_quarantine_phase(attempt: Any, phases: Sequence[Mapping[str, Any]]) -> bool:
    if (getattr(attempt, "status", None) != "failed"
            or getattr(attempt, "committed_phase", None) != "failed"
            or [p.get("phase") for p in phases] != ["claimed", "context", "failed"]):
        return False
    terminal = phases[-1]
    expected = {
        "cross_store_reconciled": True,
        "preparation_digest": "",
        "reconciliation": {
            "attempt_id": attempt.attempt_id,
            "claim_id": attempt.claim_id,
            "disposition": "quarantined",
            "effect_evidence_reused": False,
            "provider_evidence_reused": False,
            "provider_invocation_receipt_present": True,
            "reason": "portal_neutral_failure",
            "retry_required": False,
            "status": "failed",
            "task_cid": attempt.task_cid,
        },
    }
    body = terminal.get("body")
    reconciliation = body.get("reconciliation") if isinstance(body, Mapping) else None
    if not isinstance(reconciliation, Mapping):
        return False
    return bool(
        body.get("cross_store_reconciled") is True
        and all(reconciliation.get(key) is expected["reconciliation"][key]
                for key in ("effect_evidence_reused", "provider_evidence_reused",
                            "provider_invocation_receipt_present", "retry_required"))
        and body == expected
        and all(type(terminal.get(key)) is int and terminal[key] == getattr(attempt, key)
                for key in ("revision", "fencing_token", "fence_epoch"))
        and type(terminal.get("committed_at_ms")) is int
        and terminal["committed_at_ms"] == attempt.finished_at_ms
    )
