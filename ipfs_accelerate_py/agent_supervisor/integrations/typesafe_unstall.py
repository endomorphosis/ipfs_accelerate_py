"""Nominate a closed stall-recovery action. Fail-open preserve. Never writes board."""

from __future__ import annotations

import threading
from typing import Any, Mapping

from ipfs_accelerate_py.typesafe_inference import Choice, Noul
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    LOW_CONFIDENCE,
    typesafe_permitted,
)

UNSTALL_ACTIONS: tuple[str, ...] = (
    "preserve",
    "invalidate_stale_evidence",
    "replan_suffix",
    "retry_provider",
    "request_human",
)
_META = {
    "preserve": "NO_OP",
    "invalidate_stale_evidence": "NO_OP",
    "replan_suffix": "REPLAN_AFFECTED_SUFFIX",
    "retry_provider": "NO_OP",
    "request_human": "REQUEST_HUMAN_DECISION",
}
_LAST = threading.local()


def last_unstall_nomination() -> dict[str, Any]:
    value = getattr(_LAST, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _receipt(
    action: str,
    *,
    reason_codes: tuple[str, ...] = (),
    confidence: float = 0.0,
) -> dict[str, Any]:
    chosen = action if action in UNSTALL_ACTIONS else "preserve"
    payload = {
        "action": chosen,
        "meta_action": _META[chosen],
        "writes_board": False,
        "writes_locks": False,
        "may_complete_task": False,
        "accepted_as_authority": False,
        "confidence": round(float(confidence), 4),
        "reason_codes": list(reason_codes or ("preserve",)),
    }
    _LAST.value = dict(payload)
    return payload


def nominate_unstall_action(
    state: Mapping[str, Any] | None = None,
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Closed recovery nomination. No key / low conf / unknown → preserve."""

    redacted = {
        "reason": str((state or {}).get("reason") or "")[:64],
        "wake_kind": str((state or {}).get("wake_kind") or "")[:32],
        "audit_label": str((state or {}).get("audit_label") or "")[:32],
    }
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return _receipt("preserve", reason_codes=("privacy_or_unconfigured",))
    from ipfs_accelerate_py.typesafe_inference import system_one

    questions = {
        "stale_evidence": Noul(
            instructions={
                "question": "Is the stall explained by stale evidence?",
                "inspect": "`reason`",
            },
        ),
        "provider_down": Noul(
            instructions={
                "question": "Is the stall a provider outage or rate limit?",
                "inspect": "`audit_label`",
            },
        ),
        "needs_human": Noul(
            instructions={
                "question": "Is a human choice required to continue?",
            },
        ),
        "answer": Choice(
            instructions={
                "question": "Which closed recovery should code consider?",
                "focus": "Select one listed action. Code will not complete a task.",
            },
            criteria={
                "preserve": {"what": "Leave the stall; existing rules apply"},
                "invalidate_stale_evidence": {
                    "what": "Nominate invalidating stale evidence",
                    "not_for": "Completing the task",
                },
                "replan_suffix": {"what": "Nominate REPLAN_AFFECTED_SUFFIX"},
                "retry_provider": {"what": "Retry allocation; do not complete"},
                "request_human": {"what": "Nominate REQUEST_HUMAN_DECISION"},
            },
        ),
    }
    try:
        result = system_one(redacted, questions, timeout=timeout)
    except Exception:
        return _receipt("preserve", reason_codes=("typesafe_error",))
    choices = getattr(result, "choices", None) or {}
    nouls = getattr(result, "nouls", None) or {}
    picked = str(getattr(choices.get("answer"), "choice", "") or "")
    conf = float(getattr(choices.get("answer"), "confidence", 0.0) or 0.0)

    def noul(name: str) -> float:
        item = nouls.get(name)
        try:
            return float(getattr(item, "noul", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    reasons = ["composed_in_code"]
    action = picked if picked in UNSTALL_ACTIONS else "preserve"
    if picked and picked not in UNSTALL_ACTIONS:
        reasons.append("unknown_choice_preserve")
        action = "preserve"
    if conf < LOW_CONFIDENCE and action != "preserve":
        reasons.append("low_confidence_preserve")
        action = "preserve"
    elif noul("stale_evidence") >= 0.7:
        action = "invalidate_stale_evidence"
        reasons.append("stale_evidence_noul")
    elif noul("provider_down") >= 0.7:
        action = "retry_provider"
        reasons.append("provider_down_noul")
    elif noul("needs_human") >= 0.7:
        action = "request_human"
        reasons.append("needs_human_noul")
    if redacted["reason"] == "stale_evidence" and action == "preserve":
        reasons.append("blocked_stale_existing_path")
    return _receipt(action, reason_codes=tuple(reasons), confidence=conf)


__all__ = [
    "UNSTALL_ACTIONS",
    "last_unstall_nomination",
    "nominate_unstall_action",
]
