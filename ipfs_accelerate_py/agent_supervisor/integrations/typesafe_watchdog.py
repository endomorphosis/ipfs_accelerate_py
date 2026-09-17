"""Classify watchdog symptoms. Never kills. Fail-open = existing argv watchdog."""

from __future__ import annotations

import threading
from typing import Any, Mapping

from ipfs_accelerate_py.typesafe_inference import Choice, Noul
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    typesafe_permitted,
)

WATCHDOG_LABELS: tuple[str, ...] = (
    "healthy",
    "stalled",
    "provider_down",
    "false_missing",
    "needs_human",
)
_LAST = threading.local()


def last_watchdog_classification() -> dict[str, Any]:
    value = getattr(_LAST, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


_UNSTALL_REASON = {
    "stalled": "stale_evidence",
    "provider_down": "provider_down",
    "needs_human": "needs_human",
}


def _payload(
    label: str,
    *,
    reason_codes: tuple[str, ...] = (),
    unstall_action: str = "preserve",
) -> dict[str, Any]:
    chosen = label if label in WATCHDOG_LABELS else "healthy"
    body = {
        "label": chosen,
        "kills_process": False,
        "rewrites_supervisor": False,
        "writes_board": False,
        "writes_locks": False,
        "may_complete_task": False,
        "accepted_as_authority": False,
        "unstall_action": str(unstall_action or "preserve"),
        "unstall_writes_board": False,
        "reason_codes": list(reason_codes or ("watchdog_process_health_authoritative",)),
    }
    _LAST.value = dict(body)
    return body


def classify_watchdog_symptom(
    state: Mapping[str, Any] | None = None,
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Closed symptom label. No key → empty skip. Never SIGKILL."""

    redacted = {
        "phase": str((state or {}).get("phase") or "")[:64],
        "stalled": bool((state or {}).get("stalled")),
        "worker_count": int((state or {}).get("worker_count") or 0),
        "stall_evidence_available": bool(
            (state or {}).get("stall_evidence_available")
        ),
    }
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return _payload("healthy", reason_codes=("privacy_or_unconfigured",))
    from ipfs_accelerate_py.typesafe_inference import system_one

    questions = {
        "false_missing": Noul(
            instructions={
                "question": "Is a healthy worker likely misclassified as missing?",
                "inspect": "`worker_count`",
            },
        ),
        "label": Choice(
            instructions={
                "question": "Which closed watchdog symptom fits this redacted status?",
                "inspect": "`stalled`",
            },
            criteria={
                "healthy": {"what": "Workers present or not in a guarded phase"},
                "stalled": {"what": "Guarded phase aged out with no workers"},
                "provider_down": {"what": "Stall looks like provider outage"},
                "false_missing": {"what": "Watchdog argv false positive"},
                "needs_human": {"what": "Human policy needed; do not kill"},
            },
        ),
    }
    try:
        result = system_one(redacted, questions, timeout=timeout)
    except Exception:
        return _payload("healthy", reason_codes=("typesafe_error",))
    choices = getattr(result, "choices", None) or {}
    nouls = getattr(result, "nouls", None) or {}
    picked = str(getattr(choices.get("label"), "choice", "") or "")
    if picked not in WATCHDOG_LABELS:
        picked = "healthy"
    false_missing = float(getattr(nouls.get("false_missing"), "noul", 0.0) or 0.0)
    reasons = ["composed_in_code", "watchdog_process_health_authoritative"]
    if false_missing >= 0.7:
        picked = "false_missing"
        reasons.append("false_missing_noul")
    elif redacted["stalled"] and picked == "healthy":
        picked = "stalled"
        reasons.append("stalled_flag")
    unstall_action = "preserve"
    reason = _UNSTALL_REASON.get(picked)
    if reason:
        try:
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall import (
                last_unstall_nomination,
                nominate_unstall_action,
            )

            payload = state or {}
            nominate_unstall_action(
                {
                    "reason": reason,
                    "wake_kind": "watchdog",
                    "audit_label": picked,
                },
                possible_resolution_action_ids=payload.get(
                    "possible_resolution_action_ids"
                )
                or (),
                declared_meta_actions=payload.get("declared_meta_actions") or (),
                action_meta_by_id=payload.get("action_meta_by_id"),
                privacy_class=privacy_class,
                remote_disclosure_permitted=remote_disclosure_permitted,
                timeout=timeout,
            )
            unstall_action = str(
                last_unstall_nomination().get("action") or "preserve"
            )
            reasons.append("unstall_nominated")
        except Exception:
            unstall_action = "preserve"
    return _payload(
        picked,
        reason_codes=tuple(reasons),
        unstall_action=unstall_action,
    )


def observe_watchdog_symptom(state: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Sidecar classify. Never changes watchdog kill/restart decisions."""

    try:
        return classify_watchdog_symptom(state)
    except Exception:
        return None


__all__ = [
    "WATCHDOG_LABELS",
    "classify_watchdog_symptom",
    "last_watchdog_classification",
    "observe_watchdog_symptom",
]
