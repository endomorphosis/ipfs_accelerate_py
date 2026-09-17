"""Nominate a closed stall-recovery action. Fail-open preserve. Never writes board."""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.typesafe_inference import Choice, Noul, noul_yes_no
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
UNSTALL_CHOICE_CRITERIA: dict[str, Any] = {
    "preserve": {
        "what": "Leave the stall; existing rules apply",
        "not_for": "Replanning, retrying a provider, or asking a human",
        "examples": ["Unknown stall", "Low confidence"],
        "maps_to": {"NO_OP": ["does not complete a task"]},
    },
    "invalidate_stale_evidence": {
        "what": "Nominate invalidating stale evidence",
        "not_for": "Completing the task or inventing a MetaAction",
        "examples": ["reason is stale_evidence"],
        "maps_to": {"NO_OP": ["does not complete a task"]},
    },
    "replan_suffix": {
        "what": "Nominate REPLAN_AFFECTED_SUFFIX",
        "not_for": "A suffix that still matches the tree",
        "examples": ["Mandatory check failed and the plan suffix is stale"],
        "maps_to": {
            "REPLAN_AFFECTED_SUFFIX": [
                "already declared only",
                "does not complete a task",
            ]
        },
    },
    "retry_provider": {
        "what": "Retry allocation; do not complete",
        "not_for": "Completing the task or killing a worker",
        "examples": ["Provider outage or rate limit"],
        "maps_to": {"NO_OP": ["does not complete a task"]},
    },
    "request_human": {
        "what": "Nominate REQUEST_HUMAN_DECISION",
        "not_for": "A closed software recovery that already exists",
        "examples": ["A WHETHER/WHICH question needs a human choice"],
        "maps_to": {
            "REQUEST_HUMAN_DECISION": [
                "already declared only",
                "does not complete a task",
            ]
        },
    },
}
_MAPPED_METAS = frozenset(_META.values())
_LAST = threading.local()


def _meta_name(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def collect_declared_meta_actions(
    possible_resolution_action_ids: Sequence[str] = (),
    *,
    declared_meta_actions: Sequence[Any] = (),
    action_meta_by_id: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """MetaActions already declared for the question. Unknown ids are ignored."""

    declared: set[str] = set()
    by_id = {
        str(key).strip(): _meta_name(value)
        for key, value in dict(action_meta_by_id or {}).items()
        if str(key).strip() and _meta_name(value) in _MAPPED_METAS
    }
    for ident in _as_sequence(possible_resolution_action_ids):
        text = str(ident or "").strip()
        if not text:
            continue
        if text in _META:
            declared.add(_META[text])
        elif text in _MAPPED_METAS:
            declared.add(text)
        elif text in by_id:
            declared.add(by_id[text])
    for item in _as_sequence(declared_meta_actions):
        text = _meta_name(item)
        if text in _META:
            declared.add(_META[text])
        elif text in _MAPPED_METAS:
            declared.add(text)
    return frozenset(declared)


def last_unstall_nomination() -> dict[str, Any]:
    value = getattr(_LAST, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _receipt(
    action: str,
    *,
    reason_codes: tuple[str, ...] = (),
    confidence: float = 0.0,
    declared_meta_actions: Sequence[str] = (),
) -> dict[str, Any]:
    chosen = action if action in UNSTALL_ACTIONS else "preserve"
    payload = {
        "action": chosen,
        "meta_action": _META[chosen],
        "writes_board": False,
        "writes_locks": False,
        "may_complete_task": False,
        "accepted_as_authority": False,
        "invents_meta_action": False,
        "confidence": round(float(confidence), 4),
        "declared_meta_actions": sorted(
            {str(item) for item in declared_meta_actions if str(item)}
        )[:8],
        "reason_codes": list(reason_codes or ("preserve",)),
    }
    _LAST.value = dict(payload)
    return payload


def nominate_unstall_action(
    state: Mapping[str, Any] | None = None,
    *,
    possible_resolution_action_ids: Sequence[str] = (),
    declared_meta_actions: Sequence[Any] = (),
    action_meta_by_id: Mapping[str, Any] | None = None,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Closed recovery nomination. No key / low conf / unknown → preserve.

    When a mapped MetaAction is already on ``possible_resolution_action_ids``,
    keep that nomination instead of preserving. Never invents a MetaAction
    outside the declared set. Never writes board or locks.
    """

    payload = state or {}
    redacted = {
        "reason": str(payload.get("reason") or "")[:64],
        "wake_kind": str(payload.get("wake_kind") or "")[:32],
        "audit_label": str(payload.get("audit_label") or "")[:32],
    }
    id_seq = _as_sequence(
        possible_resolution_action_ids
        or payload.get("possible_resolution_action_ids")
        or ()
    )
    declared_seq = _as_sequence(
        declared_meta_actions or payload.get("declared_meta_actions") or ()
    )
    meta_by_id = action_meta_by_id or payload.get("action_meta_by_id")
    allowlist_active = bool(id_seq or declared_seq or meta_by_id)
    declared = collect_declared_meta_actions(
        id_seq,
        declared_meta_actions=declared_seq,
        action_meta_by_id=meta_by_id,
    )
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return _receipt(
            "preserve",
            reason_codes=("privacy_or_unconfigured",),
            declared_meta_actions=tuple(declared),
        )
    from ipfs_accelerate_py.typesafe_inference import system_one

    questions = {
        "stale_evidence": Noul(
            instructions={
                "question": "Is the stall explained by stale evidence?",
                "inspect": "`reason`",
            },
            criteria=noul_yes_no(
                true_what="reason is stale_evidence or evidence no longer matches the tree",
                true_examples=["reason is stale_evidence"],
                false_what="The stall has another cause",
                false_examples=["provider_down", "needs_human"],
            ),
        ),
        "provider_down": Noul(
            instructions={
                "question": "Is the stall a provider outage or rate limit?",
                "inspect": "`audit_label`",
            },
            criteria=noul_yes_no(
                true_what="Audit label indicates outage or rate limit",
                true_examples=["rate_limit", "provider_outage"],
                false_what="Workers or evidence are the issue, not the provider",
                false_examples=["stale_evidence"],
            ),
        ),
        "needs_human": Noul(
            instructions={
                "question": "Is a human choice required to continue?",
            },
            criteria=noul_yes_no(
                true_what="A WHETHER/WHICH question needs a human choice",
                true_examples=["REQUEST_HUMAN_DECISION is already declared"],
                false_what="A closed software recovery already exists",
                false_examples=["REPLAN_AFFECTED_SUFFIX is already declared"],
            ),
        ),
        "answer": Choice(
            instructions={
                "question": "Which closed recovery should code consider?",
                "focus": "Select one listed action. Code will not complete a task.",
            },
            criteria=UNSTALL_CHOICE_CRITERIA,
        ),
    }
    try:
        result = system_one(redacted, questions, timeout=timeout)
    except Exception:
        return _receipt(
            "preserve",
            reason_codes=("typesafe_error",),
            declared_meta_actions=tuple(declared),
        )
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
    mapped = _META[action]
    declared_hit = mapped in declared and mapped != "NO_OP"
    if declared_hit:
        reasons.append("prefer_declared_meta_action")
    elif conf < LOW_CONFIDENCE and action != "preserve":
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
    mapped = _META[action]
    if allowlist_active and mapped != "NO_OP" and mapped not in declared:
        reasons.append("undeclared_meta_preserve")
        action = "preserve"
    if redacted["reason"] == "stale_evidence" and action == "preserve":
        reasons.append("blocked_stale_existing_path")
    return _receipt(
        action,
        reason_codes=tuple(reasons),
        confidence=conf,
        declared_meta_actions=tuple(declared),
    )


__all__ = [
    "UNSTALL_ACTIONS",
    "collect_declared_meta_actions",
    "last_unstall_nomination",
    "nominate_unstall_action",
]
