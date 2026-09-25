"""Closed stall recoveries. TypeSafe is optional fuzzy triage, never required.

Without a key, code maps stall reason onto the catalog. TypeSafe may only
choose among already-declared MetaActions. It cannot complete a task or
invent a recovery.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

from .contracts import MetaAction, ResolutionCandidate

_LAST = threading.local()

RECOVERY_ACTIONS = (
    "preserve",
    "invalidate_stale_evidence",
    "replan_suffix",
    "retry_provider",
    "request_human",
)
RECOVERY_META = {
    "preserve": MetaAction.NO_OP.value,
    "invalidate_stale_evidence": MetaAction.NO_OP.value,
    "retry_provider": MetaAction.NO_OP.value,
    "replan_suffix": MetaAction.REPLAN_AFFECTED_SUFFIX.value,
    "request_human": MetaAction.REQUEST_HUMAN_DECISION.value,
}
STALL_REASONS = frozenset(
    {
        "stale_evidence",
        "stale",
        "provider_down",
        "provider",
        "needs_human",
        "human",
        "lease_held",
        "false_missing",
    }
)
_ADMITTABLE = frozenset({"replan_suffix", "request_human"})
_WATCHDOG_REASON = {
    "stalled": "stale_evidence",
    "provider_down": "provider_down",
    "needs_human": "needs_human",
    "false_missing": "false_missing",
}


def last_closed_recovery() -> dict[str, Any]:
    value = getattr(_LAST, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def outstanding_recovery_work(
    recovery: Mapping[str, Any] | None = None,
) -> bool:
    """True when a recovery PlanDelta still blocks board completion.

    Queue-empty and an idle question graph are not success while this is true.
    """

    payload = recovery if isinstance(recovery, Mapping) else last_closed_recovery()
    if payload.get("consumed") is True:
        return False
    action = str(payload.get("action") or "").strip()
    cids = payload.get("delta_item_cids") or ()
    return bool(cids) and action in {"invalidate_stale_evidence", "replan_suffix"}


def consume_recovery_plan_delta(
    coordinator: Any | None = None,
    *,
    now_ms: int | None = None,
) -> None:
    """Mark the recovery PlanDelta consumed on the existing board owner.

    Thread-local state is updated even when the coordinator is absent.
    TypeSafe is not this owner.
    """

    prior = last_closed_recovery()
    if not prior:
        return
    prior["consumed"] = True
    _LAST.value = prior
    consume = getattr(coordinator, "consume_recovery_plan_delta", None)
    if not callable(consume):
        return
    event_id = str(prior.get("delta_event_id") or "").strip()
    if not event_id:
        return
    try:
        kwargs: dict[str, Any] = {
            "delta_event_id": event_id,
            "task_cid": str(prior.get("task_cid") or "").strip(),
        }
        if now_ms is not None:
            kwargs["now_ms"] = now_ms
        consume(**kwargs)
    except TypeError:
        try:
            consume(delta_event_id=event_id)
        except Exception:
            return
    except Exception:
        return


def _action_name(candidate: ResolutionCandidate) -> str:
    action = getattr(getattr(candidate, "resolution_action", None), "action", "")
    return str(getattr(action, "value", action) or "").strip()


def declared_meta_names(
    candidates: Sequence[ResolutionCandidate] = (),
    extra: Sequence[str] = (),
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    for item in extra:
        ident = str(getattr(item, "value", item) or "").strip()
        if ident and ident not in seen:
            seen.add(ident)
            names.append(ident)
    for candidate in candidates:
        ident = _action_name(candidate)
        if ident and ident not in seen:
            seen.add(ident)
            names.append(ident)
    return tuple(names)


def filter_recovery_candidates(
    candidates: Sequence[ResolutionCandidate],
    meta_action: str,
) -> tuple[ResolutionCandidate, ...]:
    wanted = str(meta_action or "").strip()
    return tuple(item for item in candidates if _action_name(item) == wanted)


def collect_recovery_bindings(
    candidates: Sequence[ResolutionCandidate],
    *,
    question: Any = None,
    state: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...]]:
    """Collect declared metas and action ids. Never invents names."""

    possible_ids: list[str] = []
    extra_declared: list[str] = []
    action_meta_by_id: dict[str, str] = {}
    payload = state or {}
    if question is not None:
        for item in getattr(question, "possible_resolution_action_ids", ()) or ():
            ident = str(item).strip()
            if ident:
                possible_ids.append(ident)
    extra = payload.get("possible_resolution_action_ids") or ()
    if isinstance(extra, (list, tuple)):
        possible_ids.extend(str(item).strip() for item in extra if str(item).strip())
    declared_extra = payload.get("declared_meta_actions") or ()
    if isinstance(declared_extra, (list, tuple)):
        extra_declared.extend(
            str(item).strip() for item in declared_extra if str(item).strip()
        )
    for candidate in candidates:
        resolution = getattr(candidate, "resolution_action", None)
        ident = str(getattr(resolution, "action_id", "") or "").strip()
        meta = str(
            getattr(
                getattr(resolution, "action", None),
                "value",
                getattr(resolution, "action", ""),
            )
            or ""
        ).strip()
        if ident and meta:
            action_meta_by_id[ident] = meta
    return tuple(possible_ids), action_meta_by_id, tuple(extra_declared)


def stall_reason_from_wake(
    *,
    stale: bool,
    wake_kind: str = "",
    reason: str = "",
    state: Mapping[str, Any] | None = None,
) -> str:
    """Closed stall label. Watchdog sidecar is optional and never required."""

    payload = state or {}
    hinted = str(payload.get("reason") or reason or "").strip().lower()
    if hinted in STALL_REASONS:
        stall = hinted
    elif stale:
        stall = "stale_evidence"
    else:
        stall = ""
    label = str(payload.get("watchdog_label") or "").strip()
    mapped = _WATCHDOG_REASON.get(label)
    if mapped:
        stall = mapped
    return stall


def should_recover(*, stale: bool, wake_kind: str = "", reason: str = "", state: Mapping[str, Any] | None = None) -> bool:
    stall = stall_reason_from_wake(
        stale=stale, wake_kind=wake_kind, reason=reason, state=state
    )
    return bool(stale or stall in STALL_REASONS)


def _deterministic_action(
    *,
    reason: str,
    wake_kind: str,
    declared: Sequence[str],
) -> str:
    declared_set = set(declared)
    kind = str(wake_kind or "").strip().lower()
    stall = str(reason or "").strip().lower()
    if stall == "false_missing":
        return "preserve"
    if stall in {"needs_human", "human"} or kind == "human":
        if MetaAction.REQUEST_HUMAN_DECISION.value in declared_set:
            return "request_human"
        return "preserve"
    if stall in {"provider_down", "provider"} or kind == "provider":
        return "retry_provider"
    if stall in {"stale_evidence", "stale", "lease_held"}:
        if MetaAction.REPLAN_AFFECTED_SUFFIX.value in declared_set:
            return "replan_suffix"
        return "invalidate_stale_evidence"
    return "preserve"


def plan_closed_recovery(
    *,
    reason: str,
    wake_kind: str = "",
    candidates: Sequence[ResolutionCandidate] = (),
    extra_declared: Sequence[str] = (),
    possible_resolution_action_ids: Sequence[str] = (),
    action_meta_by_id: Mapping[str, str] | None = None,
    typesafe_state: Mapping[str, Any] | None = None,
    call_typesafe: bool = True,
    stale: bool = False,
) -> dict[str, Any]:
    """Return one catalog recovery. TypeSafe is skippable triage."""

    stall = stall_reason_from_wake(
        stale=stale,
        wake_kind=wake_kind,
        reason=reason,
        state=typesafe_state,
    ) or str(reason or "").strip().lower()
    declared = declared_meta_names(candidates, extra=tuple(extra_declared))
    action = _deterministic_action(
        reason=stall, wake_kind=wake_kind, declared=declared
    )
    reasons = ["composed_in_code", "closed_recovery_catalog"]
    typesafe_used = False
    if call_typesafe:
        try:
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
                typesafe_permitted,
            )
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_unstall import (
                last_unstall_nomination,
                nominate_unstall_action,
            )

            if typesafe_permitted():
                payload = dict(typesafe_state or {})
                payload.setdefault("reason", stall or reason)
                payload.setdefault("wake_kind", wake_kind)
                nominate_unstall_action(
                    payload,
                    possible_resolution_action_ids=possible_resolution_action_ids,
                    declared_meta_actions=declared,
                    action_meta_by_id=action_meta_by_id,
                )
                picked = str(last_unstall_nomination().get("action") or "")
                typesafe_used = True
                reasons.append("typesafe_optional_triage")
                if last_unstall_nomination().get("uncertain"):
                    reasons.append("uncertain_band_preserve")
                    action = "preserve"
                elif picked in RECOVERY_ACTIONS:
                    mapped = RECOVERY_META[picked]
                    if picked in _ADMITTABLE and mapped not in set(declared):
                        reasons.append("undeclared_meta_ignored")
                    else:
                        action = picked
                else:
                    reasons.append("typesafe_unknown_preserve")
        except Exception:
            reasons.append("typesafe_error_fail_open")

    mapped = RECOVERY_META[action]
    admit = action in _ADMITTABLE and mapped in set(declared)
    if action in _ADMITTABLE and not admit:
        action = (
            "invalidate_stale_evidence"
            if stall in {"stale_evidence", "stale", "lease_held"}
            else "preserve"
        )
        mapped = RECOVERY_META[action]
        reasons.append("undeclared_meta_coarsened")
    if action == "preserve" and (stale or stall in {"stale_evidence", "stale"}):
        action = "invalidate_stale_evidence"
        mapped = RECOVERY_META[action]
        reasons.append("stale_preserve_invalidates")
    execute = "admit" if admit else action
    payload = {
        "accepted_as_authority": False,
        "invents_meta_action": False,
        "completes_task": False,
        "kills_process": False,
        "typesafe_required": False,
        "typesafe_used": typesafe_used,
        "action": action,
        "meta_action": mapped,
        "admit": admit,
        "execute": execute,
        "stall_reason": stall,
        "declared_meta_actions": list(declared),
        "reason_codes": reasons,
    }
    _LAST.value = {
        key: value
        for key, value in payload.items()
        if key not in {"declared_meta_actions"}
    }
    return payload


def _attach_board_fence(
    recovery: Mapping[str, Any],
    reasons: tuple[str, ...],
    *,
    target_cid: str,
    coordinator: Any | None,
    claim: Any | None,
    now_ms: int | None,
    lifecycle: str,
) -> tuple[str, ...]:
    """Persist DOEP PlanDelta / ASEH claim fence on the existing board owner."""

    from .recovery_board_fence import apply_recovery_board_fence

    fence = apply_recovery_board_fence(
        recovery,
        target_cid=target_cid,
        coordinator=coordinator,
        claim=claim,
        now_ms=now_ms,
        lifecycle=lifecycle,
    )
    prior = getattr(_LAST, "value", None)
    merged = dict(prior) if isinstance(prior, Mapping) else dict(recovery)
    merged.update(
        {
            "accepted_as_authority": False,
            "completes_task": False,
            "delta_item_cids": list(fence.get("delta_item_cids") or ()),
            "delta_operations": list(fence.get("delta_operations") or ()),
            "claim_fenced": bool(fence.get("claim_fenced")),
            "claim_fence_reason": str(fence.get("claim_fence_reason") or ""),
            "delta_event_id": str(fence.get("delta_event_id") or ""),
            "task_cid": str(target_cid or ""),
        }
    )
    _LAST.value = {
        key: value
        for key, value in merged.items()
        if key not in {"declared_meta_actions"}
    }
    extra = tuple(fence.get("reason_codes") or ())
    if not extra:
        return reasons
    seen = set(reasons)
    appended = tuple(item for item in extra if item not in seen)
    return reasons + appended


def apply_recovery_plan(
    recovery: Mapping[str, Any],
    candidates: Sequence[ResolutionCandidate],
    *,
    stale: bool,
    target_cid: str = "",
    coordinator: Any | None = None,
    claim: Any | None = None,
    now_ms: int | None = None,
    lifecycle: str = "",
) -> tuple[str, tuple[ResolutionCandidate, ...], tuple[str, ...]]:
    """Map a plan onto ``admit`` / ``idle`` / ``continue``. Never invents candidates.

    When the catalog invalidates stale evidence or admits a suffix replan, a
    bounded PlanDelta is recorded on the existing board owner. TypeSafe is
    not that owner.
    """

    action = str(recovery.get("action") or "preserve")
    if recovery.get("admit"):
        matching = filter_recovery_candidates(
            candidates, str(recovery.get("meta_action") or "")
        )
        if matching:
            reasons = _attach_board_fence(
                recovery,
                ("closed_recovery_admit", action),
                target_cid=target_cid,
                coordinator=coordinator,
                claim=claim,
                now_ms=now_ms,
                lifecycle=lifecycle,
            )
            return "admit", matching, reasons
        action = "invalidate_stale_evidence" if stale else "preserve"
        recovery = dict(recovery)
        recovery["action"] = action
        recovery["admit"] = False
    if action == "retry_provider":
        return "idle", tuple(candidates), ("retry_provider",)
    if action == "invalidate_stale_evidence" or stale:
        reasons = _attach_board_fence(
            recovery
            if action == "invalidate_stale_evidence"
            else {**dict(recovery), "action": "invalidate_stale_evidence"},
            ("stale_invalidated", action),
            target_cid=target_cid,
            coordinator=coordinator,
            claim=claim,
            now_ms=now_ms,
            lifecycle=lifecycle,
        )
        return "idle", tuple(candidates), reasons
    return "continue", tuple(candidates), ("closed_recovery_preserve",)


__all__ = [
    "apply_recovery_plan",
    "collect_recovery_bindings",
    "consume_recovery_plan_delta",
    "last_closed_recovery",
    "outstanding_recovery_work",
    "RECOVERY_ACTIONS",
    "RECOVERY_META",
    "STALL_REASONS",
    "declared_meta_names",
    "filter_recovery_candidates",
    "plan_closed_recovery",
    "should_recover",
    "stall_reason_from_wake",
]
