"""Map closed recovery onto the existing board owner.

SPAR / SAWM / ASEH / PCTDD / DOEP are methods on
``configured_board_scheduler`` + ``implementation_daemon`` + DuckDB claims.
This module does not launch a second writable path. TypeSafe is never
authority.

DOEP: changed state invalidates affected work via a bounded, history-preserving
PlanDelta (claimed-safe ops only). ASEH: ``expire_task_claim`` remains claim
authority; file leases stay fencing only. A still-live claim is not mutated.
"""

from __future__ import annotations

from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    DeltaEffectClass,
    LifecycleState,
    PlanDeltaItem,
    PlanDeltaOperation,
)

_CLAIMED_SAFE = frozenset(
    {
        PlanDeltaOperation.RECORD_UNCERTAINTY,
        PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
    }
)
_LIVE_CLAIM_MARKERS = ("expired", "still-live", "still live", "without mutation")


def recovery_needs_plan_delta(recovery: Mapping[str, Any]) -> bool:
    """True when closed recovery must leave a PlanDelta on the board owner."""

    action = str(recovery.get("action") or "").strip()
    if action == "invalidate_stale_evidence":
        return True
    return action == "replan_suffix" and bool(recovery.get("admit"))


def _lifecycle(value: Any) -> LifecycleState:
    if isinstance(value, LifecycleState):
        return value
    text = str(value or "").strip()
    if text:
        try:
            return LifecycleState(text)
        except ValueError:
            pass
    return LifecycleState.CLAIMED


def _target_cid(value: Any) -> str:
    text = str(value or "").strip()
    if not text or any(char.isspace() for char in text) or "\x00" in text:
        return ""
    return text


def build_recovery_delta_items(
    recovery: Mapping[str, Any],
    *,
    target_cid: str = "",
    lifecycle: LifecycleState | str = LifecycleState.CLAIMED,
) -> tuple[PlanDeltaItem, ...]:
    """Return claimed-safe PlanDelta items. Never edits completed history."""

    if not recovery_needs_plan_delta(recovery):
        return ()
    action = str(recovery.get("action") or "").strip()
    stall = str(recovery.get("stall_reason") or "").strip() or "stale_evidence"
    target = _target_cid(target_cid)
    state = _lifecycle(lifecycle)
    items: list[PlanDeltaItem] = []
    provenance = {
        "source": "closed-recovery",
        "action": action,
        "stall_reason": stall,
        "accepted_as_authority": False,
        "completes_task": False,
    }
    stale = stall in {"stale_evidence", "stale", "lease_held"}
    if action == "invalidate_stale_evidence" or stale:
        items.append(
            PlanDeltaItem(
                item_key="recovery:record-uncertainty",
                operation=PlanDeltaOperation.RECORD_UNCERTAINTY,
                target_cid=target,
                expected_target_lifecycle=state,
                expected_target_spec_revision="",
                before_digest="",
                after_record_cid="",
                effect_class=DeltaEffectClass.EVIDENCE_ONLY,
                rationale="Stale evidence is not current; do not complete from it.",
                provenance=provenance,
                expected_effects=("invalidate-stale-evidence",),
                affected_task_cids=(target,) if target else (),
            )
        )
    if action == "invalidate_stale_evidence":
        items.append(
            PlanDeltaItem(
                item_key="recovery:request-lifecycle",
                operation=PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
                target_cid=target,
                expected_target_lifecycle=state,
                expected_target_spec_revision="",
                before_digest="",
                after_record_cid="",
                effect_class=DeltaEffectClass.LIFECYCLE_REQUEST,
                rationale="Request suffix invalidation without rewriting history.",
                provenance=provenance,
                expected_effects=("invalidate-stale-evidence",),
                affected_task_cids=(target,) if target else (),
            )
        )
    if action == "replan_suffix":
        items.append(
            PlanDeltaItem(
                item_key="recovery:replan-suffix",
                operation=PlanDeltaOperation.REQUEST_LIFECYCLE_ACTION,
                target_cid=target,
                expected_target_lifecycle=state,
                expected_target_spec_revision="",
                before_digest="",
                after_record_cid="",
                effect_class=DeltaEffectClass.LIFECYCLE_REQUEST,
                rationale="Admitted suffix replan; prefix history stays intact.",
                provenance=provenance,
                expected_effects=("replan-suffix",),
                affected_task_cids=(target,) if target else (),
            )
        )
    for item in items:
        if item.operation not in _CLAIMED_SAFE:
            raise RuntimeError(
                f"recovery PlanDelta op {item.operation.value} is not claimed-safe"
            )
    return tuple(items)


def _claim_still_live(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "expired" in name:
        return True
    return any(marker in text for marker in _LIVE_CLAIM_MARKERS)


def apply_recovery_board_fence(
    recovery: Mapping[str, Any],
    *,
    target_cid: str = "",
    lifecycle: LifecycleState | str = LifecycleState.CLAIMED,
    coordinator: Any | None = None,
    claim: Any | None = None,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Persist a bounded PlanDelta and optional DuckDB claim fence.

    File leases are not this path. ``expire_task_claim`` is attempted only for
    ``invalidate_stale_evidence``. A live claim fails without mutation.
    Missing coordinator is deferred, not invented authority.
    """

    payload: dict[str, Any] = {
        "accepted_as_authority": False,
        "completes_task": False,
        "kills_process": False,
        "typesafe_required": False,
        "action": str(recovery.get("action") or "preserve"),
        "delta_item_cids": [],
        "delta_operations": [],
        "delta_event_id": "",
        "claim_fenced": False,
        "claim_fence_reason": "not_required",
        "reason_codes": (),
        "items": (),
    }
    if not recovery_needs_plan_delta(recovery):
        return payload
    try:
        items = build_recovery_delta_items(
            recovery, target_cid=target_cid, lifecycle=lifecycle
        )
    except Exception:
        payload["reason_codes"] = ("recovery_delta_error",)
        return payload
    payload["items"] = items
    payload["delta_item_cids"] = [item.content_id for item in items]
    payload["delta_operations"] = [item.operation.value for item in items]
    reasons: list[str] = ["recovery_plan_delta"]
    action = str(recovery.get("action") or "").strip()
    if action == "invalidate_stale_evidence":
        expire = getattr(coordinator, "expire_task_claim", None)
        if coordinator is None or not callable(expire) or claim is None:
            payload["claim_fence_reason"] = "deferred"
            reasons.append("claim_fence_deferred")
        else:
            try:
                kwargs: dict[str, Any] = {}
                if now_ms is not None:
                    kwargs["now_ms"] = now_ms
                expire(claim, **kwargs)
                payload["claim_fenced"] = True
                payload["claim_fence_reason"] = "expired"
                reasons.append("claim_fenced")
            except Exception as exc:
                if _claim_still_live(exc):
                    payload["claim_fence_reason"] = "claim_still_live"
                    reasons.append("claim_still_live")
                else:
                    payload["claim_fence_reason"] = "claim_fence_error"
                    reasons.append("claim_fence_error")
    else:
        payload["claim_fence_reason"] = "not_required"
    record = getattr(coordinator, "record_recovery_plan_delta", None)
    if callable(record) and items:
        try:
            kwargs: dict[str, Any] = {
                "task_cid": _target_cid(target_cid),
                "items": tuple(item.to_dict() for item in items),
            }
            if now_ms is not None:
                kwargs["now_ms"] = now_ms
            recorded = record(**kwargs)
            if isinstance(recorded, Mapping):
                payload["delta_event_id"] = str(recorded.get("event_id") or "")
            reasons.append("recovery_delta_recorded")
        except TypeError:
            try:
                recorded = record(
                    task_cid=_target_cid(target_cid),
                    items=tuple(item.to_dict() for item in items),
                )
                if isinstance(recorded, Mapping):
                    payload["delta_event_id"] = str(recorded.get("event_id") or "")
                reasons.append("recovery_delta_recorded")
            except Exception:
                reasons.append("recovery_delta_record_error")
        except Exception:
            reasons.append("recovery_delta_record_error")
    payload["reason_codes"] = tuple(reasons)
    return payload


__all__ = [
    "apply_recovery_board_fence",
    "build_recovery_delta_items",
    "recovery_needs_plan_delta",
]
