"""Elapsed task authority cannot settle an unknown production callback."""

from collections.abc import Mapping
from typing import Any


class ExpiredExecutionCustodyPending(RuntimeError):
    """A running real execution attempt still requires native settlement."""

    def __init__(self, evidence: Mapping[str, Any]) -> None:
        super().__init__("production attempt requires settlement before expiry retry")
        self.evidence = dict(evidence)


def guard_generic_retirement(daemon: Any, attempt: Any, *, reason: str) -> None:
    """Only existing native settlement paths may retire a real execution.

    This source has no durable pre-dispatch journal. Missing result rows, a
    claimed/context phase, or an elapsed lease cannot prove non-dispatch.
    Successful completion and exact terminal failure are reconciled before
    the generic expiry pass; any still-running production attempt stays held.
    """
    from .database_portal_bridge import DatabasePortalExecutionBridge

    real_execution = bool(getattr(daemon, "require_real_execution", False)) or any(
        isinstance(
            getattr(getattr(daemon, name, None), "__self__", None),
            DatabasePortalExecutionBridge,
        )
        for name in ("_provider_fn", "_effect_fn", "_validation_fn")
    )
    if not real_execution:
        return
    raise ExpiredExecutionCustodyPending(
        {
            "reason": reason,
            **{
                name: getattr(attempt, name)
                for name in (
                    "task_cid",
                    "claim_id",
                    "attempt_id",
                    "lease_id",
                    "owner_session_id",
                    "attempt_number",
                    "fencing_token",
                    "fence_epoch",
                )
            },
        }
    )


def expired_execution_deferral(error: BaseException) -> dict[str, Any] | None:
    if not isinstance(error, ExpiredExecutionCustodyPending):
        return None
    evidence = getattr(error, "evidence", None)
    names = ("task_cid", "claim_id", "attempt_id", "lease_id", "owner_session_id")
    numbers = ("attempt_number", "fencing_token", "fence_epoch")
    if (
        type(evidence) is not dict
        or set(evidence) != {"reason", *names, *numbers}
        or evidence["reason"]
        not in {"claim_history_missing", "claim_authority_expired"}
        or any(
            type(evidence[name]) is not str
            or not evidence[name].strip()
            or len(evidence[name]) > 512
            for name in names
        )
        or any(
            type(evidence[name]) is not int or not 1 <= evidence[name] < 2**63
            for name in numbers
        )
    ):
        return None
    return {
        "deferred": True,
        "skipped": True,
        "reason": "production_attempt_settlement_required",
        "selection_idle_reason": "expired_attempt_settlement_unavailable",
        "implementation_result": None,
        "active_task_id": "",
        "unchanged": None,
        "attempt_consumed": "unknown",
        "provider_dispatched": "unknown",
        "recovery_attempt_consumed": False,
        "recovery_provider_dispatched": False,
        "backoff_seconds": 5,
        "completion_authority": False,
        "coordination_mutation_authority": False,
        "retry_authorized": False,
        "retained_attempt_evidence": dict(evidence),
    }
