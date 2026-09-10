"""Keep missing settlement evidence pending without inventing an outcome."""

from collections.abc import Mapping
from typing import Any

from ..merge.database_coordination import DatabaseCoordinationNotReadyError


def missing_completion_deferral(error: BaseException) -> dict[str, Any] | None:
    """Recognize the coordinator's bounded diagnostic, never a recovery grant.

    The caller must leave the current tick immediately. In particular it must
    not fail an attempt, release a claim, requeue a task, or dispatch work based
    on this missing row. A later tick must rerun ordinary reconciliation.
    """
    if not isinstance(error, DatabaseCoordinationNotReadyError):
        return None
    evidence = getattr(error, "evidence", None)
    if not isinstance(evidence, Mapping) or evidence.get("reason") != "completion_missing":
        return None
    identity = {}
    for name in ("task_cid", "claim_id", "attempt_id"):
        value = evidence.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            return None
        identity[name] = value
    return {
        "deferred": True,
        "skipped": True,
        "reason": "completion_missing",
        "selection_idle_reason": "completion_evidence_unavailable",
        "implementation_result": None,
        "active_task_id": "",
        # The exception can follow committed work in this tick. Missing
        # settlement evidence cannot establish that there were no effects.
        "unchanged": None,
        "attempt_consumed": "unknown",
        "provider_dispatched": "unknown",
        "recovery_attempt_consumed": False,
        "recovery_provider_dispatched": False,
        "backoff_seconds": 5,
        "completion_authority": False,
        "completion_evidence": identity,
    }
