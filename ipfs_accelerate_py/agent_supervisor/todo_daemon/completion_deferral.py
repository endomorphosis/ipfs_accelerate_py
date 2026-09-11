"""Keep missing settlement evidence pending without inventing an outcome."""

from collections.abc import Mapping
from typing import Any

from ..merge.database_coordination import (
    DatabaseCoordinationNotReadyError,
    DatabaseCoordinationTaskFenceMismatchError,
)


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


def task_fence_mismatch_deferral(error: BaseException) -> dict[str, Any] | None:
    """Keep an unproven task fence pending without repeating callback work.

    The caller must end this tick. A later tick reruns the ordinary authority
    checks; this observation never refreshes a fence or settles an effect.
    """
    if not isinstance(error, DatabaseCoordinationTaskFenceMismatchError):
        return None
    evidence = getattr(error, "evidence", None)
    if not isinstance(evidence, Mapping) or evidence.get("reason") != "task_claim_latest_fence_mismatch":
        return None
    identity = {}
    for name in ("task_cid", "claim_id", "attempt_id"):
        value = evidence.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            return None
        identity[name] = value
    for name in (
        "expected_fencing_token", "expected_fence_epoch",
        "observed_fencing_token", "observed_fence_epoch",
    ):
        value = evidence.get(name)
        if type(value) is not int or not (0 <= value < 2**63):
            return None
        if name.startswith("expected_") and value == 0:
            return None
        identity[name] = value
    if all(identity[f"expected_{key}"] == identity[f"observed_{key}"]
           for key in ("fencing_token", "fence_epoch")):
        return None
    return {
        "deferred": True,
        "skipped": True,
        "reason": "task_claim_latest_fence_mismatch",
        "selection_idle_reason": "task_fence_evidence_unavailable",
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
        "task_fence_evidence": identity,
    }
