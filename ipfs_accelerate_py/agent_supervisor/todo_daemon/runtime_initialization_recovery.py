"""Admission for closed runtime initialization failures, before retry mutation.

This module does not confer source/owner authority and performs no mutation.
The native recovery adapter must supply its qualified source check and execute
admission again inside the existing task-fenced queue/status transition.  A
matching exception string alone is deliberately insufficient.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..task_sources.control_plane_contracts import content_identity

PORTAL_INITIALIZATION_FAILURE = (
    "'PortalImplementationDaemon' object has no attribute "
    "'isolate_merge_queue_to_task_projection'"
)


class RuntimeInitializationRecoveryRejected(ValueError):
    """The failure is not a closed, exactly bound recovery candidate."""


def admit_closed_initialization_failure(
    *,
    attempt: Mapping[str, Any],
    latest_attempt: Mapping[str, Any],
    task: Mapping[str, Any],
    failed_phase: Mapping[str, Any],
    callback: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Check native observations without asserting absence of prior effects.

    ``callback`` and ``source`` are observations from native authority, never
    a requester-supplied receipt. Unknown outcomes remain unknown even when
    the last error text names the repaired constructor. This only admits a
    retry proposal; it neither replaces the failure receipt nor accepts work.
    """

    def require(condition: bool, reason: str) -> None:
        if not condition:
            raise RuntimeInitializationRecoveryRejected(reason)

    identity_fields = (
        "task_cid",
        "attempt_id",
        "claim_id",
        "lease_id",
        "owner_session_id",
        "attempt_number",
        "fencing_token",
        "fence_epoch",
    )
    for key in identity_fields:
        value = attempt.get(key)
        if key in {"attempt_number", "fencing_token", "fence_epoch"}:
            require(type(value) is int and value > 0, "invalid attempt " + key)
        else:
            require(type(value) is str and bool(value), "missing attempt " + key)
        require(
            type(latest_attempt.get(key)) is type(value)
            and latest_attempt.get(key) == value,
            "superseded attempt " + key,
        )
        require(
            type(callback.get(key)) is type(value) and callback.get(key) == value,
            "foreign callback " + key,
        )
    require(attempt.get("status") == "failed", "attempt is not failed")
    require(attempt.get("committed_phase") == "failed", "failure is not committed")
    require(
        type(attempt.get("revision")) is int and attempt["revision"] > 0,
        "missing execution revision",
    )
    require(
        latest_attempt.get("revision") == attempt["revision"],
        "execution revision advanced",
    )
    require(task.get("task_cid") == attempt["task_cid"], "foreign task")
    require(task.get("status") == "blocked", "task is not blocked")
    require(
        type(task.get("revision")) is int and task["revision"] > 0,
        "missing task revision",
    )
    body = task.get("body")
    require(isinstance(body, Mapping), "missing task body")
    receipt = body.get("completion_receipt")
    require(isinstance(receipt, Mapping), "missing terminal receipt")
    require(
        receipt.get("operation") == "database_portal_terminal_failure",
        "foreign terminal operation",
    )
    for key in identity_fields:
        require(
            type(receipt.get(key)) is type(attempt[key])
            and receipt.get(key) == attempt[key],
            "foreign terminal " + key,
        )
    require(
        receipt.get("execution_revision") == attempt["revision"],
        "foreign terminal execution revision",
    )
    require(receipt.get("execution_phase") == "failed", "foreign terminal phase")
    require(receipt.get("retryable") is False, "not a terminal initialization failure")
    require(
        receipt.get("reason") == PORTAL_INITIALIZATION_FAILURE,
        "unrecognized terminal reason",
    )
    require(failed_phase.get("phase") == "failed", "missing failed phase")
    phase_body = failed_phase.get("body")
    require(isinstance(phase_body, Mapping), "missing failure body")
    require(
        phase_body.get("portal_terminal_failure") is True
        and phase_body.get("portal_retryable_failure") is False
        and phase_body.get("reason") == PORTAL_INITIALIZATION_FAILURE,
        "foreign failed phase",
    )
    require(
        callback.get("state") == "closed_failure",
        "callback outcome is unknown or active",
    )
    require(
        callback.get("execution_revision") == attempt["revision"],
        "callback closure revision differs",
    )
    require(callback.get("active") is False, "callback is not quiescent")
    require(callback.get("unknown_outcome") is False, "unknown callback outcome")
    require(source.get("qualified") is True, "source has not been qualified")
    require(
        source.get("constructor_probe_passed") is True, "constructor repair is unproved"
    )
    require(source.get("retention_guard_passed") is True, "retention guard is unproved")
    for key in ("head", "tree"):
        value = source.get(key)
        require(
            type(value) is str
            and len(value) == 40
            and all(c in "0123456789abcdef" for c in value),
            "invalid source " + key,
        )
    require(
        type(source.get("qualification_receipt_cid")) is str
        and bool(source["qualification_receipt_cid"]),
        "missing source qualification binding",
    )
    proposal = {
        "schema": "ipfs_accelerate_py/agent-supervisor/runtime-initialization-retry-proposal@1",
        "attempt": {key: attempt[key] for key in identity_fields},
        "execution_revision": attempt["revision"],
        "task_revision": task["revision"],
        "failure_receipt_cid": content_identity(receipt),
        "callback_observation_cid": content_identity(callback),
        "source_observation_cid": content_identity(source),
        "completion_authority": False,
    }
    proposal["proposal_cid"] = content_identity(proposal)
    return proposal
