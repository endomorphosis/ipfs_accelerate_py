"""Recognize an ordinary finalizer's exact canonical CAS after interruption.

This pure classifier grants neither retry nor completion authority. Callers must
retain native claim, phase, callback, receipt and source validation. In particular
its result cannot excuse an inconsistent historical terminal phase.
"""
from collections.abc import Mapping
from typing import Any

RETRY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/database-retry-budget@1"
CONTROL_FIELDS = {"task_cid", "revision", "execution_spec_cid", "validation_spec_cid"}
ATTEMPT_FIELDS = {
    "attempt_id", "claim_id", "task_cid", "attempt_number", "owner_session_id",
    "fencing_token", "fence_epoch", "lease_id",
}


def ordinary_finalized_disposition(
    *, attempt_identity: Mapping[str, Any], control_claim: Mapping[str, Any],
    task_identity: Mapping[str, Any], task_status: str,
    receipt: Mapping[str, Any],
) -> str:
    """Return the committed disposition, or empty when exact proof is absent.

    An ordinary finalizer advances the task revision by exactly one, retaining
    its execution and validation inputs and its exact fenced attempt receipt.
    A later revision, non-consuming refund or reconciliation saga must use its
    own native recovery path instead.
    """
    if (
        task_status not in {"retrying", "blocked"}
        or set(attempt_identity) != ATTEMPT_FIELDS
        or set(control_claim) != CONTROL_FIELDS
        or set(task_identity) != CONTROL_FIELDS
        or attempt_identity.get("task_cid") != control_claim.get("task_cid")
        or receipt.get("schema") != RETRY_SCHEMA
        or any(type(receipt.get(k)) is not type(v) or receipt[k] != v
               for k, v in attempt_identity.items())
        or type(control_claim.get("revision")) is not int
        or type(task_identity.get("revision")) is not int
        or task_identity["revision"] != control_claim["revision"] + 1
        or any(type(task_identity[k]) is not type(control_claim[k])
               or task_identity[k] != control_claim[k]
               for k in CONTROL_FIELDS - {"revision"})
        or receipt.get("validation_spec_cid") != control_claim["validation_spec_cid"]
        or any(name in receipt and type(receipt[name]) is not bool
               for name in ("attempt_consumed", "forced_block"))
        or receipt.get("attempt_consumed") is False
        or "terminal_reconciliation" in receipt
        or type(receipt.get("retry_exhausted")) is not bool
    ):
        return ""
    operation = receipt.get("operation")
    if operation == "database_unknown_outcome_blocked":
        if (task_status == "blocked" and receipt.get("forced_block") is True
                and receipt["retry_exhausted"]):
            return "blocked_unknown_outcome"
    elif operation in {"database_retry_rearmed", "database_retry_exhausted"}:
        if receipt["retry_exhausted"] != (operation == "database_retry_exhausted"):
            return ""
        expected_status = "blocked" if receipt["retry_exhausted"] else "retrying"
        if task_status == expected_status and not receipt.get("forced_block"):
            return "terminalized_for_retry"
    return ""
