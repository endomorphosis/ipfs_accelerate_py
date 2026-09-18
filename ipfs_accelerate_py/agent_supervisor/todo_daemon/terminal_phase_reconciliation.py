"""Qualify failed-phase actual/intended mismatches without rewriting receipts.

A proven ordinary-finalizer actual of ``terminalized_for_retry`` is already a
consumed dead attempt. A later intended ``superseded_attempt_revoked`` is not
an alias and must not replace that actual, refund credit, or grant completion.
Independent work may continue because the attempt cannot resume. Callers still
retain native claim, callback, fence and source validation.
"""
from collections.abc import Mapping, Sequence
from typing import Any

from .ordinary_finalizer_replay import ordinary_finalized_disposition

CONSUMED_RETRY = "terminalized_for_retry"
INTENDED_SUPERSESSION = "superseded_attempt_revoked"
PHASE_FAILED = "failed"


def _ordinary_proof(ordinary_finalizer: Mapping[str, Any] | None) -> str:
    if not isinstance(ordinary_finalizer, Mapping):
        return ""
    try:
        proof = ordinary_finalized_disposition(
            attempt_identity=ordinary_finalizer["attempt_identity"],
            control_claim=ordinary_finalizer["control_claim"],
            task_identity=ordinary_finalizer["task_identity"],
            task_status=ordinary_finalizer["task_status"],
            receipt=ordinary_finalizer["receipt"],
        )
    except (KeyError, TypeError):
        return ""
    return proof if type(proof) is str else ""


def qualify_failed_phase(
    *,
    phase: Any,
    actual: Any,
    intended: Any,
    attempt_consumed: Any = None,
    ordinary_finalizer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a sealed decision for one phase row. Never mutates inputs."""
    if (
        type(phase) is not str
        or (actual is not None and type(actual) is not str)
        or (intended is not None and type(intended) is not str)
        or (attempt_consumed is not None and type(attempt_consumed) is not bool)
    ):
        return {
            "blocked": True,
            "reconciled": False,
            "independent_work_admitted": False,
            "skip_mutation": False,
            "preserved_actual": actual if type(actual) is str else None,
            "attempt_consumed": attempt_consumed if type(attempt_consumed) is bool else None,
            "reason": "phase_dispositions_malformed",
        }
    if actual == intended:
        return {
            "blocked": False,
            "reconciled": True,
            "independent_work_admitted": True,
            "skip_mutation": False,
            "preserved_actual": actual,
            "attempt_consumed": attempt_consumed,
            "reason": "phase_dispositions_match",
        }
    if (
        phase == PHASE_FAILED
        and actual == CONSUMED_RETRY
        and intended == INTENDED_SUPERSESSION
        and attempt_consumed is not False
        and _ordinary_proof(ordinary_finalizer) == CONSUMED_RETRY
    ):
        return {
            "blocked": False,
            "reconciled": False,
            "independent_work_admitted": True,
            "skip_mutation": True,
            "preserved_actual": CONSUMED_RETRY,
            "attempt_consumed": True,
            "reason": "preserved_consumed_retry_actual",
        }
    return {
        "blocked": True,
        "reconciled": False,
        "independent_work_admitted": False,
        "skip_mutation": False,
        "preserved_actual": actual,
        "attempt_consumed": attempt_consumed,
        "reason": "terminal_phase_changed_its_actual_database_disposition",
    }


def qualify_terminal_repair_batch(
    phases: Sequence[Mapping[str, Any]],
    *,
    ordinary_finalizer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit independent work when every mismatch is a proven skip-mutation.

    A blocked row fails the whole batch. Skip-mutation preserves the consumed
    retry actual and does not mark reconciliation or completion complete.
    """
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)):
        return {
            "blocked": True,
            "reconciled": False,
            "independent_work_admitted": False,
            "repair_batch_pending": False,
            "reconciliation_complete": False,
            "completion_authorized": False,
            "reason": "phase_dispositions_malformed",
            "phases": [],
        }
    rows = []
    for raw in phases:
        if not isinstance(raw, Mapping):
            return {
                "blocked": True,
                "reconciled": False,
                "independent_work_admitted": False,
                "repair_batch_pending": False,
                "reconciliation_complete": False,
                "completion_authorized": False,
                "reason": "phase_dispositions_malformed",
                "phases": [],
            }
        rows.append(
            qualify_failed_phase(
                phase=raw.get("phase"),
                actual=raw.get("actual"),
                intended=raw.get("intended"),
                attempt_consumed=raw.get("attempt_consumed"),
                ordinary_finalizer=ordinary_finalizer,
            )
        )
    if any(row["blocked"] for row in rows):
        return {
            "blocked": True,
            "reconciled": False,
            "independent_work_admitted": False,
            "repair_batch_pending": False,
            "reconciliation_complete": False,
            "completion_authorized": False,
            "reason": "terminal_reconciliation_receipt_repair_failed",
            "phases": rows,
        }
    skipped = any(row["skip_mutation"] for row in rows)
    return {
        "blocked": False,
        "reconciled": not skipped,
        "independent_work_admitted": True,
        "repair_batch_pending": False,
        "reconciliation_complete": not skipped,
        "completion_authorized": False,
        "reason": (
            "preserved_consumed_retry_actual" if skipped else "phase_dispositions_match"
        ),
        "phases": rows,
    }
