"""Closed queue lineage for one retained callback append failure.

These predicates grant no completion authority. Callers also need the sealed
attempt projection, verified event chain, exact Git artifacts, native consumer
lease, current task preauthorization and normal callback settlement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

APPEND_FAILURE = "merge_queue_reconciliation_append_unverified"
REVIVAL_REASON = (
    "merge train proved quarantined candidate already integrated into exact target"
)
RECOVERED_CLAIM_REASON = "merge train consumer exited; claim recovered"


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def append_quarantine_lineage(request: Any, *, max_attempts: int) -> bool:
    """Recognize the original row and a single native revival's crash states."""
    metadata = getattr(request, "metadata", None)
    if not isinstance(metadata, Mapping) or "completion" in metadata:
        return False
    quarantine = metadata.get("quarantine")
    if not isinstance(quarantine, Mapping):
        return False
    if set(quarantine) != {
        "acceptance_pending",
        "accepted",
        "canonical_task_id",
        "commit_sha",
        "failure_count",
        "finished_at",
        "integrated",
        "max_attempts",
        "merge_result",
        "merged",
        "reason",
        "request_id",
        "retryable",
        "started_at",
        "status",
        "target_branch",
        "task_id",
    }:
        return False
    expected = {
        "acceptance_pending": False,
        "accepted": False,
        "integrated": False,
        "merged": False,
        "retryable": False,
        "status": "quarantined",
        "reason": APPEND_FAILURE,
        "request_id": getattr(request, "request_id", ""),
        "task_id": getattr(request, "task_id", ""),
        "canonical_task_id": getattr(request, "canonical_task_key", ""),
        "commit_sha": getattr(request, "commit_sha", ""),
        "target_branch": metadata.get("target_branch"),
    }
    if any(
        type(quarantine[k]) is not type(v) or quarantine[k] != v
        for k, v in expected.items()
    ):
        return False
    started, finished = quarantine["started_at"], quarantine["finished_at"]
    if (
        any(
            not expected[k]
            for k in (
                "request_id",
                "task_id",
                "canonical_task_id",
                "commit_sha",
                "target_branch",
            )
        )
        or type(quarantine["failure_count"]) is not int
        or quarantine["failure_count"] != 1
        or type(max_attempts) is not int
        or max_attempts < 1
        or type(quarantine["max_attempts"]) is not int
        or quarantine["max_attempts"] != max_attempts
        or not _finite(started)
        or not _finite(finished)
        or started > finished
    ):
        return False
    merge = quarantine["merge_result"]
    if not isinstance(merge, Mapping) or not (
        merge.get("attempted") is True
        and merge.get("already_merged") is False
        and merge.get("merged") is False
        and merge.get("integration_occurred") is True
        and merge.get("completion_skipped") is True
        and type(merge.get("returncode")) is int
        and merge["returncode"] == 2
        and merge.get("reason") == APPEND_FAILURE
        and merge.get("target_branch") == expected["target_branch"]
        and merge.get("merge_commit") == merge.get("target_commit")
        and merge.get("merge_reconciliation_receipt")
        == {
            "recorded": False,
            "reason": APPEND_FAILURE,
            "reconciliation_count": 1,
        }
    ):
        return False
    receipt = merge["merge_reconciliation_receipt"]
    if (
        receipt["recorded"] is not False
        or type(receipt["reconciliation_count"]) is not int
    ):
        return False
    status = getattr(request, "status", "")
    failure = getattr(request, "failure_reason", "")
    attempt = getattr(request, "attempt", None)
    failures = getattr(request, "failure_count", None)
    if type(attempt) is not int or type(failures) is not int:
        return False
    revivals = metadata.get("revivals", [])
    if not isinstance(revivals, list):
        return False
    if status == "quarantined":
        return bool(
            not revivals
            and failure == APPEND_FAILURE
            and attempt == failures == 1
            and not getattr(request, "consumer_id", "")
            and not getattr(request, "claim_token", "")
        )
    if len(revivals) != 1:
        return False
    revival = revivals[0]
    if not isinstance(revival, Mapping) or set(revival) != {
        "at",
        "reason",
        "previous_enqueued_at",
        "previous_failure_count",
        "previous_failure_reason",
    }:
        return False
    if (
        revival.get("reason") != REVIVAL_REASON
        or revival.get("previous_failure_reason") != APPEND_FAILURE
        or type(revival.get("previous_failure_count")) is not int
        or revival["previous_failure_count"] != 1
        or not _finite(revival.get("at"))
        or not _finite(revival.get("previous_enqueued_at"))
        or not revival["previous_enqueued_at"] <= started <= finished <= revival["at"]
        or getattr(request, "enqueued_at", None) != revival["at"]
        or not 0 <= failures < min(max_attempts, 3)
        or attempt != failures + 1
        or failure != (RECOVERED_CLAIM_REASON if failures else "")
    ):
        return False
    if status == "processing":
        return bool(
            str(getattr(request, "consumer_id", "")).startswith("merge-train:")
            and getattr(request, "claim_token", "")
        )
    return bool(
        status in {"pending", "completed"}
        and not getattr(request, "consumer_id", "")
        and not getattr(request, "claim_token", "")
    )
