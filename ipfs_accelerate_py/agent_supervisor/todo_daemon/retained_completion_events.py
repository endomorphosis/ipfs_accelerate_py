"""Correlate legacy merge observations without granting completion authority."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def legacy_merge_observation_is_anchored(
    events: Sequence[Mapping[str, Any]],
    *,
    event_index: int,
    completion_index: int,
    alias: str,
    task_cid: str,
    task_key: str,
) -> bool:
    """Allow an incomplete auxiliary event to be ignored, never used as proof.

    The caller supplies an already verified event chain. A later fully bound
    implementation result remains mandatory and goes through the ordinary
    completion verifier. Missing-key observations are correlated to that
    result, its exact enqueue and its fresh worktree-validation start.
    """
    event = events[event_index]
    completion = events[completion_index]
    kind = event.get("type")
    if event.get("canonical_task_key") is not None or kind not in {
        "worktree_reconciliation_candidate_queued",
        "merge_reconciled",
    }:
        return False
    commit = event.get("implementation_commit")
    branch = event.get("branch")
    stream = event.get("stream_id")
    if (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or not isinstance(branch, str)
        or not branch.startswith("implementation/")
        or not stream
        or completion.get("stream_id") != stream
        or completion.get("implementation_commit") != commit
    ):
        return False
    identity = {
        "task_id": alias,
        "canonical_task_cid": task_cid,
        "branch": branch,
        "baseline_ref": event.get("baseline_ref"),
        "implementation_commit": commit,
        "attempt": event.get("attempt"),
        "stream_id": stream,
    }
    if (
        any(event.get(k) != v for k, v in identity.items())
        or type(identity["attempt"]) is not int
        or identity["attempt"] < 1
        or not isinstance(identity["baseline_ref"], str)
        or re.fullmatch(r"[0-9a-f]{40}", identity["baseline_ref"]) is None
    ):
        return False

    def matches(value):
        return (
            all(value.get(k) == v for k, v in identity.items())
            and value.get("canonical_task_key") == task_key
        )

    finals = [
        value
        for value in events[event_index + 1 : completion_index]
        if value.get("type") == "implementation_finished" and matches(value)
    ]
    if len(finals) != 1:
        return False
    final = finals[0]
    validation = final.get("validation_result")
    merge = final.get("merge_result")
    if (
        not isinstance(validation, Mapping)
        or validation.get("passed") is not True
        or final.get("returncode") != 0
        or final.get("provider_dispatched") is not False
        or not isinstance(merge, Mapping)
        or merge.get("merged") is not True
        or merge.get("canonical_task_cid") != task_cid
        or merge.get("canonical_task_key") != task_key
        or merge.get("implementation_commit") != commit
    ):
        return False
    candidate = validation.get("candidate_binding")
    workspace = (
        candidate.get("validated_workspace") if isinstance(candidate, Mapping) else None
    )
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("verified") is not True
        or not candidate.get("expected_fingerprint")
        or candidate.get("expected_fingerprint") != candidate.get("current_fingerprint")
        or not isinstance(workspace, Mapping)
        or workspace.get("verified") is not True
        or workspace.get("head") != commit
        or workspace.get("branch") != branch
        or workspace.get("status_clean") is not True
    ):
        return False
    recovery_key = final.get("recovery_key")
    worktree = final.get("worktree_path")
    request_id = merge.get("request_id")
    if (
        not isinstance(worktree, str)
        or not worktree
        or not request_id
        or not isinstance(recovery_key, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", recovery_key) is None
    ):
        return False
    starts = [
        (i, value)
        for i, value in enumerate(events[:event_index])
        if value.get("type") == "worktree_reconciliation_validation_started"
        and matches(value)
        and value.get("worktree_path") == worktree
        and value.get("recovery_key") == recovery_key
        and value.get("provider_dispatched") is False
    ]
    enqueues = [
        (i, value)
        for i, value in enumerate(events[:event_index])
        if value.get("type") == "merge_candidate_enqueued"
        and matches(value)
        and value.get("request_id") == request_id
        and value.get("worktree_path") == worktree
        and value.get("completion_task_cids") == {alias: task_cid}
        and value.get("queued") is True
        and value.get("merged") is False
    ]
    if len(starts) != 1 or len(enqueues) != 1 or starts[0][0] >= enqueues[0][0]:
        return False
    auxiliary_merge = event.get("merge_result")
    if (
        not isinstance(auxiliary_merge, Mapping)
        or auxiliary_merge.get("request_id") != request_id
    ):
        return False
    if kind == "worktree_reconciliation_candidate_queued":
        provenance = event.get("merge_queue_synchronous_source")
        return bool(
            isinstance(provenance, Mapping)
            and provenance.get("merge_candidate_enqueued_event_id")
            == enqueues[0][1].get("event_id")
            and auxiliary_merge.get("canonical_task_key") == task_key
            and auxiliary_merge.get("canonical_task_cid") == task_cid
            and auxiliary_merge.get("implementation_commit") == commit
            and auxiliary_merge.get("queued") is True
            and event.get("provider_dispatched") is False
            and event.get("attempt_consumed") is False
        )
    reconciliation = merge.get("merge_reconciliation_receipt")
    return bool(
        event.get("request_id") == request_id
        and event.get("resolved") is True
        and auxiliary_merge.get("merged") is True
        and auxiliary_merge.get("merge_commit") == merge.get("merge_commit")
        and event.get("completion_task_cids") == {alias: task_cid}
        and isinstance(reconciliation, Mapping)
        and reconciliation.get("event_id") == event.get("event_id")
        and reconciliation.get("recorded") is True
    )


def retained_callback_confirmation_is_exact(
    events: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
    completion: Mapping[str, Any],
    exact_completion: Mapping[str, Any] | None,
    receipt: Mapping[str, Any],
    worktree_path: str,
    alias: str,
    task_cid: str,
    task_key: str,
    verify_reconciliation: Any,
) -> bool:
    """Recognize the closed retained-candidate producer after native integration.

    The caller has verified the event chain and full native train receipt.
    Missing auxiliary keys are interpreted only for structural comparison
    after exact launch/worktree validation. Original events and receipt IDs
    remain unchanged; no comparison view can issue completion authority.
    """
    fields = {
        "attempt",
        "attempt_consumed",
        "baseline_ref",
        "branch",
        "canonical_task_cid",
        "canonical_task_key",
        "commit_result",
        "event_id",
        "implementation_commit",
        "log_path",
        "merge_result",
        "previous_event_id",
        "protected_path_violation",
        "provider_dispatched",
        "recovery_key",
        "returncode",
        "sequence",
        "snapshot_id",
        "stream_id",
        "task_cid",
        "task_id",
        "timestamp",
        "type",
        "validation_result",
        "worktree_lifecycle_finalize_result",
        "worktree_lifecycle_reconciliation",
        "worktree_path",
    }
    merge = source.get("merge_result")
    candidate = source.get("implementation_commit")
    baseline = source.get("baseline_ref")
    request_id = receipt.get("request_id")
    integration = receipt.get("merge_commit")
    train_merge = receipt.get("merge_result")
    if (
        set(source) != fields
        or source.get("type") != "implementation_finished"
        or source.get("task_id") != alias
        or source.get("canonical_task_cid") != task_cid
        or source.get("canonical_task_key") != task_key
        or source.get("attempt_consumed") is not False
        or source.get("provider_dispatched") is not False
        or not worktree_path
        or source.get("worktree_path") != worktree_path
        or source.get("protected_path_violation") != {}
        or source.get("commit_result")
        != {
            "baseline_ref": baseline,
            "commit": candidate,
            "committed": True,
            "reason": "existing_commit",
        }
        or source.get("worktree_lifecycle_reconciliation")
        != {
            "attempt": 0,
            "attempted": False,
            "blocked": False,
            "finalized": False,
            "reason": "no_lifecycle_record",
            "record_absence_verified": True,
            "task_index_absence_verified": False,
        }
        or source.get("worktree_lifecycle_finalize_result")
        != {
            "finalized": None,
            "reason": "lifecycle_not_adopted",
        }
        or not isinstance(merge, Mapping)
        or not isinstance(train_merge, Mapping)
        or merge.get("train_result") != receipt
        or set(merge)
        != set(train_merge)
        | {
            "implementation_commit",
            "train_result",
            "reason",
            "worktree_lifecycle_handoff",
            "canonical_task_cid",
            "queue_dir",
            "completion_task_cids",
            "request_id",
            "queued",
            "canonical_task_key",
            "target_repository_id",
        }
        or any(merge.get(key) != value for key, value in train_merge.items())
        or merge.get("attempted") is not True
        or merge.get("queued") is not False
        or merge.get("merged") is not True
        or merge.get("reason") != "merged"
        or merge.get("request_id") != request_id
        or merge.get("merge_commit") != integration
        or merge.get("target_commit") != integration
        or merge.get("canonical_task_cid") != task_cid
        or merge.get("canonical_task_key") != task_key
        or completion.get("reason") != "protected_recovery_merge_completed"
        or completion.get("completion_receipt_repair") is not True
        or completion.get("previous_event_id") != source.get("event_id")
        or type(source.get("sequence")) is not int
        or completion.get("sequence") != source["sequence"] + 1
        or exact_completion
        != {
            "implementation_commit": candidate,
            "baseline_commit": baseline,
            "completion_source_event_id": source.get("event_id"),
            "completion_source_event_type": "implementation_finished",
            "completion_source_portal_attempt": source.get("attempt"),
            "completion_event_id": completion.get("event_id"),
        }
    ):
        return False
    completion_indices = [
        i
        for i, event in enumerate(events)
        if event.get("event_id") == completion.get("event_id")
    ]
    if len(completion_indices) != 1:
        return False
    auxiliary = {}
    for kind in ("worktree_reconciliation_candidate_queued", "merge_reconciled"):
        matches = [
            (i, event)
            for i, event in enumerate(events)
            if event.get("type") == kind
            and isinstance(event.get("merge_result"), Mapping)
            and event["merge_result"].get("request_id") == request_id
        ]
        if len(matches) != 1:
            return False
        index, event = matches[0]
        if not legacy_merge_observation_is_anchored(
            events,
            event_index=index,
            completion_index=completion_indices[0],
            alias=alias,
            task_cid=task_cid,
            task_key=task_key,
        ):
            return False
        auxiliary[kind] = event
    # These are derived comparison views only. The original verified event
    # hashes and nested content-addressed receipts are not rewritten.
    return bool(
        verify_reconciliation(
            {**auxiliary["merge_reconciled"], "canonical_task_key": task_key},
            {
                **auxiliary["worktree_reconciliation_candidate_queued"],
                "canonical_task_key": task_key,
            },
        )
    )
