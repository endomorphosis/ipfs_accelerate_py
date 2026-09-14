"""Deny zero-provider recovery unless a complete native callback proves it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def verified_zero_provider_events(
    events: Sequence[Mapping[str, Any]], *, identity: Mapping[str, Any],
) -> bool:
    """Check the native chain and an explicit undispatched terminal callback.

    This additional precondition carries no recovery or completion authority.
    The owner must still verify the exact failed attempt, source, coordinator,
    retry budget and task CAS. Empty/missing evidence is never non-dispatch.
    """
    if (
        not events or len(events) > 65_536
        or set(identity) != {"task_id", "canonical_task_cid", "canonical_task_key"}
        or any(type(value) is not str or not value for value in identity.values())
    ):
        return False
    previous = ""
    stream = snapshot = ""
    starts: list[Mapping[str, Any]] = []
    finishes: list[Mapping[str, Any]] = []
    for ordinal, event in enumerate(events, 1):
        if not isinstance(event, Mapping):
            return False
        body = dict(event)
        claimed_id = body.pop("event_id", None)
        try:
            encoded = json.dumps(
                body, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False, allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError):
            return False
        current_stream = event.get("stream_id")
        current_snapshot = event.get("snapshot_id")
        if (
            claimed_id != "sha256:" + hashlib.sha256(encoded).hexdigest()
            or type(event.get("sequence")) is not int
            or event["sequence"] != ordinal
            or event.get("previous_event_id") != previous
            or type(current_stream) is not str or not current_stream
            or type(current_snapshot) is not str or not current_snapshot
            or (stream and current_stream != stream)
            or (snapshot and current_snapshot != snapshot)
        ):
            return False
        previous, stream, snapshot = claimed_id, current_stream, current_snapshot
        kind = str(event.get("type") or "")
        if (
            any(name in event and type(event[name]) is not bool for name in (
                "provider_dispatched", "attempt_consumed",
            ))
            or event.get("provider_dispatched") is True
            or event.get("attempt_consumed") is True
            or ("provider" in kind and kind.endswith(("_started", "_dispatched")))
            or kind.startswith(("validation_", "implementation_validation"))
            or ("merge" in kind and kind.endswith(("_started", "_queued", "_finished")))
            or kind in {
                "implementation_proposal_validated", "task_completed",
                "implementation_committed", "failed_validation_worktree_preserved",
            }
        ):
            return False
        for name in ("validation_result", "validation"):
            value = event.get(name)
            if isinstance(value, Mapping) and (
                value.get("attempted") is True or value.get("passed") is True
            ):
                return False
        if kind in {"implementation_started", "implementation_finished"}:
            if any(event.get(key) != value for key, value in identity.items()):
                return False
            if type(event.get("attempt")) is not int or event["attempt"] < 1:
                return False
            (starts if kind == "implementation_started" else finishes).append(event)
    if len(starts) != 1 or len(finishes) != 1:
        return False
    start, finish = starts[0], finishes[0]
    validation = finish.get("validation_result")
    commit = finish.get("commit_result")
    merge = finish.get("merge_result")
    if not all(isinstance(value, Mapping) for value in (validation, commit, merge)):
        return False
    return bool(
        start["sequence"] < finish["sequence"]
        and start["attempt"] == finish["attempt"]
        and start.get("provider_dispatched") is False
        and finish.get("provider_dispatched") is False
        and finish.get("attempt_consumed") is False
        and type(finish.get("returncode")) is int
        and finish["returncode"] != 0
        and finish.get("implementation_commit") == ""
        and validation.get("attempted") is False
        and validation.get("passed") is False
        and commit.get("committed") is False
        and commit.get("commit") in (None, "")
        and merge.get("merged") is False
        and (merge.get("queued") is None or merge.get("queued") is False)
        and merge.get("reason") == "not_attempted"
    )
