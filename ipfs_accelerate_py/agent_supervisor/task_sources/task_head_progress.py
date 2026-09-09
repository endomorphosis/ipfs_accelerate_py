"""Replay admitted intent head changes for a stopped-run resume qualification.

This pure verifier does not authenticate its inputs, read a replica, settle a
claim, or authorize launch/completion. The caller must obtain the event suffix
and heads in one authenticated owner snapshot and bind the result to its native
source/owner admission. A bootstrap watermark remains the immutable anchor;
ordinary admitted task progress need not be mistaken for bootstrap corruption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .control_plane_contracts import content_identity
from .intent_repository import INTENT_EVENT_SCHEMA, INTENT_STREAM_ID

MAX_PROGRESS_EVENTS = 4096
MAX_PROGRESS_TASKS = 512
_HEAD_KEYS = frozenset({"task_cid", "task_alias", "goal_cid", "status", "revision"})
_STATUS_EVENTS = frozenset({"intent.task_status_changed", "intent.completion_recorded"})
_EVIDENCE_EVENTS = frozenset({"intent.validation_recorded", "intent.evidence_recorded"})
# Closed intent status vocabulary, independent of a particular owner transport.
# The verifier observes these existing transitions; it never admits a mutation.
_STATUS_SUCCESSORS = {
    **{status: frozenset({"in_progress"}) for status in (
        "todo", "ready", "open", "pending", "queued", "proposed", "admitted",
    )},
    "retrying": frozenset({"in_progress", "blocked"}),
    "claimed": frozenset({"in_progress", "ready", "blocked"}),
    "running": frozenset({"ready", "completed", "blocked", "retrying"}),
    "in_progress": frozenset({"ready", "completed", "blocked", "retrying"}),
    "blocked": frozenset({"retrying", "ready"}),
}


class TaskHeadProgressError(ValueError):
    """The event suffix cannot prove the observed task-head projection."""


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TaskHeadProgressError(f"invalid {label}")
    return value


def _heads(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not 0 < len(rows) <= MAX_PROGRESS_TASKS:
        raise TaskHeadProgressError("task population bound")
    result = {}
    aliases = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _HEAD_KEYS:
            raise TaskHeadProgressError("task head shape")
        _integer(row["revision"], "task revision", minimum=1)
        if any(type(row[k]) is not str or not row[k] for k in _HEAD_KEYS - {"revision"}):
            raise TaskHeadProgressError("task identity shape")
        cid, alias = row["task_cid"], row["task_alias"]
        if cid in result or alias in aliases:
            raise TaskHeadProgressError("duplicate task identity")
        result[cid] = dict(row)
        aliases.add(alias)
    return result


def verify_task_head_progress(
    *,
    anchor_heads: Sequence[Mapping[str, Any]],
    observed_heads: Sequence[Mapping[str, Any]],
    anchor_cursor: int,
    anchor_stream_sequence: int,
    observed_cursor: int,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify a bounded, contiguous suffix without modifying either input.

    Only status transitions and task evidence are admitted here. Definition,
    acceptance, goal, plan, foreign-stream and unknown events require another
    native transition. Completion events are replayed as existing head facts;
    their hashes do not replace the caller's completion-receipt verification.
    """
    anchor = _integer(anchor_cursor, "anchor cursor")
    cursor = _integer(observed_cursor, "observed cursor")
    stream_sequence = _integer(anchor_stream_sequence, "anchor stream sequence")
    if cursor < anchor or cursor - anchor != len(events) or len(events) > MAX_PROGRESS_EVENTS:
        raise TaskHeadProgressError("event suffix bound or gap")
    original = _heads(anchor_heads)
    replayed = {cid: dict(head) for cid, head in original.items()}
    observed = _heads(observed_heads)
    event_ids = []
    for offset, event in enumerate(events, 1):
        if not isinstance(event, Mapping):
            raise TaskHeadProgressError("event shape")
        global_sequence = _integer(event.get("global_sequence"), "event cursor", minimum=1)
        sequence = _integer(event.get("sequence"), "stream sequence", minimum=1)
        kind = event.get("event_type")
        wrapper = event.get("body")
        if (
            global_sequence != anchor + offset
            or sequence != stream_sequence + offset
            or event.get("stream_id") != INTENT_STREAM_ID
            or kind not in _STATUS_EVENTS | _EVIDENCE_EVENTS
            or not isinstance(wrapper, Mapping)
            or set(wrapper) != {"schema", "event_type", "subject_id", "body", "recorded_at", "owner_id"}
            or wrapper.get("schema") != INTENT_EVENT_SCHEMA
            or wrapper.get("event_type") != kind
            or wrapper.get("recorded_at") != event.get("recorded_at")
        ):
            raise TaskHeadProgressError("event binding or unsupported transition")
        expected_id = content_identity({
            "stream_id": INTENT_STREAM_ID, "sequence": sequence,
            "global_sequence": global_sequence, "event_type": kind, "body": wrapper,
        })
        if event.get("event_id") != expected_id:
            raise TaskHeadProgressError("event content identity")
        body = wrapper.get("body")
        if not isinstance(body, Mapping):
            raise TaskHeadProgressError("event payload")
        cid = body.get("task_cid")
        if not isinstance(cid, str) or cid not in replayed:
            raise TaskHeadProgressError("foreign event task")
        subject = (
            body.get("result_id") if kind == "intent.validation_recorded"
            else body.get("evidence_id") if kind == "intent.evidence_recorded"
            else cid
        )
        if type(subject) is not str or not subject or wrapper.get("subject_id") != subject:
            raise TaskHeadProgressError("event subject binding")
        # Some native intent appenders leave the optional outer task index
        # empty. If populated, it must identify the same subject.
        if event.get("task_cid", "") not in ("", cid):
            raise TaskHeadProgressError("event task index")
        if kind in _STATUS_EVENTS:
            head = replayed[cid]
            revision = _integer(body.get("revision"), "transition revision", minimum=1)
            status = body.get("status")
            if (
                body.get("task_alias") != head["task_alias"]
                or body.get("goal_cid") != head["goal_cid"]
                or body.get("previous_status") != head["status"]
                or revision != head["revision"] + 1
                or status not in _STATUS_SUCCESSORS.get(head["status"], ())
                or (kind == "intent.completion_recorded") != (status == "completed")
            ):
                raise TaskHeadProgressError("task transition chain")
            if kind == "intent.completion_recorded" and any(
                type(body.get(key)) is not str or not body[key]
                for key in ("completion_receipt_cid", "evidence_digest")
            ):
                raise TaskHeadProgressError("completion event binding missing")
            head.update(status=status, revision=revision)
        event_ids.append(expected_id)
    if replayed != observed:
        raise TaskHeadProgressError("observed heads differ from event replay")
    material = {
        "anchor_cursor": anchor, "anchor_stream_sequence": stream_sequence,
        "observed_cursor": cursor, "event_ids": event_ids,
        "anchor_heads": [original[cid] for cid in sorted(original)],
        "observed_heads": [observed[cid] for cid in sorted(observed)],
    }
    return {
        "schema": "ipfs_accelerate_py/intent-task-head-progress@1",
        **material, "progress_cid": content_identity(material),
        "task_completion_authority": False, "launch_authority": False,
    }
