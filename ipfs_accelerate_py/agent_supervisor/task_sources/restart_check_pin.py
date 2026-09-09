"""Derive a restart check from retained intent history, without moving its anchor.

The returned pin is an observation for a native launch verifier. It is never
owner, source, effect-settlement or completion authority. In particular a Quack
read replica remains a replica: callers must retain their owner/source gates.
No pin file is trusted on restart; each check derives the pin again.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .control_plane_contracts import content_identity
from .intent_repository import COMPLETION_EVIDENCE_SCHEMA
from .task_head_progress import (
    MAX_PROGRESS_EVENTS, MAX_PROGRESS_TASKS, TaskHeadProgressError,
    verify_task_head_progress,
)


class RestartCheckPinChanged(TaskHeadProgressError):
    """The source advanced between native preflight observations; read again."""


def refresh_restart_check(check: Callable[[], Any], *, attempts: int = 3) -> Any:
    """Retry only a native observation race, before any launch side effects.

    The native check must raise RestartCheckPinChanged before retiring a token
    handoff or launching a worker. Integrity/source/owner failures are not retries.
    """
    if type(attempts) is not int or not 1 <= attempts <= 5:
        raise ValueError("restart check attempts must be between 1 and 5")
    for attempt in range(attempts):
        try:
            return check()
        except RestartCheckPinChanged:
            if attempt + 1 == attempts:
                raise


def inspect_restart_check_pin(
    connection: Any, *, anchor_heads: Sequence[Mapping[str, Any]], anchor_cursor: int,
) -> dict[str, Any]:
    """Replay one bounded, consistent read through the caller's admitted source.

    One SQL statement reads heads, suffix, and revision/receipt evidence from
    the same database snapshot. A moving task must not mix two refreshes of a
    Quack replica. Unknown events, lost history and changed definitions fail
    closed; ordinary task progress advances the effective check automatically.
    """
    if type(anchor_cursor) is not int or anchor_cursor < 0:
        raise TaskHeadProgressError("invalid restart anchor")
    # Reject invalid/bulky anchors before constructing a database query.
    verify_task_head_progress(
        anchor_heads=anchor_heads, observed_heads=anchor_heads,
        anchor_cursor=anchor_cursor, anchor_stream_sequence=0,
        observed_cursor=anchor_cursor, events=[],
    )
    query = """
        WITH suffix AS (
          SELECT * FROM domain_events WHERE global_sequence > ?
          ORDER BY global_sequence LIMIT ?
        ), heads AS (
          SELECT task_cid,task_alias,goal_cid,status,revision,updated_at,body_json
          FROM tasks ORDER BY task_cid LIMIT ?
        ), revisions AS (
          SELECT r.* FROM task_revisions r JOIN suffix e
          ON r.task_cid = json_extract_string(e.body_json,'$.body.task_cid')
          AND r.revision = TRY_CAST(json_extract_string(e.body_json,'$.body.revision') AS BIGINT)
          WHERE e.event_type IN ('intent.task_status_changed','intent.completion_recorded')
        ), receipts AS (
          SELECT r.* FROM completion_receipts r JOIN suffix e
          ON r.receipt_cid = json_extract_string(e.body_json,'$.body.completion_receipt_cid')
          WHERE e.event_type = 'intent.completion_recorded'
        )
        SELECT
          (SELECT COALESCE(MAX(global_sequence),0) FROM domain_events),
          (SELECT COALESCE(MAX(sequence),0) FROM domain_events
           WHERE stream_id='stream:intent' AND global_sequence<=?),
          (SELECT to_json(list(h ORDER BY h.task_cid)) FROM heads h),
          (SELECT to_json(list(e ORDER BY e.global_sequence)) FROM suffix e),
          (SELECT to_json(list(r)) FROM revisions r),
          (SELECT to_json(list(r)) FROM receipts r)
        """
    parameters = [anchor_cursor, MAX_PROGRESS_EVENTS + 1, MAX_PROGRESS_TASKS + 1, anchor_cursor]
    uri = getattr(connection, "_quack_uri", "")
    if uri:
        # Quack ATTACH cannot run multiple streaming table scans in one query.
        # Execute this closed read on the remote database instead. Every inner
        # parameter is a validated integer; credentials stay bound parameters.
        token = getattr(connection, "_quack_mutation_token", "")
        if not token:
            raise TaskHeadProgressError("restart Quack observation lacks credentials")
        for value in parameters:
            query = query.replace("?", str(value), 1)
        try:
            row = connection.execute(
                "SELECT * FROM quack_query(?, ?, token := ?, disable_ssl := true)",
                [uri, query, token],
            ).fetchone()
        except Exception:
            raise TaskHeadProgressError("restart Quack snapshot query unavailable") from None
    else:
        row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise TaskHeadProgressError("restart observation missing")
    heads, raw_events, revisions, receipts = [json.loads(row[index] or "[]") for index in range(2, 6)]
    events = [{**e, "body": json.loads(e["body_json"])} for e in raw_events]
    observed = [{key: h[key] for key in (
        "task_cid", "task_alias", "goal_cid", "status", "revision",
    )} for h in heads]
    progress = verify_task_head_progress(
        anchor_heads=anchor_heads, observed_heads=observed,
        anchor_cursor=anchor_cursor, anchor_stream_sequence=int(row[1]),
        observed_cursor=int(row[0]), events=events,
    )
    by_revision = {(r["task_cid"], r["revision"]): r for r in revisions}
    by_receipt = {r["receipt_cid"]: r for r in receipts}
    if len(by_revision) != len(revisions) or len(by_receipt) != len(receipts):
        raise TaskHeadProgressError("duplicate restart evidence")
    current = {h["task_cid"]: h for h in heads}
    for event in events:
        kind = event["event_type"]
        if kind not in {"intent.task_status_changed", "intent.completion_recorded"}:
            continue
        body = event["body"]["body"]
        cid, revision = body["task_cid"], body["revision"]
        historical = by_revision.get((cid, revision))
        if (
            historical is None or historical["status"] != body["status"]
            or historical["recorded_at"] != body.get("recorded_at")
            or (current[cid]["revision"] == revision
                and (current[cid]["updated_at"] != historical["recorded_at"]
                     or json.loads(current[cid]["body_json"]) != json.loads(historical["body_json"])))
        ):
            reason = (
                "missing" if historical is None else "status"
                if historical["status"] != body["status"] else "recorded_at"
                if historical["recorded_at"] != body.get("recorded_at") else "body"
            )
            raise TaskHeadProgressError(f"restart revision evidence differs: {cid}@{revision}: {reason}")
        if kind == "intent.completion_recorded":
            receipt = by_receipt.get(body["completion_receipt_cid"])
            if receipt is None:
                raise TaskHeadProgressError("restart completion receipt missing")
            payload = json.loads(receipt["body_json"])
            if (
                not isinstance(payload, dict)
                or set(payload) != {"schema", "receipt", "evidence_digests", "revision"}
                or payload.get("schema") != COMPLETION_EVIDENCE_SCHEMA
                or type(payload.get("revision")) is not int
                or not isinstance(payload.get("receipt"), dict)
                or payload["receipt"] != body.get("receipt")
                or not isinstance(payload.get("evidence_digests"), list)
                or any(type(value) is not str or not value for value in payload["evidence_digests"])
            ):
                raise TaskHeadProgressError("restart completion receipt payload differs")
            evidence = content_identity({
                "task_cid": cid, "revision": revision,
                "receipt": payload.get("receipt"),
                "evidence_digests": payload.get("evidence_digests"),
            })
            expected_receipt = content_identity({
                "namespace": "completion-receipt", "task_cid": cid,
                "revision": revision, "evidence_digest": evidence,
            })
            if (
                receipt["task_cid"] != cid or receipt["goal_cid"] != body["goal_cid"]
                or receipt["completed_at"] != body.get("recorded_at")
                or payload.get("revision") != revision
                or receipt["evidence_digest"] != evidence
                or body["evidence_digest"] != evidence
                or receipt["receipt_cid"] != expected_receipt
                or not isinstance(payload.get("receipt"), dict)
                or not isinstance(payload.get("evidence_digests"), list)
            ):
                raise TaskHeadProgressError("restart completion receipt differs")
    material = {
        "schema": "ipfs_accelerate_py/task-head-restart-check-pin@1",
        "anchor_cursor": anchor_cursor, "event_cursor": int(row[0]),
        "advanced": int(row[0]) > anchor_cursor,
        "progress": progress,
        "launch_authority": False, "completion_authority": False,
        "source_change_authority": False, "effect_settlement_authority": False,
    }
    return {**material, "pin_cid": content_identity(material)}
