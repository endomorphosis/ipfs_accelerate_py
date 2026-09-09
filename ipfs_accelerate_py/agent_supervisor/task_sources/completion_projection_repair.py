"""Exact owner-local repair of legacy inherited completion receipt counters.

This repairs a projection of an already admitted completion event. It cannot
issue completion, change task status/revision, or relax receipt verification.
The caller owns the native owner lock and transaction; no file backend exists.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .control_plane_contracts import content_identity

EVENT = "intent.completion_projection_repaired"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/completion-projection-repair@1"
COUNTER = "unknown_callback_reopen_count"


def prepare_repair(connection: Any, *, task_cid: str, expected_revision: int,
                   expected_body_cid: str) -> dict[str, Any]:
    from .intent_repository import (
        IntentRepository, IntentRepositoryIntegrityError as Error,
        INTENT_STREAM_ID, INTENT_EVENT_SCHEMA, _decode_json,
        _receipt_with_preserved_reopen_count,
        _store_control_receipt_preserving_reopen_budget,
    )
    def require(condition, message):
        if not condition:
            raise Error("completion projection repair: " + message)
    row = connection.execute(
        "SELECT task_alias,goal_cid,status,revision,body_json FROM tasks WHERE task_cid=?",
        [task_cid]).fetchone()
    require(row is not None, "task unavailable")
    body = _decode_json(row[4])
    require(type(expected_revision) is int and row[3] == expected_revision
            and content_identity(body) == expected_body_cid, "task CAS changed")
    require(row[2] in ("complete", "completed"), "task is not completed")
    revisions = connection.execute(
        "SELECT revision,status,body_json FROM task_revisions WHERE task_cid=? AND revision IN (?,?) ORDER BY revision",
        [task_cid, expected_revision - 1, expected_revision]).fetchall()
    require(len(revisions) == 2 and revisions[0][0] == expected_revision - 1
            and revisions[1][0] == expected_revision
            and revisions[1][1] == row[2]
            and _decode_json(revisions[1][2]) == body, "revision history differs")
    predecessor = _decode_json(revisions[0][2])
    prior_receipt = predecessor.get("completion_receipt", {})
    require(type(prior_receipt.get(COUNTER)) is int and prior_receipt[COUNTER] >= 0,
            "no exact inherited counter")
    events = connection.execute(
        "SELECT event_id,stream_id,sequence,global_sequence,event_type,body_json FROM domain_events WHERE task_cid=? AND stream_id=? AND event_type=? ORDER BY global_sequence LIMIT 513",
        [task_cid, INTENT_STREAM_ID, "intent.completion_recorded"]).fetchall()
    require(len(events) <= 512, "event population exceeds bound")
    matching = []
    for e in events:
        wrapper = _decode_json(e[5])
        if wrapper.get("body", {}).get("revision") == expected_revision:
            matching.append((e, wrapper))
    require(len(matching) == 1, "completion event population is not exact")
    e, wrapper = matching[0]
    require(e[0] == content_identity(dict(stream_id=e[1], sequence=e[2],
            global_sequence=e[3], event_type=e[4], body=wrapper)), "event identity differs")
    payload = wrapper.get("body", {})
    require(wrapper.get("schema") == INTENT_EVENT_SCHEMA
            and wrapper.get("subject_id") == task_cid
            and wrapper.get("event_type") == e[4]
            and payload.get("task_cid") == task_cid
            and payload.get("goal_cid") == row[1]
            and payload.get("task_alias") == row[0]
            and payload.get("status") == row[2]
            and payload.get("previous_status") == revisions[0][1], "event binding differs")
    receipt = payload.get("receipt")
    require(isinstance(receipt, dict) and receipt and COUNTER not in receipt,
            "not an inherited-counter completion")
    legacy = deepcopy(predecessor)
    legacy["completion_receipt"] = _receipt_with_preserved_reopen_count(receipt, prior_receipt)
    require(legacy == body, "body differs beyond known legacy projection")
    corrected = deepcopy(predecessor)
    _store_control_receipt_preserving_reopen_budget(corrected, receipt, completing=True)
    task = dict(task_cid=task_cid, task_alias=row[0], goal_cid=row[1], status=row[2],
                revision=expected_revision, body=corrected)
    receipts = connection.execute(
        "SELECT receipt_cid,task_cid,goal_cid,attempt_id,claim_cid,fencing_token,completed_at,validation_run_id,evidence_digest,body_json FROM completion_receipts WHERE task_cid=? LIMIT 513",
        [task_cid]).fetchall()
    require(len(receipts) <= 512, "receipt population exceeds bound")
    binding, reasons = IntentRepository._current_task_completion_binding(task, receipts)
    require(binding is not None and not reasons, "original receipt does not verify")
    matched = next(r for r in receipts if r[0] == binding["completion_receipt_cid"])
    evidence = _decode_json(matched[9])
    require(payload.get("completion_receipt_cid") == matched[0]
            and payload.get("evidence_digest") == matched[8]
            and payload.get("evidence_digests") == evidence["evidence_digests"]
            and payload.get("receipt") == evidence["receipt"], "event and receipt differ")
    return dict(schema=SCHEMA, task_cid=task_cid, revision=expected_revision,
                before_body_cid=expected_body_cid, after_body_cid=content_identity(corrected),
                body=corrected, completion_event_id=e[0], completion_receipt_cid=matched[0])


def apply_projection(connection: Any, repair: Mapping[str, Any]) -> None:
    """Replay only the exact body correction; preserve all receipt/history rows."""
    from .intent_repository import IntentRepositoryIntegrityError, _decode_json, _canonical
    row = connection.execute("SELECT revision,body_json FROM tasks WHERE task_cid=?",
                             [repair["task_cid"]]).fetchone()
    if (repair.get("schema") != SCHEMA or row is None
            or row[0] != repair["revision"]
            or content_identity(repair["body"]) != repair["after_body_cid"]
            or content_identity(_decode_json(row[1])) not in
                (repair["before_body_cid"], repair["after_body_cid"])):
        raise IntentRepositoryIntegrityError("completion projection repair replay CAS differs")
    current = _decode_json(row[1])
    if content_identity(current) == repair["before_body_cid"]:
        # Even an admitted replay event cannot replace arbitrary task data.
        corrected = deepcopy(current)
        receipt = corrected.get("completion_receipt", {})
        counter = receipt.pop(COUNTER, None)
        old_budget = corrected.get(COUNTER, 0)
        if type(counter) is not int or counter < 0 or type(old_budget) is not int or old_budget < 0:
            raise IntentRepositoryIntegrityError("completion projection repair has no exact counter delta")
        corrected[COUNTER] = max(counter, old_budget)
        if corrected != repair["body"]:
            raise IntentRepositoryIntegrityError("completion projection repair changed unrelated task data")
    connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=? AND revision=?",
        [_canonical(repair["body"]), repair["task_cid"], repair["revision"]])


def recover_on_owner(connection: Any, *, owner_identity: Mapping[str, Any],
                     task_cids: list[str]) -> list[dict[str, Any]]:
    """Called solely by a bound native owner at qualified launcher startup."""
    from .closeout_snapshot import capture_closeout_facts
    from .intent_repository import IntentRepository, IntentRepositoryIntegrityError, _decode_json
    generation = connection.execute(
        "SELECT database_uuid,generation,fence_epoch,birth_id,revision FROM store_generations ORDER BY generation DESC LIMIT 1"
    ).fetchone()
    expected = tuple(owner_identity.get(k) for k in
                     ("database_uuid", "generation", "fence_epoch", "process_birth_id"))
    if generation is None or tuple(generation[i] for i in range(4)) != expected:
        raise IntentRepositoryIntegrityError("completion projection repair owner generation differs")
    facts = capture_closeout_facts(connection)
    if not facts["all_relations_available"] or facts["truncated"]:
        raise IntentRepositoryIntegrityError("completion projection repair population unavailable")
    for name in ("task_claims", "leases", "resource_claims", "path_claims", "effect_claims", "merge_queue_entries"):
        if facts["relations"][name]["rows"]:
            raise IntentRepositoryIntegrityError("completion projection repair unsettled " + name)
    counts = IntentRepository._goal_settlement_counts(connection)
    if any(counts.values()):
        raise IntentRepositoryIntegrityError("completion projection repair unsettled native runtime: "
            + ",".join(sorted(k for k, v in counts.items() if v)))
    repo = IntentRepository(bound_connection=connection, install_schema=False,
                            owner_id=owner_identity["process_birth_id"])
    results = []
    for task in facts["relations"]["tasks"]["rows"]:
        if task["task_cid"] not in task_cids or task["status"] not in ("complete", "completed"):
            continue
        body = _decode_json(task["body_json"])
        if COUNTER not in body.get("completion_receipt", {}):
            continue
        try:
            repair = prepare_repair(connection, task_cid=task["task_cid"],
                expected_revision=task["revision"], expected_body_cid=content_identity(body))
        except IntentRepositoryIntegrityError as exc:
            results.append(dict(task_cid=task["task_cid"], changed=False, reason=str(exc)))
            continue
        repair["owner_identity"] = dict(owner_identity)
        repair["store_revision_before"] = generation[4]
        apply_projection(connection, repair)
        event = repo._append_event(connection, event_type=EVENT,
            subject_id=task["task_cid"], task_cid=task["task_cid"], body=repair)
        results.append(dict(task_cid=task["task_cid"], changed=True, event_id=event.event_id,
                            completion_receipt_cid=repair["completion_receipt_cid"]))
    if any(r["changed"] for r in results):
        updated = connection.execute(
            "UPDATE store_generations SET revision=revision+1 WHERE generation=? AND revision=? RETURNING revision",
            [generation[1], generation[4]]).fetchone()
        if updated is None or updated[0] != generation[4] + 1:
            raise IntentRepositoryIntegrityError("completion projection repair store CAS changed")
    return results
