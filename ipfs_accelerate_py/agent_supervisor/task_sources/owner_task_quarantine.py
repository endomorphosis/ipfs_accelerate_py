"""Retained-custody fences, separate from task and callback dispositions.

The existing authenticated owner transaction appends these events. A fence
never changes a task, receipt, lease or attempt and never authorizes its retry.
Revocation removes independent-work admission; it does not release custody.
There is deliberately no filesystem receipt importer or expiry-based release.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .control_plane_contracts import canonical_json_bytes, content_identity

SCHEMA = "ipfs_accelerate_py/agent-supervisor/owner-task-quarantine@1"
EVENT = "intent.owner_task_quarantine"
OPERATION = "owner_task_quarantine@1"
DIAGNOSES = frozenset(
    {"terminal_disposition_conflict", "entered_callback_state_missing"}
)
ANCHOR_KEY = "owner_task_quarantine_head@1"
ANCHOR_SQL = """INSERT INTO control_plane_metadata (key, value, updated_at) VALUES (?, ?, ?)
ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at"""
MAX_HISTORY = 4096
MAX_BYTES = 1_048_576
FIELDS = frozenset(
    {
        "schema",
        "state",
        "revision",
        "previous_event_id",
        "task_cid",
        "task_revision",
        "attempt_id",
        "execution_store_id",
        "execution_owner_id",
        "retained_state_cid",
        "owner_binding",
        "diagnosis",
        "workspace_custody_cid",
        "workspace_root",
        "fresh_workspace_root",
    }
)


class QuarantineDenied(RuntimeError):
    """A retained quarantine cannot authorize this operation."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise QuarantineDenied(reason)


def validate_body(value: Any) -> dict[str, Any]:
    # Import lazily: IntentRepository delegates to this module too.
    from .intent_repository import _fenced_provider_outer_owner_binding_valid

    require(type(value) is dict and set(value) == FIELDS, "quarantine_shape_invalid")
    require(
        value["schema"] == SCHEMA
        and value["state"] in {"active", "revoked"}
        and value["diagnosis"] in DIAGNOSES,
        "quarantine_state_invalid",
    )
    for key in ("revision", "task_revision"):
        require(
            type(value[key]) is int and 1 <= value[key] < 2**63,
            "quarantine_revision_invalid",
        )
    for key in (
        "task_cid",
        "attempt_id",
        "execution_store_id",
        "execution_owner_id",
        "retained_state_cid",
        "previous_event_id",
        "workspace_custody_cid",
        "workspace_root",
        "fresh_workspace_root",
    ):
        item = value[key]
        require(
            type(item) is str
            and len(item.encode()) <= 4096
            and not any(c in item for c in "\0\r\n")
            and (bool(item) or key == "previous_event_id"),
            "quarantine_identity_invalid",
        )
    require(
        _fenced_provider_outer_owner_binding_valid(value["owner_binding"]),
        "quarantine_owner_invalid",
    )
    require(len(canonical_json_bytes(value)) <= 32768, "quarantine_body_bound")
    return dict(value)


def heads(connection: Any, *, task_cid: str | None = None) -> dict[str, dict[str, Any]]:
    """Read the entire bounded immutable fence history; uncertainty never empties it."""
    # Always validate the complete fence population before applying a task filter.
    requested_task = task_cid
    where = "event_type = ?"
    parameters: list[Any] = [EVENT]
    population = connection.execute(
        f"SELECT count(*), COALESCE(sum(octet_length(encode(body_json))), 0) "
        f"FROM domain_events WHERE {where}",
        parameters,
    ).fetchone()
    count, size = population[0], population[1]
    require(
        type(count) is int
        and type(size) is int
        and count <= MAX_HISTORY
        and size <= MAX_BYTES,
        "quarantine_population_bound",
    )
    rows = connection.execute(
        "SELECT event_id, stream_id, sequence, global_sequence, event_type, "
        "task_cid, attempt_id, session_id, recorded_at, body_json "
        f"FROM domain_events WHERE {where} ORDER BY global_sequence LIMIT ?",
        [*parameters, MAX_HISTORY + 1],
    ).fetchall()
    require(len(rows) == count, "quarantine_population_changed")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        envelope = strict_json(row[9])
        require(
            type(envelope) is dict
            and set(envelope)
            == {
                "schema",
                "event_type",
                "subject_id",
                "body",
                "recorded_at",
                "owner_id",
            },
            "quarantine_event_shape_invalid",
        )
        body = validate_body(envelope["body"])
        require(
            row[1] == "stream:intent"
            and row[4] == EVENT
            and row[5] == body["task_cid"]
            and row[6] == body["attempt_id"]
            and envelope["schema"]
            == "ipfs_accelerate_py/agent-supervisor/intent-event@1"
            and envelope["subject_id"] == body["task_cid"]
            and envelope["event_type"] == EVENT
            and envelope["recorded_at"] == row[8]
            and row[0]
            == content_identity(
                {
                    "stream_id": row[1],
                    "sequence": row[2],
                    "global_sequence": row[3],
                    "event_type": row[4],
                    "body": envelope,
                }
            ),
            "quarantine_event_binding_invalid",
        )
        prior = result.get(body["attempt_id"])
        require(
            body["revision"] == (prior["revision"] + 1 if prior else 1)
            and body["previous_event_id"] == (prior["event_id"] if prior else ""),
            "quarantine_history_incomplete",
        )
        if prior:
            require(
                prior["state"] == "active"
                and all(
                    body[key] == prior[key]
                    for key in (
                        "task_cid",
                        "attempt_id",
                        "execution_store_id",
                        "execution_owner_id",
                        "retained_state_cid",
                        "diagnosis",
                        "workspace_custody_cid",
                        "workspace_root",
                        "fresh_workspace_root",
                    )
                ),
                "quarantine_history_rebound",
            )
        result[body["attempt_id"]] = {**body, "event_id": str(row[0])}
    anchor = connection.execute(
        "SELECT value FROM control_plane_metadata WHERE key = ?", [ANCHOR_KEY]
    ).fetchone()
    require(
        (not rows and anchor is None)
        or (anchor is not None and strict_json(anchor[0]) == anchor_value(rows)),
        "quarantine_history_anchor_mismatch",
    )
    return {
        key: value
        for key, value in result.items()
        if requested_task is None or value["task_cid"] == requested_task
    }


def strict_json(raw: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "quarantine_duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise QuarantineDenied("quarantine_nonfinite_json")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise QuarantineDenied("quarantine_json_invalid") from exc


def anchor_value(rows: Any) -> dict[str, Any]:
    return {
        "schema": ANCHOR_KEY,
        "count": len(rows),
        "event_ids_cid": content_identity([str(row[0]) for row in rows]),
    }


def next_anchor(connection: Any, event_id: str) -> dict[str, Any]:
    rows = connection.execute(
        "SELECT event_id FROM domain_events WHERE event_type = ? ORDER BY global_sequence LIMIT ?",
        [EVENT, MAX_HISTORY + 1],
    ).fetchall()
    require(len(rows) < MAX_HISTORY, "quarantine_population_bound")
    return anchor_value([*rows, (event_id,)])


def assert_task_unfenced(connection: Any, task_cid: str) -> None:
    # Revoked fences still forbid changing the retained task. A future exact
    # adjudication protocol is required to lift this denial, not this API.
    require(not heads(connection, task_cid=task_cid), "task_custody_quarantined")


def validate_append(
    connection: Any, body: Mapping[str, Any], owner_binding: Mapping[str, Any]
) -> dict[str, Any]:
    value = validate_body(dict(body))
    require(value["owner_binding"] == dict(owner_binding), "quarantine_owner_changed")
    row = connection.execute(
        "SELECT revision FROM tasks WHERE task_cid = ?", [value["task_cid"]]
    ).fetchone()
    require(
        row is not None and type(row[0]) is int and row[0] == value["task_revision"],
        "quarantine_task_revision_changed",
    )
    prior = heads(connection).get(value["attempt_id"])
    require(
        value["revision"] == (prior["revision"] + 1 if prior else 1)
        and value["previous_event_id"] == (prior["event_id"] if prior else ""),
        "quarantine_head_conflict",
    )
    if prior is None:
        require(value["state"] == "active", "quarantine_initial_state_invalid")
    else:
        require(
            prior["state"] == "active"
            and all(
                value[key] == prior[key]
                for key in (
                    "task_cid",
                    "attempt_id",
                    "execution_store_id",
                    "execution_owner_id",
                    "retained_state_cid",
                    "diagnosis",
                    "workspace_custody_cid",
                    "workspace_root",
                    "fresh_workspace_root",
                )
            ),
            "quarantine_retained_identity_changed",
        )
    return value


def append(
    repository: Any,
    *,
    retained: Mapping[str, Any],
    expected_event_id: str = "",
    revoke: bool = False,
) -> Mapping[str, Any]:
    """Native retained-owner caller only; no admission from an arbitrary report.

    The local execution owner supplies its freshly read retained identity. The
    central transaction independently binds its live owner and canonical task
    revision, and serializes the event head with every ordinary task CAS.
    """
    from .intent_repository import _fenced_provider_outer_normalize_binding

    with repository._connection(write=True) as connection:
        binding = _fenced_provider_outer_normalize_binding(connection)
        prior = heads(connection).get(str(retained["attempt_id"]))
        require(
            (prior["event_id"] if prior else "") == expected_event_id,
            "quarantine_head_conflict",
        )
        value = {
            **dict(retained),
            "schema": SCHEMA,
            "owner_binding": binding,
            "state": "revoked" if revoke else "active",
            "revision": prior["revision"] + 1 if prior else 1,
            "previous_event_id": expected_event_id,
        }
        validate_append(connection, value, binding)
        receipt = repository._append_event(
            connection,
            event_type=EVENT,
            subject_id=value["task_cid"],
            task_cid=value["task_cid"],
            attempt_id=value["attempt_id"],
            body=value,
        )
        # Bind a separate immutable-event population commitment in this same
        # owner transaction. Removing the only/last event cannot empty a fence.
        rows = connection.execute(
            "SELECT event_id FROM domain_events WHERE event_type = ? ORDER BY global_sequence LIMIT ?",
            [EVENT, MAX_HISTORY + 1],
        ).fetchall()
        if not rows or str(rows[-1][0]) != receipt.event_id:
            # A Quack adapter stages writes; its read view remains the preceding snapshot.
            rows = [*rows, (receipt.event_id,)]
        require(len(rows) <= MAX_HISTORY, "quarantine_population_bound")
        connection.execute(
            ANCHOR_SQL,
            [
                ANCHOR_KEY,
                canonical_json_bytes(anchor_value(rows)).decode(),
                receipt.recorded_at,
            ],
        )
    return {**value, "event_id": receipt.event_id}
