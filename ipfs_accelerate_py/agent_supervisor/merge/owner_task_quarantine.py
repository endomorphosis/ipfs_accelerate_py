"""Retained task custody in a lane's native coordination store.

The daemon installs this denial only after central acknowledgement. The record
stores an exact bounded commitment to existing rows; neither expiry nor another
claim may rewrite those rows. There is no release, timeout or import API.
"""

from __future__ import annotations

from typing import Any

from ..task_sources.owner_task_quarantine import require, strict_json
from ..task_sources.control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
)

KEY = "owner_task_quarantine@1"
TABLES = (
    "coordination_tasks",
    "task_dependencies",
    "task_completions",
    "fenced_leases",
    "task_claims",
    "task_attempts",
    "resource_claims",
)
MAX_ROWS = 8192
MAX_BYTES = 4_194_304


def snapshot(connection: Any, task_cid: str) -> dict[str, Any]:
    result = {}
    remaining = MAX_ROWS
    bytes_remaining = MAX_BYTES
    for table in (*TABLES, "lease_events", "token_history"):
        if table in TABLES:
            where, parameters = "task_cid = ?", [task_cid]
        else:
            where = (
                "scope_key IN (SELECT scope_key FROM fenced_leases WHERE task_cid = ?)"
            )
            parameters = [task_cid]
        population = connection.execute(
            f"SELECT count(*), COALESCE(sum(octet_length(encode(to_json(t)))),0) FROM (SELECT * FROM {table} WHERE {where}) AS t",
            parameters,
        ).fetchone()
        count, size = population[0], population[1]
        require(
            type(count) is int
            and type(size) is int
            and 0 <= count <= remaining
            and 0 <= size <= bytes_remaining,
            "coordination_quarantine_population_bound",
        )
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE {where} ORDER BY ALL LIMIT ?",
            [*parameters, remaining + 1],
        ).fetchall()
        require(len(rows) == count, "coordination_quarantine_population_changed")
        # DuckDBRow iterates column names; copy values explicitly.
        values = [[row[index] for index in range(len(row))] for row in rows]
        encoded = canonical_json_bytes(values)
        require(len(encoded) <= bytes_remaining, "coordination_quarantine_byte_bound")
        remaining -= count
        bytes_remaining -= len(encoded)
        result[table] = {
            "rows": count,
            "bytes": len(encoded),
            "cid": content_identity(values),
        }
    return result


def records(connection: Any) -> dict[str, Any]:
    row = connection.execute(
        "SELECT value FROM coordination_metadata WHERE key = ?", [KEY]
    ).fetchone()
    if row is None:
        return {}
    require(
        type(row[0]) is str and len(row[0].encode()) <= MAX_BYTES,
        "coordination_quarantine_record_bound",
    )
    value = strict_json(row[0])
    require(
        type(value) is dict
        and set(value) == {"schema", "records", "cid"}
        and value["schema"] == KEY
        and type(value["records"]) is dict
        and 0 < len(value["records"]) <= 256
        and value["cid"] == content_identity(value["records"]),
        "coordination_quarantine_record_invalid",
    )
    for task_cid, entry in value["records"].items():
        require(
            type(task_cid) is str
            and bool(task_cid)
            and type(entry) is dict
            and set(entry) == {"event_id", "retained_state_cid", "snapshot"}
            and type(entry["event_id"]) is str
            and bool(entry["event_id"])
            and type(entry["retained_state_cid"]) is str
            and bool(entry["retained_state_cid"])
            and type(entry["snapshot"]) is dict
            and set(entry["snapshot"])
            == set((*TABLES, "lease_events", "token_history")),
            "coordination_quarantine_entry_invalid",
        )
    return value["records"]


def verify(connection: Any) -> dict[str, Any]:
    value = records(connection)
    for task_cid, entry in value.items():
        require(
            snapshot(connection, task_cid) == entry["snapshot"],
            "coordination_quarantine_custody_changed",
        )
    return value


def assert_unfenced(connection: Any, task_cid: str) -> None:
    require(task_cid not in records(connection), "coordination_task_quarantined")


def install(
    connection: Any,
    *,
    task_cid: str,
    event_id: str,
    retained_state_cid: str,
    expected_snapshot: dict[str, Any],
) -> None:
    prior = verify(connection)
    current = snapshot(connection, task_cid)
    require(current == expected_snapshot, "coordination_quarantine_admission_changed")
    entry = {
        "event_id": event_id,
        "retained_state_cid": retained_state_cid,
        "snapshot": current,
    }
    if task_cid in prior:
        require(
            prior[task_cid]["retained_state_cid"] == retained_state_cid
            and prior[task_cid]["snapshot"] == current,
            "coordination_quarantine_rebound",
        )
    updated = {**prior, task_cid: entry}
    value = {"schema": KEY, "records": updated, "cid": content_identity(updated)}
    connection.execute(
        "INSERT INTO coordination_metadata (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        [KEY, canonical_json_bytes(value).decode()],
    )
