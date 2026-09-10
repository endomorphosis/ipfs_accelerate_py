"""Bounded diagnostic history, distinct from a complete lifecycle projection."""

from __future__ import annotations

import json
from collections.abc import Mapping

from .control_plane_contracts import content_identity

DIAGNOSTIC_HISTORY_SCHEMA = "ipfs_accelerate_py/task-diagnostic-history-window@1"
MAX_DIAGNOSTIC_HISTORY_ROWS = 32
MAX_DIAGNOSTIC_HISTORY_BYTES = 262_144
MAX_DIAGNOSTIC_HEAD_REVISION = 10_000


def diagnostic_window_start(revision: int) -> int:
    if type(revision) is not int or not 1 <= revision <= MAX_DIAGNOSTIC_HEAD_REVISION:
        raise ValueError("diagnostic history head exceeds admitted offset bound")
    return max(1, revision - MAX_DIAGNOSTIC_HISTORY_ROWS + 1)


def diagnostic_history_window(task_cid: str, revision: int, rows: list) -> dict:
    start = diagnostic_window_start(revision)
    if len(rows) != revision - start + 1:
        raise ValueError("diagnostic history window is incomplete")
    for expected, row in enumerate(rows, start):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"revision", "status", "body"}
            or type(row["revision"]) is not int
            or row["revision"] != expected
            or type(row["status"]) is not str
            or not isinstance(row["body"], Mapping)
        ):
            raise ValueError("diagnostic history window has a gap or malformed row")
    material = {
        "schema": DIAGNOSTIC_HISTORY_SCHEMA,
        "task_cid": task_cid,
        "head_revision": revision,
        "start_revision": start,
        "revisions": rows,
    }
    if (
        len(
            json.dumps(
                material, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        > MAX_DIAGNOSTIC_HISTORY_BYTES
    ):
        raise ValueError("diagnostic history window exceeds byte bound")
    return {**material, "projection_cid": content_identity(material)}
