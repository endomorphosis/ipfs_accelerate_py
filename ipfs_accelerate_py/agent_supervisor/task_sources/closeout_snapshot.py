"""Bounded native closeout facts; observation never settles a goal or task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .control_plane_contracts import content_identity

SCHEMA = "ipfs_accelerate_py/agent-supervisor/closeout-progress-snapshot@1"
OPERATION = "completion.closeout.snapshot"
MAX_ROWS = 512
# Only fixed in-tree table/column identities reach SQL. Unknown states remain
# visible; no expiry timestamp or projection can silently discard a claim.
RELATIONS = {
    "tasks": ("task_cid", ""),
    "goals": ("goal_cid", ""),
    "goal_edges": ("parent_goal_cid", ""),
    "task_dependencies": ("task_cid", ""),
    "task_claims": (
        "claim_id",
        "state NOT IN ('released', 'cancelled', 'canceled', 'completed', 'expired')",
    ),
    "leases": (
        "task_cid",
        "state NOT IN ('released', 'cancelled', 'canceled', 'completed', 'expired')",
    ),
    "resource_claims": (
        "claim_id",
        "state NOT IN ('released', 'cancelled', 'canceled', 'completed', 'expired')",
    ),
    "path_claims": (
        "claim_id",
        "state NOT IN ('released', 'cancelled', 'canceled', 'completed', 'expired')",
    ),
    "effect_claims": (
        "effect_id",
        "state NOT IN ('released', 'cancelled', 'canceled', 'completed', 'settled')",
    ),
    "task_blocks": (
        "block_id",
        "state NOT IN ('cleared', 'resolved', 'cancelled', 'canceled')",
    ),
    "merge_queue_entries": (
        "entry_id",
        "status NOT IN ('settled', 'cancelled', 'canceled')",
    ),
    "proof_obligations": (
        "obligation_id",
        "status NOT IN ('proved', 'proven', 'discharged', 'accepted', 'resolved')",
    ),
}


def capture_closeout_facts(connection: Any) -> dict[str, Any]:
    """Called only inside the caller-owned snapshot transaction and owner lock."""
    available = {
        str(row[0])
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    relations = {}
    for table, (order, predicate) in RELATIONS.items():
        if table not in available:
            relations[table] = {"available": False, "rows": [], "truncated": False}
            continue
        if predicate:
            state_column = predicate.split(" ", 1)[0]
            where = f" WHERE ({state_column} IS NULL OR ({predicate}))"
        else:
            where = ""
        result = connection.execute(
            f"SELECT * FROM {table}{where} ORDER BY {order} LIMIT {MAX_ROWS + 1}"
        )
        from .typed_state_owner import _result_columns

        columns = _result_columns(result)
        raw = result.fetchall()
        rows = [
            dict(row) if isinstance(row, Mapping) else dict(zip(columns, row))
            for row in raw[:MAX_ROWS]
        ]
        relations[table] = {
            "available": True,
            "rows": rows,
            "truncated": len(raw) > MAX_ROWS,
        }
    return {
        "relations": relations,
        "all_relations_available": all(r["available"] for r in relations.values()),
        "truncated": any(r["truncated"] for r in relations.values()),
        "goal_contracts_evaluated": False,
        "completion_authority": False,
        "external_semantic_obligations_verified": False,
    }


def seal_closeout_snapshot(
    completion_snapshot: Mapping[str, Any], facts: Mapping[str, Any]
) -> dict[str, Any]:
    material = {
        "schema": SCHEMA,
        "completion_snapshot": dict(completion_snapshot),
        "closeout_facts": dict(facts),
        "completion_authority": False,
    }
    return {**material, "snapshot_cid": content_identity(material)}


def validate_closeout_snapshot(
    value: Any, *, request: Mapping[str, Any]
) -> Mapping[str, Any]:
    from .typed_state_owner import (
        TypedStateOwnerProtocolError,
        validate_completion_progress_snapshot,
    )

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "completion_snapshot",
        "closeout_facts",
        "completion_authority",
        "snapshot_cid",
    }:
        raise TypedStateOwnerProtocolError("closeout snapshot shape differs")
    material = {k: v for k, v in value.items() if k != "snapshot_cid"}
    if (
        value.get("schema") != SCHEMA
        or value.get("completion_authority") is not False
        or content_identity(material) != value.get("snapshot_cid")
    ):
        raise TypedStateOwnerProtocolError("closeout snapshot seal differs")
    validate_completion_progress_snapshot(value["completion_snapshot"], request=request)
    facts = value["closeout_facts"]
    if (
        not isinstance(facts, Mapping)
        or set(facts.get("relations", {})) != set(RELATIONS)
        or facts.get("completion_authority") is not False
    ):
        raise TypedStateOwnerProtocolError("closeout relation inventory differs")
    for row in facts["relations"].values():
        if (
            not isinstance(row, Mapping)
            or type(row.get("available")) is not bool
            or type(row.get("truncated")) is not bool
            or not isinstance(row.get("rows"), list)
            or len(row["rows"]) > MAX_ROWS
        ):
            raise TypedStateOwnerProtocolError(
                "closeout relation population exceeds bound"
            )
    return value
