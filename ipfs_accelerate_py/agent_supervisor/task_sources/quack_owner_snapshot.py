"""Fresh canonical state observations through the existing owner inbox.

This operation is read-only and has no completion, drain or launch authority.
It commits every table in the native schema inventory, including claims,
leases, fences, effects and receipts, in one writer transaction. A replica
consumer must match these commitments before using its rows in a native resume
qualification. No connection or mutable state fallback is created here.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .control_plane_contracts import StateSnapshot, canonical_json_bytes
from .control_plane_migrations import compute_schema_fingerprint
from .control_plane_schema import BOOKKEEPING_TABLES, DOMAIN_TABLES
from .quack_owner_mutation import (
    QuackOwnerMutationError,
    _row_tuple,
    build_mutation_request,
    execute_owner_request,
    validate_mutation_result,
)

OWNER_SNAPSHOT_OPERATION = "owner_snapshot@1"
OWNER_SNAPSHOT_SCHEMA = "ipfs_accelerate_py/quack-owner-snapshot@1"
MAX_SNAPSHOT_ROWS = 100_000
SNAPSHOT_TABLES = tuple(
    sorted(
        set(BOOKKEEPING_TABLES).union(
            table for tables in DOMAIN_TABLES.values() for table in tables
        )
    )
)


def snapshot_request(*, binding: Mapping[str, Any], token: str) -> dict[str, Any]:
    """Use a fresh challenge so an earlier successful read cannot be replayed."""
    return build_mutation_request(
        steps=[
            {"template_id": OWNER_SNAPSHOT_OPERATION, "parameters": [uuid.uuid4().hex]}
        ],
        binding=binding,
        token=token,
    )


def table_commitments(connection: Any) -> dict[str, Any]:
    """Closed schema projection; caller owns the surrounding transaction.

    Row hashes avoid exporting credential or task payloads. Column order and
    types are bound alongside a sorted multiset of row hashes (duplicates are
    retained). The encoding is tied to the admitted DuckDB/extension runtime.
    Missing tables and exceeded bounds fail closed; no pagination truncation
    is accepted as a full snapshot.
    """
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_type = 'BASE TABLE'"
        ).fetchall()
    }
    if tables != set(SNAPSHOT_TABLES):
        raise QuackOwnerMutationError("snapshot_schema_inventory_drifted")
    result = {}
    remaining = MAX_SNAPSHOT_ROWS
    for table in SNAPSHOT_TABLES:
        columns = [
            list(_row_tuple(row))
            for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        ]
        if not columns:
            raise QuackOwnerMutationError("snapshot_table_missing")
        rows = connection.execute(
            f'SELECT sha256(to_json(r)) FROM "{table}" AS r LIMIT {remaining + 1}'
        ).fetchall()
        if len(rows) > remaining:
            raise QuackOwnerMutationError("snapshot_population_exceeded")
        remaining -= len(rows)
        hashes = sorted(str(row[0]) for row in rows)
        result[table] = {
            "row_count": len(rows),
            "sha256": "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(
                    {
                        "columns": columns,
                        "rows": hashes,
                    }
                )
            ).hexdigest(),
        }
    return result


def execute_owner_snapshot(
    connection: Any, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Service an authenticated read on the exclusive writer, never a replica."""
    if getattr(connection, "_quack_uri", ""):
        raise QuackOwnerMutationError("snapshot_requires_exclusive_writer")
    # Native owners admit both historical CID and SHA-256 schema fingerprints.
    from ..runtime.quack_state_server import _schema_fingerprint_digest

    binding = request["binding"]
    connection.execute("BEGIN TRANSACTION")
    try:
        fields = (
            "server_id",
            "store_id",
            "database_uuid",
            "process_birth_id",
            "listen_uri",
            "extension_fingerprint",
            "schema_revision",
            "generation",
        )
        rows = connection.execute(
            "SELECT " + ", ".join(fields) + ", status, stopped_at FROM state_servers "
            "WHERE server_id = ?",
            [binding["server_id"]],
        ).fetchall()
        generation = connection.execute(
            "SELECT generation, schema_revision, fence_epoch, revision, database_uuid, birth_id "
            "FROM store_generations ORDER BY generation DESC LIMIT 1"
        ).fetchone()
        fingerprint = connection.execute(
            "SELECT value FROM control_plane_metadata WHERE key = 'schema_fingerprint'"
        ).fetchone()
        if (
            len(rows) != 1
            or _row_tuple(rows[0])[:8] != tuple(binding[k] for k in fields)
            or rows[0][8] != "ready"
            or rows[0][9] is not None
            or generation is None
            or (generation[0], generation[1], generation[4], generation[5])
            != (
                binding["generation"],
                binding["schema_revision"],
                binding["database_uuid"],
                binding["process_birth_id"],
            )
            or fingerprint is None
            or _schema_fingerprint_digest(str(fingerprint[0]))
            != _schema_fingerprint_digest(binding["schema_fingerprint"])
        ):
            raise QuackOwnerMutationError("snapshot_owner_identity_drifted")
        if _schema_fingerprint_digest(
            compute_schema_fingerprint(connection)
        ) != _schema_fingerprint_digest(binding["schema_fingerprint"]):
            raise QuackOwnerMutationError("snapshot_physical_schema_drifted")
        watermark = int(
            connection.execute(
                "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events"
            ).fetchone()[0]
        )
        tables = table_commitments(connection)
        digest = (
            "sha256:"
            + hashlib.sha256(
                canonical_json_bytes(
                    {
                        "binding": binding,
                        "tables": tables,
                    }
                )
            ).hexdigest()
        )
        snapshot = StateSnapshot(
            snapshot_id="snapshot:" + request["steps"][0]["parameters"][0],
            store_id=binding["store_id"],
            database_uuid=binding["database_uuid"],
            generation=generation[0],
            schema_revision=generation[1],
            fence_epoch=generation[2],
            revision=generation[3],
            event_watermark=watermark,
            snapshot_digest=digest,
        )
        observed = {
            "schema": OWNER_SNAPSHOT_SCHEMA,
            "snapshot": snapshot.to_dict(),
            "tables": tables,
            "sampled_at_ms": int(time.time() * 1000),
            "completion_authority": False,
            "launch_authority": False,
            "drain_authority": False,
        }
        if (
            not request["issued_at_ms"]
            <= observed["sampled_at_ms"]
            <= request["expires_at_ms"]
        ):
            raise QuackOwnerMutationError("snapshot_request_expired")
    finally:
        # Even a successful observation has no state effects to commit.
        connection.execute("ROLLBACK")
    return observed


def validate_owner_snapshot(
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    token: str,
    now_ms: int | None = None,
) -> StateSnapshot:
    """Authenticate a fresh result; this does not accept any task or source."""
    validate_mutation_result(result, request=request, token=token)
    now = int(time.time() * 1000) if now_ms is None else now_ms
    observed = result.get("observed")
    if (
        request.get("operation") != OWNER_SNAPSHOT_OPERATION
        or result.get("rowcounts") != [0]
        or not isinstance(observed, Mapping)
        or set(observed)
        != {
            "schema",
            "snapshot",
            "tables",
            "sampled_at_ms",
            "completion_authority",
            "launch_authority",
            "drain_authority",
        }
        or observed.get("schema") != OWNER_SNAPSHOT_SCHEMA
        or any(
            observed.get(k) is not False
            for k in ("completion_authority", "launch_authority", "drain_authority")
        )
        or type(observed.get("sampled_at_ms")) is not int
        or not request["issued_at_ms"]
        <= observed["sampled_at_ms"]
        <= now
        <= request["expires_at_ms"]
    ):
        raise QuackOwnerMutationError("snapshot_result_invalid_or_stale")
    tables = observed["tables"]
    if not isinstance(tables, Mapping) or set(tables) != set(SNAPSHOT_TABLES):
        raise QuackOwnerMutationError("snapshot_inventory_invalid")
    population = 0
    for table in tables.values():
        if (
            not isinstance(table, Mapping)
            or set(table) != {"row_count", "sha256"}
            or type(table["row_count"]) is not int
            or table["row_count"] < 0
            or not isinstance(table["sha256"], str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", table["sha256"])
        ):
            raise QuackOwnerMutationError("snapshot_inventory_invalid")
        population += table["row_count"]
    if population > MAX_SNAPSHOT_ROWS:
        raise QuackOwnerMutationError("snapshot_population_exceeded")
    snapshot = StateSnapshot.from_dict(observed["snapshot"])
    binding = request["binding"]
    if (
        snapshot.snapshot_id != "snapshot:" + request["steps"][0]["parameters"][0]
        or any(
            getattr(snapshot, key) != binding[key]
            for key in ("store_id", "database_uuid", "generation", "schema_revision")
        )
        or snapshot.authority_class.value != "authoritative"
        or snapshot.snapshot_digest
        != "sha256:"
        + hashlib.sha256(
            canonical_json_bytes(
                {
                    "binding": binding,
                    "tables": tables,
                }
            )
        ).hexdigest()
    ):
        raise QuackOwnerMutationError("snapshot_binding_invalid")
    return snapshot


def observe_owner(
    *, binding: Mapping[str, Any], token: str, inbox: Path
) -> dict[str, Any]:
    """Obtain and verify a fresh signed snapshot from the configured owner.

    An old owner without this operation fails closed. No local database is
    opened and no unsigned replica read can satisfy this request.
    """
    request = snapshot_request(binding=binding, token=token)
    result = execute_owner_request(request, binding=binding, token=token, inbox=inbox)
    validate_owner_snapshot(result, request=request, token=token)
    return {"request": request, "result": result}
