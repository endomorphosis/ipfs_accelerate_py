"""DuckDB + Quack + DuckLake catalog for semantic world-model records.

This is not the live extra-gate exclusive ``control.duckdb``. Records inform
agent-supervisor decisions and never admit task completion. DuckLake history
is observational only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "ipfs_accelerate_py/program-world-database@1"
INTERFACE = "ProgramWorldDatabase@1"
TABLE = "program_world_records"
ENV_DUCKDB = "IPFS_ACCELERATE_PROGRAM_WORLD_DUCKDB"
ENV_DUCKLAKE = "IPFS_ACCELERATE_PROGRAM_WORLD_DUCKLAKE"
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    record_id VARCHAR PRIMARY KEY,
    record_cid VARCHAR NOT NULL UNIQUE,
    task_id VARCHAR NOT NULL,
    board VARCHAR NOT NULL,
    operation VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL,
    proposal_only BOOLEAN NOT NULL,
    completion_authority BOOLEAN NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS program_world_records_task_idx
    ON {TABLE}(task_id, operation, recorded_at);
"""


class ProgramWorldDatabaseError(ValueError):
    """Closed world-model catalog contract violation."""


def _cid(payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return cid_for_bytes(body.encode("utf-8"))


def _refuse_extra_gate_control_plane(path: Path) -> None:
    resolved = path.resolve()
    if resolved.name == "control.duckdb":
        raise ProgramWorldDatabaseError(
            "program-world catalog refuses extra-gate exclusive control.duckdb"
        )


def _connect(path: Path):
    import duckdb

    _refuse_extra_gate_control_plane(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path), config={"threads": 1})
    connection.execute(DDL)
    return connection


@dataclass
class ProgramWorldDatabase:
    duckdb_path: Path
    ducklake_root: Path | None = None

    @classmethod
    def from_env(cls) -> "ProgramWorldDatabase | None":
        configured = str(os.environ.get(ENV_DUCKDB) or "").strip()
        if not configured:
            return None
        lake = str(os.environ.get(ENV_DUCKLAKE) or "").strip()
        return cls(Path(configured), Path(lake) if lake else None)

    def persist(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise ProgramWorldDatabaseError("world-model record must be an object")
        if record.get("completion_authority") is True or record.get("cas_completed") is True:
            raise ProgramWorldDatabaseError("world-model catalog cannot persist admitted completions")
        payload = dict(record)
        payload.setdefault("proposal_only", True)
        payload["completion_authority"] = False
        payload["cas_completed"] = False
        payload["admitted"] = False
        recorded_at = str(payload.get("recorded_at") or datetime.now(UTC).isoformat())
        record_cid = _cid(
            {
                "task_id": payload.get("task_id") or "",
                "board": payload.get("board") or "",
                "operation": payload.get("operation") or payload.get("surface") or "",
                "payload": payload,
            }
        )
        row = {
            "record_id": str(payload.get("record_id") or f"program-world:{record_cid}"),
            "record_cid": record_cid,
            "task_id": str(payload.get("task_id") or "unspecified"),
            "board": str(payload.get("board") or "sawm"),
            "operation": str(payload.get("operation") or payload.get("surface") or "world"),
            "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            "proposal_only": True,
            "completion_authority": False,
            "recorded_at": recorded_at,
        }
        connection = _connect(self.duckdb_path)
        try:
            connection.execute(
                f"""
                INSERT INTO {TABLE} (
                    record_id, record_cid, task_id, board, operation,
                    payload_json, proposal_only, completion_authority, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, TRUE, FALSE, ?)
                ON CONFLICT (record_cid) DO NOTHING
                """,
                [
                    row["record_id"],
                    row["record_cid"],
                    row["task_id"],
                    row["board"],
                    row["operation"],
                    row["payload_json"],
                    row["recorded_at"],
                ],
            )
        finally:
            connection.close()
        projection = self.project_ducklake((row,))
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                mirror_world_model_record,
            )

            meta = mirror_world_model_record({**payload, "record_cid": record_cid})
        except Exception:
            meta = {"status": "unavailable", "completion_authority": False}
        return {
            "schema": SCHEMA,
            "interface": INTERFACE,
            "record_cid": record_cid,
            "stored": True,
            "completion_authority": False,
            "ducklake": projection,
            "meta_index": meta,
        }

    def records_for_decision(
        self,
        *,
        task_id: str | None = None,
        operation: str | None = None,
        limit: int = 64,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 256:
            raise ProgramWorldDatabaseError("decision query limit must be 1..256")
        connection = _connect(self.duckdb_path)
        try:
            clauses = []
            params: list[Any] = []
            if task_id:
                clauses.append("task_id = ?")
                params.append(str(task_id))
            if operation:
                clauses.append("operation = ?")
                params.append(str(operation))
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = connection.execute(
                f"""
                SELECT record_id, record_cid, task_id, board, operation,
                       payload_json, recorded_at
                FROM {TABLE}{where}
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                [*params, int(limit)],
            ).fetchall()
        finally:
            connection.close()
        records = []
        for row in rows:
            payload = json.loads(row[5])
            payload["completion_authority"] = False
            records.append(
                {
                    "record_id": row[0],
                    "record_cid": row[1],
                    "task_id": row[2],
                    "board": row[3],
                    "operation": row[4],
                    "payload": payload,
                    "recorded_at": row[6],
                }
            )
        return {
            "schema": SCHEMA,
            "interface": INTERFACE,
            "records": records,
            "n": len(records),
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
        }

    def project_ducklake(
        self, rows: Sequence[Mapping[str, Any]] | None = None
    ) -> dict[str, Any]:
        root = self.ducklake_root
        if root is None:
            return {
                "status": "unconfigured",
                "reason_code": "ducklake_unconfigured",
                "completion_authority": False,
                "authoritative": False,
            }
        if not root.is_absolute():
            return {
                "status": "unavailable",
                "reason_code": "ducklake_root_must_be_absolute",
                "completion_authority": False,
                "authoritative": False,
            }
        try:
            import duckdb
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason_code": type(exc).__name__,
                "completion_authority": False,
                "authoritative": False,
            }
        catalog = root / "metadata.ducklake"
        data = root / "parquet"
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        data.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "256MB"})
        try:
            connection.execute("SET autoinstall_known_extensions=false")
            connection.execute("SET autoload_known_extensions=false")
            connection.execute("LOAD ducklake")
            catalog_sql = str(catalog).replace("'", "''")
            data_sql = str(data).replace("'", "''")
            connection.execute(
                "ATTACH 'ducklake:"
                + catalog_sql
                + "' AS world_model_lake (DATA_PATH '"
                + data_sql
                + "', DATA_INLINING_ROW_LIMIT 0)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS world_model_lake.program_world_records (
                    record_cid VARCHAR,
                    task_id VARCHAR,
                    board VARCHAR,
                    operation VARCHAR,
                    payload_json VARCHAR,
                    recorded_at VARCHAR,
                    completion_authority BOOLEAN
                )
                """
            )
            exported = rows
            if exported is None:
                source = _connect(self.duckdb_path)
                try:
                    exported = [
                        {
                            "record_cid": item[0],
                            "task_id": item[1],
                            "board": item[2],
                            "operation": item[3],
                            "payload_json": item[4],
                            "recorded_at": item[5],
                        }
                        for item in source.execute(
                            f"""
                            SELECT record_cid, task_id, board, operation,
                                   payload_json, recorded_at
                            FROM {TABLE}
                            """
                        ).fetchall()
                    ]
                finally:
                    source.close()
            connection.execute("BEGIN TRANSACTION")
            try:
                for item in exported:
                    connection.execute(
                        """
                        INSERT INTO world_model_lake.program_world_records
                        SELECT ?, ?, ?, ?, ?, ?, FALSE
                        WHERE NOT EXISTS (
                            SELECT 1 FROM world_model_lake.program_world_records
                            WHERE record_cid = ?
                        )
                        """,
                        [
                            item["record_cid"],
                            item["task_id"],
                            item["board"],
                            item["operation"],
                            item["payload_json"],
                            item["recorded_at"],
                            item["record_cid"],
                        ],
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            count = connection.execute(
                "SELECT count(*) FROM world_model_lake.program_world_records"
            ).fetchone()[0]
        except Exception as exc:
            return {
                "status": "unavailable",
                "reason_code": type(exc).__name__,
                "error": str(exc)[:200],
                "completion_authority": False,
                "authoritative": False,
            }
        finally:
            connection.close()
        return {
            "status": "projected",
            "stored_records": int(count),
            "completion_authority": False,
            "authoritative": False,
        }


def persist_program_world_record(
    record: Mapping[str, Any],
    *,
    store: ProgramWorldDatabase | None = None,
) -> dict[str, Any]:
    active = store if store is not None else ProgramWorldDatabase.from_env()
    if active is None:
        return {
            "status": "skip",
            "reason_code": "world_model_database_unconfigured",
            "completion_authority": False,
        }
    return active.persist(record)


def records_for_decision(
    *,
    task_id: str | None = None,
    operation: str | None = None,
    store: ProgramWorldDatabase | None = None,
    limit: int = 64,
) -> dict[str, Any]:
    active = store if store is not None else ProgramWorldDatabase.from_env()
    if active is None:
        return {
            "schema": SCHEMA,
            "records": [],
            "n": 0,
            "reason_code": "world_model_database_unconfigured",
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
        }
    return active.records_for_decision(task_id=task_id, operation=operation, limit=limit)
