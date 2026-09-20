"""DuckDB + Quack + DuckLake catalog for proof-cache and ZKP certificates.

This is not the live extra-gate exclusive ``control.duckdb``. Records inform
agent-supervisor decisions and never admit task completion. DuckLake history
is observational only. Private witnesses and secrets are refused.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "ipfs_accelerate_py/proof-certificate-database@1"
INTERFACE = "ProofCertificateDatabase@1"
TABLE = "proof_certificate_records"
ENV_DUCKDB = "IPFS_ACCELERATE_PROOF_CERTIFICATE_DUCKDB"
ENV_DUCKLAKE = "IPFS_ACCELERATE_PROOF_CERTIFICATE_DUCKLAKE"
KINDS = frozenset({"proof_cache", "zkp_certificate"})
FORBIDDEN_FIELDS = frozenset(
    {
        "witness",
        "private_witness",
        "secret",
        "secrets",
        "credential",
        "credentials",
        "prompt",
        "prompts",
        "raw_source",
        "source",
        "provider_payload",
    }
)
DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    record_id VARCHAR PRIMARY KEY,
    record_cid VARCHAR NOT NULL UNIQUE,
    kind VARCHAR NOT NULL,
    key_id VARCHAR NOT NULL,
    receipt_id VARCHAR NOT NULL,
    payload_json VARCHAR NOT NULL,
    proposal_only BOOLEAN NOT NULL,
    completion_authority BOOLEAN NOT NULL,
    recorded_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS proof_certificate_records_kind_idx
    ON {TABLE}(kind, key_id, recorded_at);
"""


class ProofCertificateDatabaseError(ValueError):
    """Closed proof-certificate catalog contract violation."""


def _cid(payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return cid_for_bytes(body.encode("utf-8"))


def _public_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in dict(record).items()
        if key not in FORBIDDEN_FIELDS
    }
    payload.setdefault("proposal_only", True)
    payload["completion_authority"] = False
    payload["cas_completed"] = False
    payload["admitted"] = False
    payload["decision_authority"] = False
    return payload


def _refuse_extra_gate_control_plane(path: Path) -> None:
    resolved = path.resolve()
    if resolved.name == "control.duckdb":
        raise ProofCertificateDatabaseError(
            "proof-certificate catalog refuses extra-gate exclusive control.duckdb"
        )


def _connect(path: Path):
    import duckdb

    _refuse_extra_gate_control_plane(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path), config={"threads": 1})
    connection.execute(DDL)
    return connection


@dataclass
class ProofCertificateDatabase:
    duckdb_path: Path
    ducklake_root: Path | None = None

    @classmethod
    def from_env(cls) -> "ProofCertificateDatabase | None":
        configured = str(os.environ.get(ENV_DUCKDB) or "").strip()
        if not configured:
            return None
        lake = str(os.environ.get(ENV_DUCKLAKE) or "").strip()
        return cls(Path(configured), Path(lake) if lake else None)

    def persist(self, record: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(record, Mapping):
            raise ProofCertificateDatabaseError("proof-certificate record must be an object")
        if record.get("completion_authority") is True or record.get("cas_completed") is True:
            raise ProofCertificateDatabaseError(
                "proof-certificate catalog cannot persist admitted completions"
            )
        if any(field in record for field in FORBIDDEN_FIELDS):
            raise ProofCertificateDatabaseError("private witness or secret fields are forbidden")
        payload = _public_payload(record)
        kind = str(payload.get("kind") or "")
        if kind not in KINDS:
            raise ProofCertificateDatabaseError("kind must be proof_cache or zkp_certificate")
        recorded_at = str(payload.get("recorded_at") or datetime.now(UTC).isoformat())
        record_cid = _cid(
            {
                "kind": kind,
                "key_id": payload.get("key_id") or "",
                "receipt_id": payload.get("receipt_id") or "",
                "payload": payload,
            }
        )
        row = {
            "record_id": str(payload.get("record_id") or f"proof-certificate:{record_cid}"),
            "record_cid": record_cid,
            "kind": kind,
            "key_id": str(payload.get("key_id") or "unspecified"),
            "receipt_id": str(payload.get("receipt_id") or payload.get("proof_receipt_id") or ""),
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
                    record_id, record_cid, kind, key_id, receipt_id,
                    payload_json, proposal_only, completion_authority, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, TRUE, FALSE, ?)
                ON CONFLICT (record_cid) DO NOTHING
                """,
                [
                    row["record_id"],
                    row["record_cid"],
                    row["kind"],
                    row["key_id"],
                    row["receipt_id"],
                    row["payload_json"],
                    row["recorded_at"],
                ],
            )
        finally:
            connection.close()
        projection = self.project_ducklake((row,))
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                mirror_proof_certificate_record,
            )

            meta = mirror_proof_certificate_record({**payload, "record_cid": record_cid})
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
        kind: str | None = None,
        key_id: str | None = None,
        receipt_id: str | None = None,
        limit: int = 64,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 256:
            raise ProofCertificateDatabaseError("decision query limit must be 1..256")
        if kind is not None and kind not in KINDS:
            raise ProofCertificateDatabaseError("kind must be proof_cache or zkp_certificate")
        connection = _connect(self.duckdb_path)
        try:
            clauses = []
            params: list[Any] = []
            if kind:
                clauses.append("kind = ?")
                params.append(kind)
            if key_id:
                clauses.append("key_id = ?")
                params.append(str(key_id))
            if receipt_id:
                clauses.append("receipt_id = ?")
                params.append(str(receipt_id))
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = connection.execute(
                f"""
                SELECT record_id, record_cid, kind, key_id, receipt_id,
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
                    "kind": row[2],
                    "key_id": row[3],
                    "receipt_id": row[4],
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
                + "' AS proof_lake (DATA_PATH '"
                + data_sql
                + "', DATA_INLINING_ROW_LIMIT 0)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS proof_lake.proof_certificate_records (
                    record_cid VARCHAR,
                    kind VARCHAR,
                    key_id VARCHAR,
                    receipt_id VARCHAR,
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
                            "kind": item[1],
                            "key_id": item[2],
                            "receipt_id": item[3],
                            "payload_json": item[4],
                            "recorded_at": item[5],
                        }
                        for item in source.execute(
                            f"""
                            SELECT record_cid, kind, key_id, receipt_id,
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
                        INSERT INTO proof_lake.proof_certificate_records
                        SELECT ?, ?, ?, ?, ?, ?, FALSE
                        WHERE NOT EXISTS (
                            SELECT 1 FROM proof_lake.proof_certificate_records
                            WHERE record_cid = ?
                        )
                        """,
                        [
                            item["record_cid"],
                            item["kind"],
                            item["key_id"],
                            item["receipt_id"],
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
                "SELECT count(*) FROM proof_lake.proof_certificate_records"
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


def persist_proof_certificate_record(
    record: Mapping[str, Any],
    *,
    store: ProofCertificateDatabase | None = None,
) -> dict[str, Any]:
    active = store if store is not None else ProofCertificateDatabase.from_env()
    if active is None:
        return {
            "status": "skip",
            "reason_code": "proof_certificate_database_unconfigured",
            "completion_authority": False,
        }
    return active.persist(record)


def records_for_decision(
    *,
    kind: str | None = None,
    key_id: str | None = None,
    receipt_id: str | None = None,
    store: ProofCertificateDatabase | None = None,
    limit: int = 64,
) -> dict[str, Any]:
    active = store if store is not None else ProofCertificateDatabase.from_env()
    if active is None:
        return {
            "schema": SCHEMA,
            "records": [],
            "n": 0,
            "reason_code": "proof_certificate_database_unconfigured",
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
        }
    return active.records_for_decision(
        kind=kind, key_id=key_id, receipt_id=receipt_id, limit=limit
    )


def mirror_proof_cache_entry(
    *,
    key_id: str,
    receipt_id: str,
    verdict: str = "",
    assurance: str = "",
    complete: bool = True,
) -> dict[str, Any]:
    try:
        return persist_proof_certificate_record(
            {
                "kind": "proof_cache",
                "key_id": key_id,
                "receipt_id": receipt_id,
                "verdict": verdict,
                "assurance": assurance,
                "complete": bool(complete),
                "proposal_only": True,
                "completion_authority": False,
            }
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason_code": type(exc).__name__,
            "completion_authority": False,
        }


def mirror_zkp_certificate(
    *,
    key_id: str,
    receipt_id: str,
    envelope_id: str = "",
    circuit_id: str = "",
    backend_id: str = "",
    public_input_digest: str = "",
    proof_digest: str = "",
    ipfs_cid: str = "",
    simulated: bool = False,
) -> dict[str, Any]:
    try:
        return persist_proof_certificate_record(
            {
                "kind": "zkp_certificate",
                "key_id": key_id,
                "receipt_id": receipt_id,
                "envelope_id": envelope_id,
                "circuit_id": circuit_id,
                "backend_id": backend_id,
                "public_input_digest": public_input_digest,
                "proof_digest": proof_digest,
                "ipfs_cid": ipfs_cid,
                "simulated": bool(simulated),
                "proposal_only": True,
                "completion_authority": False,
            }
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason_code": type(exc).__name__,
            "completion_authority": False,
        }
