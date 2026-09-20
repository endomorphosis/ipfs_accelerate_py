"""DuckDB + Quack + DuckLake meta-index over supervisor catalogs.

Links filesystem/mtime, AST, BM25, knowledge-graph, vector, proof-cache,
proof-certificate, world-model, capsule, and taskboard catalogs by content
identity. Extra-gate exclusive ``control.duckdb`` is registered as a locator
only and is never attached. DuckLake history is observational. Records inform
event-driven orchestration and capsule composition; they never admit
completion.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "ipfs_accelerate_py/supervisor-meta-index@1"
INTERFACE = "SupervisorMetaIndex@1"
ENV_DUCKDB = "IPFS_ACCELERATE_META_INDEX_DUCKDB"
ENV_DUCKLAKE = "IPFS_ACCELERATE_META_INDEX_DUCKLAKE"

CATALOG_KINDS = frozenset(
    {
        "filesystem_mtime",
        "ast",
        "bm25",
        "knowledge_graph",
        "vector",
        "proof_cache",
        "proof_certificate",
        "world_model",
        "taskboard",
        "capsule",
        "metadata",
    }
)
SUBJECT_KINDS = frozenset(
    {
        "path",
        "content_cid",
        "tree_id",
        "task_id",
        "obligation_ref",
        "capsule_cid",
        "record_cid",
        "key_id",
        "receipt_id",
    }
)
DDL = """
CREATE TABLE IF NOT EXISTS catalogs (
    catalog_id VARCHAR PRIMARY KEY,
    catalog_cid VARCHAR NOT NULL UNIQUE,
    kind VARCHAR NOT NULL,
    locator_ref VARCHAR NOT NULL,
    exclusive_owner VARCHAR NOT NULL,
    attach_permitted BOOLEAN NOT NULL,
    repository_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    completion_authority BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS identity_links (
    link_id VARCHAR PRIMARY KEY,
    link_cid VARCHAR NOT NULL UNIQUE,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    catalog_id VARCHAR NOT NULL,
    record_kind VARCHAR NOT NULL,
    record_ref VARCHAR NOT NULL,
    freshness_mtime_ns BIGINT,
    recorded_at VARCHAR NOT NULL,
    completion_authority BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS identity_links_subject_idx
    ON identity_links(subject_kind, subject_ref, catalog_id);
CREATE TABLE IF NOT EXISTS capsule_bindings (
    binding_id VARCHAR PRIMARY KEY,
    capsule_cid VARCHAR NOT NULL,
    catalog_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    subject_ref VARCHAR NOT NULL,
    recorded_at VARCHAR NOT NULL,
    completion_authority BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS capsule_bindings_capsule_idx
    ON capsule_bindings(capsule_cid, catalog_id);
"""


class SupervisorMetaIndexError(ValueError):
    """Closed meta-index contract violation."""


def _cid(payload: Mapping[str, Any]) -> str:
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return cid_for_bytes(body.encode("utf-8"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _refuse_extra_gate_control_plane(path: Path) -> None:
    if path.resolve().name == "control.duckdb":
        raise SupervisorMetaIndexError(
            "meta-index refuses extra-gate exclusive control.duckdb"
        )


def _connect(path: Path):
    import duckdb

    _refuse_extra_gate_control_plane(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path), config={"threads": 1})
    connection.execute(DDL)
    return connection


def _attach_permitted(locator_ref: str, exclusive_owner: str) -> bool:
    name = Path(str(locator_ref or "")).name
    if name == "control.duckdb":
        return False
    if exclusive_owner:
        return False
    return bool(locator_ref)


@dataclass
class SupervisorMetaIndex:
    duckdb_path: Path
    ducklake_root: Path | None = None

    @classmethod
    def from_env(cls) -> "SupervisorMetaIndex | None":
        configured = str(os.environ.get(ENV_DUCKDB) or "").strip()
        if not configured:
            return None
        lake = str(os.environ.get(ENV_DUCKLAKE) or "").strip()
        return cls(Path(configured), Path(lake) if lake else None)

    def register_catalog(
        self,
        *,
        kind: str,
        locator_ref: str,
        exclusive_owner: str = "",
        repository_id: str = "",
        tree_id: str = "",
    ) -> dict[str, Any]:
        if kind not in CATALOG_KINDS:
            raise SupervisorMetaIndexError(f"unknown catalog kind {kind}")
        attach = _attach_permitted(locator_ref, exclusive_owner)
        recorded_at = _now()
        catalog_cid = _cid(
            {
                "kind": kind,
                "locator_ref": locator_ref,
                "exclusive_owner": exclusive_owner,
                "repository_id": repository_id,
                "tree_id": tree_id,
            }
        )
        catalog_id = f"catalog:{kind}:{catalog_cid}"
        connection = _connect(self.duckdb_path)
        try:
            connection.execute(
                """
                INSERT INTO catalogs (
                    catalog_id, catalog_cid, kind, locator_ref, exclusive_owner,
                    attach_permitted, repository_id, tree_id, recorded_at,
                    completion_authority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE)
                ON CONFLICT (catalog_cid) DO UPDATE SET
                    recorded_at=excluded.recorded_at
                """,
                [
                    catalog_id,
                    catalog_cid,
                    kind,
                    locator_ref,
                    exclusive_owner,
                    attach,
                    repository_id,
                    tree_id,
                    recorded_at,
                ],
            )
        finally:
            connection.close()
        self.project_ducklake()
        return {
            "schema": SCHEMA,
            "interface": INTERFACE,
            "catalog_id": catalog_id,
            "catalog_cid": catalog_cid,
            "attach_permitted": attach,
            "completion_authority": False,
        }

    def link_identity(
        self,
        *,
        subject_kind: str,
        subject_ref: str,
        catalog_id: str,
        record_kind: str,
        record_ref: str,
        freshness_mtime_ns: int | None = None,
        capsule_cid: str = "",
    ) -> dict[str, Any]:
        if subject_kind not in SUBJECT_KINDS:
            raise SupervisorMetaIndexError(f"unknown subject kind {subject_kind}")
        if not subject_ref or not catalog_id or not record_ref:
            raise SupervisorMetaIndexError("subject, catalog, and record refs are required")
        recorded_at = _now()
        link_cid = _cid(
            {
                "subject_kind": subject_kind,
                "subject_ref": subject_ref,
                "catalog_id": catalog_id,
                "record_kind": record_kind,
                "record_ref": record_ref,
            }
        )
        link_id = f"link:{link_cid}"
        connection = _connect(self.duckdb_path)
        try:
            connection.execute(
                """
                INSERT INTO identity_links (
                    link_id, link_cid, subject_kind, subject_ref, catalog_id,
                    record_kind, record_ref, freshness_mtime_ns, recorded_at,
                    completion_authority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE)
                ON CONFLICT (link_cid) DO UPDATE SET
                    freshness_mtime_ns=excluded.freshness_mtime_ns,
                    recorded_at=excluded.recorded_at
                """,
                [
                    link_id,
                    link_cid,
                    subject_kind,
                    subject_ref,
                    catalog_id,
                    record_kind,
                    record_ref,
                    freshness_mtime_ns,
                    recorded_at,
                ],
            )
            if capsule_cid:
                binding_cid = _cid(
                    {
                        "capsule_cid": capsule_cid,
                        "catalog_id": catalog_id,
                        "subject_ref": subject_ref,
                    }
                )
                connection.execute(
                    """
                    INSERT INTO capsule_bindings (
                        binding_id, capsule_cid, catalog_id, subject_kind,
                        subject_ref, recorded_at, completion_authority
                    ) VALUES (?, ?, ?, ?, ?, ?, FALSE)
                    ON CONFLICT (binding_id) DO UPDATE SET
                        recorded_at=excluded.recorded_at
                    """,
                    [
                        f"binding:{binding_cid}",
                        capsule_cid,
                        catalog_id,
                        subject_kind,
                        subject_ref,
                        recorded_at,
                    ],
                )
        finally:
            connection.close()
        self.project_ducklake()
        return {
            "schema": SCHEMA,
            "link_id": link_id,
            "link_cid": link_cid,
            "completion_authority": False,
        }

    def compose_for_subject(
        self,
        *,
        subject_kind: str,
        subject_ref: str,
        limit: int = 64,
    ) -> dict[str, Any]:
        if subject_kind not in SUBJECT_KINDS:
            raise SupervisorMetaIndexError(f"unknown subject kind {subject_kind}")
        if limit < 1 or limit > 256:
            raise SupervisorMetaIndexError("compose limit must be 1..256")
        connection = _connect(self.duckdb_path)
        try:
            rows = connection.execute(
                """
                SELECT l.subject_kind, l.subject_ref, l.record_kind, l.record_ref,
                       l.freshness_mtime_ns, c.kind, c.catalog_id, c.locator_ref,
                       c.attach_permitted, c.exclusive_owner, l.recorded_at
                FROM identity_links l
                JOIN catalogs c ON c.catalog_id = l.catalog_id
                WHERE l.subject_kind = ? AND l.subject_ref = ?
                ORDER BY l.recorded_at DESC
                LIMIT ?
                """,
                [subject_kind, subject_ref, int(limit)],
            ).fetchall()
        finally:
            connection.close()
        linked = []
        for row in rows:
            linked.append(
                {
                    "subject_kind": row[0],
                    "subject_ref": row[1],
                    "record_kind": row[2],
                    "record_ref": row[3],
                    "freshness_mtime_ns": row[4],
                    "catalog_kind": row[5],
                    "catalog_id": row[6],
                    "locator_ref": row[7],
                    "attach_permitted": bool(row[8]),
                    "exclusive_owner": row[9],
                    "recorded_at": row[10],
                }
            )
        return {
            "schema": SCHEMA,
            "interface": INTERFACE,
            "subject_kind": subject_kind,
            "subject_ref": subject_ref,
            "linked": linked,
            "n": len(linked),
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
            "event_driven_qualified": True,
        }

    def orchestration_view(self, *, tree_id: str = "", limit: int = 64) -> dict[str, Any]:
        if limit < 1 or limit > 256:
            raise SupervisorMetaIndexError("orchestration limit must be 1..256")
        connection = _connect(self.duckdb_path)
        try:
            if tree_id:
                rows = connection.execute(
                    """
                    SELECT catalog_id, kind, locator_ref, exclusive_owner,
                           attach_permitted, tree_id, recorded_at
                    FROM catalogs
                    WHERE tree_id = ? OR tree_id = ''
                    ORDER BY kind, recorded_at DESC
                    LIMIT ?
                    """,
                    [tree_id, int(limit)],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT catalog_id, kind, locator_ref, exclusive_owner,
                           attach_permitted, tree_id, recorded_at
                    FROM catalogs
                    ORDER BY kind, recorded_at DESC
                    LIMIT ?
                    """,
                    [int(limit)],
                ).fetchall()
        finally:
            connection.close()
        catalogs = [
            {
                "catalog_id": row[0],
                "kind": row[1],
                "locator_ref": row[2],
                "exclusive_owner": row[3],
                "attach_permitted": bool(row[4]),
                "tree_id": row[5],
                "recorded_at": row[6],
            }
            for row in rows
        ]
        return {
            "schema": SCHEMA,
            "interface": INTERFACE,
            "catalogs": catalogs,
            "n": len(catalogs),
            "kinds": sorted({item["kind"] for item in catalogs}),
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
            "event_driven_qualified": True,
            "extra_gate_attached": False,
        }

    def project_ducklake(self) -> dict[str, Any]:
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
                + "' AS meta_lake (DATA_PATH '"
                + data_sql
                + "', DATA_INLINING_ROW_LIMIT 0)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_lake.catalogs (
                    catalog_cid VARCHAR,
                    kind VARCHAR,
                    locator_ref VARCHAR,
                    exclusive_owner VARCHAR,
                    attach_permitted BOOLEAN,
                    tree_id VARCHAR,
                    recorded_at VARCHAR,
                    completion_authority BOOLEAN
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS meta_lake.identity_links (
                    link_cid VARCHAR,
                    subject_kind VARCHAR,
                    subject_ref VARCHAR,
                    catalog_id VARCHAR,
                    record_kind VARCHAR,
                    record_ref VARCHAR,
                    freshness_mtime_ns BIGINT,
                    recorded_at VARCHAR,
                    completion_authority BOOLEAN
                )
                """
            )
            source = _connect(self.duckdb_path)
            try:
                catalogs = source.execute(
                    """
                    SELECT catalog_cid, kind, locator_ref, exclusive_owner,
                           attach_permitted, tree_id, recorded_at
                    FROM catalogs
                    """
                ).fetchall()
                links = source.execute(
                    """
                    SELECT link_cid, subject_kind, subject_ref, catalog_id,
                           record_kind, record_ref, freshness_mtime_ns, recorded_at
                    FROM identity_links
                    """
                ).fetchall()
            finally:
                source.close()
            connection.execute("BEGIN TRANSACTION")
            try:
                for row in catalogs:
                    connection.execute(
                        """
                        INSERT INTO meta_lake.catalogs
                        SELECT ?, ?, ?, ?, ?, ?, ?, FALSE
                        WHERE NOT EXISTS (
                            SELECT 1 FROM meta_lake.catalogs WHERE catalog_cid = ?
                        )
                        """,
                        [*row, row[0]],
                    )
                for row in links:
                    connection.execute(
                        """
                        INSERT INTO meta_lake.identity_links
                        SELECT ?, ?, ?, ?, ?, ?, ?, ?, FALSE
                        WHERE NOT EXISTS (
                            SELECT 1 FROM meta_lake.identity_links WHERE link_cid = ?
                        )
                        """,
                        [*row, row[0]],
                    )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            count = connection.execute("SELECT count(*) FROM meta_lake.catalogs").fetchone()[0]
            link_count = connection.execute(
                "SELECT count(*) FROM meta_lake.identity_links"
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
            "stored_catalogs": int(count),
            "stored_links": int(link_count),
            "completion_authority": False,
            "authoritative": False,
        }


def _active() -> SupervisorMetaIndex | None:
    return SupervisorMetaIndex.from_env()


def register_catalog(**fields: Any) -> dict[str, Any]:
    index = _active()
    if index is None:
        return {
            "status": "skip",
            "reason_code": "meta_index_unconfigured",
            "completion_authority": False,
        }
    return index.register_catalog(**fields)


def link_identity(**fields: Any) -> dict[str, Any]:
    index = _active()
    if index is None:
        return {
            "status": "skip",
            "reason_code": "meta_index_unconfigured",
            "completion_authority": False,
        }
    return index.link_identity(**fields)


def compose_for_subject(
    *,
    subject_kind: str,
    subject_ref: str,
    store: SupervisorMetaIndex | None = None,
    limit: int = 64,
) -> dict[str, Any]:
    index = store if store is not None else _active()
    if index is None:
        return {
            "schema": SCHEMA,
            "linked": [],
            "n": 0,
            "reason_code": "meta_index_unconfigured",
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
            "event_driven_qualified": True,
        }
    return index.compose_for_subject(
        subject_kind=subject_kind, subject_ref=subject_ref, limit=limit
    )


def orchestration_view(
    *,
    tree_id: str = "",
    store: SupervisorMetaIndex | None = None,
    limit: int = 64,
) -> dict[str, Any]:
    index = store if store is not None else _active()
    if index is None:
        return {
            "schema": SCHEMA,
            "catalogs": [],
            "n": 0,
            "reason_code": "meta_index_unconfigured",
            "completion_authority": False,
            "decision_authority": False,
            "ducklake_authoritative": False,
            "event_driven_qualified": True,
            "extra_gate_attached": False,
        }
    return index.orchestration_view(tree_id=tree_id, limit=limit)


def mirror_world_model_record(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        catalog = register_catalog(
            kind="world_model",
            locator_ref=str(os.environ.get("IPFS_ACCELERATE_PROGRAM_WORLD_DUCKDB") or "world_model"),
            repository_id=str(record.get("board") or "sawm"),
        )
        if catalog.get("status") == "skip":
            return catalog
        return link_identity(
            subject_kind="task_id",
            subject_ref=str(record.get("task_id") or record.get("record_cid") or ""),
            catalog_id=str(catalog.get("catalog_id") or ""),
            record_kind="world_model",
            record_ref=str(record.get("record_cid") or record.get("task_id") or ""),
            capsule_cid=str(record.get("capsule_cid") or ""),
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason_code": type(exc).__name__,
            "completion_authority": False,
        }


def mirror_proof_certificate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        kind = "proof_certificate" if record.get("kind") == "zkp_certificate" else "proof_cache"
        catalog = register_catalog(
            kind=kind,
            locator_ref=str(
                os.environ.get("IPFS_ACCELERATE_PROOF_CERTIFICATE_DUCKDB") or "proof_certificates"
            ),
        )
        if catalog.get("status") == "skip":
            return catalog
        subject_kind = "receipt_id" if record.get("receipt_id") else "key_id"
        subject_ref = str(record.get("receipt_id") or record.get("key_id") or "")
        return link_identity(
            subject_kind=subject_kind,
            subject_ref=subject_ref,
            catalog_id=str(catalog.get("catalog_id") or ""),
            record_kind=kind,
            record_ref=str(record.get("record_cid") or subject_ref),
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason_code": type(exc).__name__,
            "completion_authority": False,
        }


def register_taskboard(
    *,
    board_id: str,
    exclusive_owner: str,
    locator_ref: str = "",
    tree_id: str = "",
) -> dict[str, Any]:
    return register_catalog(
        kind="taskboard",
        locator_ref=locator_ref or f"quack://{board_id}",
        exclusive_owner=exclusive_owner,
        repository_id=board_id,
        tree_id=tree_id,
    )
