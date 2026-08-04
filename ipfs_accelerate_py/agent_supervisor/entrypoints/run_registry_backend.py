"""DuckDB backend for mutable run-registry state.

Immutable IPLD history (run roots + content-addressed handle snapshots) remains
on the filesystem.  This module stores only *mutable* registry state:

* CAS heads (revision + handle CID + status fields)
* run_id -> namespace index
* namespace current-run pointers
* migration bookkeeping

Compare-and-swap is enforced with conditional SQL updates so two concurrent
writers with the same expected revision cannot both commit.

Immutable replicas open the database read-only and reject all mutating calls.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

DUCKDB_SCHEMA_VERSION: Final = 1

_SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS registry_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS run_heads (
    run_id VARCHAR PRIMARY KEY,
    run_namespace VARCHAR NOT NULL,
    run_revision BIGINT NOT NULL,
    handle_cid VARCHAR NOT NULL,
    semantic_id VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    health VARCHAR NOT NULL,
    event_cursor VARCHAR NOT NULL DEFAULT '',
    updated_at_ms BIGINT NOT NULL,
    previous_handle_cid VARCHAR NOT NULL DEFAULT '',
    previous_revision BIGINT NOT NULL DEFAULT 0,
    head_json VARCHAR NOT NULL,
    integrity_cid VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS run_index (
    run_id VARCHAR PRIMARY KEY,
    run_namespace VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS namespace_current (
    run_namespace VARCHAR PRIMARY KEY,
    repository_id VARCHAR NOT NULL,
    checkout_id VARCHAR NOT NULL DEFAULT '',
    selected_run_id VARCHAR NOT NULL DEFAULT '',
    integrity_cid VARCHAR NOT NULL DEFAULT '',
    pointer_revision BIGINT NOT NULL,
    updated_at_ms BIGINT NOT NULL,
    current_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS migration_log (
    id BIGINT PRIMARY KEY,
    migrated_at_ms BIGINT NOT NULL,
    detail_json VARCHAR NOT NULL
);
"""


class DuckDBRunRegistryBackendError(RuntimeError):
    """Raised for DuckDB backend operational failures."""


class DuckDBRunRegistryBackend:
    """Process-local DuckDB store for mutable run-registry pointers."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.read_only = bool(read_only)
        self._lock = threading.RLock()
        self._conn: Any = None
        self._open()

    def _open(self) -> None:
        try:
            import duckdb
        except ImportError as exc:
            raise DuckDBRunRegistryBackendError(
                "duckdb package is required for backend='duckdb'"
            ) from exc

        if self.read_only:
            if not self.db_path.exists():
                # Read-only open of a missing file: create an empty in-memory
                # view so queries succeed with zero rows (immutable empty replica).
                self._conn = duckdb.connect(database=":memory:")
                self._conn.execute(_SCHEMA_SQL)
                return
            # DuckDB read-only file open.
            self._conn = duckdb.connect(
                database=str(self.db_path), read_only=True
            )
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._conn = duckdb.connect(database=str(self.db_path))
        with self._lock:
            self._conn.execute(_SCHEMA_SQL)
            self._conn.execute(
                "INSERT OR REPLACE INTO registry_meta (key, value) VALUES (?, ?)",
                ["schema_version", str(DUCKDB_SCHEMA_VERSION)],
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def _ensure_open(self) -> Any:
        if self._conn is None:
            raise DuckDBRunRegistryBackendError("backend is closed")
        return self._conn

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise DuckDBRunRegistryBackendError(
                "immutable replica cannot mutate DuckDB run registry state"
            )

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def load_index(self) -> dict[str, str]:
        conn = self._ensure_open()
        with self._lock:
            rows = conn.execute(
                "SELECT run_id, run_namespace FROM run_index ORDER BY run_id"
            ).fetchall()
        return {str(run_id): str(namespace) for run_id, namespace in rows}

    def save_index(self, mapping: Mapping[str, str]) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        with self._lock:
            conn.execute("DELETE FROM run_index")
            for run_id in sorted(mapping):
                conn.execute(
                    "INSERT INTO run_index (run_id, run_namespace) VALUES (?, ?)",
                    [run_id, mapping[run_id]],
                )

    def index_put(self, run_id: str, run_namespace: str) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        with self._lock:
            conn.execute(
                "INSERT OR REPLACE INTO run_index (run_id, run_namespace) "
                "VALUES (?, ?)",
                [run_id, run_namespace],
            )

    def index_remove(self, run_id: str) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        with self._lock:
            conn.execute("DELETE FROM run_index WHERE run_id = ?", [run_id])

    def list_run_ids(self) -> list[tuple[str, str]]:
        conn = self._ensure_open()
        with self._lock:
            rows = conn.execute(
                "SELECT run_id, run_namespace FROM run_index ORDER BY run_id"
            ).fetchall()
        return [(str(run_id), str(namespace)) for run_id, namespace in rows]

    # ------------------------------------------------------------------
    # Heads
    # ------------------------------------------------------------------

    def load_head(self, run_id: str) -> dict[str, Any] | None:
        conn = self._ensure_open()
        with self._lock:
            row = conn.execute(
                "SELECT head_json FROM run_heads WHERE run_id = ?", [run_id]
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def store_head(
        self,
        run_id: str,
        run_namespace: str,
        head_payload: Mapping[str, Any],
    ) -> None:
        """Unconditional upsert of a head record (create / repair / migrate)."""

        self._ensure_writable()
        conn = self._ensure_open()
        payload = dict(head_payload)
        head_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_heads (
                    run_id, run_namespace, run_revision, handle_cid, semantic_id,
                    state, health, event_cursor, updated_at_ms,
                    previous_handle_cid, previous_revision, head_json, integrity_cid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    run_namespace,
                    int(payload["run_revision"]),
                    str(payload["handle_cid"]),
                    str(payload["semantic_id"]),
                    str(payload["state"]),
                    str(payload["health"]),
                    str(payload.get("event_cursor", "") or ""),
                    int(payload["updated_at_ms"]),
                    str(payload.get("previous_handle_cid", "") or ""),
                    int(payload.get("previous_revision", 0) or 0),
                    head_json,
                    str(payload.get("content_id") or payload.get("integrity_cid") or ""),
                ],
            )
            conn.execute(
                "INSERT OR REPLACE INTO run_index (run_id, run_namespace) VALUES (?, ?)",
                [run_id, run_namespace],
            )

    def cas_store_head(
        self,
        *,
        run_id: str,
        run_namespace: str,
        expected_revision: int,
        expected_handle_cid: str,
        head_payload: Mapping[str, Any],
    ) -> bool:
        """Conditional CAS update.  Returns True iff exactly one row changed.

        Two concurrent callers with the same expected revision cannot both win:
        the second UPDATE matches zero rows after the first commits.
        """

        self._ensure_writable()
        conn = self._ensure_open()
        payload = dict(head_payload)
        head_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        integrity = str(
            payload.get("content_id") or payload.get("integrity_cid") or ""
        )
        with self._lock:
            # Ensure the row exists (create path uses store_head; CAS assumes it).
            existing = conn.execute(
                "SELECT run_revision, handle_cid FROM run_heads WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if existing is None:
                return False
            if int(existing[0]) != int(expected_revision):
                return False
            if str(existing[1]) != str(expected_handle_cid):
                return False

            result = conn.execute(
                """
                UPDATE run_heads SET
                    run_namespace = ?,
                    run_revision = ?,
                    handle_cid = ?,
                    semantic_id = ?,
                    state = ?,
                    health = ?,
                    event_cursor = ?,
                    updated_at_ms = ?,
                    previous_handle_cid = ?,
                    previous_revision = ?,
                    head_json = ?,
                    integrity_cid = ?
                WHERE run_id = ?
                  AND run_revision = ?
                  AND handle_cid = ?
                """,
                [
                    run_namespace,
                    int(payload["run_revision"]),
                    str(payload["handle_cid"]),
                    str(payload["semantic_id"]),
                    str(payload["state"]),
                    str(payload["health"]),
                    str(payload.get("event_cursor", "") or ""),
                    int(payload["updated_at_ms"]),
                    str(payload.get("previous_handle_cid", "") or ""),
                    int(payload.get("previous_revision", 0) or 0),
                    head_json,
                    integrity,
                    run_id,
                    int(expected_revision),
                    str(expected_handle_cid),
                ],
            )
            # DuckDB returns a relation; verify post-condition.
            check = conn.execute(
                "SELECT run_revision, handle_cid FROM run_heads WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if check is None:
                return False
            return (
                int(check[0]) == int(payload["run_revision"])
                and str(check[1]) == str(payload["handle_cid"])
            )

    def delete_head(self, run_id: str) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        with self._lock:
            conn.execute("DELETE FROM run_heads WHERE run_id = ?", [run_id])
            conn.execute("DELETE FROM run_index WHERE run_id = ?", [run_id])

    # ------------------------------------------------------------------
    # Namespace current pointers
    # ------------------------------------------------------------------

    def load_current(self, run_namespace: str) -> dict[str, Any] | None:
        conn = self._ensure_open()
        with self._lock:
            row = conn.execute(
                "SELECT current_json FROM namespace_current WHERE run_namespace = ?",
                [run_namespace],
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def store_current(self, payload: Mapping[str, Any]) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        data = dict(payload)
        current_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
        with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO namespace_current (
                    run_namespace, repository_id, checkout_id, selected_run_id,
                    integrity_cid, pointer_revision, updated_at_ms, current_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(data["run_namespace"]),
                    str(data["repository_id"]),
                    str(data.get("checkout_id", "") or ""),
                    str(data.get("selected_run_id", "") or ""),
                    str(data.get("integrity_cid", "") or ""),
                    int(data["pointer_revision"]),
                    int(data["updated_at_ms"]),
                    current_json,
                ],
            )

    def cas_store_current(
        self,
        *,
        run_namespace: str,
        expected_pointer_revision: int,
        payload: Mapping[str, Any],
    ) -> bool:
        """CAS the namespace current pointer.

        ``expected_pointer_revision == 0`` means the pointer must be absent
        (insert).  Otherwise the existing revision must match exactly.
        """

        self._ensure_writable()
        conn = self._ensure_open()
        data = dict(payload)
        current_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
        with self._lock:
            existing = conn.execute(
                "SELECT pointer_revision FROM namespace_current "
                "WHERE run_namespace = ?",
                [run_namespace],
            ).fetchone()
            if expected_pointer_revision == 0:
                if existing is not None:
                    return False
                conn.execute(
                    """
                    INSERT INTO namespace_current (
                        run_namespace, repository_id, checkout_id, selected_run_id,
                        integrity_cid, pointer_revision, updated_at_ms, current_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        str(data["run_namespace"]),
                        str(data["repository_id"]),
                        str(data.get("checkout_id", "") or ""),
                        str(data.get("selected_run_id", "") or ""),
                        str(data.get("integrity_cid", "") or ""),
                        int(data["pointer_revision"]),
                        int(data["updated_at_ms"]),
                        current_json,
                    ],
                )
                return True

            if existing is None:
                return False
            if int(existing[0]) != int(expected_pointer_revision):
                return False
            conn.execute(
                """
                UPDATE namespace_current SET
                    repository_id = ?,
                    checkout_id = ?,
                    selected_run_id = ?,
                    integrity_cid = ?,
                    pointer_revision = ?,
                    updated_at_ms = ?,
                    current_json = ?
                WHERE run_namespace = ?
                  AND pointer_revision = ?
                """,
                [
                    str(data["repository_id"]),
                    str(data.get("checkout_id", "") or ""),
                    str(data.get("selected_run_id", "") or ""),
                    str(data.get("integrity_cid", "") or ""),
                    int(data["pointer_revision"]),
                    int(data["updated_at_ms"]),
                    current_json,
                    run_namespace,
                    int(expected_pointer_revision),
                ],
            )
            check = conn.execute(
                "SELECT pointer_revision FROM namespace_current "
                "WHERE run_namespace = ?",
                [run_namespace],
            ).fetchone()
            return (
                check is not None
                and int(check[0]) == int(data["pointer_revision"])
            )

    def delete_current(self, run_namespace: str) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        with self._lock:
            conn.execute(
                "DELETE FROM namespace_current WHERE run_namespace = ?",
                [run_namespace],
            )

    # ------------------------------------------------------------------
    # Migration bookkeeping
    # ------------------------------------------------------------------

    def mark_migration(self, detail: Mapping[str, Any]) -> None:
        self._ensure_writable()
        conn = self._ensure_open()
        detail_json = json.dumps(dict(detail), sort_keys=True, separators=(",", ":"))
        stamp = int(detail.get("migrated_at_ms") or 0)
        with self._lock:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM migration_log"
            ).fetchone()
            next_id = int(row[0]) + 1 if row else 1
            conn.execute(
                "INSERT INTO migration_log (id, migrated_at_ms, detail_json) "
                "VALUES (?, ?, ?)",
                [next_id, stamp, detail_json],
            )
            conn.execute(
                "INSERT OR REPLACE INTO registry_meta (key, value) VALUES (?, ?)",
                ["last_migration_at_ms", str(stamp)],
            )

    def migration_history(self) -> list[dict[str, Any]]:
        conn = self._ensure_open()
        with self._lock:
            rows = conn.execute(
                "SELECT id, migrated_at_ms, detail_json FROM migration_log "
                "ORDER BY id"
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row_id, stamp, detail_json in rows:
            try:
                detail = json.loads(detail_json)
            except (TypeError, json.JSONDecodeError):
                detail = {}
            if not isinstance(detail, dict):
                detail = {}
            results.append(
                {
                    "id": int(row_id),
                    "migrated_at_ms": int(stamp),
                    "detail": detail,
                }
            )
        return results

    def meta_get(self, key: str) -> str | None:
        conn = self._ensure_open()
        with self._lock:
            row = conn.execute(
                "SELECT value FROM registry_meta WHERE key = ?", [key]
            ).fetchone()
        if row is None:
            return None
        return str(row[0])


__all__ = (
    "DUCKDB_SCHEMA_VERSION",
    "DuckDBRunRegistryBackend",
    "DuckDBRunRegistryBackendError",
)
