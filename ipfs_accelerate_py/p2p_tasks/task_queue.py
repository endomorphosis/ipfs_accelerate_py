"""DuckDB-backed task queue for distributed inference.

This is a lightweight task delegation mechanism used by both ipfs_datasets_py and
ipfs_accelerate_py. Schema is stable and backwards compatible.

Environment:
- IPFS_ACCELERATE_PY_TASK_QUEUE_PATH (preferred)
- IPFS_DATASETS_PY_TASK_QUEUE_PATH (compat)
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class QueuedTask:
    task_id: str
    task_type: str
    model_name: str
    payload: Dict[str, Any]
    created_at: float
    status: str
    assigned_worker: Optional[str] = None


def default_queue_path() -> str:
    return os.environ.get(
        "IPFS_ACCELERATE_PY_TASK_QUEUE_PATH",
        os.environ.get(
            "IPFS_DATASETS_PY_TASK_QUEUE_PATH",
            os.path.join(os.path.expanduser("~"), ".cache", "ipfs_datasets_py", "task_queue.duckdb"),
        ),
    )


class TaskQueue:
    """DuckDB-backed task queue.

    Concurrency model:
    - multiple workers may poll concurrently
    - claiming uses an atomic UPDATE guarded by a transaction
    """

    def __init__(self, path: Optional[str] = None):
        self.path = path or default_queue_path()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        # DuckDB connection management:
        # - In practice, p2p stream handlers may invoke TaskQueue concurrently.
        # - DuckDB 1.4.x can intermittently raise binder errors like
        #   "Unique file handle conflict: Cannot attach ... already attached"
        #   when connections are created concurrently.
        # Use a single shared connection per TaskQueue instance and serialize
        # access with a lock to keep behavior deterministic.
        self._conn_lock = threading.RLock()
        self._conn: object | None = None
        self._init_db()

    def _connect(self):
        try:
            import duckdb  # type: ignore
        except Exception as exc:
            raise RuntimeError("duckdb is required for TaskQueue") from exc

        # Best-effort retries to handle transient connection races.
        last_exc: Exception | None = None
        for attempt in range(8):
            try:
                return duckdb.connect(self.path)
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                low = msg.lower()
                if (
                    "unique file handle conflict" in low
                    or "already attached" in low
                    or "catalog" in low
                    and "conflict" in low
                ):
                    time.sleep(0.02 * (attempt + 1))
                    continue
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("duckdb.connect failed")

    def _get_conn(self):
        with self._conn_lock:
            if self._conn is None:
                self._conn = self._connect()
            return self._conn

    def close(self) -> None:
        with self._conn_lock:
            conn = self._conn
            self._conn = None
        if conn is not None:
            try:
                conn.close()  # type: ignore[attr-defined]
            except Exception:
                pass

    def _init_db(self) -> None:
        # DuckDB can throw transient write-write conflicts if multiple processes
        # (or threads) try to create the schema at the same time.
        last_exc: Exception | None = None
        for attempt in range(12):
            with self._conn_lock:
                # Force a fresh connection if init previously failed.
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    conn = self._conn
                except Exception as exc:
                    last_exc = exc
                    time.sleep(0.05 * (attempt + 1))
                    continue

                try:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS tasks (
                            task_id VARCHAR PRIMARY KEY,
                            task_type VARCHAR NOT NULL,
                            model_name VARCHAR NOT NULL,
                            payload_json VARCHAR NOT NULL,
                            status VARCHAR NOT NULL,
                            assigned_worker VARCHAR,
                            created_at DOUBLE NOT NULL,
                            updated_at DOUBLE NOT NULL,
                            result_json VARCHAR,
                            error VARCHAR
                        )
                        """
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at)")
                    return
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    # Reset connection and retry on transient write/attach conflicts.
                    if (
                        "write-write conflict" in msg
                        or "unique file handle conflict" in msg
                        or "already attached" in msg
                        or "catalog" in msg
                        and "conflict" in msg
                    ):
                        try:
                            if self._conn is not None:
                                self._conn.close()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        self._conn = None
                        time.sleep(0.05 * (attempt + 1))
                        continue
                    raise

        if last_exc is not None:
            raise last_exc

    def submit(
        self,
        *,
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
    ) -> str:
        tid = task_id or uuid.uuid4().hex
        now = time.time()
        payload_json = json.dumps(payload, sort_keys=True)

        with self._conn_lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO tasks(task_id, task_type, model_name, payload_json, status, assigned_worker, created_at, updated_at)
                VALUES(?, ?, ?, ?, 'queued', NULL, ?, ?)
                """,
                (tid, str(task_type), str(model_name), payload_json, now, now),
            )
        return tid

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not task_id:
            return None

        # Use a fresh connection so readers reliably observe updates from other
        # TaskQueue instances (e.g. worker heartbeats) across threads/processes.
        with self._conn_lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        if row is None:
            return None

        (
            _task_id,
            task_type,
            model_name,
            payload_json,
            status,
            assigned_worker,
            created_at,
            updated_at,
            result_json,
            error,
        ) = row

        result: Any = None
        if isinstance(result_json, str) and result_json:
            try:
                result = json.loads(result_json)
            except Exception:
                result = result_json
        return {
            "task_id": _task_id,
            "task_type": task_type,
            "model_name": model_name,
            "payload": json.loads(payload_json),
            "status": status,
            "assigned_worker": assigned_worker,
            "created_at": created_at,
            "updated_at": updated_at,
            "result": result,
            "error": error,
        }

    def list(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        task_types: Optional[Iterable[str]] = None,
    ) -> list[Dict[str, Any]]:
        """List tasks (best-effort, for debugging/visibility).

        Args:
            status: Optional status filter (e.g. queued/running/completed/failed)
            limit: Max rows returned
            task_types: Optional task_type allowlist
        """

        lim = max(1, min(int(limit or 100), 1000))
        status_norm = str(status).strip().lower() if status is not None else ""
        types = [t for t in (task_types or []) if isinstance(t, str) and t.strip()]

        with self._conn_lock:
            conn = self._get_conn()
            if status_norm and types:
                placeholders = ",".join(["?"] * len(types))
                rows = conn.execute(
                    f"SELECT * FROM tasks WHERE status=? AND task_type IN ({placeholders}) ORDER BY created_at ASC LIMIT ?",
                    (status_norm, *types, lim),
                ).fetchall()
            elif status_norm:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY created_at ASC LIMIT ?",
                    (status_norm, lim),
                ).fetchall()
            elif types:
                placeholders = ",".join(["?"] * len(types))
                rows = conn.execute(
                    f"SELECT * FROM tasks WHERE task_type IN ({placeholders}) ORDER BY created_at ASC LIMIT ?",
                    (*types, lim),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at ASC LIMIT ?",
                    (lim,),
                ).fetchall()

        out: list[Dict[str, Any]] = []
        for row in rows or []:
            (
                _task_id,
                task_type,
                model_name,
                payload_json,
                st,
                assigned_worker,
                created_at,
                updated_at,
                result_json,
                error,
            ) = row

            try:
                payload = json.loads(payload_json)
            except Exception:
                payload = {"raw": payload_json}

            result: Any = None
            if isinstance(result_json, str) and result_json:
                try:
                    result = json.loads(result_json)
                except Exception:
                    result = result_json

            out.append(
                {
                    "task_id": str(_task_id),
                    "task_type": str(task_type),
                    "model_name": str(model_name),
                    "payload": payload if isinstance(payload, dict) else {"payload": payload},
                    "status": str(st),
                    "assigned_worker": str(assigned_worker) if assigned_worker else None,
                    "created_at": float(created_at),
                    "updated_at": float(updated_at),
                    "result": result,
                    "error": str(error) if error else None,
                }
            )
        return out

    def counts_by_task_type(
        self,
        *,
        status: Optional[str] = None,
        task_types: Optional[Iterable[str]] = None,
    ) -> Dict[str, int]:
        """Return counts grouped by task_type.

        This is intended for lightweight monitoring/autoscaling logic.
        """

        status_norm = str(status).strip().lower() if status is not None else ""
        types = [t for t in (task_types or []) if isinstance(t, str) and t.strip()]

        where = []
        params: list[Any] = []
        if status_norm:
            where.append("status = ?")
            params.append(status_norm)
        if types:
            where.append("task_type IN (%s)" % ",".join(["?"] * len(types)))
            params.extend([str(t) for t in types])

        sql = "SELECT task_type, COUNT(*) AS n FROM tasks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY task_type"

        # Use a fresh connection so readers reliably observe updates from other
        # TaskQueue instances across threads/processes.
        with self._conn_lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        out: Dict[str, int] = {}
        for row in rows or []:
            try:
                ttype, n = row
                out[str(ttype)] = int(n)
            except Exception:
                continue
        return out

    def count(
        self,
        *,
        status: Optional[str] = None,
        task_types: Optional[Iterable[str]] = None,
    ) -> int:
        """Return a total count of tasks matching filters."""

        counts = self.counts_by_task_type(status=status, task_types=task_types)
        return int(sum(int(v) for v in counts.values()))

    def claim_next(
        self,
        *,
        worker_id: str,
        supported_task_types: Optional[Iterable[str]] = None,
    ) -> Optional[QueuedTask]:
        if not worker_id:
            raise ValueError("worker_id is required")

        task_types = [t for t in (supported_task_types or []) if isinstance(t, str) and t.strip()]
        now = time.time()

        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")

            if task_types:
                placeholders = ",".join(["?"] * len(task_types))
                row = conn.execute(
                    f"SELECT task_id FROM tasks WHERE status='queued' AND task_type IN ({placeholders}) ORDER BY created_at ASC LIMIT 1",
                    tuple(task_types),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT task_id FROM tasks WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
                ).fetchone()

            if row is None:
                conn.execute("COMMIT")
                return None

            task_id = str(row[0])
            conn.execute(
                """
                UPDATE tasks
                SET status='running', assigned_worker=?, updated_at=?
                WHERE task_id=? AND status='queued'
                """,
                (str(worker_id), now, task_id),
            )

            row2 = conn.execute(
                "SELECT * FROM tasks WHERE task_id=? AND status='running' AND assigned_worker=?",
                (task_id, str(worker_id)),
            ).fetchone()
            if row2 is None:
                conn.execute("COMMIT")
                return None

            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

        try:
            payload = json.loads(row2[3])
        except Exception:
            payload = {"raw": row2[3]}

        return QueuedTask(
            task_id=str(row2[0]),
            task_type=str(row2[1]),
            model_name=str(row2[2]),
            payload=payload if isinstance(payload, dict) else {"payload": payload},
            created_at=float(row2[6]),
            status=str(row2[4]),
            assigned_worker=str(row2[5]) if row2[5] else None,
        )

    def claim(
        self,
        *,
        task_id: str,
        worker_id: str,
    ) -> Optional[QueuedTask]:
        """Atomically claim a specific queued task by id."""

        if not task_id:
            return None
        if not worker_id:
            raise ValueError("worker_id is required")

        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            conn.execute(
                """
                UPDATE tasks
                SET status='running', assigned_worker=?, updated_at=?
                WHERE task_id=? AND status='queued'
                """,
                (str(worker_id), now, str(task_id)),
            )

            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id=? AND status='running' AND assigned_worker=?",
                (str(task_id), str(worker_id)),
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if row is None:
            return None

        try:
            payload = json.loads(row[3])
        except Exception:
            payload = {"raw": row[3]}

        return QueuedTask(
            task_id=str(row[0]),
            task_type=str(row[1]),
            model_name=str(row[2]),
            payload=payload if isinstance(payload, dict) else {"payload": payload},
            created_at=float(row[6]),
            status=str(row[4]),
            assigned_worker=str(row[5]) if row[5] else None,
        )

    def complete(
        self,
        *,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> bool:
        if not task_id:
            return False
        status_norm = (status or "").strip().lower()
        if status_norm not in {"completed", "failed", "cancelled"}:
            status_norm = "failed"

        def _json_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return value
            if isinstance(value, str) and value:
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
            return {}

        now = time.time()

        # Merge any existing progress/logs with the final result so peers can
        # keep observing stdout/stderr after completion.
        with self._conn_lock:
            conn = self._get_conn()
            try:
                existing_row = conn.execute(
                    "SELECT result_json FROM tasks WHERE task_id=?",
                    (str(task_id),),
                ).fetchone()
                existing = _json_dict(existing_row[0]) if existing_row else {}
                incoming = result if isinstance(result, dict) else {}

                merged: Dict[str, Any] = {}
                if isinstance(existing, dict):
                    merged.update(existing)
                if isinstance(incoming, dict):
                    merged.update(incoming)

                # Preserve existing logs/progress if incoming doesn't provide them.
                if "logs" in existing and "logs" not in incoming:
                    merged["logs"] = existing.get("logs")
                if "progress" in existing and "progress" not in incoming:
                    merged["progress"] = existing.get("progress")

                result_json = json.dumps(merged, sort_keys=True) if merged else None

                conn.execute(
                    """
                    UPDATE tasks
                    SET status=?, updated_at=?, result_json=?, error=?
                    WHERE task_id=?
                    """,
                    (status_norm, now, result_json, str(error) if error else None, str(task_id)),
                )
                return True
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "write-write conflict" in msg
                    or "catalog" in msg
                    and "conflict" in msg
                    or "conflict on tuple" in msg
                    or "transactioncontext error" in msg
                ):
                    return False
                raise


    def update(
        self,
        *,
        task_id: str,
        status: Optional[str] = None,
        result_patch: Optional[Dict[str, Any]] = None,
        append_log: Optional[str] = None,
        log_stream: str = "stdout",
        error: Optional[str] = None,
        max_logs: int = 200,
    ) -> bool:
        """Best-effort task progress update.

        This is intended for long-running tasks (e.g. docker) so peers can poll
        `get`/`wait` and observe heartbeats + stdout/stderr incrementally.
        """

        if not task_id:
            return False

        status_norm = str(status).strip().lower() if status is not None else ""
        if status_norm and status_norm not in {"queued", "running", "completed", "failed", "cancelled"}:
            status_norm = ""

        def _json_dict(value: Any) -> Dict[str, Any]:
            if isinstance(value, dict):
                return value
            if isinstance(value, str) and value:
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
            return {}

        now = time.time()
        max_keep = max(0, min(int(max_logs or 200), 2000))

        with self._conn_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT status, result_json FROM tasks WHERE task_id=?",
                    (str(task_id),),
                ).fetchone()
                if not row:
                    return False

                current_status = str(row[0] or "")
                current = _json_dict(row[1])

                if isinstance(result_patch, dict) and result_patch:
                    # Shallow merge patches into result dict.
                    for k, v in result_patch.items():
                        try:
                            key = str(k)
                            if isinstance(current.get(key), dict) and isinstance(v, dict):
                                # Prefer merging nested dicts (e.g. progress) so
                                # independent updaters can cooperate.
                                merged = dict(current.get(key) or {})
                                merged.update(v)
                                current[key] = merged
                            else:
                                current[key] = v
                        except Exception:
                            continue

                if append_log is not None:
                    entry = {
                        "ts": float(now),
                        "stream": str(log_stream or "stdout"),
                        "message": str(append_log),
                    }
                    logs = current.get("logs")
                    if not isinstance(logs, list):
                        logs = []
                    logs.append(entry)
                    if max_keep and len(logs) > max_keep:
                        logs = logs[-max_keep:]
                    current["logs"] = logs

                result_json = json.dumps(current, sort_keys=True) if current else None
                new_status = status_norm or current_status

                conn.execute(
                    """
                    UPDATE tasks
                    SET status=?, updated_at=?, result_json=?, error=?
                    WHERE task_id=?
                    """,
                    (new_status, now, result_json, str(error) if error else None, str(task_id)),
                )
                return True
            except Exception as exc:
                msg = str(exc).lower()
                if (
                    "write-write conflict" in msg
                    or "catalog" in msg
                    and "conflict" in msg
                    or "conflict on tuple" in msg
                    or "transactioncontext error" in msg
                ):
                    return False
                raise
