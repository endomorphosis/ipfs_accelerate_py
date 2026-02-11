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
        self._init_db()

    def _connect(self):
        try:
            import duckdb  # type: ignore
        except Exception as exc:
            raise RuntimeError("duckdb is required for TaskQueue") from exc

        return duckdb.connect(self.path)

    def _init_db(self) -> None:
        # DuckDB can throw transient write-write conflicts if multiple processes
        # (or threads) try to create the schema at the same time.
        last_exc: Exception | None = None
        for attempt in range(12):
            conn = self._connect()
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
                if "write-write conflict" in msg or "catalog" in msg and "conflict" in msg:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

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

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO tasks(task_id, task_type, model_name, payload_json, status, assigned_worker, created_at, updated_at)
                VALUES(?, ?, ?, ?, 'queued', NULL, ?, ?)
                """,
                (tid, str(task_type), str(model_name), payload_json, now, now),
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return tid

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not task_id:
            return None

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

        conn = self._connect()
        try:
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
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
        conn = self._connect()
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

            # Preserve existing logs if incoming doesn't provide them.
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
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True


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

        last_exc: Exception | None = None
        for attempt in range(8):
            conn = self._connect()
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
                            current[str(k)] = v
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
                last_exc = exc
                msg = str(exc).lower()
                if "write-write conflict" in msg or "catalog" in msg and "conflict" in msg:
                    time.sleep(0.02 * (attempt + 1))
                    continue
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        if last_exc is not None:
            raise last_exc
        return False
