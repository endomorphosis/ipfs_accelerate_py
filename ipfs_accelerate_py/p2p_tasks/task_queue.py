"""DuckDB-backed task queue for distributed inference.

This is a lightweight task delegation mechanism used by both ipfs_datasets_py and
ipfs_accelerate_py. Schema is stable and backwards compatible.

Voice-job reliability (ABBY-VOICE-G016) depends on persisted attempt/backoff/lease
state and owner heartbeats: claim ownership is recorded in DuckDB as ``attempt``,
``max_attempts``, ``next_attempt_at``, ``lease_until``, and ``heartbeat_at`` so a
worker crash recovers without duplicate provider execution.
An explicit ``max_attempts=0`` means no attempt ceiling; omitted values retain
the historical finite default of three attempts.

Environment:
- IPFS_ACCELERATE_PY_TASK_QUEUE_PATH (preferred)
- IPFS_DATASETS_PY_TASK_QUEUE_PATH (compat)
"""

from __future__ import annotations

import json
import os
import random
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
    priority: int = 5
    attempt: int = 0
    max_attempts: int = 3
    next_attempt_at: float = 0.0
    lease_until: Optional[float] = None
    heartbeat_at: Optional[float] = None


_TASK_SELECT_COLUMNS = (
    "task_id, task_type, model_name, payload_json, status, assigned_worker, "
    "created_at, updated_at, result_json, error, priority, attempt, max_attempts, "
    "next_attempt_at, lease_until, heartbeat_at, idempotency_key"
)


def _priority(value: Any) -> int:
    """Normalize public queue priorities to the existing 1..10 contract."""

    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 5


def _attempt_limit(value: Any) -> int:
    """Normalize retry limits, preserving zero as the unlimited sentinel."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 3
    if parsed == 0:
        return 0
    return max(1, min(1000, parsed))


_ATTEMPTS_AVAILABLE_SQL = (
    "(coalesce(max_attempts, 3) = 0 OR coalesce(attempt, 0) < coalesce(max_attempts, 3))"
)


def _claim_exprs() -> tuple[str, str, str]:
    """SQL fragments for claim eligibility shared by the Quack claim paths."""
    required_expr = (
        "coalesce("
        "nullif(json_extract_string(payload_json, '$.session_id'), ''), "
        "nullif(json_extract_string(payload_json, '$.session'), ''), "
        "nullif(json_extract_string(payload_json, '$.p2p_session'), '')"
        ")"
    )
    sticky_expr = "nullif(json_extract_string(payload_json, '$.sticky_worker_id'), '')"
    priority_expr = "coalesce(priority, 5)"
    return required_expr, sticky_expr, priority_expr


def _eligibility_where(
    *,
    task_types: Iterable[str],
    worker_id: str,
    session: str,
    max_priority: Optional[int],
    now: float,
) -> tuple[str, list[Any], str]:
    """Build ``(where_sql, params, priority_expr)`` for claim eligibility."""
    required_expr, sticky_expr, priority_expr = _claim_exprs()
    types = [str(t) for t in (task_types or []) if isinstance(t, str) and t.strip()]
    where: list[str] = [
        "status='queued'",
        "coalesce(next_attempt_at, 0) <= ?",
        _ATTEMPTS_AVAILABLE_SQL,
    ]
    params: list[Any] = [now]
    if types:
        where.append(f"task_type IN ({','.join(['?'] * len(types))})")
        params.extend(types)
    where.append(f"({sticky_expr} IS NULL OR {sticky_expr} = ?)")
    params.append(str(worker_id))
    if session:
        where.append(f"({required_expr} IS NULL OR {required_expr} = ?)")
        params.append(str(session))
    if max_priority is not None:
        cap = max(1, min(10, int(max_priority)))
        where.append(f"({priority_expr} <= ?)")
        params.append(cap)
    return " AND ".join(where), params, priority_expr


def _transient_conflict(exc: Exception) -> bool:
    low = str(exc or "").lower()
    return (
        "conflict on tuple" in low
        or "transactioncontext" in low
        or ("transaction" in low and "conflict" in low)
    )


def _queued_task_from_row(row: Any) -> QueuedTask:
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
        priority=int(row[10] if row[10] is not None else 5),
        attempt=int(row[11] if row[11] is not None else 0),
        max_attempts=int(row[12] if row[12] is not None else 3),
        next_attempt_at=float(row[13] if row[13] is not None else 0.0),
        lease_until=float(row[14]) if row[14] is not None else None,
        heartbeat_at=float(row[15]) if row[15] is not None else None,
    )


def default_queue_path() -> str:
    return os.environ.get(
        "IPFS_ACCELERATE_PY_TASK_QUEUE_PATH",
        os.environ.get(
            "IPFS_DATASETS_PY_TASK_QUEUE_PATH",
            os.path.join(
                os.path.expanduser("~"), ".cache", "ipfs_datasets_py", "task_queue.duckdb"
            ),
        ),
    )


def is_quack_path(path: Optional[str]) -> bool:
    """True when *path* is a Quack endpoint URI (``quack:host:port``)."""
    return str(path or "").strip().lower().startswith("quack:")


def quack_token() -> str:
    """Auth token for Quack queue endpoints.

    Read from ``TASK_QUEUE_QUACK_TOKEN`` directly, or from the file named by
    ``TASK_QUEUE_QUACK_TOKEN_FILE``. The file indirection exists for isolated
    subprocesses (e.g. the agent supervisor's LLM provider children) whose
    environment is scrubbed of ``*_QUACK_TOKEN`` variables but which are
    allowed to read a token *file path*: the path grants no authority by
    itself, and only the loopback queue token — never supervisor state
    authority — lives in that file.
    """
    tok = os.environ.get("TASK_QUEUE_QUACK_TOKEN", "").strip()
    if tok:
        return tok
    token_file = os.environ.get("TASK_QUEUE_QUACK_TOKEN_FILE", "").strip()
    if token_file:
        try:
            with open(token_file, "r", encoding="utf-8") as fh:
                tok = fh.read().strip()
            if tok:
                return tok
        except OSError:
            pass
    return ""


def _sql_literal(value: Any) -> str:
    """Render a Python value as a DuckDB SQL literal.

    Used to interpolate parameters into statements executed owner-side via
    ``quack_query_by_name`` (which takes the whole statement as one string).
    Strings are single-quoted with embedded quotes doubled; every other type
    is generated, never parsed, so there is no injection vector.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite float cannot be a SQL literal")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _interpolate_params(sql: str, params: Iterable[Any]) -> str:
    """Replace each ``?`` placeholder in *sql* with a SQL literal."""
    parts = sql.split("?")
    values = list(params)
    if len(parts) - 1 != len(values):
        raise ValueError(
            f"placeholder/parameter mismatch: {len(parts) - 1} placeholders, "
            f"{len(values)} parameters"
        )
    out = [parts[0]]
    for value, rest in zip(values, parts[1:]):
        out.append(_sql_literal(value))
        out.append(rest)
    return "".join(out)


class TaskQueue:
    """DuckDB-backed task queue with persisted attempt/backoff/lease state.

    Concurrency model:
    - multiple workers may poll concurrently
    - claiming uses an atomic UPDATE guarded by a transaction
    - owner heartbeats renew ``lease_until`` only for the assigned worker
    - expired-lease recovery requeues or fails without double-claiming

    Transport:
    - a filesystem path opens the DuckDB file directly (multi-process use is
      limited by the DuckDB file lock);
    - a ``quack:host:port`` URI connects as a Quack client to a queue owner
      serving the database (``CALL quack_serve(...)``), which is the intended
      mode for shared multi-process / multi-peer queues. The token comes from
      the ``TASK_QUEUE_QUACK_TOKEN`` environment variable (or the file named
      by ``TASK_QUEUE_QUACK_TOKEN_FILE``).

    Quack execution model: reads (SELECT) run over the attached catalog, but
    every mutation (INSERT/UPDATE/DELETE) is executed *owner-side* via
    ``quack_query_by_name`` — remote attachments reject UPDATE/DELETE with
    "Can only update base table", while owner-side execution runs against the
    real base tables. Each mutation is a single statement, hence atomic;
    multi-statement transactions are therefore unnecessary (and unavailable)
    in Quack mode.
    """

    def __init__(self, path: Optional[str] = None, *, default_lease_seconds: float = 300.0):
        self.path = path or default_queue_path()
        self._quack = is_quack_path(self.path)
        if not self._quack:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.default_lease_seconds = max(1.0, float(default_lease_seconds))

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

        if self._quack:
            # Quack client mode: one persistent client connection per TaskQueue
            # instance. All SQL (schema init, claim, complete, polls) is
            # executed by the queue owner; no DuckDB file lock is ever taken
            # by clients.
            with self._conn_lock:
                if self._conn is None:
                    self._conn = self._connect_quack(duckdb)
                return self._conn

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

    def _connect_quack(self, duckdb):
        """Open a DuckDB client connection to the Quack queue owner."""
        import re

        uri = str(self.path).strip()
        # Strict allowlist: loopback or explicit hostnames only, no quotes or
        # statement separators — the URI is interpolated into ATTACH below.
        # Normalize quack://host:port to the quack:host:port form the
        # extension's ATTACH handler accepts.
        if not re.fullmatch(
            r"quack:(?://)?[A-Za-z0-9._-]+(?::\d{1,5})?", uri
        ):
            raise ValueError(f"invalid quack endpoint URI: {uri!r}")
        uri = re.sub(r"^quack://", "quack:", uri)
        token = quack_token()
        if token and not re.fullmatch(r"[A-Za-z0-9_-]+", token):
            raise ValueError("TASK_QUEUE_QUACK_TOKEN contains unsafe characters")
        conn = duckdb.connect()
        try:
            conn.execute("LOAD quack")
            if token:
                conn.execute(f"CREATE SECRET (TYPE quack, TOKEN '{token}')")
            # Attach the owner's session as catalog ``taskqueue`` and make it
            # the default so all existing unqualified queue SQL works as-is.
            conn.execute(f"ATTACH '{uri}' AS taskqueue")
            conn.execute("USE taskqueue")
            return conn
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

    def _quack_write(self, sql: str, params: Iterable[Any] = ()) -> list[tuple]:
        """Execute one mutation statement owner-side via Quack.

        The statement (with ``?`` placeholders interpolated as SQL literals)
        runs in the queue owner's session through ``quack_query_by_name``,
        where the queue tables are base tables and UPDATE/DELETE/INSERT work.
        Returns the rows produced by the statement (e.g. ``RETURNING``).
        """
        if not self._quack:
            raise RuntimeError("_quack_write is only valid in Quack client mode")
        stmt = _interpolate_params(sql, params)
        with self._conn_lock:
            conn = self._get_conn()
            return conn.execute(
                "SELECT * FROM quack_query_by_name('taskqueue', ?)", (stmt,)
            ).fetchall()

    def _quack_recover_expired(self, now: float, limit: int = 1000) -> int:
        """Owner-side expired-lease recovery for Quack client mode.

        Mirrors :meth:`_recover_expired_in_transaction` but as separate
        single statements: one read plus up to two guarded UPDATEs. The
        UPDATEs re-check ``status='running'`` and the lease deadline, so two
        racing recovery loops cannot double-assign the same claim.
        """
        bounded_limit = max(1, min(int(limit or 1000), 10000))
        with self._conn_lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT task_id, attempt, max_attempts
                FROM tasks
                WHERE status='running' AND lease_until IS NOT NULL AND lease_until <= ?
                ORDER BY lease_until ASC, created_at ASC
                LIMIT ?
                """,
                (float(now), bounded_limit),
            ).fetchall()
        retry_ids = [
            str(row[0])
            for row in rows
            if int(row[2] if row[2] is not None else 3) == 0
            or int(row[1] or 0) < int(row[2] if row[2] is not None else 3)
        ]
        exhausted_ids = [
            str(row[0])
            for row in rows
            if int(row[2] if row[2] is not None else 3) > 0
            and int(row[1] or 0) >= int(row[2] if row[2] is not None else 3)
        ]
        deadline = float(now)
        if retry_ids:
            id_list = ", ".join(_sql_literal(i) for i in retry_ids)
            self._quack_write(
                "UPDATE tasks SET status='queued', assigned_worker=NULL, "
                "lease_until=NULL, heartbeat_at=NULL, updated_at=? "
                f"WHERE task_id IN ({id_list}) AND status='running' "
                "AND lease_until IS NOT NULL AND lease_until <= ?",
                (deadline, deadline),
            )
        if exhausted_ids:
            id_list = ", ".join(_sql_literal(i) for i in exhausted_ids)
            self._quack_write(
                "UPDATE tasks SET status='failed', assigned_worker=NULL, "
                "lease_until=NULL, heartbeat_at=NULL, updated_at=?, "
                "error=CASE "
                "WHEN error IS NULL OR error='' "
                "THEN 'claim lease expired after maximum attempts' "
                "ELSE error || '; claim lease expired after maximum attempts' END "
                f"WHERE task_id IN ({id_list}) AND status='running' "
                "AND lease_until IS NOT NULL AND lease_until <= ?",
                (deadline, deadline),
            )
        return len(retry_ids) + len(exhausted_ids)

    def _claim_next_quack(
        self,
        *,
        worker_id: str,
        task_types: list,
        session: str,
        max_priority: Optional[int],
        now: float,
        lease_duration: float,
    ) -> Optional[QueuedTask]:
        """Quack client mode single claim: one atomic UPDATE...RETURNING."""
        wid = str(worker_id)
        self._quack_recover_expired(now)
        inner_where, inner_params, priority_expr = _eligibility_where(
            task_types=task_types,
            worker_id=wid,
            session=session,
            max_priority=max_priority,
            now=now,
        )
        # The outer guard re-checks the single chosen row; no type filter is
        # needed there since the candidate came from the filtered subselect.
        outer_where, outer_params, _ = _eligibility_where(
            task_types=[],
            worker_id=wid,
            session=session,
            max_priority=max_priority,
            now=now,
        )
        sql = (
            "UPDATE tasks SET status='running', assigned_worker=?, updated_at=?, "
            "attempt=coalesce(attempt, 0)+1, heartbeat_at=?, lease_until=? "
            "WHERE task_id = ("
            f"SELECT task_id FROM tasks WHERE {inner_where} "
            f"ORDER BY {priority_expr} DESC, created_at ASC, task_id ASC LIMIT 1"
            ") "
            f"AND {outer_where} "
            f"RETURNING {_TASK_SELECT_COLUMNS}"
        )
        params = [wid, now, now, now + lease_duration] + inner_params + outer_params
        for _ in range(6):
            try:
                rows = self._quack_write(sql, params)
            except Exception as exc:
                if _transient_conflict(exc):
                    time.sleep(0.005 + random.random() * 0.02)
                    continue
                raise
            if rows:
                return _queued_task_from_row(rows[0])
            return None
        return None

    def _claim_next_many_quack(
        self,
        *,
        worker_id: str,
        task_types: list,
        limit: int,
        same_task_type: bool,
        session: str,
        max_priority: Optional[int],
        now: float,
        lease_duration: float,
    ) -> list[QueuedTask]:
        """Quack client mode batch claim: select ids, then guarded UPDATE."""
        wid = str(worker_id)
        self._quack_recover_expired(now)
        effective_types = list(task_types)
        with self._conn_lock:
            conn = self._get_conn()
            if same_task_type:
                where, params, priority_expr = _eligibility_where(
                    task_types=effective_types,
                    worker_id=wid,
                    session=session,
                    max_priority=max_priority,
                    now=now,
                )
                row0 = conn.execute(
                    f"SELECT task_type FROM tasks WHERE {where} "
                    f"ORDER BY {priority_expr} DESC, created_at ASC, task_id ASC LIMIT 1",
                    tuple(params),
                ).fetchone()
                if row0 is None:
                    return []
                effective_types = [str(row0[0])]
            where, params, priority_expr = _eligibility_where(
                task_types=effective_types,
                worker_id=wid,
                session=session,
                max_priority=max_priority,
                now=now,
            )
            rows = conn.execute(
                f"SELECT task_id FROM tasks WHERE {where} "
                f"ORDER BY {priority_expr} DESC, created_at ASC, task_id ASC "
                f"LIMIT {int(limit)}",
                tuple(params),
            ).fetchall()
            ids = [str(r[0]) for r in (rows or []) if r and r[0]]
            if not ids:
                return []
        guard_where, guard_params, _ = _eligibility_where(
            task_types=effective_types,
            worker_id=wid,
            session=session,
            max_priority=max_priority,
            now=now,
        )
        id_list = ", ".join(_sql_literal(i) for i in ids)
        upd = (
            "UPDATE tasks SET status='running', assigned_worker=?, updated_at=?, "
            "attempt=coalesce(attempt, 0)+1, heartbeat_at=?, lease_until=? "
            f"WHERE task_id IN ({id_list}) AND {guard_where} "
            f"RETURNING {_TASK_SELECT_COLUMNS}"
        )
        got = self._quack_write(upd, [wid, now, now, now + lease_duration] + guard_params)
        out = [_queued_task_from_row(r) for r in got]
        out.sort(key=lambda t: (-t.priority, t.created_at, t.task_id))
        return out

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
        # Quack client mode: the queue owner already owns the schema and runs
        # the DDL itself. DDL (ALTER/CREATE INDEX) is not implemented over a
        # quack attachment, so skip schema init entirely for clients.
        if self._quack:
            return
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
                            error VARCHAR,
                            priority INTEGER DEFAULT 5,
                            attempt INTEGER DEFAULT 0,
                            max_attempts INTEGER DEFAULT 3,
                            next_attempt_at DOUBLE DEFAULT 0,
                            lease_until DOUBLE,
                            heartbeat_at DOUBLE,
                            idempotency_key VARCHAR
                        )
                        """
                    )
                    existing_columns = {
                        str(row[1])
                        for row in conn.execute("PRAGMA table_info('tasks')").fetchall()
                        if len(row) > 1
                    }
                    had_priority_column = "priority" in existing_columns
                    # ADD COLUMN IF NOT EXISTS keeps databases created by earlier
                    # releases readable without rebuilding or copying task rows.
                    migrations = (
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS priority INTEGER DEFAULT 5",
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS attempt INTEGER DEFAULT 0",
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS max_attempts INTEGER DEFAULT 3",
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS next_attempt_at DOUBLE DEFAULT 0",
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS lease_until DOUBLE",
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS heartbeat_at DOUBLE",
                        "ALTER TABLE tasks ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR",
                    )
                    for statement in migrations:
                        conn.execute(statement)
                    if had_priority_column:
                        conn.execute("UPDATE tasks SET priority=5 WHERE priority IS NULL")
                    else:
                        # Older queues carried priority only in payload JSON.
                        # Preserve that ordering/cap behavior during migration.
                        conn.execute(
                            """
                            UPDATE tasks
                            SET priority=greatest(
                                1,
                                least(
                                    10,
                                    coalesce(
                                        TRY_CAST(
                                            json_extract_string(payload_json, '$.priority')
                                            AS INTEGER
                                        ),
                                        5
                                    )
                                )
                            )
                            """
                        )
                    conn.execute("UPDATE tasks SET attempt=0 WHERE attempt IS NULL")
                    conn.execute("UPDATE tasks SET max_attempts=3 WHERE max_attempts IS NULL")
                    conn.execute("UPDATE tasks SET next_attempt_at=0 WHERE next_attempt_at IS NULL")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_tasks_status_created ON tasks(status, created_at)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_tasks_claim "
                        "ON tasks(status, priority, next_attempt_at, created_at)"
                    )
                    conn.execute(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_idempotency "
                        "ON tasks(idempotency_key)"
                    )
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

    def _submit_with_outcome(
        self,
        *,
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        priority: Optional[int] = None,
        max_attempts: Optional[int] = None,
        next_attempt_at: Optional[float] = None,
    ) -> tuple[str, bool]:
        """Submit work and report whether an identical task already existed.

        ``idempotency_key`` provides atomic submit-once behavior across process
        restarts. Reusing a key with different work is rejected instead of
        silently aliasing two provider operations. The boolean result is true
        when the returned task was replayed rather than inserted by this call.
        """

        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        tid = task_id or uuid.uuid4().hex
        now = time.time()
        payload_json = json.dumps(payload, sort_keys=True)
        identity = str(idempotency_key or payload.get("idempotency_key") or "").strip() or None
        task_priority = _priority(priority if priority is not None else payload.get("priority", 5))
        attempts_limit = _attempt_limit(
            max_attempts if max_attempts is not None else payload.get("max_attempts", 3)
        )
        try:
            eligible_at = float(
                next_attempt_at
                if next_attempt_at is not None
                else payload.get("next_attempt_at", 0.0)
            )
        except (TypeError, ValueError):
            eligible_at = 0.0
        eligible_at = max(0.0, eligible_at)

        if self._quack:
            return self._submit_quack(
                task_type=str(task_type),
                model_name=str(model_name),
                payload_json=payload_json,
                tid=tid,
                identity=identity,
                task_priority=task_priority,
                attempts_limit=attempts_limit,
                eligible_at=eligible_at,
                now=now,
            )

        with self._conn_lock:
            conn = self._get_conn()
            conn.execute("BEGIN TRANSACTION")
            try:
                existing = None
                if identity is not None:
                    existing = conn.execute(
                        "SELECT task_id, task_type, model_name, payload_json "
                        "FROM tasks WHERE idempotency_key=?",
                        (identity,),
                    ).fetchone()
                if existing is None and task_id:
                    existing = conn.execute(
                        "SELECT task_id, task_type, model_name, payload_json "
                        "FROM tasks WHERE task_id=?",
                        (tid,),
                    ).fetchone()
                if existing is not None:
                    same_work = (
                        str(existing[1]) == str(task_type)
                        and str(existing[2]) == str(model_name)
                        and str(existing[3]) == payload_json
                    )
                    if not same_work:
                        raise ValueError(
                            "idempotency key or task_id already identifies different work"
                        )
                    conn.execute("COMMIT")
                    return str(existing[0]), True

                conn.execute(
                    """
                    INSERT INTO tasks(
                        task_id,
                        task_type,
                        model_name,
                        payload_json,
                        status,
                        assigned_worker,
                        created_at,
                        updated_at,
                        priority,
                        attempt,
                        max_attempts,
                        next_attempt_at,
                        lease_until,
                        heartbeat_at,
                        idempotency_key
                    )
                    VALUES(?, ?, ?, ?, 'queued', NULL, ?, ?, ?, 0, ?, ?, NULL, NULL, ?)
                    """,
                    (
                        tid,
                        str(task_type),
                        str(model_name),
                        payload_json,
                        now,
                        now,
                        task_priority,
                        attempts_limit,
                        eligible_at,
                        identity,
                    ),
                )
                conn.execute("COMMIT")
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                low = str(exc).lower()
                duplicate_or_race = (
                    "duplicate key" in low
                    or "unique constraint" in low
                    or "conflict on tuple" in low
                    or "transactioncontext" in low
                    or ("transaction" in low and "conflict" in low)
                )
                if duplicate_or_race and (identity is not None or task_id):
                    # A second process may have committed the same identity
                    # after our initial read. Observe that winner and apply the
                    # same exact-work check as the uncontended path.
                    for retry_index in range(8):
                        try:
                            if identity is not None:
                                existing = conn.execute(
                                    "SELECT task_id, task_type, model_name, payload_json "
                                    "FROM tasks WHERE idempotency_key=?",
                                    (identity,),
                                ).fetchone()
                            else:
                                existing = conn.execute(
                                    "SELECT task_id, task_type, model_name, payload_json "
                                    "FROM tasks WHERE task_id=?",
                                    (tid,),
                                ).fetchone()
                            if existing is not None:
                                if (
                                    str(existing[1]) == str(task_type)
                                    and str(existing[2]) == str(model_name)
                                    and str(existing[3]) == payload_json
                                ):
                                    return str(existing[0]), True
                                raise ValueError(
                                    "idempotency key or task_id already identifies different work"
                                ) from exc
                        except ValueError:
                            raise
                        except Exception:
                            pass
                        time.sleep(0.002 * (retry_index + 1))
                raise
        return tid, False

    def _submit_quack(
        self,
        *,
        task_type: str,
        model_name: str,
        payload_json: str,
        tid: str,
        identity: Optional[str],
        task_priority: int,
        attempts_limit: int,
        eligible_at: float,
        now: float,
    ) -> tuple[str, bool]:
        """Quack client mode submit: reads over ATTACH, INSERT owner-side.

        The idempotency read-then-insert race is handled the same way as the
        file mode: a duplicate-key error from the owner triggers a re-read
        and the exact-work check.
        """
        with self._conn_lock:
            conn = self._get_conn()
            existing = None
            if identity is not None:
                existing = conn.execute(
                    "SELECT task_id, task_type, model_name, payload_json "
                    "FROM tasks WHERE idempotency_key=?",
                    (identity,),
                ).fetchone()
            if existing is None and tid:
                existing = conn.execute(
                    "SELECT task_id, task_type, model_name, payload_json "
                    "FROM tasks WHERE task_id=?",
                    (tid,),
                ).fetchone()
            if existing is not None:
                same_work = (
                    str(existing[1]) == str(task_type)
                    and str(existing[2]) == str(model_name)
                    and str(existing[3]) == payload_json
                )
                if not same_work:
                    raise ValueError(
                        "idempotency key or task_id already identifies different work"
                    )
                return str(existing[0]), True
            try:
                self._quack_write(
                    """
                    INSERT INTO tasks(
                        task_id,
                        task_type,
                        model_name,
                        payload_json,
                        status,
                        assigned_worker,
                        created_at,
                        updated_at,
                        priority,
                        attempt,
                        max_attempts,
                        next_attempt_at,
                        lease_until,
                        heartbeat_at,
                        idempotency_key
                    )
                    VALUES(?, ?, ?, ?, 'queued', NULL, ?, ?, ?, 0, ?, ?, NULL, NULL, ?)
                    """,
                    (
                        tid,
                        str(task_type),
                        str(model_name),
                        payload_json,
                        now,
                        now,
                        task_priority,
                        attempts_limit,
                        eligible_at,
                        identity,
                    ),
                )
            except Exception as exc:
                low = str(exc).lower()
                duplicate_or_race = (
                    "duplicate key" in low
                    or "unique constraint" in low
                    or "conflict on tuple" in low
                )
                if duplicate_or_race and (identity is not None or tid):
                    for retry_index in range(8):
                        try:
                            if identity is not None:
                                existing = conn.execute(
                                    "SELECT task_id, task_type, model_name, payload_json "
                                    "FROM tasks WHERE idempotency_key=?",
                                    (identity,),
                                ).fetchone()
                            else:
                                existing = conn.execute(
                                    "SELECT task_id, task_type, model_name, payload_json "
                                    "FROM tasks WHERE task_id=?",
                                    (tid,),
                                ).fetchone()
                            if existing is not None:
                                if (
                                    str(existing[1]) == str(task_type)
                                    and str(existing[2]) == str(model_name)
                                    and str(existing[3]) == payload_json
                                ):
                                    return str(existing[0]), True
                                raise ValueError(
                                    "idempotency key or task_id already identifies different work"
                                ) from exc
                        except ValueError:
                            raise
                        except Exception:
                            pass
                        time.sleep(0.002 * (retry_index + 1))
                raise
        return tid, False

    def submit(
        self,
        *,
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        priority: Optional[int] = None,
        max_attempts: Optional[int] = None,
        next_attempt_at: Optional[float] = None,
    ) -> str:
        """Submit work, returning the existing task for an identical identity.

        Existing callers retain the historical string return value. Consumers
        that need to distinguish a new insert from an exact replay can use
        :meth:`submit_with_outcome`.
        """

        submitted_id, _ = self._submit_with_outcome(
            task_type=task_type,
            model_name=model_name,
            payload=payload,
            task_id=task_id,
            idempotency_key=idempotency_key,
            priority=priority,
            max_attempts=max_attempts,
            next_attempt_at=next_attempt_at,
        )
        return submitted_id

    def submit_with_outcome(
        self,
        *,
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        task_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        priority: Optional[int] = None,
        max_attempts: Optional[int] = None,
        next_attempt_at: Optional[float] = None,
    ) -> tuple[str, bool]:
        """Atomically return ``(task_id, replayed)`` for a submission."""

        return self._submit_with_outcome(
            task_type=task_type,
            model_name=model_name,
            payload=payload,
            task_id=task_id,
            idempotency_key=idempotency_key,
            priority=priority,
            max_attempts=max_attempts,
            next_attempt_at=next_attempt_at,
        )

    def submit_once(
        self,
        *,
        idempotency_key: str,
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        priority: Optional[int] = None,
        max_attempts: Optional[int] = None,
        next_attempt_at: Optional[float] = None,
    ) -> str:
        """Explicit submit-once convenience API."""

        identity = str(idempotency_key or "").strip()
        if not identity:
            raise ValueError("idempotency_key is required")
        return self.submit(
            task_type=task_type,
            model_name=model_name,
            payload=payload,
            idempotency_key=identity,
            priority=priority,
            max_attempts=max_attempts,
            next_attempt_at=next_attempt_at,
        )

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not task_id:
            return None

        # Use a fresh connection so readers reliably observe updates from other
        # TaskQueue instances (e.g. worker heartbeats) across threads/processes.
        # In Quack client mode the shared attached connection is reused instead
        # (every query round-trips to the owner) and must NOT be closed here.
        with self._conn_lock:
            if self._quack:
                row = self._get_conn().execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            else:
                conn = self._connect()
                try:
                    row = conn.execute(
                        f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
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
            priority,
            attempt,
            max_attempts,
            next_attempt_at,
            lease_until,
            heartbeat_at,
            idempotency_key,
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
            "priority": int(priority),
            "attempt": int(attempt),
            "max_attempts": int(max_attempts),
            "next_attempt_at": float(next_attempt_at),
            "lease_until": float(lease_until) if lease_until is not None else None,
            "heartbeat_at": float(heartbeat_at) if heartbeat_at is not None else None,
            "idempotency_key": str(idempotency_key) if idempotency_key else None,
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
                    (
                        f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks WHERE status=? "
                        f"AND task_type IN ({placeholders}) "
                        "ORDER BY created_at ASC LIMIT ?"
                    ),
                    (status_norm, *types, lim),
                ).fetchall()
            elif status_norm:
                rows = conn.execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks "
                    "WHERE status=? ORDER BY created_at ASC LIMIT ?",
                    (status_norm, lim),
                ).fetchall()
            elif types:
                placeholders = ",".join(["?"] * len(types))
                rows = conn.execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks "
                    f"WHERE task_type IN ({placeholders}) ORDER BY created_at ASC LIMIT ?",
                    (*types, lim),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks ORDER BY created_at ASC LIMIT ?",
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
                priority,
                attempt,
                max_attempts,
                next_attempt_at,
                lease_until,
                heartbeat_at,
                idempotency_key,
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
                    "priority": int(priority),
                    "attempt": int(attempt),
                    "max_attempts": int(max_attempts),
                    "next_attempt_at": float(next_attempt_at),
                    "lease_until": float(lease_until) if lease_until is not None else None,
                    "heartbeat_at": float(heartbeat_at) if heartbeat_at is not None else None,
                    "idempotency_key": str(idempotency_key) if idempotency_key else None,
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
        # TaskQueue instances across threads/processes. In Quack client mode
        # the shared attached connection is reused and must NOT be closed.
        with self._conn_lock:
            if self._quack:
                rows = self._get_conn().execute(sql, params).fetchall()
            else:
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

    @staticmethod
    def _recover_expired_in_transaction(conn: Any, now: float, limit: int) -> int:
        rows = conn.execute(
            """
            SELECT task_id, attempt, max_attempts
            FROM tasks
            WHERE status='running' AND lease_until IS NOT NULL AND lease_until <= ?
            ORDER BY lease_until ASC, created_at ASC
            LIMIT ?
            """,
            (float(now), int(limit)),
        ).fetchall()
        retry_ids = [
            str(row[0])
            for row in rows
            if int(row[2] if row[2] is not None else 3) == 0
            or int(row[1] or 0) < int(row[2] if row[2] is not None else 3)
        ]
        exhausted_ids = [
            str(row[0])
            for row in rows
            if int(row[2] if row[2] is not None else 3) > 0
            and int(row[1] or 0) >= int(row[2] if row[2] is not None else 3)
        ]

        if retry_ids:
            placeholders = ",".join("?" for _ in retry_ids)
            conn.execute(
                (
                    "UPDATE tasks SET status='queued', assigned_worker=NULL, "
                    "lease_until=NULL, heartbeat_at=NULL, updated_at=? "
                    f"WHERE task_id IN ({placeholders}) AND status='running' "
                    "AND lease_until IS NOT NULL AND lease_until <= ?"
                ),
                tuple([float(now), *retry_ids, float(now)]),
            )
        if exhausted_ids:
            placeholders = ",".join("?" for _ in exhausted_ids)
            conn.execute(
                (
                    "UPDATE tasks SET status='failed', assigned_worker=NULL, "
                    "lease_until=NULL, heartbeat_at=NULL, updated_at=?, "
                    "error=CASE "
                    "WHEN error IS NULL OR error='' "
                    "THEN 'claim lease expired after maximum attempts' "
                    "ELSE error || '; claim lease expired after maximum attempts' END "
                    f"WHERE task_id IN ({placeholders}) AND status='running' "
                    "AND lease_until IS NOT NULL AND lease_until <= ?"
                ),
                tuple([float(now), *exhausted_ids, float(now)]),
            )
        return len(retry_ids) + len(exhausted_ids)

    def recover_expired_leases(
        self,
        *,
        now: Optional[float] = None,
        limit: int = 1000,
    ) -> int:
        """Requeue expired claims, terminally failing exhausted tasks.

        Recovery is one DuckDB transaction, so two recovery loops cannot both
        assign the same expired claim. Attempts are incremented only by claims,
        not by the recovery scan.
        """

        recovered_at = time.time() if now is None else float(now)
        bounded_limit = max(1, min(int(limit or 1000), 10000))
        if self._quack:
            return self._quack_recover_expired(recovered_at, bounded_limit)
        with self._conn_lock:
            conn = self._get_conn()
            conn.execute("BEGIN TRANSACTION")
            try:
                count = self._recover_expired_in_transaction(conn, recovered_at, bounded_limit)
                conn.execute("COMMIT")
                return count
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise

    def heartbeat(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_seconds: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Renew owner heartbeats for a running claim only when the worker owns it."""

        tid = str(task_id or "").strip()
        wid = str(worker_id or "").strip()
        if not tid or not wid:
            return False
        heartbeat_at = time.time() if now is None else float(now)
        duration = (
            self.default_lease_seconds if lease_seconds is None else max(1.0, float(lease_seconds))
        )
        if self._quack:
            rows = self._quack_write(
                """
                UPDATE tasks
                SET heartbeat_at=?, lease_until=?, updated_at=?
                WHERE task_id=? AND status='running' AND assigned_worker=?
                RETURNING task_id
                """,
                (
                    heartbeat_at,
                    heartbeat_at + duration,
                    heartbeat_at,
                    tid,
                    wid,
                ),
            )
            return len(rows) > 0
        with self._conn_lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                UPDATE tasks
                SET heartbeat_at=?, lease_until=?, updated_at=?
                WHERE task_id=? AND status='running' AND assigned_worker=?
                RETURNING task_id
                """,
                (
                    heartbeat_at,
                    heartbeat_at + duration,
                    heartbeat_at,
                    tid,
                    wid,
                ),
            ).fetchone()
        return row is not None

    def retry(
        self,
        *,
        task_id: str,
        worker_id: str,
        delay_seconds: float = 0.0,
        error: Optional[str] = None,
        now: Optional[float] = None,
    ) -> bool:
        """Release owned retryable work into persisted attempt/backoff/lease state."""

        tid = str(task_id or "").strip()
        wid = str(worker_id or "").strip()
        if not tid or not wid:
            return False
        retry_at = time.time() if now is None else float(now)
        retry_at += max(0.0, float(delay_seconds))
        if self._quack:
            rows = self._quack_write(
                """
                UPDATE tasks
                SET status=CASE
                        WHEN max_attempts = 0 OR attempt < max_attempts
                        THEN 'queued'
                        ELSE 'failed'
                    END,
                    assigned_worker=NULL,
                    next_attempt_at=?,
                    lease_until=NULL,
                    heartbeat_at=NULL,
                    updated_at=?,
                    error=?
                WHERE task_id=? AND status='running' AND assigned_worker=?
                RETURNING task_id
                """,
                (
                    retry_at,
                    time.time() if now is None else float(now),
                    str(error) if error else None,
                    tid,
                    wid,
                ),
            )
            return len(rows) > 0
        with self._conn_lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                UPDATE tasks
                SET status=CASE
                        WHEN max_attempts = 0 OR attempt < max_attempts
                        THEN 'queued'
                        ELSE 'failed'
                    END,
                    assigned_worker=NULL,
                    next_attempt_at=?,
                    lease_until=NULL,
                    heartbeat_at=NULL,
                    updated_at=?,
                    error=?
                WHERE task_id=? AND status='running' AND assigned_worker=?
                RETURNING task_id
                """,
                (
                    retry_at,
                    time.time() if now is None else float(now),
                    str(error) if error else None,
                    tid,
                    wid,
                ),
            ).fetchone()
        return row is not None

    def claim_next(
        self,
        *,
        worker_id: str,
        supported_task_types: Optional[Iterable[str]] = None,
        session_id: str | None = None,
        max_priority: Optional[int] = None,
        lease_seconds: Optional[float] = None,
    ) -> Optional[QueuedTask]:
        """Claim the next queued task for ``worker_id``.

        Args:
            worker_id: Identifier of the claiming worker.
            supported_task_types: If given, only tasks of these types are eligible.
            session_id: If given, restrict to tasks that require this session (or
                have no session requirement).
            max_priority: Upper bound on the ``priority`` field stored inside the
                task payload JSON (1-10, where 10 is highest priority).  When set,
                only tasks whose payload priority is **at most** this value are
                eligible.  Use this to implement trust-tiered queue access: lower
                the cap for baseline (untrusted) peers so that high-priority tasks
                are reserved for trusted peers.  ``None`` (default) means no cap.
            lease_seconds: Claim lifetime. The owner must call :meth:`heartbeat`
                before this deadline for long-running work.
        """
        if not worker_id:
            raise ValueError("worker_id is required")

        task_types = [t for t in (supported_task_types or []) if isinstance(t, str) and t.strip()]
        session = str(session_id or "").strip()
        now = time.time()
        lease_duration = (
            self.default_lease_seconds if lease_seconds is None else max(1.0, float(lease_seconds))
        )

        if self._quack:
            return self._claim_next_quack(
                worker_id=worker_id,
                task_types=task_types,
                session=session,
                max_priority=max_priority,
                now=now,
                lease_duration=lease_duration,
            )

        required_expr = (
            "coalesce("
            "nullif(json_extract_string(payload_json, '$.session_id'), ''), "
            "nullif(json_extract_string(payload_json, '$.session'), ''), "
            "nullif(json_extract_string(payload_json, '$.p2p_session'), '')"
            ")"
        )

        sticky_expr = "nullif(json_extract_string(payload_json, '$.sticky_worker_id'), '')"

        priority_expr = "coalesce(priority, 5)"

        def _is_transient_conflict(exc: Exception) -> bool:
            msg = str(exc or "")
            low = msg.lower()
            return (
                "conflict on tuple" in low
                or "transactioncontext" in low
                or ("transaction" in low and "conflict" in low)
            )

        row2 = None
        for attempt in range(8):
            conn = self._connect()
            try:
                conn.execute("BEGIN TRANSACTION")
                self._recover_expired_in_transaction(conn, now, 1000)

                where: list[str] = [
                    "status='queued'",
                    "coalesce(next_attempt_at, 0) <= ?",
                    _ATTEMPTS_AVAILABLE_SQL,
                ]
                params: list[object] = [now]

                if task_types:
                    placeholders = ",".join(["?"] * len(task_types))
                    where.append(f"task_type IN ({placeholders})")
                    params.extend([str(t) for t in task_types])

                # Optional per-task sticky routing (session resume affinity).
                where.append(f"({sticky_expr} IS NULL OR {sticky_expr} = ?)")
                params.append(str(worker_id))

                if session:
                    where.append(f"({required_expr} IS NULL OR {required_expr} = ?)")
                    params.append(str(session))

                # Trust-tiered priority cap: baseline peers only see low-priority tasks.
                if max_priority is not None:
                    cap = max(1, min(10, int(max_priority)))
                    where.append(f"({priority_expr} <= ?)")
                    params.append(cap)

                where_sql = " AND ".join(where)
                row = conn.execute(
                    f"SELECT task_id FROM tasks WHERE {where_sql} "
                    f"ORDER BY {priority_expr} DESC, created_at ASC, task_id ASC LIMIT 1",
                    tuple(params),
                ).fetchone()

                if row is None:
                    conn.execute("COMMIT")
                    return None

                task_id = str(row[0])

                # Re-check every eligibility guard at update time to avoid races.
                update_sql = (
                    f"UPDATE tasks SET status='running', assigned_worker=?, updated_at=?, "
                    f"attempt=coalesce(attempt, 0)+1, heartbeat_at=?, lease_until=? "
                    f"WHERE task_id=? AND status='queued' "
                    f"AND coalesce(next_attempt_at, 0) <= ? "
                    f"AND {_ATTEMPTS_AVAILABLE_SQL} "
                    f"AND ({sticky_expr} IS NULL OR {sticky_expr} = ?)"
                )
                update_params: list[object] = [
                    str(worker_id),
                    now,
                    now,
                    now + lease_duration,
                    task_id,
                    now,
                    str(worker_id),
                ]
                if session:
                    update_sql += f" AND ({required_expr} IS NULL OR {required_expr} = ?)"
                    update_params.append(str(session))
                if max_priority is not None:
                    update_sql += f" AND ({priority_expr} <= ?)"
                    update_params.append(cap)
                conn.execute(update_sql, tuple(update_params))

                row2 = conn.execute(
                    f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks "
                    "WHERE task_id=? AND status='running' AND assigned_worker=?",
                    (task_id, str(worker_id)),
                ).fetchone()
                conn.execute("COMMIT")

                if row2 is None:
                    return None
                break
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                if attempt < 7 and _is_transient_conflict(exc):
                    time.sleep(0.002 + random.random() * 0.02)
                    continue
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        if row2 is None:
            return None

        return _queued_task_from_row(row2)

    def claim_next_many(
        self,
        *,
        worker_id: str,
        supported_task_types: Optional[Iterable[str]] = None,
        max_tasks: int = 1,
        same_task_type: bool = True,
        session_id: str | None = None,
        max_priority: Optional[int] = None,
        lease_seconds: Optional[float] = None,
    ) -> list[QueuedTask]:
        """Atomically claim up to `max_tasks` queued tasks.

        When `same_task_type=True`, the first claimed task determines the
        `task_type`, and the method claims additional queued tasks of that same
        type (FIFO by created_at). This is useful for batching homogeneous work
        (e.g., text-generation).

        Args:
            max_priority: Upper bound on the ``priority`` field inside the task
                payload JSON (1-10).  When set, only tasks with payload priority
                at most this value are eligible.  Use together with
                ``PeerTrustLevel`` to gate high-priority tasks for trusted peers.
            lease_seconds: Shared claim lifetime for every member.
        """

        if not worker_id:
            raise ValueError("worker_id is required")

        try:
            limit = int(max_tasks)
        except Exception:
            limit = 1
        limit = max(1, min(limit, 128))

        task_types = [t for t in (supported_task_types or []) if isinstance(t, str) and t.strip()]
        session = str(session_id or "").strip()
        now = time.time()
        lease_duration = (
            self.default_lease_seconds if lease_seconds is None else max(1.0, float(lease_seconds))
        )

        if self._quack:
            return self._claim_next_many_quack(
                worker_id=worker_id,
                task_types=task_types,
                limit=limit,
                same_task_type=same_task_type,
                session=session,
                max_priority=max_priority,
                now=now,
                lease_duration=lease_duration,
            )

        required_expr = (
            "coalesce("
            "nullif(json_extract_string(payload_json, '$.session_id'), ''), "
            "nullif(json_extract_string(payload_json, '$.session'), ''), "
            "nullif(json_extract_string(payload_json, '$.p2p_session'), '')"
            ")"
        )

        sticky_expr = "nullif(json_extract_string(payload_json, '$.sticky_worker_id'), '')"

        priority_expr = "coalesce(priority, 5)"

        def _is_transient_conflict(exc: Exception) -> bool:
            msg = str(exc or "")
            low = msg.lower()
            return (
                "conflict on tuple" in low
                or "transactioncontext" in low
                or ("transaction" in low and "conflict" in low)
            )

        rows2: list[object] = []
        for attempt in range(8):
            conn = self._connect()
            try:
                conn.execute("BEGIN TRANSACTION")
                self._recover_expired_in_transaction(conn, now, 1000)

                # Pick the oldest queued task (optionally filtered by supported types)
                # to establish the batch's task_type.
                picked_type: str | None = None
                if same_task_type:
                    where0: list[str] = [
                        "status='queued'",
                        "coalesce(next_attempt_at, 0) <= ?",
                        _ATTEMPTS_AVAILABLE_SQL,
                    ]
                    params0: list[object] = [now]
                    if task_types:
                        placeholders = ",".join(["?"] * len(task_types))
                        where0.append(f"task_type IN ({placeholders})")
                        params0.extend([str(t) for t in task_types])
                    where0.append(f"({sticky_expr} IS NULL OR {sticky_expr} = ?)")
                    params0.append(str(worker_id))
                    if session:
                        where0.append(f"({required_expr} IS NULL OR {required_expr} = ?)")
                        params0.append(str(session))
                    # Trust-tiered priority cap for same_task_type probe.
                    if max_priority is not None:
                        cap = max(1, min(10, int(max_priority)))
                        where0.append(f"({priority_expr} <= ?)")
                        params0.append(cap)
                    where0_sql = " AND ".join(where0)
                    row0 = conn.execute(
                        f"SELECT task_type FROM tasks WHERE {where0_sql} "
                        f"ORDER BY {priority_expr} DESC, created_at ASC, task_id ASC LIMIT 1",
                        tuple(params0),
                    ).fetchone()
                    if row0 is None:
                        conn.execute("COMMIT")
                        return []
                    picked_type = str(row0[0])

                # Select task_ids to claim.
                params: list[object] = [now]
                where = [
                    "status='queued'",
                    "coalesce(next_attempt_at, 0) <= ?",
                    _ATTEMPTS_AVAILABLE_SQL,
                ]
                if task_types:
                    placeholders = ",".join(["?"] * len(task_types))
                    where.append(f"task_type IN ({placeholders})")
                    params.extend([str(t) for t in task_types])
                if same_task_type and picked_type:
                    where.append("task_type = ?")
                    params.append(str(picked_type))
                where.append(f"({sticky_expr} IS NULL OR {sticky_expr} = ?)")
                params.append(str(worker_id))
                if session:
                    where.append(f"({required_expr} IS NULL OR {required_expr} = ?)")
                    params.append(str(session))
                # Trust-tiered priority cap for the main select.
                if max_priority is not None:
                    cap = max(1, min(10, int(max_priority)))
                    where.append(f"({priority_expr} <= ?)")
                    params.append(cap)

                where_sql = " AND ".join(where)
                rows = conn.execute(
                    (
                        f"SELECT task_id FROM tasks WHERE {where_sql} "
                        f"ORDER BY {priority_expr} DESC, created_at ASC, task_id ASC "
                        f"LIMIT {int(limit)}"
                    ),
                    tuple(params),
                ).fetchall()
                ids = [str(r[0]) for r in (rows or []) if r and r[0]]
                if not ids:
                    conn.execute("COMMIT")
                    return []

                id_placeholders = ",".join(["?"] * len(ids))
                if session:
                    # NOTE: this is best-effort; it prevents accidental claims even
                    # if the initial SELECT raced with another session.
                    sql = (
                        "UPDATE tasks SET status='running', assigned_worker=?, updated_at=?, "
                        "attempt=coalesce(attempt, 0)+1, heartbeat_at=?, lease_until=? "
                        f"WHERE task_id IN ({id_placeholders}) AND status='queued' "
                        "AND coalesce(next_attempt_at, 0) <= ? "
                        f"AND {_ATTEMPTS_AVAILABLE_SQL} "
                        f"AND ({sticky_expr} IS NULL OR {sticky_expr} = ?) "
                        f"AND ({required_expr} IS NULL OR {required_expr} = ?)"
                    )
                    update_params: list[object] = [
                        str(worker_id),
                        now,
                        now,
                        now + lease_duration,
                        *ids,
                        now,
                        str(worker_id),
                        str(session),
                    ]
                    if max_priority is not None:
                        sql += f" AND ({priority_expr} <= ?)"
                        update_params.append(cap)
                    conn.execute(
                        sql,
                        tuple(update_params),
                    )
                else:
                    sql = (
                        "UPDATE tasks SET status='running', assigned_worker=?, updated_at=?, "
                        "attempt=coalesce(attempt, 0)+1, heartbeat_at=?, lease_until=? "
                        f"WHERE task_id IN ({id_placeholders}) AND status='queued' "
                        "AND coalesce(next_attempt_at, 0) <= ? "
                        f"AND {_ATTEMPTS_AVAILABLE_SQL} "
                        f"AND ({sticky_expr} IS NULL OR {sticky_expr} = ?)"
                    )
                    update_params = [
                        str(worker_id),
                        now,
                        now,
                        now + lease_duration,
                        *ids,
                        now,
                        str(worker_id),
                    ]
                    if max_priority is not None:
                        sql += f" AND ({priority_expr} <= ?)"
                        update_params.append(cap)
                    conn.execute(
                        sql,
                        tuple(update_params),
                    )

                rows2 = conn.execute(
                    (
                        f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks "
                        f"WHERE task_id IN ({id_placeholders}) "
                        "AND status='running' AND assigned_worker=? "
                        "ORDER BY priority DESC, created_at ASC, task_id ASC"
                    ),
                    tuple(ids + [str(worker_id)]),
                ).fetchall()

                conn.execute("COMMIT")
                break
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                if attempt < 7 and _is_transient_conflict(exc):
                    time.sleep(0.002 + random.random() * 0.02)
                    continue
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        out: list[QueuedTask] = []
        for row2 in rows2 or []:
            out.append(_queued_task_from_row(row2))
        return out

    def claim(
        self,
        *,
        task_id: str,
        worker_id: str,
        session_id: str | None = None,
        lease_seconds: Optional[float] = None,
    ) -> Optional[QueuedTask]:
        """Atomically claim a specific queued task by id."""

        if not task_id:
            return None
        if not worker_id:
            raise ValueError("worker_id is required")

        now = time.time()
        lease_duration = (
            self.default_lease_seconds if lease_seconds is None else max(1.0, float(lease_seconds))
        )
        session = str(session_id or "").strip()

        if self._quack:
            wid = str(worker_id)
            self._quack_recover_expired(now)
            required_expr_q, sticky_expr_q, _ = _claim_exprs()
            sql = (
                "UPDATE tasks SET status='running', assigned_worker=?, updated_at=?, "
                "attempt=coalesce(attempt, 0)+1, heartbeat_at=?, lease_until=? "
                "WHERE task_id=? AND status='queued' "
                "AND coalesce(next_attempt_at, 0) <= ? "
                f"AND {_ATTEMPTS_AVAILABLE_SQL} "
                f"AND ({sticky_expr_q} IS NULL OR {sticky_expr_q} = ?)"
            )
            params: list = [wid, now, now, now + lease_duration, str(task_id), now, wid]
            if session:
                sql += f" AND ({required_expr_q} IS NULL OR {required_expr_q} = ?)"
                params.append(session)
            sql += f" RETURNING {_TASK_SELECT_COLUMNS}"
            rows = self._quack_write(sql, params)
            if not rows:
                return None
            return _queued_task_from_row(rows[0])

        required_expr = (
            "coalesce("
            "nullif(json_extract_string(payload_json, '$.session_id'), ''), "
            "nullif(json_extract_string(payload_json, '$.session'), ''), "
            "nullif(json_extract_string(payload_json, '$.p2p_session'), '')"
            ")"
        )
        sticky_expr = "nullif(json_extract_string(payload_json, '$.sticky_worker_id'), '')"
        conn = self._connect()
        try:
            conn.execute("BEGIN TRANSACTION")
            self._recover_expired_in_transaction(conn, now, 1000)
            if session:
                conn.execute(
                    f"""
                    UPDATE tasks
                    SET status='running', assigned_worker=?, updated_at=?,
                        attempt=coalesce(attempt, 0)+1,
                        heartbeat_at=?, lease_until=?
                    WHERE task_id=? AND status='queued'
                      AND coalesce(next_attempt_at, 0) <= ?
                      AND {_ATTEMPTS_AVAILABLE_SQL}
                      AND ({sticky_expr} IS NULL OR {sticky_expr} = ?)
                      AND ({required_expr} IS NULL OR {required_expr} = ?)
                    """,
                    (
                        str(worker_id),
                        now,
                        now,
                        now + lease_duration,
                        str(task_id),
                        now,
                        str(worker_id),
                        str(session),
                    ),
                )
            else:
                conn.execute(
                    f"""
                    UPDATE tasks
                    SET status='running', assigned_worker=?, updated_at=?,
                        attempt=coalesce(attempt, 0)+1,
                        heartbeat_at=?, lease_until=?
                    WHERE task_id=? AND status='queued'
                      AND coalesce(next_attempt_at, 0) <= ?
                      AND {_ATTEMPTS_AVAILABLE_SQL}
                      AND ({sticky_expr} IS NULL OR {sticky_expr} = ?)
                    """.strip(),
                    (
                        str(worker_id),
                        now,
                        now,
                        now + lease_duration,
                        str(task_id),
                        now,
                        str(worker_id),
                    ),
                )

            row = conn.execute(
                f"SELECT {_TASK_SELECT_COLUMNS} FROM tasks "
                "WHERE task_id=? AND status='running' AND assigned_worker=?",
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

        return _queued_task_from_row(row)

    def complete(
        self,
        *,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        worker_id: Optional[str] = None,
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
        if self._quack:
            with self._conn_lock:
                conn = self._get_conn()
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
                if "logs" in existing and "logs" not in incoming:
                    merged["logs"] = existing.get("logs")
                if "progress" in existing and "progress" not in incoming:
                    merged["progress"] = existing.get("progress")
                result_json = json.dumps(merged, sort_keys=True) if merged else None
                owner = str(worker_id or "").strip()
                try:
                    if owner:
                        rows = self._quack_write(
                            """
                            UPDATE tasks
                            SET status=?, updated_at=?, result_json=?,
                                error=?, lease_until=NULL, heartbeat_at=NULL
                            WHERE task_id=? AND status='running' AND assigned_worker=?
                            RETURNING task_id
                            """,
                            (
                                status_norm,
                                now,
                                result_json,
                                str(error) if error else None,
                                str(task_id),
                                owner,
                            ),
                        )
                    else:
                        rows = self._quack_write(
                            """
                            UPDATE tasks
                            SET status=?, updated_at=?, result_json=?,
                                error=?, lease_until=NULL, heartbeat_at=NULL
                            WHERE task_id=?
                            RETURNING task_id
                            """,
                            (
                                status_norm,
                                now,
                                result_json,
                                str(error) if error else None,
                                str(task_id),
                            ),
                        )
                    return len(rows) > 0
                except Exception as exc:
                    msg = str(exc).lower()
                    if (
                        "write-write conflict" in msg
                        or "conflict on tuple" in msg
                        or "transactioncontext error" in msg
                        or ("catalog" in msg and "conflict" in msg)
                    ):
                        return False
                    raise
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

                owner = str(worker_id or "").strip()
                if owner:
                    updated = conn.execute(
                        """
                        UPDATE tasks
                        SET status=?, updated_at=?, result_json=?,
                            error=?, lease_until=NULL, heartbeat_at=NULL
                        WHERE task_id=? AND status='running' AND assigned_worker=?
                        RETURNING task_id
                        """,
                        (
                            status_norm,
                            now,
                            result_json,
                            str(error) if error else None,
                            str(task_id),
                            owner,
                        ),
                    ).fetchone()
                else:
                    updated = conn.execute(
                        """
                        UPDATE tasks
                        SET status=?, updated_at=?, result_json=?,
                            error=?, lease_until=NULL, heartbeat_at=NULL
                        WHERE task_id=?
                        RETURNING task_id
                        """,
                        (
                            status_norm,
                            now,
                            result_json,
                            str(error) if error else None,
                            str(task_id),
                        ),
                    ).fetchone()
                return updated is not None
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

    def cancel(self, *, task_id: str, reason: str | None = None) -> bool:
        """Cancel a queued task.

        This is used by higher-level orchestrators when a task is pinned to an
        offline worker (e.g., session resume) and needs to be retried elsewhere.

        Only tasks in status='queued' are cancelled.
        """

        if not task_id:
            return False

        now = time.time()

        def _merge_progress(existing: Any) -> Dict[str, Any]:
            base: Dict[str, Any] = {}
            if isinstance(existing, dict):
                base = dict(existing)
            progress = base.get("progress")
            if not isinstance(progress, dict):
                progress = {}
            base["progress"] = progress
            return base

        if self._quack:
            with self._conn_lock:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT result_json FROM tasks WHERE task_id=? AND status='queued'",
                    (str(task_id),),
                ).fetchone()
                if row is None:
                    return False
                try:
                    existing = json.loads(row[0]) if row[0] else {}
                except Exception:
                    existing = {}
                result_obj = _merge_progress(existing)
                result_obj["progress"]["cancelled_at"] = now
                if isinstance(reason, str) and reason.strip():
                    result_obj["progress"]["cancel_reason"] = reason.strip()
                rows = self._quack_write(
                    "UPDATE tasks SET status='cancelled', assigned_worker=NULL, "
                    "lease_until=NULL, heartbeat_at=NULL, result_json=?, updated_at=? "
                    "WHERE task_id=? AND status='queued' RETURNING task_id",
                    (json.dumps(result_obj, sort_keys=True), now, str(task_id)),
                )
                return len(rows) > 0

        with self._conn_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN TRANSACTION")
                row = conn.execute(
                    "SELECT result_json FROM tasks WHERE task_id=? AND status='queued'",
                    (str(task_id),),
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return False

                try:
                    existing = json.loads(row[0]) if row[0] else {}
                except Exception:
                    existing = {}

                result_obj = _merge_progress(existing)
                result_obj["progress"]["cancelled_at"] = now
                if isinstance(reason, str) and reason.strip():
                    result_obj["progress"]["cancel_reason"] = reason.strip()

                conn.execute(
                    (
                        "UPDATE tasks SET status='cancelled', assigned_worker=NULL, "
                        "lease_until=NULL, heartbeat_at=NULL, result_json=?, updated_at=? "
                        "WHERE task_id=? AND status='queued'"
                    ),
                    (json.dumps(result_obj, sort_keys=True), now, str(task_id)),
                )
                conn.execute("COMMIT")
                return True
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

    def delete(self, *, task_id: str) -> bool:
        """Delete a task row by id.

        This is intended for internal/ephemeral tasks where the caller already
        returned the result to the requester (e.g. p2p call_tool -> tool.call).
        """

        if not task_id:
            return False

        if self._quack:
            try:
                self._quack_write(
                    "DELETE FROM tasks WHERE task_id=? RETURNING task_id",
                    (str(task_id),),
                )
                return True
            except Exception:
                return False

        with self._conn_lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM tasks WHERE task_id=?", (str(task_id),))
                return True
            except Exception:
                return False

    def prune_terminal(
        self,
        *,
        older_than_s: float,
        limit: int = 1000,
        statuses: Optional[Iterable[str]] = None,
    ) -> int:
        """Prune terminal tasks (completed/failed/cancelled) older than a cutoff."""

        keep_s = float(older_than_s)
        if keep_s <= 0:
            return 0
        lim = max(1, min(int(limit or 1000), 10000))

        st_default = {"completed", "failed", "cancelled"}
        st_in = [str(s).strip().lower() for s in (statuses or st_default) if str(s).strip()]
        st_in = [s for s in st_in if s in st_default]
        if not st_in:
            st_in = sorted(st_default)

        cutoff = time.time() - keep_s
        placeholders = ",".join(["?"] * len(st_in))

        if self._quack:
            quack_sql = (
                "DELETE FROM tasks WHERE task_id IN ("
                "  SELECT task_id FROM tasks"
                f"  WHERE status IN ({placeholders}) AND updated_at < ?"
                "  ORDER BY updated_at ASC"
                "  LIMIT ?"
                ") RETURNING task_id"
            )
            prune_params: list = [*st_in, float(cutoff), int(lim)]
            try:
                rows = self._quack_write(quack_sql, tuple(prune_params))
                return len(rows)
            except Exception:
                return 0

        # Delete in small batches to avoid long write locks.
        sql = (
            "DELETE FROM tasks WHERE task_id IN ("
            "  SELECT task_id FROM tasks"
            f"  WHERE status IN ({placeholders}) AND updated_at < ?"
            "  ORDER BY updated_at ASC"
            "  LIMIT ?"
            ")"
        )
        params: list[Any] = [*st_in, float(cutoff), int(lim)]

        with self._conn_lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(sql, tuple(params))
                try:
                    return int(getattr(cur, "rowcount", 0) or 0)
                except Exception:
                    return 0
            except Exception:
                return 0

    def release(
        self,
        *,
        task_id: str,
        worker_id: str,
        reason: str | None = None,
    ) -> bool:
        """Best-effort: release a running task back to queued.

        This is used when a worker claims a task but determines it should not
        execute it (e.g., session-affinity mismatch).

        Only releases tasks currently in status='running' and assigned to the
        provided worker_id.
        """

        tid = str(task_id or "").strip()
        wid = str(worker_id or "").strip()
        if not tid or not wid:
            return False

        now = time.time()
        try:
            note = str(reason or "released")
        except Exception:
            note = "released"

        with self._conn_lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT result_json FROM tasks WHERE task_id=?",
                    (tid,),
                ).fetchone()
                existing = {}
                if row and isinstance(row[0], str) and row[0]:
                    try:
                        parsed = json.loads(row[0])
                        existing = parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        existing = {}

                merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
                progress = merged.get("progress")
                if not isinstance(progress, dict):
                    progress = {}
                progress = dict(progress)
                progress.setdefault("release_count", 0)
                try:
                    progress["release_count"] = int(progress.get("release_count") or 0) + 1
                except Exception:
                    progress["release_count"] = 1
                progress["last_release_reason"] = note
                progress["last_release_ts"] = float(now)
                merged["progress"] = progress

                result_json = json.dumps(merged, sort_keys=True) if merged else None

                if self._quack:
                    rows = self._quack_write(
                        """
                        UPDATE tasks
                        SET status='queued', assigned_worker=NULL, updated_at=?, result_json=?,
                            next_attempt_at=?, lease_until=NULL, heartbeat_at=NULL
                        WHERE task_id=? AND status='running' AND assigned_worker=?
                        RETURNING task_id
                        """,
                        (float(now), result_json, float(now), tid, wid),
                    )
                    return len(rows) > 0

                updated = conn.execute(
                    """
                    UPDATE tasks
                    SET status='queued', assigned_worker=NULL, updated_at=?, result_json=?,
                        next_attempt_at=?, lease_until=NULL, heartbeat_at=NULL
                    WHERE task_id=? AND status='running' AND assigned_worker=?
                    RETURNING task_id
                    """,
                    (float(now), result_json, float(now), tid, wid),
                ).fetchone()
                return updated is not None
            except Exception:
                return False

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
        if status_norm and status_norm not in {
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
        }:
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

                if self._quack:
                    self._quack_write(
                        """
                        UPDATE tasks
                        SET status=?, updated_at=?, result_json=?, error=?
                        WHERE task_id=?
                        """,
                        (
                            new_status,
                            now,
                            result_json,
                            str(error) if error else None,
                            str(task_id),
                        ),
                    )
                    return True

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
