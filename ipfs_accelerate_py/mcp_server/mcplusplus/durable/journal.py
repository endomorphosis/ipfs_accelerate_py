"""SQLite durable execution journal (DurableJournalRecord@1 authority).

Interface label: ``DurableJournal@1``

This module owns the append-only journal store used by the mandatory
``SqliteDurableExecutor@1`` adapter (ADR-0005 / MCPP-051):

* SQLite with WAL and ``PRAGMA synchronous=FULL``
* Monotonic per-execution ``journal_seq`` starting at 1
* Content-addressed journal records (``mcpp-jcs-v1`` / Kubo CIDv1)
* Idempotency-key index for committed side effects
* Execution projection rows for status, fencing, cancel, timers, obligations

The journal is the recovery authority for step commit, cancel persistence,
fencing epochs, and resume checkpoints. Event DAG emission is optional
provenance and is not required for crash recovery.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

JOURNAL_RECORD_SCHEMA = "mcp++/durable/journal-record@1"
CANONICALIZATION = "mcpp-jcs-v1"
INTERFACE_LABEL = "DurableJournal@1"
ADAPTER_ID = "sqlite-journal@1"
SCHEMA_MARKER = "mcp++/durable/journal@1"

JOURNAL_TRANSITIONS = frozenset(
    {
        "started",
        "resumed",
        "signalled",
        "cancel_requested",
        "cancelled",
        "checkpointed",
        "side_effect_committed",
        "retried",
        "timer_scheduled",
        "timer_fired",
        "timer_cancelled",
        "compensation_started",
        "compensation_completed",
        "recovered",
        "finalized",
    }
)

TRANSITION_EVENT_TYPE: Dict[str, str] = {
    "started": "envelope",
    "resumed": "invocation",
    "signalled": "intent",
    "cancel_requested": "error",
    "cancelled": "result",
    "checkpointed": "decision",
    "side_effect_committed": "result",
    "retried": "invocation",
    "timer_scheduled": "intent",
    "timer_fired": "invocation",
    "timer_cancelled": "error",
    "compensation_started": "intent",
    "compensation_completed": "result",
    "recovered": "envelope",
    "finalized": "receipt",
}

TERMINAL_STATUSES = frozenset(
    {
        "cancelled",
        "compensated",
        "succeeded",
        "failed",
        "rejected",
        "timed_out",
    }
)

ACTIVE_STATUSES = frozenset(
    {
        "pending",
        "running",
        "paused",
        "cancelling",
        "compensating",
    }
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DurableJournalError(RuntimeError):
    """Base error for the durable journal store."""

    code = "durable_journal_error"


class ExecutionNotFoundError(DurableJournalError, KeyError):
    """Raised when an execution_id has no durable row."""

    code = "execution_not_found"

    def __init__(self, execution_id: str, message: Optional[str] = None) -> None:
        self.execution_id = execution_id
        super().__init__(message or f"execution not found: {execution_id!r}")


class IdempotencyConflictError(DurableJournalError):
    """An idempotency key was reused with a conflicting payload."""

    code = "idempotency_conflict"

    def __init__(
        self,
        *,
        scope: str,
        idempotency_key: str,
        message: Optional[str] = None,
    ) -> None:
        self.scope = scope
        self.idempotency_key = idempotency_key
        super().__init__(
            message
            or (
                f"idempotency conflict in scope {scope!r} for key "
                f"{idempotency_key!r}"
            )
        )


class JournalIntegrityError(DurableJournalError):
    """Journal sequence or parent linkage is inconsistent."""

    code = "journal_integrity"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Return deterministic JSON text (mcpp-jcs-v1 style sort/separators)."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not JSON-serializable: {exc}") from exc


def cid_for_mapping(value: Mapping[str, Any]) -> str:
    """Mint a Kubo-compatible CIDv1 for a canonical JSON mapping."""

    return cid_for_bytes(canonical_json(dict(value)).encode("utf-8"))


def is_portable_cid(value: object) -> bool:
    """Return True when ``value`` looks like a portable CIDv0/CIDv1 string."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if len(text) < 46 or len(text) > 128:
        return False
    if text.startswith("Qm"):
        alphabet = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
        return len(text) == 46 and all(ch in alphabet for ch in text)
    if text.startswith("b") and len(text) >= 59:
        alphabet = set("abcdefghijklmnopqrstuvwxyz234567")
        return all(ch in alphabet for ch in text[1:])
    return False


def require_portable_cid(value: object, *, field: str = "cid") -> str:
    """Validate and return a portable CID string."""

    if not is_portable_cid(value):
        raise ValueError(f"{field} must be a portable CIDv0/CIDv1 string")
    return str(value).strip()


def _require_non_empty_str(value: object, name: str, *, max_len: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_len:
        raise ValueError(f"{name} must be a non-empty string up to {max_len} characters")
    return value.strip()


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _loads_json(raw: Optional[str], default: Any = None) -> Any:
    if raw is None:
        return default
    return json.loads(raw)


@dataclass(frozen=True)
class AppendResult:
    """Outcome of a successful journal append."""

    execution_id: str
    journal_seq: int
    record_cid: str
    record: Dict[str, Any]
    status: str
    fencing_token: int


# ---------------------------------------------------------------------------
# Journal store
# ---------------------------------------------------------------------------


class DurableJournal:
    """Append-only SQLite journal for DurableExecutor executions.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. Parent directories are created.
    clock_ms:
        Optional callable returning unix epoch milliseconds (injectable for tests).
    """

    DB_VERSION = 1

    def __init__(
        self,
        db_path: os.PathLike[str] | str,
        *,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._connection = self._open_database()

    # -- lifecycle ---------------------------------------------------------

    def _open_database(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            self._create_schema(connection)
            return connection
        except Exception:
            connection.close()
            raise

    @classmethod
    def _create_schema(cls, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS executions (
              execution_id TEXT PRIMARY KEY,
              envelope_cid TEXT NOT NULL,
              status TEXT NOT NULL,
              fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0),
              journal_seq INTEGER NOT NULL DEFAULT 0 CHECK (journal_seq >= 0),
              last_checkpoint_id TEXT,
              progress_cid TEXT,
              correlation_id TEXT,
              start_idempotency_key TEXT NOT NULL UNIQUE,
              cancel_state TEXT NOT NULL DEFAULT 'none',
              cancel_reason TEXT,
              obligation_cids_json TEXT NOT NULL DEFAULT '[]',
              attempt INTEGER NOT NULL DEFAULT 0 CHECK (attempt >= 0),
              receipt_cid TEXT,
              result_cid TEXT,
              parent_execution_id TEXT,
              executor_did TEXT,
              last_event_cid TEXT,
              last_record_cid TEXT,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS executions_status
              ON executions(status);
            CREATE INDEX IF NOT EXISTS executions_correlation
              ON executions(correlation_id);

            CREATE TABLE IF NOT EXISTS journal_records (
              execution_id TEXT NOT NULL,
              journal_seq INTEGER NOT NULL CHECK (journal_seq >= 1),
              record_cid TEXT NOT NULL UNIQUE,
              transition TEXT NOT NULL,
              fencing_token INTEGER NOT NULL CHECK (fencing_token >= 0),
              idempotency_key TEXT,
              event_cid TEXT,
              record_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL,
              PRIMARY KEY (execution_id, journal_seq),
              FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
            );

            CREATE INDEX IF NOT EXISTS journal_records_transition
              ON journal_records(execution_id, transition);

            CREATE TABLE IF NOT EXISTS side_effects (
              execution_id TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              effect_json TEXT NOT NULL,
              journal_seq INTEGER NOT NULL,
              committed_at_ms INTEGER NOT NULL,
              PRIMARY KEY (execution_id, idempotency_key),
              FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
            );

            CREATE TABLE IF NOT EXISTS method_idempotency (
              execution_id TEXT NOT NULL,
              scope TEXT NOT NULL,
              idempotency_key TEXT NOT NULL,
              fingerprint TEXT NOT NULL,
              result_json TEXT NOT NULL,
              journal_seq INTEGER,
              created_at_ms INTEGER NOT NULL,
              PRIMARY KEY (execution_id, scope, idempotency_key),
              FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
            );

            CREATE TABLE IF NOT EXISTS start_idempotency (
              idempotency_key TEXT PRIMARY KEY,
              execution_id TEXT NOT NULL UNIQUE,
              envelope_cid TEXT NOT NULL,
              result_json TEXT NOT NULL,
              created_at_ms INTEGER NOT NULL,
              FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
            );

            CREATE TABLE IF NOT EXISTS timers (
              execution_id TEXT NOT NULL,
              timer_id TEXT NOT NULL,
              fire_at_ms INTEGER NOT NULL,
              status TEXT NOT NULL,
              payload_cid TEXT,
              journal_seq INTEGER,
              created_at_ms INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL,
              PRIMARY KEY (execution_id, timer_id),
              FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
            );
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('version', ?)",
            (str(cls.DB_VERSION),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('interface', ?)",
            (INTERFACE_LABEL,),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('adapter', ?)",
            (ADAPTER_ID,),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema', ?)",
            (SCHEMA_MARKER,),
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._connection.close()

    def __enter__(self) -> "DurableJournal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @classmethod
    def open(
        cls,
        db_path: os.PathLike[str] | str,
        *,
        clock_ms: Optional[Callable[[], int]] = None,
    ) -> "DurableJournal":
        """Open or create a durable journal database."""

        return cls(db_path, clock_ms=clock_ms)

    def journal_mode(self) -> str:
        """Return the active SQLite journal_mode (expect ``wal``)."""

        with self._lock:
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
            return str(row[0]).lower()

    def db_version(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'version'"
            ).fetchone()
            return int(row["value"]) if row else 0

    def now_ms(self) -> int:
        return int(self._clock_ms())

    # -- reads -------------------------------------------------------------

    def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Return the durable execution projection or raise."""

        with self._lock:
            return self._get_execution_unlocked(execution_id)

    def try_get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._try_get_execution_unlocked(execution_id)

    def find_by_start_idempotency(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        key = _require_non_empty_str(idempotency_key, "idempotency_key")
        with self._lock:
            row = self._connection.execute(
                "SELECT execution_id FROM start_idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            return self._get_execution_unlocked(str(row["execution_id"]))

    def find_by_correlation(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        corr = _require_non_empty_str(correlation_id, "correlation_id")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT execution_id FROM executions
                WHERE correlation_id = ?
                ORDER BY created_at_ms DESC
                LIMIT 1
                """,
                (corr,),
            ).fetchone()
            if row is None:
                return None
            return self._get_execution_unlocked(str(row["execution_id"]))

    def list_recoverable(self) -> List[Dict[str, Any]]:
        """List non-terminal executions that recover may reconstruct."""

        with self._lock:
            rows = self._connection.execute(
                """
                SELECT execution_id FROM executions
                WHERE status NOT IN ('cancelled','compensated','succeeded','failed','rejected','timed_out')
                ORDER BY created_at_ms ASC
                """
            ).fetchall()
            return [self._get_execution_unlocked(str(r["execution_id"])) for r in rows]

    def list_records(
        self,
        execution_id: str,
        *,
        from_seq: int = 1,
        to_seq: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return journal records in sequence order for replay."""

        eid = _require_non_empty_str(execution_id, "execution_id")
        start = _require_positive_int(from_seq, "from_seq")
        with self._lock:
            if self._try_get_execution_unlocked(eid) is None:
                raise ExecutionNotFoundError(eid)
            if to_seq is None:
                rows = self._connection.execute(
                    """
                    SELECT record_json FROM journal_records
                    WHERE execution_id = ? AND journal_seq >= ?
                    ORDER BY journal_seq ASC
                    """,
                    (eid, start),
                ).fetchall()
            else:
                end = _require_positive_int(to_seq, "to_seq")
                rows = self._connection.execute(
                    """
                    SELECT record_json FROM journal_records
                    WHERE execution_id = ? AND journal_seq >= ? AND journal_seq <= ?
                    ORDER BY journal_seq ASC
                    """,
                    (eid, start, end),
                ).fetchall()
            return [json.loads(r["record_json"]) for r in rows]

    def get_record(self, execution_id: str, journal_seq: int) -> Dict[str, Any]:
        eid = _require_non_empty_str(execution_id, "execution_id")
        seq = _require_positive_int(journal_seq, "journal_seq")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT record_json FROM journal_records
                WHERE execution_id = ? AND journal_seq = ?
                """,
                (eid, seq),
            ).fetchone()
            if row is None:
                raise JournalIntegrityError(
                    f"missing journal record {eid!r} seq={seq}"
                )
            return json.loads(row["record_json"])

    def committed_side_effects(self, execution_id: str) -> List[Dict[str, Any]]:
        eid = _require_non_empty_str(execution_id, "execution_id")
        with self._lock:
            if self._try_get_execution_unlocked(eid) is None:
                raise ExecutionNotFoundError(eid)
            rows = self._connection.execute(
                """
                SELECT effect_json, idempotency_key, journal_seq
                FROM side_effects
                WHERE execution_id = ?
                ORDER BY journal_seq ASC, idempotency_key ASC
                """,
                (eid,),
            ).fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                effect = json.loads(row["effect_json"])
                effect.setdefault("idempotency_key", row["idempotency_key"])
                effect["_journal_seq"] = int(row["journal_seq"])
                out.append(effect)
            return out

    def is_side_effect_committed(self, execution_id: str, idempotency_key: str) -> bool:
        eid = _require_non_empty_str(execution_id, "execution_id")
        key = _require_non_empty_str(idempotency_key, "idempotency_key")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT 1 FROM side_effects
                WHERE execution_id = ? AND idempotency_key = ?
                """,
                (eid, key),
            ).fetchone()
            return row is not None

    def get_method_idempotency(
        self,
        execution_id: str,
        scope: str,
        idempotency_key: str,
    ) -> Optional[Dict[str, Any]]:
        eid = _require_non_empty_str(execution_id, "execution_id")
        sc = _require_non_empty_str(scope, "scope", max_len=64)
        key = _require_non_empty_str(idempotency_key, "idempotency_key")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT fingerprint, result_json, journal_seq
                FROM method_idempotency
                WHERE execution_id = ? AND scope = ? AND idempotency_key = ?
                """,
                (eid, sc, key),
            ).fetchone()
            if row is None:
                return None
            return {
                "fingerprint": row["fingerprint"],
                "result": json.loads(row["result_json"]),
                "journal_seq": row["journal_seq"],
            }

    def get_start_idempotency(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        key = _require_non_empty_str(idempotency_key, "idempotency_key")
        with self._lock:
            row = self._connection.execute(
                """
                SELECT execution_id, envelope_cid, result_json
                FROM start_idempotency
                WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
            if row is None:
                return None
            return {
                "execution_id": row["execution_id"],
                "envelope_cid": row["envelope_cid"],
                "result": json.loads(row["result_json"]),
            }

    def list_timers(self, execution_id: str) -> List[Dict[str, Any]]:
        eid = _require_non_empty_str(execution_id, "execution_id")
        with self._lock:
            if self._try_get_execution_unlocked(eid) is None:
                raise ExecutionNotFoundError(eid)
            rows = self._connection.execute(
                """
                SELECT timer_id, fire_at_ms, status, payload_cid, journal_seq
                FROM timers
                WHERE execution_id = ?
                ORDER BY timer_id ASC
                """,
                (eid,),
            ).fetchall()
            return [
                {
                    "timer_id": r["timer_id"],
                    "fire_at_ms": int(r["fire_at_ms"]),
                    "status": r["status"],
                    "payload_cid": r["payload_cid"],
                    "journal_seq": r["journal_seq"],
                }
                for r in rows
            ]

    def replay(self, execution_id: str) -> Dict[str, Any]:
        """Reconstruct execution projection solely from journal records.

        Returns a dict with ``execution`` (projected), ``records``, and
        ``side_effects_not_replayed`` (committed effect idempotency keys).
        """

        eid = _require_non_empty_str(execution_id, "execution_id")
        with self._lock:
            stored = self._try_get_execution_unlocked(eid)
            if stored is None:
                raise ExecutionNotFoundError(eid)
            records = [
                json.loads(r["record_json"])
                for r in self._connection.execute(
                    """
                    SELECT record_json FROM journal_records
                    WHERE execution_id = ?
                    ORDER BY journal_seq ASC
                    """,
                    (eid,),
                ).fetchall()
            ]
            projected = self._project_from_records(eid, records, stored)
            effects = self.committed_side_effects(eid) if records else []
            return {
                "execution": projected,
                "records": records,
                "side_effects_not_replayed": [
                    e["idempotency_key"] for e in effects if e.get("idempotency_key")
                ],
            }

    # -- writes ------------------------------------------------------------

    def create_execution(
        self,
        *,
        execution_id: str,
        envelope_cid: str,
        start_idempotency_key: str,
        fencing_token: int = 1,
        correlation_id: Optional[str] = None,
        parent_execution_id: Optional[str] = None,
        executor_did: Optional[str] = None,
        progress_cid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Insert a pending execution row (journal append follows separately)."""

        eid = _require_non_empty_str(execution_id, "execution_id")
        env = require_portable_cid(envelope_cid, field="envelope_cid")
        ikey = _require_non_empty_str(start_idempotency_key, "start_idempotency_key")
        fence = _require_non_negative_int(fencing_token, "fencing_token")
        if fence < 1:
            fence = 1
        now = self.now_ms()
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self._connection.execute(
                    "SELECT execution_id FROM start_idempotency WHERE idempotency_key = ?",
                    (ikey,),
                ).fetchone()
                if existing is not None:
                    self._connection.execute("ROLLBACK")
                    raise IdempotencyConflictError(
                        scope="start",
                        idempotency_key=ikey,
                        message=f"start idempotency key already bound: {ikey!r}",
                    )
                self._connection.execute(
                    """
                    INSERT INTO executions (
                      execution_id, envelope_cid, status, fencing_token, journal_seq,
                      last_checkpoint_id, progress_cid, correlation_id,
                      start_idempotency_key, cancel_state, cancel_reason,
                      obligation_cids_json, attempt, receipt_cid, result_cid,
                      parent_execution_id, executor_did, last_event_cid, last_record_cid,
                      created_at_ms, updated_at_ms
                    ) VALUES (
                      ?, ?, 'pending', ?, 0,
                      NULL, ?, ?,
                      ?, 'none', NULL,
                      '[]', 0, NULL, NULL,
                      ?, ?, NULL, NULL,
                      ?, ?
                    )
                    """,
                    (
                        eid,
                        env,
                        fence,
                        progress_cid,
                        correlation_id,
                        ikey,
                        parent_execution_id,
                        executor_did,
                        now,
                        now,
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            return self._get_execution_unlocked(eid)

    def append(
        self,
        *,
        execution_id: str,
        transition: str,
        fencing_token: int,
        idempotency_key: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        progress_cid: Optional[str] = None,
        side_effects: Optional[Sequence[Mapping[str, Any]]] = None,
        result_cid: Optional[str] = None,
        receipt_cid: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        event_cid: Optional[str] = None,
        status: Optional[str] = None,
        cancel_state: Optional[str] = None,
        cancel_reason: Optional[str] = None,
        obligation_cids: Optional[Sequence[str]] = None,
        attempt: Optional[int] = None,
        advance_fencing_token: Optional[int] = None,
        timer: Optional[Mapping[str, Any]] = None,
        method_idempotency: Optional[Mapping[str, Any]] = None,
        start_idempotency_result: Optional[Mapping[str, Any]] = None,
    ) -> AppendResult:
        """Append a DurableJournalRecord@1 and update the execution projection.

        The entire append + projection update is one SQLite transaction.
        """

        eid = _require_non_empty_str(execution_id, "execution_id")
        if transition not in JOURNAL_TRANSITIONS:
            raise ValueError(f"unknown journal transition: {transition!r}")
        presented_fence = _require_non_negative_int(fencing_token, "fencing_token")
        effects = [dict(e) for e in (side_effects or ())]
        for effect in effects:
            if "kind" not in effect or not isinstance(effect.get("kind"), str):
                raise ValueError("side_effect.kind is required")
            if "idempotency_key" not in effect or not isinstance(
                effect.get("idempotency_key"), str
            ):
                raise ValueError("side_effect.idempotency_key is required")

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                execution = self._try_get_execution_unlocked(eid)
                if execution is None:
                    raise ExecutionNotFoundError(eid)

                next_seq = int(execution["journal_seq"]) + 1
                parents: List[str] = []
                if execution.get("last_record_cid"):
                    parents = [str(execution["last_record_cid"])]

                now = self.now_ms()
                record: Dict[str, Any] = {
                    "schema": JOURNAL_RECORD_SCHEMA,
                    "execution_id": eid,
                    "journal_seq": next_seq,
                    "transition": transition,
                    "event_type": TRANSITION_EVENT_TYPE.get(transition),
                    "fencing_token": presented_fence
                    if advance_fencing_token is None
                    else int(advance_fencing_token),
                    "envelope_cid": execution["envelope_cid"],
                    "parents": parents,
                    "created_at_ms": now,
                    "canonicalization": CANONICALIZATION,
                }
                if idempotency_key is not None:
                    record["idempotency_key"] = _require_non_empty_str(
                        idempotency_key, "idempotency_key"
                    )
                if checkpoint_id is not None:
                    record["checkpoint_id"] = _require_non_empty_str(
                        checkpoint_id, "checkpoint_id"
                    )
                if progress_cid is not None:
                    record["progress_cid"] = require_portable_cid(
                        progress_cid, field="progress_cid"
                    )
                if effects:
                    record["side_effects"] = effects
                if result_cid is not None:
                    record["result_cid"] = require_portable_cid(
                        result_cid, field="result_cid"
                    )
                if receipt_cid is not None:
                    record["receipt_cid"] = require_portable_cid(
                        receipt_cid, field="receipt_cid"
                    )
                if payload is not None:
                    record["payload"] = dict(payload)
                if event_cid is not None:
                    record["event_cid"] = require_portable_cid(
                        event_cid, field="event_cid"
                    )

                # CID excludes event_cid if we want stable mint before event;
                # event_cid is included when provided by caller.
                record_cid = cid_for_mapping(record)
                record_for_store = dict(record)
                # Bind self-identity for storage (not part of schema required fields)
                record_for_store["record_cid"] = record_cid

                self._connection.execute(
                    """
                    INSERT INTO journal_records (
                      execution_id, journal_seq, record_cid, transition,
                      fencing_token, idempotency_key, event_cid, record_json,
                      created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eid,
                        next_seq,
                        record_cid,
                        transition,
                        int(record["fencing_token"]),
                        record.get("idempotency_key"),
                        record.get("event_cid"),
                        canonical_json(record_for_store),
                        now,
                    ),
                )

                new_status = status if status is not None else execution["status"]
                new_fence = (
                    int(advance_fencing_token)
                    if advance_fencing_token is not None
                    else int(execution["fencing_token"])
                )
                new_checkpoint = (
                    checkpoint_id
                    if checkpoint_id is not None
                    else execution.get("last_checkpoint_id")
                )
                new_progress = (
                    progress_cid
                    if progress_cid is not None
                    else execution.get("progress_cid")
                )
                new_cancel_state = (
                    cancel_state
                    if cancel_state is not None
                    else execution.get("cancel_state", "none")
                )
                new_cancel_reason = (
                    cancel_reason
                    if cancel_reason is not None
                    else execution.get("cancel_reason")
                )
                if obligation_cids is not None:
                    # Merge unique obligation CIDs.
                    existing_obs = list(execution.get("obligation_cids") or [])
                    merged = list(dict.fromkeys([*existing_obs, *obligation_cids]))
                    obligations_json = canonical_json(merged)
                else:
                    obligations_json = canonical_json(
                        list(execution.get("obligation_cids") or [])
                    )
                new_attempt = (
                    int(attempt)
                    if attempt is not None
                    else int(execution.get("attempt") or 0)
                )
                new_result = (
                    result_cid if result_cid is not None else execution.get("result_cid")
                )
                new_receipt = (
                    receipt_cid
                    if receipt_cid is not None
                    else execution.get("receipt_cid")
                )
                new_event = (
                    event_cid if event_cid is not None else execution.get("last_event_cid")
                )

                self._connection.execute(
                    """
                    UPDATE executions SET
                      status = ?,
                      fencing_token = ?,
                      journal_seq = ?,
                      last_checkpoint_id = ?,
                      progress_cid = ?,
                      cancel_state = ?,
                      cancel_reason = ?,
                      obligation_cids_json = ?,
                      attempt = ?,
                      result_cid = ?,
                      receipt_cid = ?,
                      last_event_cid = ?,
                      last_record_cid = ?,
                      updated_at_ms = ?
                    WHERE execution_id = ?
                    """,
                    (
                        new_status,
                        new_fence,
                        next_seq,
                        new_checkpoint,
                        new_progress,
                        new_cancel_state,
                        new_cancel_reason,
                        obligations_json,
                        new_attempt,
                        new_result,
                        new_receipt,
                        new_event,
                        record_cid,
                        now,
                        eid,
                    ),
                )

                for effect in effects:
                    ekey = str(effect["idempotency_key"])
                    existing_effect = self._connection.execute(
                        """
                        SELECT effect_json FROM side_effects
                        WHERE execution_id = ? AND idempotency_key = ?
                        """,
                        (eid, ekey),
                    ).fetchone()
                    if existing_effect is not None:
                        prior = json.loads(existing_effect["effect_json"])
                        if prior != effect:
                            raise IdempotencyConflictError(
                                scope="side_effect",
                                idempotency_key=ekey,
                            )
                        # Already committed — do not re-insert.
                        continue
                    self._connection.execute(
                        """
                        INSERT INTO side_effects (
                          execution_id, idempotency_key, effect_json,
                          journal_seq, committed_at_ms
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (eid, ekey, canonical_json(effect), next_seq, now),
                    )

                if timer is not None:
                    self._upsert_timer_unlocked(eid, timer, next_seq, now)

                if method_idempotency is not None:
                    scope = _require_non_empty_str(
                        method_idempotency.get("scope"), "method_idempotency.scope", max_len=64
                    )
                    mkey = _require_non_empty_str(
                        method_idempotency.get("idempotency_key"),
                        "method_idempotency.idempotency_key",
                    )
                    fingerprint = _require_non_empty_str(
                        method_idempotency.get("fingerprint"),
                        "method_idempotency.fingerprint",
                        max_len=8192,
                    )
                    result_obj = method_idempotency.get("result")
                    if not isinstance(result_obj, Mapping):
                        raise ValueError("method_idempotency.result must be a mapping")
                    self._connection.execute(
                        """
                        INSERT OR REPLACE INTO method_idempotency (
                          execution_id, scope, idempotency_key, fingerprint,
                          result_json, journal_seq, created_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            eid,
                            scope,
                            mkey,
                            fingerprint,
                            canonical_json(dict(result_obj)),
                            next_seq,
                            now,
                        ),
                    )

                if start_idempotency_result is not None:
                    skey = execution["start_idempotency_key"]
                    if not isinstance(start_idempotency_result, Mapping):
                        raise ValueError("start_idempotency_result must be a mapping")
                    self._connection.execute(
                        """
                        INSERT OR REPLACE INTO start_idempotency (
                          idempotency_key, execution_id, envelope_cid,
                          result_json, created_at_ms
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            skey,
                            eid,
                            execution["envelope_cid"],
                            canonical_json(dict(start_idempotency_result)),
                            now,
                        ),
                    )

                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

            return AppendResult(
                execution_id=eid,
                journal_seq=next_seq,
                record_cid=record_cid,
                record=record_for_store,
                status=str(new_status),
                fencing_token=int(new_fence),
            )

    def update_execution_fields(
        self,
        execution_id: str,
        **fields: Any,
    ) -> Dict[str, Any]:
        """Update selected projection fields without appending a journal record.

        Used for rare bookkeeping (e.g. caching inspect-only data). Prefer
        :meth:`append` for any durable lifecycle transition.
        """

        allowed = {
            "status",
            "fencing_token",
            "last_checkpoint_id",
            "progress_cid",
            "cancel_state",
            "cancel_reason",
            "attempt",
            "receipt_cid",
            "result_cid",
            "last_event_cid",
            "correlation_id",
            "obligation_cids",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unsupported execution fields: {sorted(unknown)}")
        eid = _require_non_empty_str(execution_id, "execution_id")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if self._try_get_execution_unlocked(eid) is None:
                    raise ExecutionNotFoundError(eid)
                now = self.now_ms()
                sets: List[str] = ["updated_at_ms = ?"]
                values: List[Any] = [now]
                if "obligation_cids" in fields:
                    sets.append("obligation_cids_json = ?")
                    values.append(canonical_json(list(fields["obligation_cids"] or [])))
                for key in (
                    "status",
                    "fencing_token",
                    "last_checkpoint_id",
                    "progress_cid",
                    "cancel_state",
                    "cancel_reason",
                    "attempt",
                    "receipt_cid",
                    "result_cid",
                    "last_event_cid",
                    "correlation_id",
                ):
                    if key in fields:
                        sets.append(f"{key} = ?")
                        values.append(fields[key])
                values.append(eid)
                self._connection.execute(
                    f"UPDATE executions SET {', '.join(sets)} WHERE execution_id = ?",
                    values,
                )
                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            return self._get_execution_unlocked(eid)

    def store_method_idempotency(
        self,
        *,
        execution_id: str,
        scope: str,
        idempotency_key: str,
        fingerprint: str,
        result: Mapping[str, Any],
        journal_seq: Optional[int] = None,
    ) -> None:
        """Persist a method-level idempotency result without a journal append."""

        eid = _require_non_empty_str(execution_id, "execution_id")
        sc = _require_non_empty_str(scope, "scope", max_len=64)
        key = _require_non_empty_str(idempotency_key, "idempotency_key")
        fp = _require_non_empty_str(fingerprint, "fingerprint", max_len=8192)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                if self._try_get_execution_unlocked(eid) is None:
                    raise ExecutionNotFoundError(eid)
                row = self._connection.execute(
                    """
                    SELECT fingerprint FROM method_idempotency
                    WHERE execution_id = ? AND scope = ? AND idempotency_key = ?
                    """,
                    (eid, sc, key),
                ).fetchone()
                if row is not None and row["fingerprint"] != fp:
                    raise IdempotencyConflictError(scope=sc, idempotency_key=key)
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO method_idempotency (
                      execution_id, scope, idempotency_key, fingerprint,
                      result_json, journal_seq, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eid,
                        sc,
                        key,
                        fp,
                        canonical_json(dict(result)),
                        journal_seq,
                        self.now_ms(),
                    ),
                )
                self._connection.execute("COMMIT")
            except Exception:
                try:
                    self._connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise

    def update_start_idempotency_result(
        self,
        idempotency_key: str,
        result: Mapping[str, Any],
    ) -> None:
        """Refresh the cached start result after journal_record_cid is known."""

        key = _require_non_empty_str(idempotency_key, "idempotency_key")
        with self._lock:
            self._connection.execute(
                """
                UPDATE start_idempotency
                SET result_json = ?
                WHERE idempotency_key = ?
                """,
                (canonical_json(dict(result)), key),
            )

    # -- internals ---------------------------------------------------------

    def _upsert_timer_unlocked(
        self,
        execution_id: str,
        timer: Mapping[str, Any],
        journal_seq: int,
        now: int,
    ) -> None:
        timer_id = _require_non_empty_str(timer.get("timer_id"), "timer.timer_id")
        fire_at = _require_non_negative_int(timer.get("fire_at_ms"), "timer.fire_at_ms")
        status = _require_non_empty_str(timer.get("status", "scheduled"), "timer.status", max_len=32)
        payload_cid = timer.get("payload_cid")
        if payload_cid is not None:
            payload_cid = require_portable_cid(payload_cid, field="timer.payload_cid")
        existing = self._connection.execute(
            """
            SELECT 1 FROM timers WHERE execution_id = ? AND timer_id = ?
            """,
            (execution_id, timer_id),
        ).fetchone()
        if existing is None:
            self._connection.execute(
                """
                INSERT INTO timers (
                  execution_id, timer_id, fire_at_ms, status, payload_cid,
                  journal_seq, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    timer_id,
                    fire_at,
                    status,
                    payload_cid,
                    journal_seq,
                    now,
                    now,
                ),
            )
        else:
            self._connection.execute(
                """
                UPDATE timers SET
                  fire_at_ms = ?, status = ?, payload_cid = ?,
                  journal_seq = ?, updated_at_ms = ?
                WHERE execution_id = ? AND timer_id = ?
                """,
                (
                    fire_at,
                    status,
                    payload_cid,
                    journal_seq,
                    now,
                    execution_id,
                    timer_id,
                ),
            )

    def _try_get_execution_unlocked(self, execution_id: str) -> Optional[Dict[str, Any]]:
        row = self._connection.execute(
            "SELECT * FROM executions WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_execution(row)

    def _get_execution_unlocked(self, execution_id: str) -> Dict[str, Any]:
        execution = self._try_get_execution_unlocked(execution_id)
        if execution is None:
            raise ExecutionNotFoundError(execution_id)
        return execution

    def _row_to_execution(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "execution_id": row["execution_id"],
            "envelope_cid": row["envelope_cid"],
            "status": row["status"],
            "fencing_token": int(row["fencing_token"]),
            "journal_seq": int(row["journal_seq"]),
            "last_checkpoint_id": row["last_checkpoint_id"],
            "progress_cid": row["progress_cid"],
            "correlation_id": row["correlation_id"],
            "start_idempotency_key": row["start_idempotency_key"],
            "cancel_state": row["cancel_state"] or "none",
            "cancel_reason": row["cancel_reason"],
            "obligation_cids": list(_loads_json(row["obligation_cids_json"], [])),
            "attempt": int(row["attempt"] or 0),
            "receipt_cid": row["receipt_cid"],
            "result_cid": row["result_cid"],
            "parent_execution_id": row["parent_execution_id"],
            "executor_did": row["executor_did"],
            "last_event_cid": row["last_event_cid"],
            "last_record_cid": row["last_record_cid"],
            "created_at_ms": int(row["created_at_ms"]),
            "updated_at_ms": int(row["updated_at_ms"]),
        }

    def _project_from_records(
        self,
        execution_id: str,
        records: Sequence[Mapping[str, Any]],
        stored: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build a recoverable projection from ordered journal records."""

        if not records:
            return dict(stored)

        status = "running"
        fencing_token = int(records[0].get("fencing_token") or 1)
        last_checkpoint_id = None
        progress_cid = stored.get("progress_cid")
        cancel_state = "none"
        cancel_reason = None
        obligation_cids: List[str] = []
        attempt = 0
        receipt_cid = None
        result_cid = None
        last_event_cid = None
        last_record_cid = None
        journal_seq = 0
        envelope_cid = stored.get("envelope_cid") or records[0].get("envelope_cid")

        for record in records:
            journal_seq = int(record["journal_seq"])
            fencing_token = int(record.get("fencing_token") or fencing_token)
            transition = record.get("transition")
            last_record_cid = record.get("record_cid") or last_record_cid
            if record.get("event_cid"):
                last_event_cid = record["event_cid"]
            if record.get("checkpoint_id"):
                last_checkpoint_id = record["checkpoint_id"]
            if record.get("progress_cid"):
                progress_cid = record["progress_cid"]
            if record.get("result_cid"):
                result_cid = record["result_cid"]
            if record.get("receipt_cid"):
                receipt_cid = record["receipt_cid"]
            payload = record.get("payload") if isinstance(record.get("payload"), Mapping) else {}

            if transition == "started":
                status = "running"
            elif transition == "resumed":
                if status not in TERMINAL_STATUSES:
                    status = "running"
            elif transition == "signalled":
                pass
            elif transition == "cancel_requested":
                status = "cancelling"
                cancel_state = "cancelling"
                cancel_reason = (payload or {}).get("reason") or cancel_reason
            elif transition == "cancelled":
                status = "cancelled"
                cancel_state = "cancelled"
                cancel_reason = (payload or {}).get("reason") or cancel_reason
            elif transition == "checkpointed":
                if status not in TERMINAL_STATUSES and status != "cancelling":
                    status = "running"
            elif transition == "retried":
                attempt = int((payload or {}).get("attempt") or (attempt + 1))
                if status not in TERMINAL_STATUSES and cancel_state not in {
                    "cancelling",
                    "cancelled",
                }:
                    status = "running"
            elif transition == "timer_scheduled":
                if status == "running":
                    status = "paused"
            elif transition == "timer_fired":
                if status == "paused":
                    status = "running"
            elif transition == "compensation_started":
                status = "compensating"
            elif transition == "compensation_completed":
                status = str((payload or {}).get("status") or "compensated")
            elif transition == "recovered":
                # Keep status; recovered is a recovery marker.
                pass
            elif transition == "finalized":
                status = str((payload or {}).get("terminal_status") or status)
                if status not in TERMINAL_STATUSES:
                    status = "succeeded"

        return {
            "execution_id": execution_id,
            "envelope_cid": envelope_cid,
            "status": status,
            "fencing_token": fencing_token,
            "journal_seq": journal_seq,
            "last_checkpoint_id": last_checkpoint_id,
            "progress_cid": progress_cid,
            "correlation_id": stored.get("correlation_id"),
            "start_idempotency_key": stored.get("start_idempotency_key"),
            "cancel_state": cancel_state,
            "cancel_reason": cancel_reason,
            "obligation_cids": obligation_cids or list(stored.get("obligation_cids") or []),
            "attempt": attempt,
            "receipt_cid": receipt_cid,
            "result_cid": result_cid,
            "parent_execution_id": stored.get("parent_execution_id"),
            "executor_did": stored.get("executor_did"),
            "last_event_cid": last_event_cid,
            "last_record_cid": last_record_cid,
            "created_at_ms": stored.get("created_at_ms"),
            "updated_at_ms": stored.get("updated_at_ms"),
        }
