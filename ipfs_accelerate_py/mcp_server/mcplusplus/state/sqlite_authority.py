"""Durable single-authority DuckDB/Quack state with CAS, leases, fencing, and restart.

Interface label: ``SqliteAuthorityState@1`` (stable wire name)

This module is the primary MCP++ 1.0 backend for ``StateRef@1`` mode
``single_authority`` (ADR-0004 §2 / 2026-08-16 correction of plan KD-9):

* DuckDB file store, with best-effort local Quack/DuckLake ``LOAD``
* Compare-and-swap on a monotonic ``version``
* Exclusive leases and monotonic fencing tokens
* Restart recovery of every acknowledged committed write
* SQLite WAL fallback via ``MCPPLUSPLUS_SQL_ENGINE=sqlite``

The DuckDB database is the authority for this mode's live keyspace (distinct
from kit coordination storage, where immutable blocks are authority and a
local SQL index is rebuildable). Concurrent writers without a valid lease/fence
produce an explicit conflict — never a silent merge.

Crash-injection boundaries (``CAS_INTERRUPTION_POINTS``) exist only so restart
tests can model process death at durable seams. Production callers leave
``crash_injector`` unset.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ipfs_accelerate_py.mcp_server.mcplusplus.storage.engine import (
    EngineConnection,
    EngineError,
    connect_sql_engine,
)


SCHEMA_MARKER = "mcp++/state/state-ref@1"
CONSISTENCY_MODE = "single_authority"
PROVIDER_ID = "duckdb-authority"
SQLITE_PROVIDER_ID = "sqlite-authority"
INTERFACE_LABEL = "SqliteAuthorityState@1"

# Named seams for the declared crash matrix (MCPP-037 acceptance).
# Ordering mirrors a real CAS write: open txn → check expectations →
# stage durable rows → commit → return.
CAS_INTERRUPTION_POINTS: Tuple[str, ...] = (
    "before_transaction",
    "after_expectation_verification",
    "after_value_persist",
    "before_sqlite_commit",
    "after_sqlite_commit",
)

# Boundaries at which a write has not yet been acknowledged as committed.
# After ``commit()`` the write is durable even if the process dies before
# returning success to the caller.
PRE_COMMIT_INTERRUPTION_POINTS = frozenset(
    {
        "before_transaction",
        "after_expectation_verification",
        "after_value_persist",
        "before_sqlite_commit",
    }
)
POST_COMMIT_INTERRUPTION_POINTS = frozenset({"after_sqlite_commit"})


# ---------------------------------------------------------------------------
# Errors (fail closed)
# ---------------------------------------------------------------------------


class SqliteAuthorityError(RuntimeError):
    """Base error for the single-authority SQLite backend."""

    code = "sqlite_authority_error"


class StateNotFoundError(SqliteAuthorityError, KeyError):
    """Raised when a state id has no durable row."""

    code = "state_not_found"


class CasMismatchError(SqliteAuthorityError):
    """Expected version does not match the durable live version."""

    code = "cas_mismatch"

    def __init__(
        self,
        state_id: str,
        *,
        expected_version: int,
        actual_version: int,
        message: Optional[str] = None,
    ) -> None:
        self.state_id = state_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            message
            or (
                f"CAS mismatch for {state_id!r}: expected version "
                f"{expected_version}, durable version is {actual_version}"
            )
        )


class StaleFenceError(SqliteAuthorityError):
    """Presented fence token is lower than the highest accepted token."""

    code = "stale_fence"

    def __init__(
        self,
        state_id: str,
        *,
        presented_token: int,
        accepted_token: int,
        message: Optional[str] = None,
    ) -> None:
        self.state_id = state_id
        self.presented_token = presented_token
        self.accepted_token = accepted_token
        super().__init__(
            message
            or (
                f"stale fence for {state_id!r}: presented token "
                f"{presented_token} < accepted token {accepted_token}"
            )
        )


class LeaseError(SqliteAuthorityError):
    """Lease acquisition, renewal, or exclusive-write check failed."""

    code = "lease_error"


class IdempotencyConflictError(SqliteAuthorityError):
    """An operation id was reused with a different write payload."""

    code = "idempotency_conflict"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
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


def _loads_optional(raw: Optional[str]) -> Any:
    if raw is None:
        return None
    return json.loads(raw)


def _require_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_state_id(state_id: object) -> str:
    if not isinstance(state_id, str) or not state_id or len(state_id) > 256:
        raise ValueError("state_id must be a non-empty string up to 256 characters")
    return state_id


def _require_principal(principal: object, name: str = "principal") -> str:
    if not isinstance(principal, str) or not principal or len(principal) > 512:
        raise ValueError(f"{name} must be a non-empty principal string")
    return principal


def _normalize_fence(fence: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if fence is None:
        return None
    if not isinstance(fence, Mapping):
        raise ValueError("fence must be a mapping or null")
    token = _require_non_negative_int(fence.get("token"), "fence.token")
    out: Dict[str, Any] = {"token": token}
    if "epoch" in fence and fence["epoch"] is not None:
        out["epoch"] = _require_non_negative_int(fence["epoch"], "fence.epoch")
    if "issued_to" in fence and fence["issued_to"] is not None:
        out["issued_to"] = _require_principal(fence["issued_to"], "fence.issued_to")
    if "fence_cid" in fence and fence["fence_cid"] is not None:
        if not isinstance(fence["fence_cid"], str) or not fence["fence_cid"]:
            raise ValueError("fence.fence_cid must be a non-empty string when present")
        out["fence_cid"] = fence["fence_cid"]
    return out


def _normalize_lease(lease: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if lease is None:
        return None
    if not isinstance(lease, Mapping):
        raise ValueError("lease must be a mapping or null")
    holder = _require_principal(lease.get("holder"), "lease.holder")
    expires_at_ms = _require_non_negative_int(lease.get("expires_at_ms"), "lease.expires_at_ms")
    out: Dict[str, Any] = {"holder": holder, "expires_at_ms": expires_at_ms}
    if "issued_at_ms" in lease and lease["issued_at_ms"] is not None:
        out["issued_at_ms"] = _require_non_negative_int(lease["issued_at_ms"], "lease.issued_at_ms")
    if "epoch" in lease and lease["epoch"] is not None:
        out["epoch"] = _require_non_negative_int(lease["epoch"], "lease.epoch")
    if "lease_cid" in lease and lease["lease_cid"] is not None:
        if not isinstance(lease["lease_cid"], str) or not lease["lease_cid"]:
            raise ValueError("lease.lease_cid must be a non-empty string when present")
        out["lease_cid"] = lease["lease_cid"]
    if "renewable" in lease and lease["renewable"] is not None:
        if not isinstance(lease["renewable"], bool):
            raise ValueError("lease.renewable must be a boolean when present")
        out["renewable"] = lease["renewable"]
    return out


def _normalize_authority(authority: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if authority is None:
        return None
    if not isinstance(authority, Mapping):
        raise ValueError("authority must be a mapping or null")
    kind = authority.get("kind")
    if kind not in {"none", "principal", "lease_holder", "quorum", "plugin"}:
        raise ValueError("authority.kind is invalid")
    out: Dict[str, Any] = {"kind": kind}
    for key in ("principal", "plugin_id", "guarantee"):
        if key in authority and authority[key] is not None:
            if not isinstance(authority[key], str) or not authority[key]:
                raise ValueError(f"authority.{key} must be a non-empty string when present")
            out[key] = authority[key]
    if "principals" in authority and authority["principals"] is not None:
        principals = authority["principals"]
        if not isinstance(principals, Sequence) or isinstance(principals, (str, bytes)):
            raise ValueError("authority.principals must be a sequence of principals")
        out["principals"] = [ _require_principal(p, "authority.principals[]") for p in principals ]
    if "threshold" in authority and authority["threshold"] is not None:
        thr = authority["threshold"]
        if isinstance(thr, bool) or not isinstance(thr, int) or thr < 1:
            raise ValueError("authority.threshold must be a positive integer")
        out["threshold"] = thr
    return out


@dataclass(frozen=True)
class CasWriteResult:
    """Outcome of a successful compare-and-swap write."""

    state_id: str
    version: int
    epoch: int
    fence_token: int
    value: Any
    state_ref: Dict[str, Any]
    status: str = "updated"
    operation_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SqliteAuthorityState:
    """Single-authority StateRef store backed by DuckDB (primary) or SQLite.

    Parameters
    ----------
    db_path:
        Path to the database file. Parent directories are created.
    clock_ms:
        Optional callable returning unix epoch milliseconds (injectable for tests).
    crash_injector:
        Optional ``callable(boundary: str) -> None`` invoked at named durable
        seams. Raising from the injector models process death; recovery is the
        reopen path, not cleanup in the injector.
    engine:
        ``duckdb`` (default) or ``sqlite``.
    """

    DB_VERSION = 1

    def __init__(
        self,
        db_path: os.PathLike[str] | str,
        *,
        clock_ms: Optional[Callable[[], int]] = None,
        crash_injector: Optional[Callable[[str], None]] = None,
        engine: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._crash_injector = crash_injector
        self._lock = threading.RLock()
        self._requested_engine = engine
        self._connection = self._open_database()

    # -- lifecycle ---------------------------------------------------------

    def _open_database(self) -> EngineConnection:
        connection = connect_sql_engine(self.db_path, engine=self._requested_engine)
        try:
            self._create_schema(connection)
            return connection
        except Exception:
            connection.close()
            raise

    @classmethod
    def _create_schema(cls, connection: EngineConnection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS state_entries (
              state_id TEXT PRIMARY KEY,
              mode TEXT NOT NULL CHECK (mode = 'single_authority'),
              version INTEGER NOT NULL CHECK (version >= 0),
              epoch INTEGER NOT NULL DEFAULT 0 CHECK (epoch >= 0),
              fence_token INTEGER NOT NULL DEFAULT 0 CHECK (fence_token >= 0),
              value_json TEXT NOT NULL,
              root_cid TEXT,
              schema_cid TEXT,
              authority_json TEXT,
              lease_json TEXT,
              fence_json TEXT,
              parents_json TEXT,
              metadata_json TEXT,
              updated_at_ms INTEGER NOT NULL,
              created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS write_ops (
              operation_id TEXT PRIMARY KEY,
              state_id TEXT NOT NULL,
              expected_version INTEGER NOT NULL,
              new_version INTEGER NOT NULL,
              value_json TEXT NOT NULL,
              fence_token INTEGER NOT NULL,
              committed INTEGER NOT NULL CHECK (committed IN (0, 1)),
              created_at_ms INTEGER NOT NULL,
              FOREIGN KEY (state_id) REFERENCES state_entries(state_id)
            );

            CREATE INDEX IF NOT EXISTS write_ops_state
              ON write_ops(state_id, new_version);
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
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('provider', ?)",
            (
                PROVIDER_ID
                if getattr(connection, "engine", "duckdb") == "duckdb"
                else SQLITE_PROVIDER_ID,
            ),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('mode', ?)",
            (CONSISTENCY_MODE,),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SqliteAuthorityState":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @classmethod
    def open(
        cls,
        db_path: os.PathLike[str] | str,
        *,
        clock_ms: Optional[Callable[[], int]] = None,
        crash_injector: Optional[Callable[[str], None]] = None,
        engine: Optional[str] = None,
    ) -> "SqliteAuthorityState":
        """Open or re-open a durable store (restart recovery path)."""

        return cls(
            db_path,
            clock_ms=clock_ms,
            crash_injector=crash_injector,
            engine=engine,
        )

    def _interrupt(self, boundary: str) -> None:
        if self._crash_injector is not None:
            self._crash_injector(boundary)

    def _now_ms(self) -> int:
        return int(self._clock_ms())

    # -- reads -------------------------------------------------------------

    def get(self, state_id: str) -> Dict[str, Any]:
        """Return the durable live record for ``state_id``.

        Result keys: ``state_id``, ``version``, ``epoch``, ``fence_token``,
        ``value``, ``lease``, ``fence``, ``authority``, ``root_cid``,
        ``schema_cid``, ``parents``, ``metadata``, ``updated_at_ms``,
        ``created_at_ms``, ``state_ref``.
        """

        state_id = _require_state_id(state_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM state_entries WHERE state_id = ?",
                (state_id,),
            ).fetchone()
        if row is None:
            raise StateNotFoundError(f"state id not found: {state_id!r}")
        return self._row_to_record(row)

    def get_ref(self, state_id: str) -> Dict[str, Any]:
        """Return a ``StateRef@1``-shaped handle for the live value."""

        return self.get(state_id)["state_ref"]

    def list_ids(self) -> Tuple[str, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT state_id FROM state_entries ORDER BY state_id"
            ).fetchall()
        return tuple(str(r["state_id"]) for r in rows)

    def _row_to_record(self, row: Any) -> Dict[str, Any]:
        value = json.loads(row["value_json"])
        lease = _loads_optional(row["lease_json"])
        fence = _loads_optional(row["fence_json"])
        authority = _loads_optional(row["authority_json"])
        parents = _loads_optional(row["parents_json"]) or []
        metadata = _loads_optional(row["metadata_json"])
        provider = (
            PROVIDER_ID
            if self._connection.engine == "duckdb"
            else SQLITE_PROVIDER_ID
        )
        state_ref: Dict[str, Any] = {
            "schema": SCHEMA_MARKER,
            "id": row["state_id"],
            "mode": CONSISTENCY_MODE,
            "version": int(row["version"]),
            "epoch": int(row["epoch"]),
            "provider": provider,
            "root_cid": row["root_cid"],
            "parents": list(parents),
        }
        if row["schema_cid"] is not None:
            state_ref["schema_cid"] = row["schema_cid"]
        if authority is not None:
            state_ref["authority"] = authority
        if lease is not None:
            state_ref["lease"] = lease
        else:
            state_ref["lease"] = None
        if fence is not None:
            state_ref["fence"] = fence
        else:
            # Always expose a fence object when a token is active.
            token = int(row["fence_token"])
            state_ref["fence"] = {"token": token} if token > 0 else None
        if metadata is not None:
            state_ref["metadata"] = metadata
        return {
            "state_id": row["state_id"],
            "version": int(row["version"]),
            "epoch": int(row["epoch"]),
            "fence_token": int(row["fence_token"]),
            "value": value,
            "lease": lease,
            "fence": state_ref["fence"],
            "authority": authority,
            "root_cid": row["root_cid"],
            "schema_cid": row["schema_cid"],
            "parents": list(parents),
            "metadata": metadata,
            "updated_at_ms": int(row["updated_at_ms"]),
            "created_at_ms": int(row["created_at_ms"]),
            "state_ref": state_ref,
        }

    # -- create ------------------------------------------------------------

    def create(
        self,
        state_id: str,
        value: Any,
        *,
        authority: Optional[Mapping[str, Any]] = None,
        schema_cid: Optional[str] = None,
        root_cid: Optional[str] = None,
        parents: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        fence_token: int = 0,
        epoch: int = 0,
        lease: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new single-authority state entry at version 0.

        Fails closed if ``state_id`` already exists.
        """

        state_id = _require_state_id(state_id)
        epoch = _require_non_negative_int(epoch, "epoch")
        fence_token = _require_non_negative_int(fence_token, "fence_token")
        authority_n = _normalize_authority(authority)
        lease_n = _normalize_lease(lease)
        value_json = _canonical_json(value)
        parents_list = list(parents or [])
        for p in parents_list:
            if not isinstance(p, str) or not p:
                raise ValueError("parents must be non-empty strings")
        now = self._now_ms()
        fence_obj: Optional[Dict[str, Any]]
        if fence_token > 0:
            fence_obj = {"token": fence_token, "epoch": epoch}
            if authority_n and authority_n.get("principal"):
                fence_obj["issued_to"] = authority_n["principal"]
        else:
            fence_obj = None

        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT state_id FROM state_entries WHERE state_id = ?",
                    (state_id,),
                ).fetchone()
                if existing is not None:
                    conn.execute("ROLLBACK")
                    raise CasMismatchError(
                        state_id,
                        expected_version=-1,
                        actual_version=self.get(state_id)["version"],
                        message=f"state id already exists: {state_id!r}",
                    )
                conn.execute(
                    """
                    INSERT INTO state_entries (
                      state_id, mode, version, epoch, fence_token, value_json,
                      root_cid, schema_cid, authority_json, lease_json, fence_json,
                      parents_json, metadata_json, updated_at_ms, created_at_ms
                    ) VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state_id,
                        CONSISTENCY_MODE,
                        epoch,
                        fence_token,
                        value_json,
                        root_cid,
                        schema_cid,
                        _canonical_json(authority_n) if authority_n is not None else None,
                        _canonical_json(lease_n) if lease_n is not None else None,
                        _canonical_json(fence_obj) if fence_obj is not None else None,
                        _canonical_json(parents_list),
                        _canonical_json(dict(metadata)) if metadata is not None else None,
                        now,
                        now,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except EngineError:
                    pass
                raise
        return self.get(state_id)

    # -- CAS write ---------------------------------------------------------

    def cas_write(
        self,
        state_id: str,
        *,
        expected_version: int,
        value: Any,
        fence_token: Optional[int] = None,
        writer: Optional[str] = None,
        operation_id: Optional[str] = None,
        root_cid: Optional[str] = None,
        parents: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        advance_fence: bool = False,
    ) -> CasWriteResult:
        """Compare-and-swap the live value under version and fence preconditions.

        Rules (fail closed):

        1. ``expected_version`` MUST equal the durable version (else
           :class:`CasMismatchError`).
        2. If a fence is active and ``fence_token`` is presented, it MUST be
           ``>=`` the accepted token; a lower token raises
           :class:`StaleFenceError`.
        3. If an exclusive lease is held and not expired, ``writer`` MUST be
           the lease holder (else :class:`LeaseError`).
        4. ``operation_id``, when provided, is an idempotency key: replaying the
           same payload after a committed write returns ``status="unchanged"``.
        """

        state_id = _require_state_id(state_id)
        expected_version = _require_non_negative_int(expected_version, "expected_version")
        if fence_token is not None:
            fence_token = _require_non_negative_int(fence_token, "fence_token")
        if writer is not None:
            writer = _require_principal(writer, "writer")
        if operation_id is not None:
            if not isinstance(operation_id, str) or not operation_id or len(operation_id) > 256:
                raise ValueError("operation_id must be a non-empty string up to 256 characters")
        value_json = _canonical_json(value)
        parents_list = list(parents) if parents is not None else None
        if parents_list is not None:
            for p in parents_list:
                if not isinstance(p, str) or not p:
                    raise ValueError("parents must be non-empty strings")

        self._interrupt("before_transaction")

        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM state_entries WHERE state_id = ?",
                    (state_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise StateNotFoundError(f"state id not found: {state_id!r}")

                # Idempotent replay of a previously committed operation.
                if operation_id is not None:
                    prior = conn.execute(
                        "SELECT * FROM write_ops WHERE operation_id = ?",
                        (operation_id,),
                    ).fetchone()
                    if prior is not None:
                        if int(prior["committed"]) != 1:
                            # Uncommitted staging should not survive a rollback;
                            # treat as absent and continue a fresh attempt.
                            pass
                        else:
                            same = (
                                prior["state_id"] == state_id
                                and int(prior["expected_version"]) == expected_version
                                and prior["value_json"] == value_json
                            )
                            if not same:
                                conn.execute("ROLLBACK")
                                raise IdempotencyConflictError(
                                    f"operation_id {operation_id!r} reused with different payload"
                                )
                            conn.execute("ROLLBACK")
                            record = self.get(state_id)
                            return CasWriteResult(
                                state_id=state_id,
                                version=int(record["version"]),
                                epoch=int(record["epoch"]),
                                fence_token=int(record["fence_token"]),
                                value=record["value"],
                                state_ref=record["state_ref"],
                                status="unchanged",
                                operation_id=operation_id,
                            )

                actual_version = int(row["version"])
                accepted_fence = int(row["fence_token"])
                epoch = int(row["epoch"])
                lease = _loads_optional(row["lease_json"])
                now = self._now_ms()

                # Fence check before CAS so stale writers fail closed even when
                # versions happen to match after a reclaim race.
                if fence_token is not None and fence_token < accepted_fence:
                    conn.execute("ROLLBACK")
                    raise StaleFenceError(
                        state_id,
                        presented_token=fence_token,
                        accepted_token=accepted_fence,
                    )

                # When a fence is active, writers must present a token.
                if accepted_fence > 0 and fence_token is None:
                    conn.execute("ROLLBACK")
                    raise StaleFenceError(
                        state_id,
                        presented_token=-1,
                        accepted_token=accepted_fence,
                        message=(
                            f"fence required for {state_id!r}: "
                            f"accepted token {accepted_fence}, none presented"
                        ),
                    )

                if actual_version != expected_version:
                    conn.execute("ROLLBACK")
                    raise CasMismatchError(
                        state_id,
                        expected_version=expected_version,
                        actual_version=actual_version,
                    )

                if lease is not None:
                    expires = int(lease["expires_at_ms"])
                    holder = str(lease["holder"])
                    if expires > now:
                        if writer is None or writer != holder:
                            conn.execute("ROLLBACK")
                            raise LeaseError(
                                f"exclusive lease held by {holder!r} until {expires}; "
                                f"writer {writer!r} is not authorized"
                            )

                self._interrupt("after_expectation_verification")

                new_version = actual_version + 1
                new_fence_token = accepted_fence
                fence_obj = _loads_optional(row["fence_json"])
                if advance_fence:
                    new_fence_token = accepted_fence + 1
                    fence_obj = {"token": new_fence_token, "epoch": epoch}
                    if writer is not None:
                        fence_obj["issued_to"] = writer
                elif fence_token is not None and fence_token > accepted_fence:
                    # Accept a higher presented fence (reclaim / epoch advance).
                    new_fence_token = fence_token
                    fence_obj = {"token": new_fence_token, "epoch": epoch}
                    if writer is not None:
                        fence_obj["issued_to"] = writer
                elif fence_obj is None and new_fence_token > 0:
                    fence_obj = {"token": new_fence_token, "epoch": epoch}

                parents_json = (
                    _canonical_json(parents_list)
                    if parents_list is not None
                    else row["parents_json"]
                )
                metadata_json = (
                    _canonical_json(dict(metadata))
                    if metadata is not None
                    else row["metadata_json"]
                )
                new_root_cid = root_cid if root_cid is not None else row["root_cid"]

                updated = conn.execute(
                    """
                    UPDATE state_entries SET
                      version = ?,
                      fence_token = ?,
                      value_json = ?,
                      root_cid = ?,
                      fence_json = ?,
                      parents_json = ?,
                      metadata_json = ?,
                      updated_at_ms = ?
                    WHERE state_id = ? AND version = ?
                    RETURNING state_id
                    """,
                    (
                        new_version,
                        new_fence_token,
                        value_json,
                        new_root_cid,
                        _canonical_json(fence_obj) if fence_obj is not None else None,
                        parents_json,
                        metadata_json,
                        now,
                        state_id,
                        expected_version,
                    ),
                ).fetchone()
                if updated is None:
                    conn.execute("ROLLBACK")
                    # Lost the row-level race inside the transaction (should be
                    # rare under BEGIN IMMEDIATE); surface as CAS mismatch.
                    fresh = conn.execute(
                        "SELECT version FROM state_entries WHERE state_id = ?",
                        (state_id,),
                    ).fetchone()
                    actual = int(fresh["version"]) if fresh is not None else -1
                    raise CasMismatchError(
                        state_id,
                        expected_version=expected_version,
                        actual_version=actual,
                    )

                if operation_id is not None:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO write_ops (
                          operation_id, state_id, expected_version, new_version,
                          value_json, fence_token, committed, created_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                        """,
                        (
                            operation_id,
                            state_id,
                            expected_version,
                            new_version,
                            value_json,
                            new_fence_token,
                            now,
                        ),
                    )

                self._interrupt("after_value_persist")
                self._interrupt("before_sqlite_commit")
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except EngineError:
                    pass
                raise

            # Commit has returned: the write is durable. A crash injector that
            # fires here models death after durability but before caller ack.
            # Do not ROLLBACK — there is no open transaction.
            self._interrupt("after_sqlite_commit")

        record = self.get(state_id)
        return CasWriteResult(
            state_id=state_id,
            version=int(record["version"]),
            epoch=int(record["epoch"]),
            fence_token=int(record["fence_token"]),
            value=record["value"],
            state_ref=record["state_ref"],
            status="updated",
            operation_id=operation_id,
        )

    # -- leases & fences ---------------------------------------------------

    def acquire_lease(
        self,
        state_id: str,
        *,
        holder: str,
        ttl_ms: int,
        fence_token: Optional[int] = None,
        renewable: bool = True,
    ) -> Dict[str, Any]:
        """Acquire or reclaim an exclusive write lease.

        If a non-expired lease is held by another principal, fails closed.
        Reclaim after expiry advances ``epoch`` and issues a new fence token
        so stale writers fail closed (Profile G ``G_STALE_FENCE`` alignment).
        """

        state_id = _require_state_id(state_id)
        holder = _require_principal(holder, "holder")
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms < 1:
            raise ValueError("ttl_ms must be a positive integer")
        if fence_token is not None:
            fence_token = _require_non_negative_int(fence_token, "fence_token")

        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM state_entries WHERE state_id = ?",
                    (state_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise StateNotFoundError(f"state id not found: {state_id!r}")

                now = self._now_ms()
                lease = _loads_optional(row["lease_json"])
                epoch = int(row["epoch"])
                accepted_fence = int(row["fence_token"])

                if fence_token is not None and fence_token < accepted_fence:
                    conn.execute("ROLLBACK")
                    raise StaleFenceError(
                        state_id,
                        presented_token=fence_token,
                        accepted_token=accepted_fence,
                    )

                if lease is not None and int(lease["expires_at_ms"]) > now:
                    if str(lease["holder"]) != holder:
                        conn.execute("ROLLBACK")
                        raise LeaseError(
                            f"lease held by {lease['holder']!r} until "
                            f"{lease['expires_at_ms']}; cannot acquire for {holder!r}"
                        )
                    # Same holder re-acquire while lease is live (acts as renew).
                    if lease.get("renewable", True) is False:
                        conn.execute("ROLLBACK")
                        raise LeaseError("lease is not renewable")
                    new_epoch = epoch
                    new_fence = accepted_fence if accepted_fence > 0 else 1
                elif lease is not None and int(lease["expires_at_ms"]) <= now:
                    # Reclaim after expiry: bump epoch and fence so stale writers fail closed.
                    new_epoch = epoch + 1
                    new_fence = accepted_fence + 1
                else:
                    # First acquire on this id.
                    new_epoch = epoch if epoch > 0 else 1
                    new_fence = accepted_fence if accepted_fence > 0 else 1

                new_lease = {
                    "holder": holder,
                    "issued_at_ms": now,
                    "expires_at_ms": now + ttl_ms,
                    "epoch": new_epoch,
                    "renewable": bool(renewable),
                }
                fence_obj = {
                    "token": new_fence,
                    "epoch": new_epoch,
                    "issued_to": holder,
                }
                authority = {
                    "kind": "lease_holder",
                    "principal": holder,
                }
                conn.execute(
                    """
                    UPDATE state_entries SET
                      epoch = ?,
                      fence_token = ?,
                      lease_json = ?,
                      fence_json = ?,
                      authority_json = ?,
                      updated_at_ms = ?
                    WHERE state_id = ?
                    """,
                    (
                        new_epoch,
                        new_fence,
                        _canonical_json(new_lease),
                        _canonical_json(fence_obj),
                        _canonical_json(authority),
                        now,
                        state_id,
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except EngineError:
                    pass
                raise
        return self.get(state_id)

    def renew_lease(
        self,
        state_id: str,
        *,
        holder: str,
        ttl_ms: int,
        fence_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Renew an unexpired lease for the same holder."""

        state_id = _require_state_id(state_id)
        holder = _require_principal(holder, "holder")
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms < 1:
            raise ValueError("ttl_ms must be a positive integer")
        if fence_token is not None:
            fence_token = _require_non_negative_int(fence_token, "fence_token")

        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM state_entries WHERE state_id = ?",
                    (state_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise StateNotFoundError(f"state id not found: {state_id!r}")

                now = self._now_ms()
                lease = _loads_optional(row["lease_json"])
                accepted_fence = int(row["fence_token"])
                if lease is None:
                    conn.execute("ROLLBACK")
                    raise LeaseError("no lease to renew")
                if str(lease["holder"]) != holder:
                    conn.execute("ROLLBACK")
                    raise LeaseError(
                        f"lease held by {lease['holder']!r}; {holder!r} cannot renew"
                    )
                if int(lease["expires_at_ms"]) <= now:
                    conn.execute("ROLLBACK")
                    raise LeaseError("lease expired; acquire a new lease instead of renewing")
                if lease.get("renewable", True) is False:
                    conn.execute("ROLLBACK")
                    raise LeaseError("lease is not renewable")
                if fence_token is not None and fence_token < accepted_fence:
                    conn.execute("ROLLBACK")
                    raise StaleFenceError(
                        state_id,
                        presented_token=fence_token,
                        accepted_token=accepted_fence,
                    )

                new_lease = dict(lease)
                new_lease["expires_at_ms"] = now + ttl_ms
                conn.execute(
                    """
                    UPDATE state_entries SET lease_json = ?, updated_at_ms = ?
                    WHERE state_id = ?
                    """,
                    (_canonical_json(new_lease), now, state_id),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except EngineError:
                    pass
                raise
        return self.get(state_id)

    def release_lease(
        self,
        state_id: str,
        *,
        holder: str,
        fence_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Release the exclusive lease if held by ``holder``."""

        state_id = _require_state_id(state_id)
        holder = _require_principal(holder, "holder")
        if fence_token is not None:
            fence_token = _require_non_negative_int(fence_token, "fence_token")

        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM state_entries WHERE state_id = ?",
                    (state_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise StateNotFoundError(f"state id not found: {state_id!r}")

                lease = _loads_optional(row["lease_json"])
                accepted_fence = int(row["fence_token"])
                if lease is None:
                    conn.execute("ROLLBACK")
                    raise LeaseError("no lease to release")
                if str(lease["holder"]) != holder:
                    conn.execute("ROLLBACK")
                    raise LeaseError(
                        f"lease held by {lease['holder']!r}; {holder!r} cannot release"
                    )
                if fence_token is not None and fence_token < accepted_fence:
                    conn.execute("ROLLBACK")
                    raise StaleFenceError(
                        state_id,
                        presented_token=fence_token,
                        accepted_token=accepted_fence,
                    )
                now = self._now_ms()
                conn.execute(
                    """
                    UPDATE state_entries SET lease_json = NULL, updated_at_ms = ?
                    WHERE state_id = ?
                    """,
                    (now, state_id),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except EngineError:
                    pass
                raise
        return self.get(state_id)

    def issue_fence(
        self,
        state_id: str,
        *,
        issued_to: Optional[str] = None,
        min_token: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Advance the fencing token (reclaim / epoch fence).

        The new token is ``max(accepted + 1, min_token or 0)``. Writes that
        present a lower token fail closed with :class:`StaleFenceError`.
        """

        state_id = _require_state_id(state_id)
        if issued_to is not None:
            issued_to = _require_principal(issued_to, "issued_to")
        if min_token is not None:
            min_token = _require_non_negative_int(min_token, "min_token")

        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM state_entries WHERE state_id = ?",
                    (state_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise StateNotFoundError(f"state id not found: {state_id!r}")
                accepted = int(row["fence_token"])
                epoch = int(row["epoch"])
                new_token = accepted + 1
                if min_token is not None and min_token > new_token:
                    new_token = min_token
                fence_obj: Dict[str, Any] = {"token": new_token, "epoch": epoch}
                if issued_to is not None:
                    fence_obj["issued_to"] = issued_to
                now = self._now_ms()
                conn.execute(
                    """
                    UPDATE state_entries SET
                      fence_token = ?, fence_json = ?, updated_at_ms = ?
                    WHERE state_id = ?
                    """,
                    (new_token, _canonical_json(fence_obj), now, state_id),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except EngineError:
                    pass
                raise
        return self.get(state_id)

    # -- diagnostics -------------------------------------------------------

    def journal_mode(self) -> str:
        """Return ``duckdb`` for the primary engine, else SQLite ``wal``."""

        with self._lock:
            if self._connection.engine == "duckdb":
                return "duckdb"
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    @property
    def engine(self) -> str:
        return self._connection.engine

    @property
    def loaded_extensions(self) -> tuple[str, ...]:
        return self._connection.loaded_extensions

    def db_version(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key = 'version'"
            ).fetchone()
        return int(row["value"]) if row is not None else 0


__all__ = [
    "CAS_INTERRUPTION_POINTS",
    "CONSISTENCY_MODE",
    "INTERFACE_LABEL",
    "POST_COMMIT_INTERRUPTION_POINTS",
    "PRE_COMMIT_INTERRUPTION_POINTS",
    "PROVIDER_ID",
    "SCHEMA_MARKER",
    "CasMismatchError",
    "CasWriteResult",
    "IdempotencyConflictError",
    "LeaseError",
    "SqliteAuthorityError",
    "SqliteAuthorityState",
    "DuckDBAuthorityState",
    "StateNotFoundError",
    "StaleFenceError",
]


DuckDBAuthorityState = SqliteAuthorityState
