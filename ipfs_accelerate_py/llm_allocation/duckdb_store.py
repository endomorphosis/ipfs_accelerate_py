"""DuckDB persistence for LLM allocation observations.

Importing this module does not open a database. Writes are fail-soft: a
missing DuckDB install or IO error never raises into generate_text.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from .limits import PROVIDER_LIMIT_HINTS, ProviderLimitHint
from .observations import CallObservation
from .paths import path_for_provider

_LOCK = threading.RLock()
_SCHEMA = """
CREATE TABLE IF NOT EXISTS call_observations (
    ts TIMESTAMP,
    provider VARCHAR,
    protocol VARCHAR,
    model VARCHAR,
    success BOOLEAN,
    latency_ms DOUBLE,
    error_kind VARCHAR,
    retryable BOOLEAN,
    status_code INTEGER,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    cached_tokens BIGINT,
    estimated_cost_usd DOUBLE,
    remaining_requests BIGINT,
    remaining_tokens BIGINT,
    limit_requests BIGINT,
    limit_tokens BIGINT,
    session_id VARCHAR,
    path VARCHAR,
    tokens_per_second DOUBLE
);
CREATE TABLE IF NOT EXISTS provider_limits (
    provider VARCHAR PRIMARY KEY,
    protocol VARCHAR,
    rpm BIGINT,
    tpm BIGINT,
    input_usd_per_1m DOUBLE,
    output_usd_per_1m DOUBLE,
    cached_input_usd_per_1m DOUBLE,
    source VARCHAR,
    updated_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS provider_health (
    provider VARCHAR PRIMARY KEY,
    protocol VARCHAR,
    available BOOLEAN,
    authenticated BOOLEAN,
    ready BOOLEAN,
    reason VARCHAR,
    probed_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS allocation_sessions (
    session_id VARCHAR PRIMARY KEY,
    path VARCHAR,
    preferred_provider VARCHAR,
    last_provider VARCHAR,
    last_model VARCHAR,
    last_error_kind VARCHAR,
    call_count BIGINT,
    success_count BIGINT,
    fail_count BIGINT,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    total_cost_usd DOUBLE,
    total_latency_ms DOUBLE,
    avg_latency_ms DOUBLE,
    tokens_per_second DOUBLE,
    provider_metadata VARCHAR,
    handoff_context VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cli_sessions (
    allocation_session_id VARCHAR,
    provider VARCHAR,
    native_session_id VARCHAR,
    resume_style VARCHAR,
    workspace VARCHAR,
    call_count BIGINT,
    metadata VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (allocation_session_id, provider)
);
CREATE TABLE IF NOT EXISTS cli_session_aliases (
    native_session_id VARCHAR,
    provider VARCHAR,
    allocation_session_id VARCHAR,
    current_provider VARCHAR,
    current_native_session_id VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (native_session_id, provider)
);
CREATE TABLE IF NOT EXISTS api_key_slots (
    backend VARCHAR,
    slot_id VARCHAR,
    key_fingerprint VARCHAR,
    secret_name VARCHAR,
    label VARCHAR,
    enabled BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    PRIMARY KEY (backend, slot_id)
);
CREATE TABLE IF NOT EXISTS api_key_stats (
    backend VARCHAR,
    slot_id VARCHAR,
    call_count BIGINT,
    success_count BIGINT,
    fail_count BIGINT,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    total_cost_usd DOUBLE,
    total_latency_ms DOUBLE,
    avg_latency_ms DOUBLE,
    tokens_per_second DOUBLE,
    last_error_kind VARCHAR,
    remaining_requests BIGINT,
    remaining_tokens BIGINT,
    remaining_balance_usd DOUBLE,
    updated_at TIMESTAMP,
    PRIMARY KEY (backend, slot_id)
);
CREATE TABLE IF NOT EXISTS session_key_bindings (
    session_id VARCHAR PRIMARY KEY,
    backend VARCHAR,
    slot_id VARCHAR,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
"""
_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_obs_session ON call_observations(session_id);
CREATE INDEX IF NOT EXISTS idx_obs_path_provider ON call_observations(path, provider);
CREATE INDEX IF NOT EXISTS idx_cli_sessions_alloc ON cli_sessions(allocation_session_id);
CREATE INDEX IF NOT EXISTS idx_cli_alias_alloc ON cli_session_aliases(allocation_session_id);
CREATE INDEX IF NOT EXISTS idx_cli_alias_current ON cli_session_aliases(current_native_session_id);
CREATE INDEX IF NOT EXISTS idx_api_key_slots_fp ON api_key_slots(key_fingerprint);
CREATE INDEX IF NOT EXISTS idx_session_key_backend ON session_key_bindings(backend, slot_id);
"""
_OBS_EXTRA_COLUMNS = (
    ("session_id", "VARCHAR"),
    ("path", "VARCHAR"),
    ("tokens_per_second", "DOUBLE"),
)
_SESSION_EXTRA_COLUMNS = (
    ("provider_metadata", "VARCHAR"),
    ("handoff_context", "VARCHAR"),
)
_MAX_HANDOFF_CHARS = 80000
_CLI_SESSION_EXTRA_COLUMNS = (
    ("metadata", "VARCHAR"),
)
_METADATA_BLOCKLIST = frozenset(
    {
        "prompt",
        "input",
        "user_prompt",
        "user_input",
        "raw_response",
        "stdout",
        "stderr",
        "api_key",
        "token",
        "authorization",
        "credential",
        "secret",
        "password",
    }
)
_MAX_METADATA_JSON_CHARS = 8192


def sanitize_session_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Drop prompts/secrets and bound values for session persistence."""
    clean: dict[str, Any] = {}
    if not payload:
        return clean
    for key, value in dict(payload).items():
        k = str(key or "").strip()
        lowered = k.lower()
        if not k or lowered in _METADATA_BLOCKLIST:
            continue
        if lowered in {"api_key_slot", "api_key_fingerprint"}:
            pass
        elif any(
            marker in lowered
            for marker in ("api_key", "password", "secret", "authorization", "credential")
        ):
            continue
        if isinstance(value, bool):
            clean[k] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            clean[k] = value
        elif isinstance(value, float):
            clean[k] = value
        else:
            text = str(value if value is not None else "").strip()
            if not text:
                continue
            clean[k] = text[:256]
        if len(clean) >= 48:
            break
    return clean


def _metadata_json(payload: Mapping[str, Any] | None) -> str:
    clean = sanitize_session_metadata(payload)
    if not clean:
        return ""
    encoded = json.dumps(clean, separators=(",", ":"), sort_keys=True, default=str)
    return encoded[:_MAX_METADATA_JSON_CHARS]


def _parse_metadata_json(raw: Any) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return sanitize_session_metadata(payload) if isinstance(payload, dict) else {}


def default_allocation_db_path() -> Path:
    override = str(os.environ.get("IPFS_ACCELERATE_LLM_ALLOCATION_DB") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / ".ipfs_accelerate" / "llm_allocation.duckdb"


def llm_observe_enabled() -> bool:
    raw = str(os.environ.get("IPFS_ACCELERATE_LLM_OBSERVE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def llm_allocate_enabled() -> bool:
    raw = str(os.environ.get("IPFS_ACCELERATE_LLM_ALLOCATE", "1")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


class AllocationStore:
    """Thin DuckDB wrapper. Missing duckdb degrades to a no-op."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else default_allocation_db_path()
        self._ready = False
        self._duckdb = None

    def _connect(self):
        if self._duckdb is None:
            try:
                import duckdb  # type: ignore
            except ImportError:
                return None
            self._duckdb = duckdb
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._duckdb.connect(str(self.path))
        if not self._ready:
            conn.execute(_SCHEMA)
            self._migrate_observations(conn)
            conn.execute(_INDEXES)
            self._seed_limits(conn)
            self._ready = True
        return conn

    def _migrate_observations(self, conn: Any) -> None:
        for name, type_name in _OBS_EXTRA_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE call_observations ADD COLUMN IF NOT EXISTS {name} {type_name}"
                )
            except Exception:
                continue
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cli_sessions (
                    allocation_session_id VARCHAR,
                    provider VARCHAR,
                    native_session_id VARCHAR,
                    resume_style VARCHAR,
                    workspace VARCHAR,
                    call_count BIGINT,
                    metadata VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (allocation_session_id, provider)
                )
                """
            )
        except Exception:
            pass
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_key_slots (
                    backend VARCHAR,
                    slot_id VARCHAR,
                    key_fingerprint VARCHAR,
                    secret_name VARCHAR,
                    label VARCHAR,
                    enabled BOOLEAN,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (backend, slot_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_key_stats (
                    backend VARCHAR,
                    slot_id VARCHAR,
                    call_count BIGINT,
                    success_count BIGINT,
                    fail_count BIGINT,
                    prompt_tokens BIGINT,
                    completion_tokens BIGINT,
                    total_tokens BIGINT,
                    total_cost_usd DOUBLE,
                    total_latency_ms DOUBLE,
                    avg_latency_ms DOUBLE,
                    tokens_per_second DOUBLE,
                    last_error_kind VARCHAR,
                    remaining_requests BIGINT,
                    remaining_tokens BIGINT,
                    remaining_balance_usd DOUBLE,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (backend, slot_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_key_bindings (
                    session_id VARCHAR PRIMARY KEY,
                    backend VARCHAR,
                    slot_id VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )
        except Exception:
            pass
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cli_session_aliases (
                    native_session_id VARCHAR,
                    provider VARCHAR,
                    allocation_session_id VARCHAR,
                    current_provider VARCHAR,
                    current_native_session_id VARCHAR,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (native_session_id, provider)
                )
                """
            )
        except Exception:
            pass
        for name, type_name in _SESSION_EXTRA_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE allocation_sessions ADD COLUMN IF NOT EXISTS {name} {type_name}"
                )
            except Exception:
                continue
        for name, type_name in _CLI_SESSION_EXTRA_COLUMNS:
            try:
                conn.execute(
                    f"ALTER TABLE cli_sessions ADD COLUMN IF NOT EXISTS {name} {type_name}"
                )
            except Exception:
                continue

    def _seed_limits(self, conn: Any) -> None:
        now = datetime.now(timezone.utc)
        for hint in PROVIDER_LIMIT_HINTS.values():
            conn.execute(
                """
                INSERT INTO provider_limits
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (provider) DO NOTHING
                """,
                [
                    hint.provider,
                    hint.protocol,
                    hint.rpm,
                    hint.tpm,
                    hint.input_usd_per_1m,
                    hint.output_usd_per_1m,
                    hint.cached_input_usd_per_1m,
                    hint.source,
                    now,
                ],
            )

    def record(self, observation: CallObservation) -> None:
        if not llm_observe_enabled():
            return
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                try:
                    ts_value = datetime.fromisoformat(
                        str(observation.ts).replace("Z", "+00:00")
                    )
                except (TypeError, ValueError):
                    ts_value = datetime.now(timezone.utc)
                path = str(observation.path or path_for_provider(observation.provider))
                latency_ms = float(observation.latency_ms)
                completion = int(observation.completion_tokens)
                total = int(observation.total_tokens)
                throughput_tokens = completion or total
                tps = float(observation.tokens_per_second)
                if tps <= 0 and latency_ms > 0 and throughput_tokens > 0:
                    tps = throughput_tokens / (latency_ms / 1000.0)
                session_id = str(observation.session_id or "").strip()[:128]
                conn.execute(
                    """
                    INSERT INTO call_observations (
                        ts, provider, protocol, model, success, latency_ms,
                        error_kind, retryable, status_code, prompt_tokens,
                        completion_tokens, total_tokens, cached_tokens,
                        estimated_cost_usd, remaining_requests, remaining_tokens,
                        limit_requests, limit_tokens, session_id, path,
                        tokens_per_second
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        ts_value,
                        observation.provider,
                        observation.protocol,
                        observation.model,
                        bool(observation.success),
                        latency_ms,
                        observation.error_kind,
                        bool(observation.retryable),
                        observation.status_code,
                        int(observation.prompt_tokens),
                        completion,
                        total,
                        int(observation.cached_tokens),
                        float(observation.estimated_cost_usd),
                        observation.remaining_requests,
                        observation.remaining_tokens,
                        observation.limit_requests,
                        observation.limit_tokens,
                        session_id,
                        path,
                        tps,
                    ],
                )
                if session_id:
                    extra = sanitize_session_metadata(
                        getattr(observation, "extra_metadata", None)
                    )
                    self._upsert_session(
                        conn,
                        session_id=session_id,
                        path=path,
                        observation=observation,
                        tokens_per_second=tps,
                        latency_ms=latency_ms,
                        ts_value=ts_value,
                        extra_metadata=extra,
                    )
                    native = str(
                        extra.get("session_id")
                        or extra.get("muse_session_id")
                        or extra.get("native_session_id")
                        or ""
                    ).strip()
                    if native:
                        self._upsert_cli_session_row(
                            conn,
                            session_id=session_id,
                            provider=observation.provider,
                            native_session_id=native,
                            resume_style="",
                            workspace="",
                            metadata=extra,
                            ts_value=ts_value,
                        )
                extra_all = sanitize_session_metadata(
                    getattr(observation, "extra_metadata", None)
                )
                slot = str(extra_all.get("api_key_slot") or "").strip()
                if slot:
                    self._upsert_api_key_stats_row(
                        conn,
                        observation=observation,
                        slot_id=slot,
                        ts_value=ts_value,
                    )
            finally:
                conn.close()

    def _upsert_session(
        self,
        conn: Any,
        *,
        session_id: str,
        path: str,
        observation: CallObservation,
        tokens_per_second: float,
        latency_ms: float,
        ts_value: datetime,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        success = 1 if observation.success else 0
        fail = 0 if observation.success else 1
        meta_json = _metadata_json(extra_metadata)
        conn.execute(
            """
            INSERT INTO allocation_sessions (
                session_id, path, preferred_provider, last_provider, last_model,
                last_error_kind, call_count, success_count, fail_count,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd,
                total_latency_ms, avg_latency_ms, tokens_per_second,
                provider_metadata, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            ON CONFLICT (session_id) DO UPDATE SET
                path = CASE
                    WHEN excluded.path <> '' THEN excluded.path ELSE allocation_sessions.path
                END,
                preferred_provider = CASE
                    WHEN excluded.success_count = 1 THEN excluded.last_provider
                    ELSE allocation_sessions.preferred_provider
                END,
                last_provider = excluded.last_provider,
                last_model = excluded.last_model,
                last_error_kind = excluded.last_error_kind,
                call_count = allocation_sessions.call_count + 1,
                success_count = allocation_sessions.success_count + excluded.success_count,
                fail_count = allocation_sessions.fail_count + excluded.fail_count,
                prompt_tokens = allocation_sessions.prompt_tokens + excluded.prompt_tokens,
                completion_tokens = allocation_sessions.completion_tokens
                    + excluded.completion_tokens,
                total_tokens = allocation_sessions.total_tokens + excluded.total_tokens,
                total_cost_usd = allocation_sessions.total_cost_usd + excluded.total_cost_usd,
                total_latency_ms = allocation_sessions.total_latency_ms
                    + excluded.total_latency_ms,
                avg_latency_ms = (allocation_sessions.total_latency_ms
                    + excluded.total_latency_ms)
                    / (allocation_sessions.call_count + 1),
                tokens_per_second = CASE
                    WHEN (allocation_sessions.total_latency_ms + excluded.total_latency_ms) > 0
                    THEN (allocation_sessions.total_tokens + excluded.total_tokens)
                        / ((allocation_sessions.total_latency_ms + excluded.total_latency_ms)
                            / 1000.0)
                    ELSE 0
                END,
                provider_metadata = CASE
                    WHEN excluded.provider_metadata <> '' THEN excluded.provider_metadata
                    ELSE allocation_sessions.provider_metadata
                END,
                updated_at = excluded.updated_at
            """,
            [
                session_id,
                path,
                observation.provider if observation.success else "",
                observation.provider,
                observation.model,
                observation.error_kind,
                success,
                fail,
                int(observation.prompt_tokens),
                int(observation.completion_tokens),
                int(observation.total_tokens),
                float(observation.estimated_cost_usd),
                latency_ms,
                latency_ms,
                tokens_per_second,
                meta_json,
                ts_value,
                ts_value,
            ],
        )

    def upsert_health(
        self,
        provider: str,
        *,
        protocol: str = "unknown",
        available: bool = False,
        authenticated: bool = False,
        ready: bool = False,
        reason: str = "",
    ) -> None:
        if not llm_observe_enabled():
            return
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                conn.execute(
                    """
                    INSERT INTO provider_health VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (provider) DO UPDATE SET
                        protocol = excluded.protocol,
                        available = excluded.available,
                        authenticated = excluded.authenticated,
                        ready = excluded.ready,
                        reason = excluded.reason,
                        probed_at = excluded.probed_at
                    """,
                    [
                        provider,
                        protocol,
                        available,
                        authenticated,
                        ready,
                        reason[:256],
                        datetime.now(timezone.utc),
                    ],
                )
            finally:
                conn.close()

    def get_session(self, session_id: str) -> dict[str, Any]:
        sid = str(session_id or "").strip()[:128]
        if not sid:
            return {}
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return {}
            try:
                row = conn.execute(
                    """
                    SELECT
                        session_id, path, preferred_provider, last_provider,
                        last_model, last_error_kind, call_count, success_count,
                        fail_count, prompt_tokens, completion_tokens, total_tokens,
                        total_cost_usd, total_latency_ms, avg_latency_ms,
                        tokens_per_second, provider_metadata, handoff_context,
                        created_at, updated_at
                    FROM allocation_sessions
                    WHERE session_id = ?
                    """,
                    [sid],
                ).fetchone()
                if not row:
                    return {}
                columns = [
                    "session_id",
                    "path",
                    "preferred_provider",
                    "last_provider",
                    "last_model",
                    "last_error_kind",
                    "call_count",
                    "success_count",
                    "fail_count",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "total_cost_usd",
                    "total_latency_ms",
                    "avg_latency_ms",
                    "tokens_per_second",
                    "provider_metadata",
                    "handoff_context",
                    "created_at",
                    "updated_at",
                ]
                payload = dict(zip(columns, row))
                payload["session_id"] = sid
                provider_metadata = _parse_metadata_json(payload.pop("provider_metadata", ""))
                payload["provider_metadata"] = provider_metadata
                payload["handoff_context"] = str(payload.get("handoff_context") or "")
                payload["handoff_pending"] = bool(payload["handoff_context"].strip())
                last_provider_name = str(
                    payload.get("last_provider") or payload.get("preferred_provider") or ""
                )
                if last_provider_name in {"muse_code", "muse"} and provider_metadata:
                    payload["muse"] = dict(provider_metadata)
                native_by_provider: dict[str, dict[str, Any]] = {}
                try:
                    cli_rows = conn.execute(
                        """
                        SELECT provider, native_session_id, resume_style, call_count, metadata
                        FROM cli_sessions
                        WHERE allocation_session_id = ?
                        """,
                        [sid],
                    ).fetchall()
                    native_by_provider = {
                        str(provider): {
                            "native_session_id": str(native or ""),
                            "resume_style": str(style or ""),
                            "call_count": int(count or 0),
                            "metadata": _parse_metadata_json(meta_raw),
                        }
                        for provider, native, style, count, meta_raw in cli_rows
                    }
                except Exception:
                    try:
                        cli_rows = conn.execute(
                            """
                            SELECT provider, native_session_id, resume_style, call_count
                            FROM cli_sessions
                            WHERE allocation_session_id = ?
                            """,
                            [sid],
                        ).fetchall()
                        native_by_provider = {
                            str(provider): {
                                "native_session_id": str(native or ""),
                                "resume_style": str(style or ""),
                                "call_count": int(count or 0),
                            }
                            for provider, native, style, count in cli_rows
                        }
                    except Exception:
                        native_by_provider = {}
                payload["cli_sessions"] = native_by_provider
                preferred = str(payload.get("preferred_provider") or "")
                last = str(payload.get("last_provider") or "")
                chosen = preferred or last
                chosen_cli = native_by_provider.get(chosen) or {}
                payload["native_cli_session_id"] = str(
                    chosen_cli.get("native_session_id") or ""
                )
                try:
                    alias_rows = conn.execute(
                        """
                        SELECT native_session_id, provider, current_provider,
                               current_native_session_id
                        FROM cli_session_aliases
                        WHERE allocation_session_id = ?
                        ORDER BY updated_at ASC
                        """,
                        [sid],
                    ).fetchall()
                    payload["session_aliases"] = [
                        {
                            "native_session_id": str(native or ""),
                            "provider": str(alias_provider or ""),
                            "current_provider": str(current_provider or ""),
                            "current_native_session_id": str(current_native or ""),
                        }
                        for native, alias_provider, current_provider, current_native in alias_rows
                    ]
                except Exception:
                    payload["session_aliases"] = []
                if chosen in {"muse_code", "muse"} and chosen_cli.get("metadata"):
                    payload.setdefault("muse", dict(chosen_cli.get("metadata") or {}))
                    if not payload.get("provider_metadata"):
                        payload["provider_metadata"] = dict(chosen_cli.get("metadata") or {})
                return payload
            finally:
                conn.close()

    def get_cli_session(self, session_id: str, provider: str) -> dict[str, Any]:
        sid = str(session_id or "").strip()[:128]
        name = str(provider or "").strip().lower()
        if not sid or not name:
            return {}
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return {}
            try:
                row = conn.execute(
                    """
                    SELECT
                        allocation_session_id, provider, native_session_id,
                        resume_style, workspace, call_count, metadata,
                        created_at, updated_at
                    FROM cli_sessions
                    WHERE allocation_session_id = ? AND provider = ?
                    """,
                    [sid, name],
                ).fetchone()
                if not row:
                    return {}
                columns = [
                    "allocation_session_id",
                    "provider",
                    "native_session_id",
                    "resume_style",
                    "workspace",
                    "call_count",
                    "metadata",
                    "created_at",
                    "updated_at",
                ]
                payload = dict(zip(columns, row))
                payload["metadata"] = _parse_metadata_json(payload.get("metadata"))
                return payload
            finally:
                conn.close()

    def upsert_cli_session(
        self,
        *,
        session_id: str,
        provider: str,
        native_session_id: str,
        resume_style: str = "",
        workspace: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        sid = str(session_id or "").strip()[:128]
        name = str(provider or "").strip().lower()
        native = str(native_session_id or "").strip()[:256]
        if not sid or not name or not native:
            return
        if not llm_observe_enabled():
            return
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                self._upsert_cli_session_row(
                    conn,
                    session_id=sid,
                    provider=name,
                    native_session_id=native,
                    resume_style=resume_style,
                    workspace=workspace,
                    metadata=metadata,
                    ts_value=datetime.now(timezone.utc),
                )
            finally:
                conn.close()

    def _upsert_cli_session_row(
        self,
        conn: Any,
        *,
        session_id: str,
        provider: str,
        native_session_id: str,
        resume_style: str = "",
        workspace: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        ts_value: Optional[datetime] = None,
    ) -> None:
        now = ts_value or datetime.now(timezone.utc)
        meta_json = _metadata_json(metadata)
        conn.execute(
            """
            INSERT INTO cli_sessions (
                allocation_session_id, provider, native_session_id,
                resume_style, workspace, call_count, metadata, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT (allocation_session_id, provider) DO UPDATE SET
                native_session_id = excluded.native_session_id,
                resume_style = CASE
                    WHEN excluded.resume_style <> '' THEN excluded.resume_style
                    ELSE cli_sessions.resume_style
                END,
                workspace = CASE
                    WHEN excluded.workspace <> '' THEN excluded.workspace
                    ELSE cli_sessions.workspace
                END,
                call_count = cli_sessions.call_count + 1,
                metadata = CASE
                    WHEN excluded.metadata <> '' THEN excluded.metadata
                    ELSE cli_sessions.metadata
                END,
                updated_at = excluded.updated_at
            """,
            [
                session_id,
                provider,
                native_session_id,
                str(resume_style or "")[:64],
                str(workspace or "")[:512],
                meta_json,
                now,
                now,
            ],
        )
        self._upsert_session_alias(
            conn,
            native_session_id=native_session_id,
            provider=provider,
            allocation_session_id=session_id,
            current_provider=provider,
            current_native_session_id=native_session_id,
            ts_value=now,
        )

    def _upsert_session_alias(
        self,
        conn: Any,
        *,
        native_session_id: str,
        provider: str,
        allocation_session_id: str,
        current_provider: str,
        current_native_session_id: str,
        ts_value: Optional[datetime] = None,
    ) -> None:
        native = str(native_session_id or "").strip()[:256]
        alloc = str(allocation_session_id or "").strip()[:128]
        from_provider = str(provider or "").strip().lower()
        current = str(current_provider or from_provider).strip().lower()
        current_native = str(current_native_session_id or native).strip()[:256]
        if not native or not alloc or not from_provider:
            return
        now = ts_value or datetime.now(timezone.utc)
        conn.execute(
            """
            INSERT INTO cli_session_aliases (
                native_session_id, provider, allocation_session_id,
                current_provider, current_native_session_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (native_session_id, provider) DO UPDATE SET
                allocation_session_id = excluded.allocation_session_id,
                current_provider = excluded.current_provider,
                current_native_session_id = CASE
                    WHEN excluded.current_native_session_id <> ''
                    THEN excluded.current_native_session_id
                    ELSE cli_session_aliases.current_native_session_id
                END,
                updated_at = excluded.updated_at
            """,
            [native, from_provider, alloc, current, current_native, now, now],
        )
        if current_native:
            conn.execute(
                """
                UPDATE cli_session_aliases
                SET current_provider = ?,
                    current_native_session_id = ?,
                    updated_at = ?
                WHERE allocation_session_id = ?
                """,
                [current, current_native, now, alloc],
            )

    def remap_session_aliases(
        self,
        session_id: str,
        *,
        current_provider: str,
        current_native_session_id: str = "",
        previous_native_session_id: str = "",
        previous_provider: str = "",
    ) -> None:
        """Point every native id for an allocation session at the current CLI."""
        sid = str(session_id or "").strip()[:128]
        current = str(current_provider or "").strip().lower()
        if not sid or not current:
            return
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                now = datetime.now(timezone.utc)
                prev = str(previous_native_session_id or "").strip()[:256]
                prev_provider = str(previous_provider or "").strip().lower()
                if prev and prev_provider:
                    self._upsert_session_alias(
                        conn,
                        native_session_id=prev,
                        provider=prev_provider,
                        allocation_session_id=sid,
                        current_provider=current,
                        current_native_session_id=str(current_native_session_id or ""),
                        ts_value=now,
                    )
                conn.execute(
                    """
                    UPDATE cli_session_aliases
                    SET current_provider = ?,
                        current_native_session_id = CASE
                            WHEN ? <> '' THEN ? ELSE current_native_session_id
                        END,
                        updated_at = ?
                    WHERE allocation_session_id = ?
                    """,
                    [
                        current,
                        str(current_native_session_id or ""),
                        str(current_native_session_id or ""),
                        now,
                        sid,
                    ],
                )
            finally:
                conn.close()

    def resolve_session(self, any_session_id: str) -> dict[str, Any]:
        """Resolve an allocation id or any native CLI id to current session state."""
        sid = str(any_session_id or "").strip()[:256]
        if not sid:
            return {}
        direct = self.get_session(sid[:128])
        if direct:
            current_provider = str(
                direct.get("preferred_provider") or direct.get("last_provider") or ""
            )
            cli_map = direct.get("cli_sessions") if isinstance(direct.get("cli_sessions"), dict) else {}
            current_native = str(
                (cli_map.get(current_provider) or {}).get("native_session_id")
                or direct.get("native_cli_session_id")
                or ""
            )
            return {
                "allocation_session_id": str(direct.get("session_id") or sid[:128]),
                "current_provider": current_provider,
                "current_native_session_id": current_native,
                "session": direct,
                "aliases": self.list_session_aliases(str(direct.get("session_id") or sid[:128])),
            }
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return {}
            try:
                row = conn.execute(
                    """
                    SELECT allocation_session_id, provider, current_provider,
                           current_native_session_id
                    FROM cli_session_aliases
                    WHERE native_session_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    [sid],
                ).fetchone()
                if not row:
                    row = conn.execute(
                        """
                        SELECT allocation_session_id, provider, provider, native_session_id
                        FROM cli_sessions
                        WHERE native_session_id = ?
                        LIMIT 1
                        """,
                        [sid],
                    ).fetchone()
                if not row:
                    return {}
                alloc, origin_provider, current_provider, current_native = [
                    str(part or "") for part in row
                ]
            except Exception:
                return {}
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        session = self.get_session(alloc) if alloc else {}
        return {
            "allocation_session_id": alloc,
            "origin_provider": origin_provider,
            "current_provider": current_provider or str(session.get("preferred_provider") or ""),
            "current_native_session_id": current_native
            or str(session.get("native_cli_session_id") or ""),
            "session": session,
            "aliases": self.list_session_aliases(alloc) if alloc else [],
        }

    def list_session_aliases(self, session_id: str) -> list[dict[str, str]]:
        sid = str(session_id or "").strip()[:128]
        if not sid:
            return []
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return []
            try:
                rows = conn.execute(
                    """
                    SELECT native_session_id, provider, current_provider,
                           current_native_session_id
                    FROM cli_session_aliases
                    WHERE allocation_session_id = ?
                    ORDER BY updated_at ASC
                    """,
                    [sid],
                ).fetchall()
                return [
                    {
                        "native_session_id": str(native or ""),
                        "provider": str(provider or ""),
                        "current_provider": str(current_provider or ""),
                        "current_native_session_id": str(current_native or ""),
                    }
                    for native, provider, current_provider, current_native in rows
                ]
            except Exception:
                return []
            finally:
                conn.close()

    def rebind_session_provider(
        self,
        session_id: str,
        provider: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        handoff_context: str = "",
    ) -> None:
        """Point an allocation session at another CLI/API provider."""
        sid = str(session_id or "").strip()[:128]
        name = str(provider or "").strip().lower()
        if not sid or not name:
            return
        path = path_for_provider(name)
        meta_json = _metadata_json(metadata)
        handoff = str(handoff_context or "")[:_MAX_HANDOFF_CHARS]
        now = datetime.now(timezone.utc)
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                existing = conn.execute(
                    "SELECT session_id FROM allocation_sessions WHERE session_id = ?",
                    [sid],
                ).fetchone()
                if existing:
                    conn.execute(
                        """
                        UPDATE allocation_sessions SET
                            path = ?,
                            preferred_provider = ?,
                            last_provider = ?,
                            provider_metadata = CASE
                                WHEN ? <> '' THEN ? ELSE provider_metadata
                            END,
                            handoff_context = CASE
                                WHEN ? <> '' THEN ? ELSE handoff_context
                            END,
                            updated_at = ?
                        WHERE session_id = ?
                        """,
                        [
                            path,
                            name,
                            name,
                            meta_json,
                            meta_json,
                            handoff,
                            handoff,
                            now,
                            sid,
                        ],
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO allocation_sessions (
                            session_id, path, preferred_provider, last_provider,
                            last_model, last_error_kind, call_count, success_count,
                            fail_count, prompt_tokens, completion_tokens, total_tokens,
                            total_cost_usd, total_latency_ms, avg_latency_ms,
                            tokens_per_second, provider_metadata, handoff_context,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, '', '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?, ?, ?, ?)
                        """,
                        [sid, path, name, name, meta_json, handoff, now, now],
                    )
            finally:
                conn.close()

    def pop_session_handoff(self, session_id: str) -> str:
        """Return and clear pending CLI handoff context for one session."""
        sid = str(session_id or "").strip()[:128]
        if not sid:
            return ""
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return ""
            try:
                row = conn.execute(
                    "SELECT handoff_context FROM allocation_sessions WHERE session_id = ?",
                    [sid],
                ).fetchone()
                text = str((row[0] if row else "") or "")
                if text:
                    conn.execute(
                        """
                        UPDATE allocation_sessions
                        SET handoff_context = '', updated_at = ?
                        WHERE session_id = ?
                        """,
                        [datetime.now(timezone.utc), sid],
                    )
                return text[:_MAX_HANDOFF_CHARS]
            except Exception:
                return ""
            finally:
                conn.close()

    def provider_stats(
        self, names: Sequence[str], *, path: Optional[str] = None
    ) -> dict[str, dict[str, Any]]:
        if not names:
            return {}
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return {}
            try:
                placeholders = ",".join(["?"] * len(names))
                path_clause = ""
                params: list[Any] = list(names)
                if path:
                    path_clause = " AND path = ?"
                    params.append(str(path))
                rows = conn.execute(
                    f"""
                    SELECT
                        provider,
                        count(*) FILTER (
                            WHERE CAST(ts AS TIMESTAMP) > now() - INTERVAL 5 MINUTE
                        ) AS recent_calls,
                        count(*) FILTER (
                            WHERE success AND CAST(ts AS TIMESTAMP) > now() - INTERVAL 5 MINUTE
                        ) AS recent_ok,
                        count(*) FILTER (
                            WHERE NOT success AND CAST(ts AS TIMESTAMP) > now() - INTERVAL 5 MINUTE
                        ) AS recent_fail,
                        coalesce(sum(total_tokens) FILTER (
                            WHERE CAST(ts AS TIMESTAMP) > now() - INTERVAL 1 MINUTE
                        ), 0) AS tokens_1m,
                        coalesce(sum(estimated_cost_usd) FILTER (
                            WHERE CAST(ts AS TIMESTAMP) > now() - INTERVAL 1 DAY
                        ), 0) AS spend_1d,
                        max(CASE WHEN NOT success THEN CAST(ts AS TIMESTAMP) END) AS last_fail_ts,
                        arg_max(error_kind, CAST(ts AS TIMESTAMP)) AS last_error_kind,
                        avg(latency_ms) FILTER (
                            WHERE success AND CAST(ts AS TIMESTAMP) > now() - INTERVAL 15 MINUTE
                        ) AS avg_latency_ms,
                        arg_max(remaining_requests, CAST(ts AS TIMESTAMP)) AS remaining_requests,
                        arg_max(remaining_tokens, CAST(ts AS TIMESTAMP)) AS remaining_tokens
                    FROM call_observations
                    WHERE provider IN ({placeholders}){path_clause}
                    GROUP BY provider
                    """,
                    params,
                ).fetchall()
                columns = [
                    "provider",
                    "recent_calls",
                    "recent_ok",
                    "recent_fail",
                    "tokens_1m",
                    "spend_1d",
                    "last_fail_ts",
                    "last_error_kind",
                    "avg_latency_ms",
                    "remaining_requests",
                    "remaining_tokens",
                ]
                stats = {str(row[0]): dict(zip(columns, row)) for row in rows}
                health_rows = conn.execute(
                    f"""
                    SELECT provider, available, authenticated, ready, reason
                    FROM provider_health
                    WHERE provider IN ({placeholders})
                    """,
                    list(names),
                ).fetchall()
                for provider, available, authenticated, ready, reason in health_rows:
                    payload = stats.setdefault(str(provider), {"provider": provider})
                    payload["available"] = bool(available)
                    payload["authenticated"] = bool(authenticated)
                    payload["ready"] = bool(ready)
                    payload["health_reason"] = reason
                limit_rows = conn.execute(
                    f"""
                    SELECT provider, rpm, tpm
                    FROM provider_limits
                    WHERE provider IN ({placeholders})
                    """,
                    list(names),
                ).fetchall()
                for provider, rpm, tpm in limit_rows:
                    payload = stats.setdefault(str(provider), {"provider": provider})
                    payload["rpm"] = rpm
                    payload["tpm"] = tpm
                return stats
            finally:
                conn.close()

    def native_session_counts(self) -> dict[str, int]:
        """Count persisted native CLI sessions per provider."""
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return {}
            try:
                rows = conn.execute(
                    """
                    SELECT provider, count(*)
                    FROM cli_sessions
                    WHERE native_session_id <> ''
                    GROUP BY provider
                    """
                ).fetchall()
                return {str(provider): int(count or 0) for provider, count in rows}
            except Exception:
                return {}
            finally:
                conn.close()

    def upsert_api_key_slot(
        self,
        *,
        backend: str,
        slot_id: str,
        key_fingerprint: str,
        secret_name: str,
        label: str = "",
        enabled: bool = True,
    ) -> None:
        now = datetime.now(timezone.utc)
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                conn.execute(
                    """
                    INSERT INTO api_key_slots (
                        backend, slot_id, key_fingerprint, secret_name, label,
                        enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (backend, slot_id) DO UPDATE SET
                        key_fingerprint = excluded.key_fingerprint,
                        secret_name = excluded.secret_name,
                        label = CASE
                            WHEN excluded.label <> '' THEN excluded.label
                            ELSE api_key_slots.label
                        END,
                        enabled = excluded.enabled,
                        updated_at = excluded.updated_at
                    """,
                    [
                        backend,
                        slot_id,
                        key_fingerprint[:64],
                        secret_name[:256],
                        str(label or "")[:128],
                        bool(enabled),
                        now,
                        now,
                    ],
                )
            finally:
                conn.close()

    def list_api_key_slots(self, backend: str = "") -> list[dict[str, Any]]:
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return []
            try:
                if backend:
                    rows = conn.execute(
                        """
                        SELECT s.backend, s.slot_id, s.key_fingerprint, s.label, s.enabled,
                               st.call_count, st.success_count, st.fail_count,
                               st.total_tokens, st.total_cost_usd, st.avg_latency_ms,
                               st.tokens_per_second, st.last_error_kind,
                               st.remaining_requests, st.remaining_tokens,
                               st.remaining_balance_usd
                        FROM api_key_slots s
                        LEFT JOIN api_key_stats st
                          ON s.backend = st.backend AND s.slot_id = st.slot_id
                        WHERE s.backend = ?
                        ORDER BY s.slot_id
                        """,
                        [backend],
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT s.backend, s.slot_id, s.key_fingerprint, s.label, s.enabled,
                               st.call_count, st.success_count, st.fail_count,
                               st.total_tokens, st.total_cost_usd, st.avg_latency_ms,
                               st.tokens_per_second, st.last_error_kind,
                               st.remaining_requests, st.remaining_tokens,
                               st.remaining_balance_usd
                        FROM api_key_slots s
                        LEFT JOIN api_key_stats st
                          ON s.backend = st.backend AND s.slot_id = st.slot_id
                        ORDER BY s.backend, s.slot_id
                        """
                    ).fetchall()
                columns = [
                    "backend",
                    "slot_id",
                    "key_fingerprint",
                    "label",
                    "enabled",
                    "call_count",
                    "success_count",
                    "fail_count",
                    "total_tokens",
                    "total_cost_usd",
                    "avg_latency_ms",
                    "tokens_per_second",
                    "last_error_kind",
                    "remaining_requests",
                    "remaining_tokens",
                    "remaining_balance_usd",
                ]
                return [dict(zip(columns, row)) for row in rows]
            except Exception:
                return []
            finally:
                conn.close()

    def bind_session_api_key(self, session_id: str, backend: str, slot_id: str) -> None:
        sid = str(session_id or "").strip()[:128]
        if not sid or not backend or not slot_id:
            return
        now = datetime.now(timezone.utc)
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return
            try:
                conn.execute(
                    """
                    INSERT INTO session_key_bindings (
                        session_id, backend, slot_id, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (session_id) DO UPDATE SET
                        backend = excluded.backend,
                        slot_id = excluded.slot_id,
                        updated_at = excluded.updated_at
                    """,
                    [sid, backend, slot_id, now, now],
                )
            finally:
                conn.close()

    def get_session_api_key(self, session_id: str, backend: str = "") -> dict[str, Any]:
        sid = str(session_id or "").strip()[:128]
        if not sid:
            return {}
        with _LOCK:
            conn = self._connect()
            if conn is None:
                return {}
            try:
                if backend:
                    row = conn.execute(
                        """
                        SELECT session_id, backend, slot_id
                        FROM session_key_bindings
                        WHERE session_id = ? AND backend = ?
                        """,
                        [sid, backend],
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT session_id, backend, slot_id
                        FROM session_key_bindings
                        WHERE session_id = ?
                        """,
                        [sid],
                    ).fetchone()
                if not row:
                    return {}
                return {
                    "session_id": str(row[0] or ""),
                    "backend": str(row[1] or ""),
                    "slot_id": str(row[2] or ""),
                }
            except Exception:
                return {}
            finally:
                conn.close()

    def _upsert_api_key_stats_row(
        self,
        conn: Any,
        *,
        observation: CallObservation,
        slot_id: str,
        ts_value: datetime,
    ) -> None:
        backend = str(observation.provider or "").strip().lower()
        slot = str(slot_id or "").strip()[:64]
        if not backend or not slot:
            return
        success = 1 if observation.success else 0
        fail = 0 if observation.success else 1
        latency_ms = float(observation.latency_ms)
        balance = observation.extra_metadata.get("remaining_balance_usd")
        try:
            balance_f = float(balance) if balance not in (None, "") else None
        except (TypeError, ValueError):
            balance_f = None
        conn.execute(
            """
            INSERT INTO api_key_stats (
                backend, slot_id, call_count, success_count, fail_count,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd,
                total_latency_ms, avg_latency_ms, tokens_per_second,
                last_error_kind, remaining_requests, remaining_tokens,
                remaining_balance_usd, updated_at
            ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (backend, slot_id) DO UPDATE SET
                call_count = api_key_stats.call_count + 1,
                success_count = api_key_stats.success_count + excluded.success_count,
                fail_count = api_key_stats.fail_count + excluded.fail_count,
                prompt_tokens = api_key_stats.prompt_tokens + excluded.prompt_tokens,
                completion_tokens = api_key_stats.completion_tokens
                    + excluded.completion_tokens,
                total_tokens = api_key_stats.total_tokens + excluded.total_tokens,
                total_cost_usd = api_key_stats.total_cost_usd + excluded.total_cost_usd,
                total_latency_ms = api_key_stats.total_latency_ms
                    + excluded.total_latency_ms,
                avg_latency_ms = (api_key_stats.total_latency_ms
                    + excluded.total_latency_ms) / (api_key_stats.call_count + 1),
                tokens_per_second = CASE
                    WHEN (api_key_stats.total_latency_ms + excluded.total_latency_ms) > 0
                    THEN (api_key_stats.total_tokens + excluded.total_tokens)
                        / ((api_key_stats.total_latency_ms + excluded.total_latency_ms)
                            / 1000.0)
                    ELSE 0
                END,
                last_error_kind = excluded.last_error_kind,
                remaining_requests = coalesce(
                    excluded.remaining_requests, api_key_stats.remaining_requests
                ),
                remaining_tokens = coalesce(
                    excluded.remaining_tokens, api_key_stats.remaining_tokens
                ),
                remaining_balance_usd = coalesce(
                    excluded.remaining_balance_usd, api_key_stats.remaining_balance_usd
                ),
                updated_at = excluded.updated_at
            """,
            [
                backend,
                slot,
                success,
                fail,
                int(observation.prompt_tokens),
                int(observation.completion_tokens),
                int(observation.total_tokens),
                float(observation.estimated_cost_usd),
                latency_ms,
                latency_ms,
                float(observation.tokens_per_second or 0.0),
                str(observation.error_kind or ""),
                observation.remaining_requests,
                observation.remaining_tokens,
                balance_f,
                ts_value,
            ],
        )


_DEFAULT_STORE: Optional[AllocationStore] = None
_STORE_LOCK = threading.Lock()


def get_allocation_store() -> AllocationStore:
    global _DEFAULT_STORE
    with _STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = AllocationStore()
        return _DEFAULT_STORE


def reset_allocation_store(store: Optional[AllocationStore] = None) -> None:
    global _DEFAULT_STORE
    with _STORE_LOCK:
        _DEFAULT_STORE = store
