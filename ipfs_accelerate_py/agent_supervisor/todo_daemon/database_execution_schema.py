"""Provider-cold schema installer for DatabaseImplementationDaemon@1.

This module owns the daemon execution-store DDL so trusted bootstrap code can
create an empty execution authority without importing the model/provider
runtime.  It is a schema utility for the existing daemon, not a second
execution store or scheduling authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.duckdb_state import open_duckdb_connection

DATABASE_IMPLEMENTATION_DAEMON_INTERFACE: Final[str] = (
    "DatabaseImplementationDaemon@1"
)
DATABASE_IMPLEMENTATION_DAEMON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1"
)

DAEMON_EXECUTION_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS daemon_execution_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS database_task_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    claim_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    task_alias VARCHAR NOT NULL DEFAULT '',
    attempt_number BIGINT NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    lease_id VARCHAR NOT NULL DEFAULT '',
    committed_phase VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    started_at_ms BIGINT NOT NULL,
    finished_at_ms BIGINT,
    revision BIGINT NOT NULL DEFAULT 1,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS database_task_attempts_task_idx
    ON database_task_attempts(task_cid, attempt_number);
CREATE INDEX IF NOT EXISTS database_task_attempts_owner_idx
    ON database_task_attempts(owner_session_id, status);
CREATE INDEX IF NOT EXISTS database_task_attempts_claim_idx
    ON database_task_attempts(claim_id);

CREATE TABLE IF NOT EXISTS attempt_phases (
    attempt_id VARCHAR NOT NULL,
    phase VARCHAR NOT NULL,
    committed_at_ms BIGINT NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    revision BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    PRIMARY KEY (attempt_id, phase)
);

CREATE TABLE IF NOT EXISTS provider_invocations (
    invocation_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    result_json VARCHAR NOT NULL DEFAULT '{}',
    UNIQUE (attempt_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS effect_claims (
    effect_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL,
    task_cid VARCHAR NOT NULL,
    effect_key VARCHAR NOT NULL,
    idempotency_key VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    result_json VARCHAR NOT NULL DEFAULT '{}',
    UNIQUE (attempt_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS daemon_execution_events (
    event_id VARCHAR PRIMARY KEY,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    task_cid VARCHAR NOT NULL DEFAULT '',
    event_type VARCHAR NOT NULL,
    recorded_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
"""

_EXECUTION_TABLES: Final[tuple[str, ...]] = (
    "daemon_execution_metadata",
    "database_task_attempts",
    "attempt_phases",
    "provider_invocations",
    "effect_claims",
    "daemon_execution_events",
)


def _statements(sql: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in sql.split(";") if part.strip())


def install_database_execution_schema(
    path: Path,
    *,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Install the existing daemon schema and bind closed bootstrap metadata."""

    required = {
        "authority_mode",
        "logical_owner_session_id",
        "process_instance_id",
        "state_schema_revision",
        "control_schema_profile_id",
        "control_schema_fingerprint",
    }
    if set(metadata) != required or any(
        not isinstance(value, str) or not value for value in metadata.values()
    ):
        raise ValueError("execution bootstrap metadata must be a closed nonempty mapping")
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = open_duckdb_connection(path)
    try:
        for statement in _statements(DAEMON_EXECUTION_SQL):
            connection.execute(statement)
        bound = {
            "interface": DATABASE_IMPLEMENTATION_DAEMON_INTERFACE,
            "schema": DATABASE_IMPLEMENTATION_DAEMON_SCHEMA,
            **dict(metadata),
        }
        for key, value in sorted(bound.items()):
            connection.execute(
                """
                INSERT OR REPLACE INTO daemon_execution_metadata(key, value)
                VALUES (?, ?)
                """,
                [key, value],
            )
        observed = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        }
        if observed != set(_EXECUTION_TABLES):
            raise RuntimeError("execution schema table inventory differs")
        receipt: dict[str, Any] = {
            "schema": "ipfs_accelerate_py/agent-supervisor/database-execution-schema-install@1",
            "daemon_schema": DATABASE_IMPLEMENTATION_DAEMON_SCHEMA,
            "tables": list(_EXECUTION_TABLES),
            "metadata": bound,
        }
        receipt["receipt_cid"] = content_identity(receipt)
        return receipt
    finally:
        connection.close()


__all__ = [
    "DAEMON_EXECUTION_SQL",
    "DATABASE_IMPLEMENTATION_DAEMON_INTERFACE",
    "DATABASE_IMPLEMENTATION_DAEMON_SCHEMA",
    "install_database_execution_schema",
]
