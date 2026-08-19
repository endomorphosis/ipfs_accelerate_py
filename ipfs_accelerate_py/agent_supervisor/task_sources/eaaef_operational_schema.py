"""Versioned EAAEF owner-transaction extension for the operational profile.

The base datasets-authoritative profile deliberately owns the canonical task,
lease, attempt, validation, provider, effect, and event relations.  This module
adds only the physical fields and the completion barrier that the external
agent bootstrap daemon needs to use those relations without opening its former
coordination and execution sidecars.

Installation is an offline materializer concern.  Runtime owner handlers may
verify this profile, but they never run DDL, open a database path, or own a
transaction lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .control_plane_migrations import (
    ControlPlaneMigration,
    ControlPlaneMigrationRunner,
    MigrationCatalog,
    MigrationRunReport,
    compute_schema_fingerprint,
    duckdb_available,
)
from .control_plane_schema import (
    DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION,
    DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
    DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
    load_datasets_authoritative_operational_catalog,
)
from .duckdb_state import open_duckdb_connection
from .task_identity import canonical_content_cid

EAAEF_OPERATIONAL_PROFILE_INTERFACE: Final = (
    "DatasetsAuthoritativeEAAEFOperationalProfile@2"
)
EAAEF_OPERATIONAL_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "datasets-authoritative-eaaef-operational-control-plane@2"
)
EAAEF_OPERATIONAL_PROFILE_ID: Final = (
    "datasets-authoritative-eaaef-operational-control-plane@2"
)
EAAEF_OPERATIONAL_MIGRATION_VERSION: Final = 2
EAAEF_OPERATIONAL_MIGRATION_ID: Final = (
    "0002_eaaef_owner_transaction_operational_extension"
)
EAAEF_BOARD_SCHEDULER_LEASE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/eaaef-board-shard-scheduler-lease@1"
)
EAAEF_BOARD_SCHEDULER_LEASE_KIND: Final = "board_shard_scheduler"
EAAEF_BOARD_SCHEDULER_LEASE_MODE: Final = "shared_scheduler"


# Keep join-, fence-, replay-, and idempotency-critical values in physical
# columns.  JSON bodies below are evidence payloads, never authority indexes.
_EAAEF_OPERATIONAL_SQL: Final = """
ALTER TABLE leases ADD COLUMN claim_id VARCHAR DEFAULT '';
ALTER TABLE leases ADD COLUMN attempt_id VARCHAR DEFAULT '';
ALTER TABLE leases ADD COLUMN attempt_number BIGINT DEFAULT 0;
ALTER TABLE leases ADD COLUMN lease_kind VARCHAR DEFAULT 'task';
ALTER TABLE leases ADD COLUMN scope_id VARCHAR DEFAULT '';
ALTER TABLE leases ADD COLUMN mode VARCHAR DEFAULT 'exclusive';

ALTER TABLE task_claims ADD COLUMN attempt_id VARCHAR DEFAULT '';
ALTER TABLE task_claims ADD COLUMN attempt_number BIGINT DEFAULT 0;
ALTER TABLE task_claims ADD COLUMN lease_id VARCHAR DEFAULT '';
ALTER TABLE task_claims ADD COLUMN worktree_id VARCHAR DEFAULT '';
ALTER TABLE task_claims ADD COLUMN claimed_at_ms BIGINT DEFAULT 0;
ALTER TABLE task_claims ADD COLUMN expires_at_ms BIGINT DEFAULT 0;
ALTER TABLE task_claims ADD COLUMN released_at_ms BIGINT;
ALTER TABLE task_claims ADD COLUMN body_json VARCHAR DEFAULT '{}';

ALTER TABLE task_attempts ADD COLUMN claim_id VARCHAR DEFAULT '';
ALTER TABLE task_attempts ADD COLUMN task_alias VARCHAR DEFAULT '';
ALTER TABLE task_attempts ADD COLUMN lease_id VARCHAR DEFAULT '';
ALTER TABLE task_attempts ADD COLUMN committed_phase VARCHAR DEFAULT 'claimed';
ALTER TABLE task_attempts ADD COLUMN started_at_ms BIGINT DEFAULT 0;
ALTER TABLE task_attempts ADD COLUMN finished_at_ms BIGINT;
ALTER TABLE task_attempts ADD COLUMN body_json VARCHAR DEFAULT '{}';

ALTER TABLE attempt_phases ADD COLUMN committed_at_ms BIGINT DEFAULT 0;
ALTER TABLE attempt_phases ADD COLUMN fencing_token BIGINT DEFAULT 0;
ALTER TABLE attempt_phases ADD COLUMN fence_epoch BIGINT DEFAULT 0;
ALTER TABLE attempt_phases ADD COLUMN revision BIGINT DEFAULT 0;
ALTER TABLE attempt_phases ADD COLUMN body_json VARCHAR DEFAULT '{}';

ALTER TABLE provider_invocations ADD COLUMN idempotency_key VARCHAR DEFAULT '';
ALTER TABLE provider_invocations ADD COLUMN owner_session_id VARCHAR DEFAULT '';
ALTER TABLE provider_invocations ADD COLUMN recorded_at_ms BIGINT DEFAULT 0;
ALTER TABLE provider_invocations ADD COLUMN result_json VARCHAR DEFAULT '{}';
ALTER TABLE provider_invocations ADD COLUMN fencing_token BIGINT DEFAULT 0;
ALTER TABLE provider_invocations ADD COLUMN fence_epoch BIGINT DEFAULT 0;

ALTER TABLE effect_claims ADD COLUMN operation_key VARCHAR DEFAULT '';
ALTER TABLE effect_claims ADD COLUMN idempotency_key VARCHAR DEFAULT '';
ALTER TABLE effect_claims ADD COLUMN owner_session_id VARCHAR DEFAULT '';
ALTER TABLE effect_claims ADD COLUMN recorded_at_ms BIGINT DEFAULT 0;
ALTER TABLE effect_claims ADD COLUMN result_json VARCHAR DEFAULT '{}';
ALTER TABLE effect_claims ADD COLUMN fencing_token BIGINT DEFAULT 0;
ALTER TABLE effect_claims ADD COLUMN fence_epoch BIGINT DEFAULT 0;

ALTER TABLE daemon_sessions ADD COLUMN metadata_json VARCHAR DEFAULT '{}';

CREATE TABLE eaaef_completion_barriers (
    task_cid VARCHAR PRIMARY KEY,
    claim_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    lease_id VARCHAR NOT NULL,
    owner_session_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    control_expected_revision BIGINT NOT NULL,
    control_expected_status VARCHAR NOT NULL,
    evidence_digest VARCHAR NOT NULL,
    preparation_digest VARCHAR NOT NULL UNIQUE,
    prepared_at_ms BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    control_receipt_json VARCHAR NOT NULL DEFAULT '{}',
    reconciliation_json VARCHAR NOT NULL DEFAULT '{}',
    body_json VARCHAR NOT NULL DEFAULT '{}',
    revision BIGINT NOT NULL DEFAULT 1
);
CREATE INDEX eaaef_completion_barriers_status_idx
    ON eaaef_completion_barriers(status, prepared_at_ms, task_cid);
CREATE UNIQUE INDEX eaaef_completion_barriers_claim_idx
    ON eaaef_completion_barriers(claim_id, attempt_id, lease_id);

CREATE TABLE eaaef_completion_barrier_history (
    history_id VARCHAR PRIMARY KEY,
    task_cid VARCHAR NOT NULL,
    claim_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL,
    attempt_number BIGINT NOT NULL,
    lease_id VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    fence_epoch BIGINT NOT NULL,
    preparation_digest VARCHAR NOT NULL,
    terminal_status VARCHAR NOT NULL,
    archived_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL
);
CREATE UNIQUE INDEX eaaef_completion_barrier_history_preparation_uidx
    ON eaaef_completion_barrier_history(preparation_digest);
CREATE INDEX eaaef_completion_barrier_history_task_idx
    ON eaaef_completion_barrier_history(task_cid, attempt_number, fencing_token);

CREATE UNIQUE INDEX eaaef_task_claim_attempt_uidx
    ON task_claims(attempt_id, attempt_number);
CREATE UNIQUE INDEX eaaef_task_claim_lease_uidx
    ON task_claims(lease_id);
CREATE UNIQUE INDEX eaaef_attempt_phase_uidx
    ON attempt_phases(attempt_id, phase_name);
CREATE UNIQUE INDEX eaaef_provider_idempotency_uidx
    ON provider_invocations(attempt_id, idempotency_key);
CREATE UNIQUE INDEX eaaef_effect_idempotency_uidx
    ON effect_claims(attempt_id, idempotency_key);

CREATE TABLE eaaef_operational_profile_seals (
    profile_id VARCHAR PRIMARY KEY,
    schema_revision BIGINT NOT NULL,
    migration_id VARCHAR NOT NULL,
    operation_vocabulary_cid VARCHAR NOT NULL,
    schema_fingerprint VARCHAR NOT NULL DEFAULT '',
    required_index_set_cid VARCHAR NOT NULL DEFAULT '',
    sealed_at VARCHAR NOT NULL DEFAULT '1970-01-01T00:00:00Z'
);
INSERT INTO eaaef_operational_profile_seals(
    profile_id, schema_revision, migration_id, operation_vocabulary_cid
) VALUES (
    'datasets-authoritative-eaaef-operational-control-plane@2',
    2,
    '0002_eaaef_owner_transaction_operational_extension',
    '__EAAEF_OPERATION_VOCABULARY_CID__'
);

INSERT INTO schema_contracts (
    contract_id, interface_name, domain_name, schema_revision,
    payload_schema, description, created_at
) VALUES (
    'contract:DatasetsAuthoritativeEAAEFOperationalProfile@2',
    'DatasetsAuthoritativeEAAEFOperationalProfile@2',
    'runtime',
    2,
    'ipfs_accelerate_py/agent-supervisor/datasets-authoritative-eaaef-operational-control-plane@2',
    'Single-owner EAAEF task coordination and execution extension; semantic and proof truth remains authoritative in ipfs_datasets_py',
    '1970-01-01T00:00:00Z'
);
"""


_REQUIRED_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "leases": (
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_kind",
            "scope_id",
            "mode",
        ),
        "task_claims": (
            "attempt_id",
            "attempt_number",
            "lease_id",
            "worktree_id",
            "claimed_at_ms",
            "expires_at_ms",
            "released_at_ms",
            "body_json",
        ),
        "task_attempts": (
            "claim_id",
            "task_alias",
            "lease_id",
            "committed_phase",
            "started_at_ms",
            "finished_at_ms",
            "body_json",
        ),
        "attempt_phases": (
            "committed_at_ms",
            "fencing_token",
            "fence_epoch",
            "revision",
            "body_json",
        ),
        "provider_invocations": (
            "idempotency_key",
            "owner_session_id",
            "recorded_at_ms",
            "result_json",
            "fencing_token",
            "fence_epoch",
        ),
        "effect_claims": (
            "operation_key",
            "idempotency_key",
            "owner_session_id",
            "recorded_at_ms",
            "result_json",
            "fencing_token",
            "fence_epoch",
        ),
        "daemon_sessions": ("metadata_json",),
        "eaaef_completion_barriers": (
            "task_cid",
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "owner_session_id",
            "fencing_token",
            "fence_epoch",
            "control_expected_revision",
            "control_expected_status",
            "evidence_digest",
            "preparation_digest",
            "prepared_at_ms",
            "status",
            "control_receipt_json",
            "reconciliation_json",
            "body_json",
            "revision",
        ),
        "eaaef_completion_barrier_history": (
            "history_id",
            "task_cid",
            "claim_id",
            "attempt_id",
            "attempt_number",
            "lease_id",
            "fencing_token",
            "fence_epoch",
            "preparation_digest",
            "terminal_status",
            "archived_at_ms",
            "body_json",
        ),
        "eaaef_operational_profile_seals": (
            "profile_id",
            "schema_revision",
            "migration_id",
            "operation_vocabulary_cid",
            "schema_fingerprint",
            "required_index_set_cid",
            "sealed_at",
        ),
    }
)

_REQUIRED_INDEX_NAMES: Final = frozenset(
    {
        "eaaef_completion_barriers_status_idx",
        "eaaef_completion_barriers_claim_idx",
        "eaaef_completion_barrier_history_preparation_uidx",
        "eaaef_completion_barrier_history_task_idx",
        "eaaef_task_claim_attempt_uidx",
        "eaaef_task_claim_lease_uidx",
        "eaaef_attempt_phase_uidx",
        "eaaef_provider_idempotency_uidx",
        "eaaef_effect_idempotency_uidx",
    }
)


class EAAEFOperationalSchemaError(RuntimeError):
    """The sealed EAAEF operational profile is absent or has drifted."""


def eaaef_board_scheduler_lease_seed(
    *,
    board_namespace: str,
    shard_id: str,
    lease_id: str,
    principal_did: str,
    owner_session_id: str,
    owner_generation: int,
    fencing_token: int,
    fence_epoch: int,
    issued_at_ms: int,
    expires_at_ms: int,
) -> Mapping[str, Any]:
    """Return the closed physical seed for the claim-selection authority.

    This is an offline provisioning contract, not an admitted lease receipt.
    The command owner still verifies signature, expiry, principal, and fence in
    its active transaction before the borrowed adapter observes this row.
    """

    compact = {
        "board_namespace": str(board_namespace or ""),
        "shard_id": str(shard_id or ""),
        "lease_id": str(lease_id or ""),
        "principal_did": str(principal_did or ""),
        "owner_session_id": str(owner_session_id or ""),
    }
    if not all(compact.values()) or not compact["principal_did"].startswith(
        "did:key:z"
    ):
        raise EAAEFOperationalSchemaError(
            "board scheduler lease identities are incomplete"
        )
    numbers = {
        "owner_generation": owner_generation,
        "fencing_token": fencing_token,
        "fence_epoch": fence_epoch,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in numbers.values()
    ) or expires_at_ms <= issued_at_ms:
        raise EAAEFOperationalSchemaError(
            "board scheduler lease bounds are invalid"
        )
    board_scope = f"board:{compact['board_namespace']}:{compact['shard_id']}"
    extension = {
        "schema": EAAEF_BOARD_SCHEDULER_LEASE_SCHEMA,
        "board_namespace": compact["board_namespace"],
        "shard_id": compact["shard_id"],
        "scope_id": board_scope,
        "lease_kind": EAAEF_BOARD_SCHEDULER_LEASE_KIND,
        "mode": EAAEF_BOARD_SCHEDULER_LEASE_MODE,
        "owner_generation": owner_generation,
    }
    row = {
        "task_cid": board_scope,
        "claim_cid": compact["lease_id"],
        "resolution_cid": "resolution:eaaef-board-shard-scheduler",
        "claimant_did": compact["principal_did"],
        "logical_epoch": owner_generation,
        "fencing_token": fencing_token,
        "expires_at_ms": expires_at_ms,
        "attempt": 1,
        "state": "accepted",
        "started_at_ms": issued_at_ms,
        "release_reason": None,
        "retry_not_before_ms": 0,
        "owner_session_id": compact["owner_session_id"],
        "fence_epoch": fence_epoch,
        "revision": 1,
        "extension_schema": EAAEF_BOARD_SCHEDULER_LEASE_SCHEMA,
        "extension_json": extension,
        "claim_id": "",
        "attempt_id": "",
        "attempt_number": 0,
        "lease_kind": EAAEF_BOARD_SCHEDULER_LEASE_KIND,
        "scope_id": board_scope,
        "mode": EAAEF_BOARD_SCHEDULER_LEASE_MODE,
    }
    seed = {
        "schema": EAAEF_BOARD_SCHEDULER_LEASE_SCHEMA,
        "board_scope": board_scope,
        "row": row,
        "offline_provision_only": True,
        "production_admitted": False,
    }
    return MappingProxyType(
        {**seed, "seed_contract_cid": canonical_content_cid(seed)}
    )


def eaaef_operational_migration() -> ControlPlaneMigration:
    """Return the exact checksum-bound profile-v2 migration."""

    sql_text = _EAAEF_OPERATIONAL_SQL.replace(
        "__EAAEF_OPERATION_VOCABULARY_CID__",
        _expected_operation_vocabulary_cid(),
    )
    return ControlPlaneMigration.from_sql(
        version=EAAEF_OPERATIONAL_MIGRATION_VERSION,
        migration_id=EAAEF_OPERATIONAL_MIGRATION_ID,
        sql_text=sql_text,
        description="EAAEF borrowed-owner-transaction operational extension",
        depends_on=(DATASETS_AUTHORITATIVE_OPERATIONAL_MIGRATION_VERSION,),
        preconditions=(
            "SELECT COUNT(*) = 1 FROM schema_contracts WHERE contract_id = "
            "'contract:DatasetsAuthoritativeOperationalControlPlane@1'",
            "SELECT COUNT(*) = 0 FROM information_schema.tables WHERE "
            "table_schema = 'main' AND table_name = 'eaaef_completion_barriers'",
        ),
        postconditions=(
            "SELECT COUNT(*) = 1 FROM information_schema.tables WHERE "
            "table_schema = 'main' AND table_name = 'eaaef_completion_barriers'",
            "SELECT COUNT(*) = 1 FROM schema_contracts WHERE contract_id = "
            "'contract:DatasetsAuthoritativeEAAEFOperationalProfile@2' "
            "AND schema_revision = 2",
        ),
        source_path="task_sources/eaaef_operational_schema.py",
    )


def load_eaaef_operational_catalog() -> MigrationCatalog:
    """Return base operational @1 plus the exact EAAEF @2 extension."""

    base = load_datasets_authoritative_operational_catalog()
    return MigrationCatalog.from_migrations(
        (*base.migrations, eaaef_operational_migration())
    )


def eaaef_operation_vocabulary_cid(operations: Any) -> str:
    """Bind an exact sorted operation vocabulary without importing its owner."""

    values = sorted(str(item) for item in operations)
    if len(values) != len(set(values)) or not values:
        raise EAAEFOperationalSchemaError(
            "EAAEF operation vocabulary must be non-empty and duplicate-free"
        )
    return canonical_content_cid(
        {
            "schema": "EAAEFBootstrapDaemonOperationVocabulary@1",
            "operations": values,
        }
    )


def _expected_operation_vocabulary_cid() -> str:
    from .eaaef_bootstrap_daemon_gateway import (
        EAAEF_BOOTSTRAP_DAEMON_OPERATIONS,
    )

    return eaaef_operation_vocabulary_cid(EAAEF_BOOTSTRAP_DAEMON_OPERATIONS)


def eaaef_operational_profile_contract(
    *, operation_vocabulary_cid: str
) -> Mapping[str, Any]:
    """Return the source-stable portion sealed by a run materializer."""

    supplied_vocabulary = str(operation_vocabulary_cid or "")
    expected_vocabulary = _expected_operation_vocabulary_cid()
    if supplied_vocabulary != expected_vocabulary:
        raise EAAEFOperationalSchemaError(
            "EAAEF operation vocabulary differs from the exact 31-operation contract"
        )
    catalog = load_eaaef_operational_catalog()
    migration = eaaef_operational_migration()
    payload = {
        "schema": EAAEF_OPERATIONAL_PROFILE_SCHEMA,
        "interface": EAAEF_OPERATIONAL_PROFILE_INTERFACE,
        "profile_id": EAAEF_OPERATIONAL_PROFILE_ID,
        "base_profile_id": DATASETS_AUTHORITATIVE_OPERATIONAL_PROFILE_ID,
        "base_profile_schema": DATASETS_AUTHORITATIVE_OPERATIONAL_SCHEMA,
        "schema_version": EAAEF_OPERATIONAL_MIGRATION_VERSION,
        "migration_id": migration.migration_id,
        "migration_checksum": migration.checksum,
        "catalog_fingerprint": catalog.fingerprint(),
        "operation_vocabulary_cid": supplied_vocabulary,
        "runtime_ddl_allowed": False,
        "direct_database_open_allowed": False,
        "sidecar_writes_allowed": False,
    }
    return MappingProxyType(
        {**payload, "profile_contract_cid": canonical_content_cid(payload)}
    )


def install_eaaef_operational_schema(
    database_path: Path | str,
    *,
    application_version: str,
    tool_version: str,
    owner_id: str,
) -> MigrationRunReport:
    """Install @1 then @2 through the sole offline migration runner."""

    if not duckdb_available():
        raise EAAEFOperationalSchemaError(
            "DuckDB is required to install the EAAEF operational profile"
        )
    runner = ControlPlaneMigrationRunner.for_database(
        database_path,
        catalog=load_eaaef_operational_catalog(),
        application_version=application_version,
        tool_version=tool_version,
        owner_id=owner_id,
    )
    report = runner.apply(target_version=EAAEF_OPERATIONAL_MIGRATION_VERSION)
    with open_duckdb_connection(database_path) as connection:
        schema_fingerprint = compute_schema_fingerprint(connection)
        index_set_cid = _required_index_set_cid(connection)
        row = connection.execute(
            "SELECT schema_fingerprint, required_index_set_cid, "
            "operation_vocabulary_cid FROM eaaef_operational_profile_seals "
            "WHERE profile_id=?",
            [EAAEF_OPERATIONAL_PROFILE_ID],
        ).fetchone()
        if row is None:
            raise EAAEFOperationalSchemaError(
                "EAAEF operational profile seal is absent after migration"
            )
        expected_vocabulary = _expected_operation_vocabulary_cid()
        if str(row[2]) != expected_vocabulary:
            raise EAAEFOperationalSchemaError(
                "EAAEF operational profile vocabulary seal drifted"
            )
        stored = (str(row[0] or ""), str(row[1] or ""))
        expected = (schema_fingerprint, index_set_cid)
        if stored == ("", ""):
            connection.execute(
                "UPDATE eaaef_operational_profile_seals SET "
                "schema_fingerprint=?, required_index_set_cid=?, sealed_at=? "
                "WHERE profile_id=? AND schema_fingerprint='' "
                "AND required_index_set_cid=''",
                [
                    schema_fingerprint,
                    index_set_cid,
                    "1970-01-01T00:00:00Z",
                    EAAEF_OPERATIONAL_PROFILE_ID,
                ],
            )
        elif stored != expected:
            raise EAAEFOperationalSchemaError(
                "EAAEF operational profile seal differs from installed schema"
            )
    return report


def _column_names(connection: Any, table: str) -> frozenset[str]:
    return frozenset(
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    )


def _required_index_set_cid(connection: Any) -> str:
    rows = connection.execute(
        "SELECT index_name, sql FROM duckdb_indexes() WHERE schema_name='main' "
        "AND index_name LIKE 'eaaef_%' ORDER BY index_name"
    ).fetchall()
    indexes = {
        str(row[0]): " ".join(str(row[1] or "").split()) for row in rows
    }
    if frozenset(indexes) != _REQUIRED_INDEX_NAMES:
        raise EAAEFOperationalSchemaError(
            "EAAEF authority index set is missing, extra, or renamed"
        )
    return canonical_content_cid(
        {
            "schema": "EAAEFOperationalRequiredIndexSet@1",
            "indexes": indexes,
        }
    )


def verify_eaaef_operational_connection(
    connection: Any,
    *,
    operation_vocabulary_cid: str,
) -> Mapping[str, Any]:
    """Verify @2 on the already-borrowed owner connection without DDL."""

    catalog = load_eaaef_operational_catalog()
    migration = eaaef_operational_migration()
    profile = dict(
        eaaef_operational_profile_contract(
            operation_vocabulary_cid=operation_vocabulary_cid
        )
    )
    migration_rows = connection.execute(
        "SELECT version, migration_id, checksum FROM schema_migrations "
        "ORDER BY version"
    ).fetchall()
    expected_migrations = [
        (item.version, item.migration_id, item.checksum) for item in catalog.migrations
    ]
    observed_migrations = [
        (int(row[0]), str(row[1]), str(row[2])) for row in migration_rows
    ]
    if observed_migrations != expected_migrations:
        raise EAAEFOperationalSchemaError(
            "EAAEF operational migration history is missing or drifted"
        )
    missing_columns: dict[str, list[str]] = {}
    for table, required in _REQUIRED_COLUMNS.items():
        observed = _column_names(connection, table)
        absent = sorted(set(required) - observed)
        if absent:
            missing_columns[table] = absent
    if missing_columns:
        raise EAAEFOperationalSchemaError(
            f"EAAEF operational columns are missing: {missing_columns}"
        )
    contract = connection.execute(
        "SELECT interface_name, schema_revision, payload_schema FROM "
        "schema_contracts WHERE contract_id = "
        "'contract:DatasetsAuthoritativeEAAEFOperationalProfile@2'"
    ).fetchone()
    if contract is None or (
        str(contract[0]) != EAAEF_OPERATIONAL_PROFILE_INTERFACE
        or int(contract[1]) != EAAEF_OPERATIONAL_MIGRATION_VERSION
        or str(contract[2]) != EAAEF_OPERATIONAL_PROFILE_SCHEMA
    ):
        raise EAAEFOperationalSchemaError(
            "EAAEF operational schema contract is missing or drifted"
        )
    fingerprint = compute_schema_fingerprint(connection)
    index_set_cid = _required_index_set_cid(connection)
    seal = connection.execute(
        "SELECT schema_revision, migration_id, operation_vocabulary_cid, "
        "schema_fingerprint, required_index_set_cid FROM "
        "eaaef_operational_profile_seals WHERE profile_id=?",
        [EAAEF_OPERATIONAL_PROFILE_ID],
    ).fetchone()
    if seal is None or (
        int(seal[0]) != EAAEF_OPERATIONAL_MIGRATION_VERSION
        or str(seal[1]) != migration.migration_id
        or str(seal[2]) != operation_vocabulary_cid
        or str(seal[3]) != fingerprint
        or str(seal[4]) != index_set_cid
    ):
        raise EAAEFOperationalSchemaError(
            "EAAEF operational schema/vocabulary/index seal is missing or drifted"
        )
    evidence = {
        **profile,
        "valid": True,
        "migration_id": migration.migration_id,
        "migration_checksum": migration.checksum,
        "catalog_fingerprint": catalog.fingerprint(),
        "schema_fingerprint": fingerprint,
        "required_index_set_cid": index_set_cid,
        "required_columns": {
            table: list(columns) for table, columns in _REQUIRED_COLUMNS.items()
        },
    }
    return MappingProxyType(
        {**evidence, "verification_cid": canonical_content_cid(evidence)}
    )


def verify_eaaef_operational_schema(
    database_path: Path | str,
    *,
    operation_vocabulary_cid: str,
) -> Mapping[str, Any]:
    """Verify exact migration identities, physical columns, and no-sidecar policy."""

    with open_duckdb_connection(database_path) as connection:
        return verify_eaaef_operational_connection(
            connection,
            operation_vocabulary_cid=operation_vocabulary_cid,
        )


__all__ = (
    "EAAEF_BOARD_SCHEDULER_LEASE_KIND",
    "EAAEF_BOARD_SCHEDULER_LEASE_MODE",
    "EAAEF_BOARD_SCHEDULER_LEASE_SCHEMA",
    "EAAEF_OPERATIONAL_MIGRATION_ID",
    "EAAEF_OPERATIONAL_MIGRATION_VERSION",
    "EAAEF_OPERATIONAL_PROFILE_ID",
    "EAAEF_OPERATIONAL_PROFILE_INTERFACE",
    "EAAEF_OPERATIONAL_PROFILE_SCHEMA",
    "EAAEFOperationalSchemaError",
    "eaaef_board_scheduler_lease_seed",
    "eaaef_operation_vocabulary_cid",
    "eaaef_operational_migration",
    "eaaef_operational_profile_contract",
    "install_eaaef_operational_schema",
    "load_eaaef_operational_catalog",
    "verify_eaaef_operational_schema",
    "verify_eaaef_operational_connection",
)
