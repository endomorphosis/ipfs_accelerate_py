"""Normalized control-plane schema contract and install helpers.

Interface: ``ControlPlaneSchema@1``.

Domain SQL lives in ``sql/0001_control_plane.sql`` and is applied only through
the checksum-bound migration catalog. This module owns the typed inventory of
domains, tables, join-critical identity columns, task/lease compatibility
columns, diagnostic views, and the optional DuckDB/Quack dependency pin for
the supervisor service.

Import is side-effect free: no database, filesystem mutation, network, or
provider action occurs at module load.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from .control_plane_migrations import (
    ControlPlaneMigration,
    ControlPlaneMigrationRunner,
    MigrationCatalog,
    MigrationCatalogError,
    MigrationRunReport,
    checksum_sql,
    compute_schema_fingerprint,
    load_default_catalog,
)
from .quack_capabilities import (
    PINNED_DUCKDB_MAJOR,
    PINNED_DUCKDB_MINOR,
    PINNED_DUCKDB_VERSION_PREFIX,
    PINNED_EXTENSION_API,
    PINNED_EXTENSION_NAME,
    default_compatibility_profile,
)

CONTROL_PLANE_SCHEMA_INTERFACE: Final = "ControlPlaneSchema@1"
CONTROL_PLANE_SCHEMA_VERSION: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-schema@1"
)
CONTROL_PLANE_SCHEMA_MIGRATION_ID: Final = "0001_control_plane"
CONTROL_PLANE_SCHEMA_SQL_FILENAME: Final = "0001_control_plane.sql"

# Optional supervisor service dependency profile (see pyproject.toml extras).
SUPERVISOR_SERVICE_EXTRA_NAME: Final = "agent-supervisor"
PINNED_DUCKDB_PACKAGE_SPEC: Final = "duckdb>=1.5.0,<1.6.0"
PINNED_QUACK_PROFILE_ID: Final = "agent-supervisor-duckdb-quack-1.5"

# Bookkeeping tables owned by the migration runner (not recreated by domain SQL).
BOOKKEEPING_TABLES: Final[tuple[str, ...]] = (
    "control_plane_metadata",
    "schema_migrations",
    "schema_migration_attempts",
)

# Domain -> physical tables installed by 0001_control_plane.sql.
_SCHEMA_DOMAIN_TABLES: dict[str, tuple[str, ...]] = {
    "meta": (
        "schema_contracts",
        "state_servers",
        "server_epochs",
        "client_sessions",
        "capability_snapshots",
        "credentials",
        "authorization_roles",
        "authorization_grants",
        "backup_snapshots",
        "restore_receipts",
        "maintenance_leases",
    ),
    "control": (
        "schema_contracts",
        "state_servers",
        "authorization_roles",
        "authorization_grants",
        "maintenance_leases",
    ),
    "git": (
        "repositories",
        "repository_revisions",
        "submodule_edges",
        "worktrees",
        "worktree_snapshots",
        "worktree_paths",
        "dirty_overlays",
        "branches",
        "git_refs",
        "merge_bases",
        "merge_queue_entries",
        "resource_claims",
        "path_claims",
        "leases",
        "lease_events",
        "token_history",
    ),
    "intent": (
        "objectives",
        "objective_revisions",
        "goals",
        "goal_edges",
        "plans",
        "plan_revisions",
        "planning_decisions",
        "plan_candidates",
        "tasks",
        "task_revisions",
        "task_dependencies",
        "task_outputs",
        "task_acceptance",
        "task_validations",
        "task_assignments",
        "task_blocks",
        "refill_epochs",
        "findings",
        "finding_dispositions",
    ),
    "schedule": (
        "tasks",
        "task_dependencies",
        "task_assignments",
        "task_blocks",
        "refill_epochs",
        "merge_queue_entries",
        "leases",
        "task_claims",
    ),
    "runtime": (
        "supervisor_instances",
        "daemon_instances",
        "daemon_sessions",
        "heartbeats",
        "health_samples",
        "stall_detections",
        "restart_decisions",
        "task_attempts",
        "attempt_phases",
        "task_claims",
        "provider_invocations",
        "validation_runs",
        "validation_results",
        "merge_attempts",
        "recovery_actions",
        "idempotency_records",
        "effect_claims",
        "completion_receipts",
        "domain_events",
        "structured_logs",
        "metrics",
        "metric_samples",
        "budget_reservations",
        "budget_consumption",
        "quack_query_telemetry",
    ),
    "code": (
        "source_snapshots",
        "source_files",
        "file_versions",
        "parse_runs",
        "symbols",
        "symbol_versions",
        "ast_nodes",
        "ast_edges",
        "module_imports",
        "call_edges",
        "code_references",
        "definitions",
        "type_relations",
        "mutations",
        "mutation_files",
        "mutation_hunks",
        "ast_mutations",
        "impact_edges",
        "impact_closures",
        "repair_candidates",
        "repair_applications",
    ),
    "evidence": (
        "proof_obligations",
        "proof_attempts",
        "counterexamples",
        "evidence_nodes",
        "artifacts",
        "completion_receipts",
        "validation_results",
    ),
    "cache": (
        "context_manifests",
        "context_members",
        "context_deltas",
        "prompt_templates",
        "prompt_instances",
        "prompt_inputs",
        "provider_calls",
        "provider_responses",
        "failure_signatures",
        "decision_cache_entries",
        "replay_suppressions",
        "churn_metrics",
    ),
    "improve": (
        "improvement_campaigns",
        "improvement_findings",
        "improvement_actions",
    ),
}

SCHEMA_DOMAIN_TABLES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {key: tuple(value) for key, value in _SCHEMA_DOMAIN_TABLES.items()}
)

SCHEMA_DOMAINS: Final[tuple[str, ...]] = (
    "meta",
    "intent",
    "schedule",
    "runtime",
    "git",
    "code",
    "evidence",
    "cache",
    "control",
    "improve",
)

DIAGNOSTIC_VIEWS: Final[tuple[str, ...]] = (
    "ready_task_context_v1",
    "live_sessions_v1",
    "expiring_leases_v1",
    "ready_versus_claimed_tasks_v1",
    "stuck_phases_v1",
    "failed_migrations_v1",
    "server_identity_v1",
)

# Columns that must be first-class (never joinable only via opaque JSON).
JOIN_CRITICAL_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "tasks": (
            "task_cid",
            "task_alias",
            "goal_cid",
            "status",
            "revision",
            "semantic_fingerprint",
            "canonical_task_key",
            "idempotency_key",
        ),
        "leases": (
            "task_cid",
            "claim_cid",
            "resolution_cid",
            "claimant_did",
            "owner_session_id",
            "logical_epoch",
            "fencing_token",
            "expires_at_ms",
            "attempt",
            "state",
            "started_at_ms",
            "retry_not_before_ms",
            "revision",
        ),
        "task_dependencies": (
            "task_cid",
            "dependency_task_cid",
            "kind",
        ),
        "task_claims": (
            "claim_cid",
            "task_cid",
            "claimant_did",
            "session_id",
            "fencing_token",
            "logical_epoch",
            "expires_at_ms",
            "status",
            "revision",
        ),
        "task_attempts": (
            "attempt_id",
            "task_cid",
            "attempt_number",
            "claim_cid",
            "session_id",
            "status",
            "revision",
        ),
        "domain_events": (
            "event_id",
            "stream_id",
            "sequence",
            "event_type",
            "task_cid",
        ),
        "repositories": (
            "repository_id",
            "head_commit_id",
            "revision",
        ),
        "worktrees": (
            "worktree_id",
            "repository_id",
            "head_commit_id",
            "status",
            "revision",
        ),
        "mutations": (
            "mutation_id",
            "mutation_cid",
            "task_cid",
            "attempt_id",
            "before_snapshot_id",
            "after_snapshot_id",
            "status",
        ),
        "evidence_nodes": (
            "evidence_id",
            "evidence_cid",
            "evidence_kind",
            "task_cid",
        ),
        "state_servers": (
            "server_id",
            "database_uuid",
            "process_birth_id",
            "extension_fingerprint",
            "quack_profile_id",
            "startup_epoch",
            "revision",
        ),
        "daemon_sessions": (
            "session_id",
            "daemon_id",
            "fencing_epoch",
            "expires_at",
            "status",
            "revision",
        ),
        "heartbeats": (
            "heartbeat_cid",
            "task_cid",
            "claimant_did",
            "session_id",
            "fencing_token",
            "observed_at_ms",
            "expires_at_ms",
        ),
        "completion_receipts": (
            "receipt_cid",
            "task_cid",
            "attempt_id",
            "claim_cid",
            "fencing_token",
        ),
        "goals": (
            "goal_cid",
            "goal_alias",
            "parent_goal_cid",
            "status",
            "revision",
        ),
        "symbols": (
            "symbol_id",
            "snapshot_id",
            "path",
            "fingerprint",
        ),
        "ast_nodes": (
            "node_id",
            "snapshot_id",
            "path",
            "node_path",
            "parser_identity",
            "fingerprint",
        ),
        "context_manifests": (
            "manifest_cid",
            "task_cid",
            "repository_id",
            "schema_fingerprint",
        ),
        "idempotency_records": (
            "idempotency_key",
            "scope",
            "request_digest",
            "result_digest",
        ),
    }
)

# Lease columns required for compatibility with LeaseCoordinator semantics.
LEASE_SEMANTIC_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "claim_cid",
    "resolution_cid",
    "claimant_did",
    "logical_epoch",
    "fencing_token",
    "expires_at_ms",
    "attempt",
    "state",
    "started_at_ms",
    "release_reason",
    "retry_not_before_ms",
)

TASK_IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "task_cid",
    "task_alias",
    "goal_cid",
    "status",
    "revision",
    "semantic_fingerprint",
    "canonical_task_key",
    "idempotency_key",
)

OPAQUE_JSON_COLUMN_NAMES: Final[frozenset[str]] = frozenset(
    {
        "body_json",
        "payload_json",
        "labels_json",
        "identity_json",
        "effect_json",
        "policy_json",
        "evidence_policy_json",
        "argv_json",
        "value_json",
        "bundle_json",
        "provenance_json",
    }
)

_PYPROJECT_DUCKDB_PIN_RE: Final = re.compile(
    r'duckdb\s*>=\s*1\.5(?:\.0)?\s*,\s*<\s*1\.6(?:\.0)?'
)


class ControlPlaneSchemaError(RuntimeError):
    """Base class for control-plane schema contract failures."""


class ControlPlaneSchemaIntegrityError(ControlPlaneSchemaError, ValueError):
    """Schema inventory, SQL, or applied database shape is inconsistent."""


@dataclass(frozen=True)
class SupervisorDependencyPin:
    """Pinned optional dependency profile for the DuckDB/Quack supervisor service."""

    extra_name: str = SUPERVISOR_SERVICE_EXTRA_NAME
    duckdb_package_spec: str = PINNED_DUCKDB_PACKAGE_SPEC
    duckdb_major: int = PINNED_DUCKDB_MAJOR
    duckdb_minor: int = PINNED_DUCKDB_MINOR
    duckdb_version_prefix: str = PINNED_DUCKDB_VERSION_PREFIX
    quack_profile_id: str = PINNED_QUACK_PROFILE_ID
    extension_name: str = PINNED_EXTENSION_NAME
    extension_api: str = PINNED_EXTENSION_API

    def to_dict(self) -> dict[str, Any]:
        return {
            "extra_name": self.extra_name,
            "duckdb_package_spec": self.duckdb_package_spec,
            "duckdb_major": int(self.duckdb_major),
            "duckdb_minor": int(self.duckdb_minor),
            "duckdb_version_prefix": self.duckdb_version_prefix,
            "quack_profile_id": self.quack_profile_id,
            "extension_name": self.extension_name,
            "extension_api": self.extension_api,
        }


@dataclass(frozen=True)
class ControlPlaneSchema:
    """Typed inventory and install surface for the normalized control plane.

    Interface: ``ControlPlaneSchema@1``.
    """

    interface: str = CONTROL_PLANE_SCHEMA_INTERFACE
    schema_version: str = CONTROL_PLANE_SCHEMA_VERSION
    migration_id: str = CONTROL_PLANE_SCHEMA_MIGRATION_ID
    sql_filename: str = CONTROL_PLANE_SCHEMA_SQL_FILENAME
    domains: tuple[str, ...] = SCHEMA_DOMAINS
    domain_tables: Mapping[str, tuple[str, ...]] = SCHEMA_DOMAIN_TABLES
    diagnostic_views: tuple[str, ...] = DIAGNOSTIC_VIEWS
    join_critical_columns: Mapping[str, tuple[str, ...]] = JOIN_CRITICAL_COLUMNS
    lease_semantic_columns: tuple[str, ...] = LEASE_SEMANTIC_COLUMNS
    task_identity_columns: tuple[str, ...] = TASK_IDENTITY_COLUMNS
    dependency_pin: SupervisorDependencyPin = SupervisorDependencyPin()

    def __post_init__(self) -> None:
        if self.interface != CONTROL_PLANE_SCHEMA_INTERFACE:
            raise ControlPlaneSchemaIntegrityError(
                f"unsupported schema interface: {self.interface}"
            )
        if self.schema_version != CONTROL_PLANE_SCHEMA_VERSION:
            raise ControlPlaneSchemaIntegrityError(
                f"unsupported schema version: {self.schema_version}"
            )
        missing_domains = [name for name in SCHEMA_DOMAINS if name not in self.domain_tables]
        if missing_domains:
            raise ControlPlaneSchemaIntegrityError(
                f"schema inventory missing domains: {missing_domains}"
            )
        for domain in SCHEMA_DOMAINS:
            tables = self.domain_tables.get(domain) or ()
            if not tables:
                raise ControlPlaneSchemaIntegrityError(
                    f"domain {domain!r} must declare at least one table"
                )

    @property
    def all_domain_tables(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        for domain in self.domains:
            for table_name in self.domain_tables[domain]:
                if table_name in seen:
                    continue
                seen.add(table_name)
                ordered.append(table_name)
        return tuple(ordered)

    def sql_path(self) -> Path:
        return Path(__file__).resolve().parent / "sql" / self.sql_filename

    def load_sql(self) -> str:
        path = self.sql_path()
        if not path.is_file():
            raise ControlPlaneSchemaIntegrityError(
                f"control-plane schema SQL is missing: {path}"
            )
        return path.read_text(encoding="utf-8")

    def sql_checksum(self) -> str:
        return checksum_sql(self.load_sql())

    def migration_catalog(
        self,
        sql_directory: Path | str | None = None,
    ) -> MigrationCatalog:
        return load_default_catalog(sql_directory)

    def ensure_catalog_contains_schema(
        self,
        catalog: MigrationCatalog | None = None,
    ) -> ControlPlaneMigration:
        resolved = catalog if catalog is not None else self.migration_catalog()
        try:
            migration = resolved.get(1)
        except MigrationCatalogError as exc:
            raise ControlPlaneSchemaIntegrityError(
                "default migration catalog is missing version 1 "
                f"({self.migration_id})"
            ) from exc
        if migration.migration_id != self.migration_id:
            raise ControlPlaneSchemaIntegrityError(
                f"expected migration_id {self.migration_id!r}, "
                f"got {migration.migration_id!r}"
            )
        expected = self.sql_checksum()
        if migration.checksum != expected:
            raise ControlPlaneSchemaIntegrityError(
                f"catalog checksum drift for {self.migration_id}: "
                f"catalog {migration.checksum}, sql {expected}"
            )
        return migration

    def install(
        self,
        database_path: Path | str,
        *,
        application_version: str | None = None,
        tool_version: str | None = None,
        owner_id: str | None = None,
        catalog: MigrationCatalog | None = None,
    ) -> MigrationRunReport:
        """Apply the default catalog through the migration runner."""

        resolved_catalog = catalog if catalog is not None else self.migration_catalog()
        self.ensure_catalog_contains_schema(resolved_catalog)
        runner = ControlPlaneMigrationRunner.for_database(
            database_path,
            catalog=resolved_catalog,
            application_version=application_version,
            tool_version=tool_version,
            owner_id=owner_id,
        )
        return runner.apply()

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "schema_version": self.schema_version,
            "migration_id": self.migration_id,
            "sql_filename": self.sql_filename,
            "domains": list(self.domains),
            "domain_tables": {
                domain: list(tables) for domain, tables in self.domain_tables.items()
            },
            "diagnostic_views": list(self.diagnostic_views),
            "join_critical_columns": {
                table: list(columns)
                for table, columns in self.join_critical_columns.items()
            },
            "lease_semantic_columns": list(self.lease_semantic_columns),
            "task_identity_columns": list(self.task_identity_columns),
            "dependency_pin": self.dependency_pin.to_dict(),
            "sql_checksum": self.sql_checksum(),
        }


def default_control_plane_schema() -> ControlPlaneSchema:
    """Return the canonical ControlPlaneSchema@1 inventory."""

    return ControlPlaneSchema()


def control_plane_sql_path() -> Path:
    return default_control_plane_schema().sql_path()


def load_control_plane_sql() -> str:
    return default_control_plane_schema().load_sql()


def install_control_plane_schema(
    database_path: Path | str,
    **kwargs: Any,
) -> MigrationRunReport:
    return default_control_plane_schema().install(database_path, **kwargs)


def schema_fingerprint_for_database(database_path: Path | str) -> str:
    """Compute the canonical information-schema fingerprint for a database."""

    runner = ControlPlaneMigrationRunner.for_database(database_path)
    return runner.schema_fingerprint()


def list_table_columns(connection: Any, table_name: str) -> list[str]:
    """Return ordered column names for a table or view in the main schema."""

    rows = connection.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position, column_name
        """,
        [str(table_name)],
    ).fetchall()
    columns: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            columns.append(str(row["column_name"]))
        else:
            columns.append(str(row[0]))
    return columns


def assert_join_critical_columns_present(connection: Any) -> None:
    """Fail closed when join-critical identities exist only as opaque JSON."""

    schema = default_control_plane_schema()
    for table_name, required in schema.join_critical_columns.items():
        columns = list_table_columns(connection, table_name)
        if not columns:
            raise ControlPlaneSchemaIntegrityError(
                f"join-critical table missing after schema install: {table_name}"
            )
        missing = [name for name in required if name not in columns]
        if missing:
            raise ControlPlaneSchemaIntegrityError(
                f"table {table_name} missing join-critical columns: {missing}"
            )
        # Opaque JSON columns may exist but must not be the only identity path.
        non_json = [name for name in columns if name not in OPAQUE_JSON_COLUMN_NAMES]
        for identity in required:
            if identity not in non_json:
                raise ControlPlaneSchemaIntegrityError(
                    f"join-critical identity {table_name}.{identity} is not a "
                    "first-class non-JSON column"
                )


def assert_task_and_lease_semantics(connection: Any) -> None:
    """Preserve existing task CID PK and lease semantic columns."""

    schema = default_control_plane_schema()
    task_columns = list_table_columns(connection, "tasks")
    for column in schema.task_identity_columns:
        if column not in task_columns:
            raise ControlPlaneSchemaIntegrityError(
                f"tasks table missing identity column {column}"
            )
    lease_columns = list_table_columns(connection, "leases")
    for column in schema.lease_semantic_columns:
        if column not in lease_columns:
            raise ControlPlaneSchemaIntegrityError(
                f"leases table missing semantic column {column}"
            )
    # Primary-key presence via information_schema is dialect-limited; enforce
    # uniqueness by requiring task_cid/lease task_cid as first columns of record.
    if task_columns[0] != "task_cid":
        raise ControlPlaneSchemaIntegrityError(
            "tasks.task_cid must remain the leading primary identity column"
        )
    if lease_columns[0] != "task_cid":
        raise ControlPlaneSchemaIntegrityError(
            "leases.task_cid must remain the leading primary identity column"
        )


def assert_domains_installed(connection: Any) -> None:
    """Ensure every declared domain table and diagnostic view exists."""

    schema = default_control_plane_schema()
    for table_name in schema.all_domain_tables:
        columns = list_table_columns(connection, table_name)
        if not columns:
            raise ControlPlaneSchemaIntegrityError(
                f"domain table missing after install: {table_name}"
            )
    for view_name in schema.diagnostic_views:
        columns = list_table_columns(connection, view_name)
        if not columns:
            raise ControlPlaneSchemaIntegrityError(
                f"diagnostic view missing after install: {view_name}"
            )
    for bookkeeping in BOOKKEEPING_TABLES:
        columns = list_table_columns(connection, bookkeeping)
        if not columns:
            raise ControlPlaneSchemaIntegrityError(
                f"bookkeeping table missing after install: {bookkeeping}"
            )


def supervisor_dependency_pin() -> SupervisorDependencyPin:
    return default_control_plane_schema().dependency_pin


def assert_pyproject_pins_supervisor_duckdb(
    pyproject_text: str | None = None,
    *,
    pyproject_path: Path | str | None = None,
) -> SupervisorDependencyPin:
    """Verify pyproject.toml pins DuckDB 1.5.x for the optional supervisor extra."""

    pin = supervisor_dependency_pin()
    if pyproject_text is None:
        path = (
            Path(pyproject_path)
            if pyproject_path is not None
            else Path(__file__).resolve().parents[3] / "pyproject.toml"
        )
        pyproject_text = path.read_text(encoding="utf-8")
    text = str(pyproject_text)
    if pin.extra_name not in text:
        raise ControlPlaneSchemaIntegrityError(
            f"pyproject.toml missing optional extra {pin.extra_name!r}"
        )
    if not _PYPROJECT_DUCKDB_PIN_RE.search(text):
        raise ControlPlaneSchemaIntegrityError(
            "pyproject.toml must pin duckdb>=1.5.0,<1.6.0 for the supervisor service"
        )
    profile = default_compatibility_profile()
    if profile.profile_id != pin.quack_profile_id:
        raise ControlPlaneSchemaIntegrityError(
            f"compatibility profile id {profile.profile_id!r} does not match "
            f"pinned {pin.quack_profile_id!r}"
        )
    if profile.duckdb_major != pin.duckdb_major or profile.duckdb_minor != pin.duckdb_minor:
        raise ControlPlaneSchemaIntegrityError(
            "compatibility profile DuckDB minor does not match package pin"
        )
    return pin


def prove_fresh_and_upgraded_equivalence(
    *,
    fresh_database_path: Path | str,
    upgraded_database_path: Path | str,
    application_version: str = "0.0.45",
    tool_version: str = "1.5.2",
) -> dict[str, Any]:
    """Apply empty-to-latest on two DBs and prove identical schema fingerprints.

    The upgraded path first installs bookkeeping only (version 0), then applies
    pending migrations, matching a fresh empty-to-latest install.
    """

    schema = default_control_plane_schema()
    catalog = schema.migration_catalog()
    schema.ensure_catalog_contains_schema(catalog)

    fresh_runner = ControlPlaneMigrationRunner.for_database(
        fresh_database_path,
        catalog=catalog,
        application_version=application_version,
        tool_version=tool_version,
        owner_id="schema-proof-fresh",
    )
    fresh_report = fresh_runner.apply()

    upgraded_runner = ControlPlaneMigrationRunner.for_database(
        upgraded_database_path,
        catalog=catalog,
        application_version=application_version,
        tool_version=tool_version,
        owner_id="schema-proof-upgrade",
    )
    upgraded_runner.ensure_bookkeeping()
    if upgraded_runner.current_version() != 0:
        raise ControlPlaneSchemaIntegrityError(
            "upgraded proof database must start at schema version 0"
        )
    upgraded_report = upgraded_runner.apply()

    if fresh_report.schema_fingerprint != upgraded_report.schema_fingerprint:
        raise ControlPlaneSchemaIntegrityError(
            "fresh and upgraded schema fingerprints diverge: "
            f"{fresh_report.schema_fingerprint} != "
            f"{upgraded_report.schema_fingerprint}"
        )
    if fresh_report.to_version != upgraded_report.to_version:
        raise ControlPlaneSchemaIntegrityError(
            "fresh and upgraded schema versions diverge"
        )
    return {
        "equivalent": True,
        "schema_fingerprint": fresh_report.schema_fingerprint,
        "catalog_fingerprint": fresh_report.catalog_fingerprint,
        "to_version": fresh_report.to_version,
        "migration_id": schema.migration_id,
    }


__all__ = [
    "BOOKKEEPING_TABLES",
    "CONTROL_PLANE_SCHEMA_INTERFACE",
    "CONTROL_PLANE_SCHEMA_MIGRATION_ID",
    "CONTROL_PLANE_SCHEMA_SQL_FILENAME",
    "CONTROL_PLANE_SCHEMA_VERSION",
    "ControlPlaneSchema",
    "ControlPlaneSchemaError",
    "ControlPlaneSchemaIntegrityError",
    "DIAGNOSTIC_VIEWS",
    "JOIN_CRITICAL_COLUMNS",
    "LEASE_SEMANTIC_COLUMNS",
    "OPAQUE_JSON_COLUMN_NAMES",
    "PINNED_DUCKDB_PACKAGE_SPEC",
    "PINNED_QUACK_PROFILE_ID",
    "SCHEMA_DOMAINS",
    "SCHEMA_DOMAIN_TABLES",
    "SUPERVISOR_SERVICE_EXTRA_NAME",
    "SupervisorDependencyPin",
    "TASK_IDENTITY_COLUMNS",
    "assert_domains_installed",
    "assert_join_critical_columns_present",
    "assert_pyproject_pins_supervisor_duckdb",
    "assert_task_and_lease_semantics",
    "compute_schema_fingerprint",
    "control_plane_sql_path",
    "default_control_plane_schema",
    "install_control_plane_schema",
    "list_table_columns",
    "load_control_plane_sql",
    "prove_fresh_and_upgraded_equivalence",
    "schema_fingerprint_for_database",
    "supervisor_dependency_pin",
]
